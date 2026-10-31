"""Business logic behind the HTTP routes (chat orchestration, SSE, exports, cleanup).

Routes in ``app.py`` stay thin; everything stateful lives here so it can be tested
without HTTP.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

from agent import AgentRuntime
from config import Settings
from database import Database, DEFAULT_TITLE
from exceptions import ConflictError, NotFoundError, ServiceNotConfiguredError, UpstreamError, ValidationFailed
from llm import build_chat_model
from rag import RagService
from schemas import ChatRequest
from security import redact

log = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15.0
MAX_STORED_SOURCES = 20


def sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatService:
    def __init__(self, settings: Settings, db: Database, rag: RagService, runtime: AgentRuntime) -> None:
        self.settings = settings
        self.db = db
        self.rag = rag
        self.runtime = runtime

    # ------------------------------------------------------------ preferences
    def effective_prefs(self, scope_id: str) -> dict[str, Any]:
        pref = self.db.get_preferences(scope_id)
        s = self.settings
        model = pref.model if pref and pref.model in s.model_options else s.groq_model
        return {
            "model": model,
            "theme": pref.theme if pref else "system",
            "temperature": pref.temperature if pref else s.default_temperature,
            "top_k": (pref.top_k if pref and pref.top_k else s.top_k),
            "response_style": pref.response_style if pref else "balanced",
        }

    # ------------------------------------------------------------------ chat
    async def stream_chat(self, scope_id: str, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Yield UI events for one turn. Persists the user message up front and the
        assistant message exactly once (full answer, or the partial text if interrupted)."""
        prefs = self.effective_prefs(scope_id)
        model = req.model or prefs["model"]
        if model not in self.settings.model_options:
            raise ValidationFailed("Unknown model. Choose one from /api/models.")
        temperature = req.temperature if req.temperature is not None else prefs["temperature"]
        top_k = req.top_k or prefs["top_k"]
        style = req.response_style or prefs["response_style"]

        # Fail fast (before touching the database) if the LLM provider is not configured.
        build_chat_model(self.settings, model, temperature)

        # ---- conversation + user message
        if req.thread_id:
            conv = self.db.get_conversation(scope_id, req.thread_id)  # NotFound if not owned
        else:
            if req.mode != "send":
                raise ValidationFailed("thread_id is required to regenerate or edit.")
            conv = self.db.create_conversation(scope_id)
        thread_id = conv.thread_id

        fork_from = None
        user_text = req.message
        if req.mode in ("regenerate", "edit"):
            msgs = self.db.list_messages(scope_id, thread_id)
            last_user = next((m for m in reversed(msgs) if m.role == "user"), None)
            if last_user is None:
                raise ValidationFailed("There is no earlier message to regenerate.")
            if req.mode == "regenerate":
                user_text = last_user.content
            self.db.delete_messages_from(scope_id, thread_id, last_user.message_id, inclusive=True)
            fork_from = await self.runtime.rewind_config(thread_id)
        elif req.client_message_id and self.db.message_exists(scope_id, thread_id, req.client_message_id):
            raise ConflictError("This message was already submitted.")

        user_msg = self.db.add_message(scope_id, thread_id, "user", user_text,
                                       message_id=req.client_message_id if req.mode == "send" else None)
        title = self.db.get_conversation(scope_id, thread_id).title
        yield {"type": "meta", "thread_id": thread_id, "user_message_id": user_msg.message_id, "title": title,
               "model": model}

        partial = ""
        persisted = False
        try:
            async with asyncio.timeout(self.settings.chat_timeout_seconds):
                async for event in self.runtime.run_turn(
                    scope_id=scope_id, thread_id=thread_id, user_text=user_text, model=model,
                    temperature=temperature, top_k=top_k, response_style=style,
                    weather_units=self.settings.weather_units, fork_from=fork_from,
                ):
                    if event["type"] == "delta":
                        partial += event["text"]
                    elif event["type"] == "reset":
                        partial = ""
                    elif event["type"] == "done":
                        content = event["content"] or partial.strip() or \
                            "I wasn't able to produce a response. Please try again."
                        meta = {"tools": [{"name": t["name"], "ok": t["ok"]} for t in event["tools"]],
                                "sources": event["sources"][:MAX_STORED_SOURCES]}
                        if event["status"] == "step_limit":
                            meta["step_limit"] = True
                        usage = event["usage"] or {}
                        saved = self.db.add_message(
                            scope_id, thread_id, "assistant", content, model=model,
                            prompt_tokens=usage.get("input_tokens") or None,
                            completion_tokens=usage.get("output_tokens") or None,
                            latency_ms=event["latency_ms"], meta=meta)
                        persisted = True
                        event = {**event, "content": content, "message_id": saved.message_id,
                                 "title": self.db.get_conversation(scope_id, thread_id).title,
                                 "thread_id": thread_id}
                        event.pop("tools", None)
                        event["tools"] = meta["tools"]
                    yield event
        except TimeoutError:
            raise UpstreamError("The request took too long and was stopped. Please try again.") from None
        finally:
            # Client disconnected / cancelled / failed mid-answer: keep what the user already saw.
            if not persisted and partial.strip():
                try:
                    self.db.add_message(scope_id, thread_id, "assistant", partial.strip(), model=model,
                                        meta={"interrupted": True})
                except Exception:  # pragma: no cover - best effort
                    log.warning("could not persist partial answer")

    async def chat_once(self, scope_id: str, req: ChatRequest) -> dict[str, Any]:
        """Non-streaming variant: run the same pipeline and return the final result."""
        result: dict[str, Any] | None = None
        meta: dict[str, Any] = {}
        async for event in self.stream_chat(scope_id, req):
            if event["type"] == "meta":
                meta = event
            elif event["type"] == "done":
                result = event
        if result is None:  # pragma: no cover
            raise UpstreamError("No response was produced.")
        return {"thread_id": meta["thread_id"], **result}

    # ------------------------------------------------------------------- SSE
    async def open_stream(self, scope_id: str, req: ChatRequest) -> tuple[dict[str, Any], AsyncIterator[dict[str, Any]]]:
        """Run validation + persistence of the user message and return ``(meta_event, rest)``.

        Raising here (before any bytes are streamed) lets the route answer with a proper
        HTTP status (404 unknown conversation, 422 bad model, 503 LLM not configured...)."""
        agen = self.stream_chat(scope_id, req).__aiter__()
        try:
            first = await agen.__anext__()
        except StopAsyncIteration:  # pragma: no cover
            raise UpstreamError("No response was produced.") from None
        return first, agen

    async def sse_frames(self, first: dict[str, Any], agen: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
        """SSE frames with heartbeats; always terminates with a ``done`` event."""
        first = dict(first)
        yield sse(first.pop("type"), first)
        finished_ok = False
        pending: asyncio.Task | None = None
        try:
            while True:
                pending = asyncio.ensure_future(agen.__anext__())
                while True:
                    done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_SECONDS)
                    if done:
                        break
                    yield ": keep-alive\n\n"
                try:
                    event = dict(pending.result())
                except StopAsyncIteration:
                    break
                finally:
                    pending = None
                kind = event.pop("type")
                yield sse(kind, event)
                if kind == "done":
                    finished_ok = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            safe = getattr(exc, "message", None)
            code = getattr(exc, "code", "internal_error")
            if safe is None:
                log.error("stream failed: %s", type(exc).__name__)
                safe = "Something went wrong while generating the answer. Please retry."
            yield sse("error", {"error": redact(str(safe)), "code": code, "retryable": True})
            yield sse("done", {"status": "error"})
            return
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
            await agen.aclose()
        if not finished_ok:  # pragma: no cover - generator ended without done
            yield sse("done", {"status": "error"})

    # ----------------------------------------------------------- housekeeping
    async def delete_conversation(self, scope_id: str, thread_id: str) -> None:
        """Retention policy: deleting a conversation removes its messages, LangGraph checkpoints,
        uploaded documents (files + FAISS vectors + chunks). Persistent memories are kept
        (they belong to the browser scope, not the conversation) until deleted explicitly."""
        self.db.get_conversation(scope_id, thread_id)
        await asyncio.to_thread(self.rag.delete_documents_for_thread, scope_id, thread_id)
        self.db.delete_conversation(scope_id, thread_id)
        await self.runtime.delete_thread(thread_id)

    async def clear_conversation(self, scope_id: str, thread_id: str) -> int:
        """Remove messages + checkpoints but keep the conversation and its documents."""
        n = self.db.clear_messages(scope_id, thread_id)
        await self.runtime.delete_thread(thread_id)
        return n

    def export_conversation(self, scope_id: str, thread_id: str, fmt: str) -> tuple[str, str]:
        conv = self.db.get_conversation(scope_id, thread_id)
        msgs = self.db.list_messages(scope_id, thread_id, roles=("user", "assistant"))
        names = {"user": "You", "assistant": "AKGPT"}
        if fmt == "md":
            parts = [f"# {conv.title}\n"]
            for m in msgs:
                parts.append(f"## {names[m.role]}\n\n{m.content}\n")
            return "\n".join(parts), "text/markdown"
        parts = [conv.title, "=" * len(conv.title), ""]
        for m in msgs:
            parts.append(f"{names[m.role]}:\n{m.content}\n")
        return "\n".join(parts), "text/plain"


def new_thread_id() -> str:
    return str(uuid.uuid4())

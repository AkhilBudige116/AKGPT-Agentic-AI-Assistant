"""AKGPT - FastAPI application.

AKGPT is a **single-user / local prototype**: there are no accounts. Each browser gets an
anonymous random session cookie; conversations, documents, memories and settings are bound
to a hash of that cookie, and every endpoint checks ownership. A thread id sent by the
browser is never treated as proof of authorisation.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
from contextlib import asynccontextmanager
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent import AgentRuntime
from config import Settings, configure_langsmith, get_settings
from database import Database
from exceptions import (
    AKGPTError,
    ConfigurationError,
    ForbiddenError,
    NotFoundError,
    PayloadTooLargeError,
    RateLimitedError,
    ValidationFailed,
)
from logging_config import new_request_id, request_id_var, setup_logging
from rag import RagService, stage_upload
from schemas import (
    ChatRequest,
    ConversationDetail,
    ConversationList,
    ConversationOut,
    CreateConversationRequest,
    DocumentList,
    DocumentOut,
    HealthOut,
    MemoryList,
    MemoryUpdate,
    ModelsOut,
    RenameRequest,
    SettingsOut,
    SettingsUpdate,
    conversation_out,
    document_out,
    memory_out,
    message_out,
)
from security import (
    ALLOWED_EXTENSIONS,
    THREAD_ID_RE,
    RateLimiter,
    is_valid_scope_id,
    validate_thread_id,
)
from services import ChatService
from tools import configure_tools

log = logging.getLogger("akgpt")
BASE_DIR = Path(__file__).resolve().parent
VERSION = "1.0.0"
SESSION_COOKIE = "akgpt_session"
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
RATE_LIMITED_PATHS = {"/api/chat", "/api/chat/stream", "/api/upload"}

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
       "form-action 'self'; frame-ancestors 'none'")


def scope_from_token(token: str) -> str:
    """Database scope id = urlsafe-base64(sha256(cookie token)); the DB never stores the bearer token."""
    return base64.urlsafe_b64encode(hashlib.sha256(token.encode()).digest()).rstrip(b"=").decode()


# =================================================================== ASGI middleware
class _BodyTooLarge(Exception):
    pass


class SecurityMiddleware:
    """Request id, session cookie, origin/CSRF check, body-size limit, rate limit, security headers."""

    def __init__(self, app, settings: Settings, limiter: RateLimiter) -> None:
        self.app = app
        self.settings = settings
        self.limiter = limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        method, path = scope["method"], scope["path"]
        rid = headers.get("x-request-id", "")
        rid = rid if rid.isalnum() and 6 <= len(rid) <= 32 else new_request_id()
        request_id_var.set(rid)

        # ---- session cookie -> scope id
        cookies = SimpleCookie()
        try:
            cookies.load(headers.get("cookie", ""))
        except Exception:
            cookies = SimpleCookie()
        token = cookies[SESSION_COOKIE].value if SESSION_COOKIE in cookies else ""
        new_cookie = None
        if not is_valid_scope_id(token):
            token = secrets.token_urlsafe(32)
            new_cookie = token
        scope_id = scope_from_token(token)
        scope.setdefault("state", {})["scope_id"] = scope_id
        scope["state"]["request_id"] = rid

        def respond_headers(extra: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
            extra = [(k, v) for k, v in extra if k.lower() not in (b"x-request-id",)]
            extra += [
                (b"x-request-id", rid.encode()),
                (b"x-content-type-options", b"nosniff"),
                (b"x-frame-options", b"DENY"),
                (b"referrer-policy", b"same-origin"),
                (b"content-security-policy", CSP.encode()),
            ]
            if path.startswith("/api/") or path == "/health":
                extra.append((b"cache-control", b"no-store"))
            if new_cookie:
                attrs = f"{SESSION_COOKIE}={new_cookie}; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000"
                if self.settings.cookie_secure:
                    attrs += "; Secure"
                extra.append((b"set-cookie", attrs.encode()))
            return extra

        async def reject(exc: AKGPTError) -> None:
            body = json.dumps({"error": exc.message, "code": exc.code, "request_id": rid}).encode()
            hdrs = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
            hdrs += [(k.lower().encode(), v.encode()) for k, v in exc.headers.items()]
            await send({"type": "http.response.start", "status": exc.status_code, "headers": respond_headers(hdrs)})
            await send({"type": "http.response.body", "body": body})

        # ---- CSRF-style origin check for state-changing requests
        if method in UNSAFE_METHODS:
            origin = headers.get("origin")
            if headers.get("sec-fetch-site") == "cross-site" or (origin and not self._origin_ok(origin, headers)):
                await reject(ForbiddenError("Cross-origin request blocked."))
                return

        # ---- rate limit (keyed by session)
        if method == "POST" and path in RATE_LIMITED_PATHS:
            try:
                self.limiter.check(scope_id)
            except RateLimitedError as exc:
                await reject(exc)
                return

        # ---- body size limits (Content-Length fast path + streaming counter)
        limit = self.settings.max_upload_bytes + 1024 * 1024 if path == "/api/upload" else self.settings.max_request_bytes
        declared = headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            await reject(PayloadTooLargeError("Request body is too large."))
            return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge()
            return message

        started = False

        async def wrapped_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message = {**message, "headers": respond_headers(list(message.get("headers", [])))}
            await send(message)

        try:
            await self.app(scope, limited_receive, wrapped_send)
        except _BodyTooLarge:
            if not started:
                await reject(PayloadTooLargeError("Request body is too large."))

    def _origin_ok(self, origin: str, headers: dict[str, str]) -> bool:
        origin = origin.rstrip("/")
        if origin in self.settings.cors_origins:
            return True
        host = headers.get("host", "")
        return bool(host) and urlparse(origin).netloc == host


# ===================================================================== app factory
def create_app(settings: Settings | None = None, *, model_factory=None, transport: httpx.AsyncBaseTransport | None = None,
               embedder=None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, json_logs=settings.is_production)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for warning in settings.startup_warnings():
            log.warning(warning)
        tracing = configure_langsmith(settings)
        log.info("AKGPT %s starting (env=%s, langsmith_tracing=%s)", VERSION, settings.app_env, tracing)
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        Path(settings.checkpoint_db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        db = Database(settings.database_url)
        db.init_schema()
        rag = RagService(settings, db, embedder=embedder)
        try:
            status = await asyncio.to_thread(rag.initialise)
            log.info("vector index ready (vectors=%s, compatible=%s)", status.ntotal, status.compatible)
        except Exception as exc:  # keep serving chat even if the index is unusable
            log.error("vector index failed to initialise: %s", type(exc).__name__)
        configure_tools(settings, db, rag, transport=transport)
        async with AsyncSqliteSaver.from_conn_string(str(Path(settings.checkpoint_db_path).expanduser())) as saver:
            runtime = AgentRuntime(settings, rag, saver, model_factory=model_factory)
            app.state.db, app.state.rag, app.state.runtime = db, rag, runtime
            app.state.chat = ChatService(settings, db, rag, runtime)
            yield
        db.dispose()

    app = FastAPI(
        title="AKGPT", version=VERSION, lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None, openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.settings = settings
    app.state.limiter = limiter

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Content-Type", "X-Request-ID"])
    app.add_middleware(SecurityMiddleware, settings=settings, limiter=limiter)  # outermost
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    # ------------------------------------------------------------ error handlers
    def err(status: int, message: str, code: str, request: Request, headers: dict | None = None) -> JSONResponse:
        return JSONResponse({"error": message, "code": code, "request_id": getattr(request.state, "request_id", None)},
                            status_code=status, headers=headers)

    @app.exception_handler(AKGPTError)
    async def _akgpt(request: Request, exc: AKGPTError):
        if exc.status_code >= 500:
            log.error("request failed: %s %s", type(exc).__name__, exc.message)
        return err(exc.status_code, exc.message, exc.code, request, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        # Report field names + messages only - never echo submitted values.
        fields = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors()[:5])
        return err(422, f"Invalid request: {fields}", "validation_error", request)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return err(exc.status_code, "Not found." if exc.status_code == 404 else str(exc.detail), code, request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.error("unhandled error: %s", type(exc).__name__)
        return err(500, "Internal server error.", "internal_error", request)

    # ------------------------------------------------------------------ helpers
    def scope_id(request: Request) -> str:
        return request.state.scope_id

    def chat(request: Request) -> ChatService:
        return request.app.state.chat

    def db_of(request: Request) -> Database:
        return request.app.state.db

    def tid(value: str) -> str:
        return validate_thread_id(value)

    def settings_out(request: Request) -> SettingsOut:
        prefs = chat(request).effective_prefs(scope_id(request))
        return SettingsOut(
            **prefs, available_models=settings.model_options, max_upload_mb=settings.max_upload_size_mb,
            supported_extensions=sorted(ALLOWED_EXTENSIONS), integrations=settings.key_status())

    # ------------------------------------------------------------------- routes
    @app.get("/", include_in_schema=False)
    async def home(request: Request):
        return templates.TemplateResponse(request=request, name="index.html", context={
            "app_name": "AKGPT", "version": VERSION})

    @app.get("/health", response_model=HealthOut)
    async def health(request: Request):
        db_ok = await asyncio.to_thread(request.app.state.db.ping) if hasattr(request.app.state, "db") else False
        rag = getattr(request.app.state, "rag", None)
        idx = rag.status() if rag else None
        index_ok = bool(idx and idx.loaded and idx.compatible)
        llm_ok = settings.groq_api_key is not None if settings.llm_provider == "groq" else True
        body = HealthOut(status="ok" if (db_ok and index_ok and llm_ok) else "degraded", database=db_ok,
                         vector_index=index_ok, llm_configured=llm_ok, version=VERSION)
        return JSONResponse(body.model_dump(), status_code=200 if db_ok else 503)

    @app.get("/api/models", response_model=ModelsOut)
    async def models():
        return ModelsOut(default=settings.groq_model, models=settings.model_options, provider=settings.llm_provider)

    # ---- chat
    @app.post("/api/chat")
    async def chat_endpoint(body: ChatRequest, request: Request):
        result = await chat(request).chat_once(scope_id(request), body)
        return {
            "thread_id": result["thread_id"],
            "message": {"message_id": result["message_id"], "role": "assistant", "content": result["content"]},
            "tools": result["tools"], "sources": result["sources"], "usage": result["usage"],
            "latency_ms": result["latency_ms"], "title": result["title"],
        }

    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest, request: Request):
        svc = chat(request)
        first, rest = await svc.open_stream(scope_id(request), body)  # HTTP errors surface here
        return StreamingResponse(
            svc.sse_frames(first, rest), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})

    # ---- conversations
    @app.get("/api/conversations", response_model=ConversationList)
    async def list_conversations(request: Request):
        rows = await asyncio.to_thread(db_of(request).list_conversations, scope_id(request))
        return ConversationList(conversations=[conversation_out(c) for c in rows])

    @app.post("/api/conversations", response_model=ConversationOut, status_code=201)
    async def create_conversation(request: Request, body: CreateConversationRequest | None = None):
        conv = db_of(request).create_conversation(scope_id(request), title=(body.title if body and body.title else "New chat"))
        return conversation_out(conv)

    @app.get("/api/conversations/{thread_id}", response_model=ConversationDetail)
    async def get_conversation(thread_id: str, request: Request):
        sid, d = scope_id(request), db_of(request)
        conv = d.get_conversation(sid, tid(thread_id))
        msgs = d.list_messages(sid, conv.thread_id, roles=("user", "assistant"))
        docs = d.list_documents(sid, conv.thread_id)
        return ConversationDetail(**conversation_out(conv).model_dump(), messages=[message_out(m) for m in msgs],
                                  documents=[document_out(x) for x in docs])

    @app.patch("/api/conversations/{thread_id}", response_model=ConversationOut)
    async def rename_conversation(thread_id: str, body: RenameRequest, request: Request):
        return conversation_out(db_of(request).rename_conversation(scope_id(request), tid(thread_id), body.title))

    @app.delete("/api/conversations/{thread_id}", status_code=204)
    async def delete_conversation(thread_id: str, request: Request):
        await chat(request).delete_conversation(scope_id(request), tid(thread_id))
        return Response(status_code=204)

    @app.delete("/api/conversations/{thread_id}/messages")
    async def clear_conversation(thread_id: str, request: Request):
        n = await chat(request).clear_conversation(scope_id(request), tid(thread_id))
        return {"cleared": n}

    @app.get("/api/conversations/{thread_id}/export")
    async def export_conversation(thread_id: str, request: Request, format: str = Query("md", pattern="^(md|txt)$")):
        text, media = chat(request).export_conversation(scope_id(request), tid(thread_id), format)
        return PlainTextResponse(text, media_type=f"{media}; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="akgpt-{thread_id[:8]}.{format}"'})

    # ---- documents
    @app.post("/api/upload", response_model=DocumentOut, status_code=201)
    async def upload(request: Request, file: UploadFile = File(...), thread_id: str | None = Form(None)):
        sid, d, rag = scope_id(request), db_of(request), request.app.state.rag
        if thread_id:
            tid(thread_id)
            d.get_conversation(sid, thread_id)  # ownership check before accepting bytes
        staged = await stage_upload(file, settings)
        try:
            if not thread_id:
                thread_id = d.create_conversation(sid, title="Uploaded document").thread_id
            result = await asyncio.to_thread(rag.ingest, sid, thread_id, staged)
        finally:
            staged.discard()
        out = document_out(result.document, duplicate=result.duplicate)
        return JSONResponse(out.model_dump(mode="json"), status_code=200 if result.duplicate else 201)

    @app.get("/api/documents", response_model=DocumentList)
    async def list_documents(request: Request, thread_id: str | None = Query(None)):
        if thread_id:
            tid(thread_id)
        rows = db_of(request).list_documents(scope_id(request), thread_id)
        return DocumentList(documents=[document_out(x) for x in rows])

    @app.delete("/api/documents/{document_id}", status_code=204)
    async def delete_document(document_id: str, request: Request):
        if not THREAD_ID_RE.match(document_id):  # same UUID shape
            raise NotFoundError("Document not found.")
        await asyncio.to_thread(request.app.state.rag.delete_document, scope_id(request), document_id)
        return Response(status_code=204)

    # ---- memories
    @app.get("/api/memories", response_model=MemoryList)
    async def list_memories(request: Request):
        return MemoryList(memories=[memory_out(m) for m in db_of(request).list_memories(scope_id(request))])

    @app.patch("/api/memories/{memory_id}")
    async def update_memory(memory_id: str, body: MemoryUpdate, request: Request):
        from security import looks_sensitive

        if not THREAD_ID_RE.match(memory_id):
            raise NotFoundError("Memory not found.")
        if looks_sensitive(body.content):
            raise ValidationFailed("That looks like a secret or payment detail and cannot be stored.")
        return memory_out(db_of(request).update_memory(scope_id(request), memory_id, body.content.strip()))

    @app.delete("/api/memories/{memory_id}", status_code=204)
    async def delete_memory(memory_id: str, request: Request):
        if not THREAD_ID_RE.match(memory_id) or not db_of(request).delete_memory(scope_id(request), memory_id):
            raise NotFoundError("Memory not found.")
        return Response(status_code=204)

    # ---- settings
    @app.get("/api/settings", response_model=SettingsOut)
    async def get_settings_endpoint(request: Request):
        return settings_out(request)

    @app.post("/api/settings", response_model=SettingsOut)
    async def update_settings(body: SettingsUpdate, request: Request):
        values = body.model_dump(exclude_none=True)
        if "model" in values and values["model"] not in settings.model_options:
            raise ValidationFailed("Unknown model. Choose one from /api/models.")
        db_of(request).save_preferences(scope_id(request), values)
        return settings_out(request)

    return app


def get_app() -> FastAPI:
    """Uvicorn factory entrypoint: ``uvicorn app:get_app --factory``."""
    return create_app()


class _LazyApp:
    """``uvicorn app:app`` support without building the app (and reading config) at import time."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    async def __call__(self, scope, receive, send):
        if self._app is None:
            try:
                self._app = create_app()
            except ConfigurationError as exc:
                raise SystemExit(f"AKGPT configuration error: {exc.message}") from None
        await self._app(scope, receive, send)


app = _LazyApp()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    try:
        cfg = get_settings()
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc.message}")
    uvicorn.run("app:app", host=cfg.host, port=cfg.port, reload=False)

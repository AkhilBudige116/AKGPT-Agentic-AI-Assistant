"""Shared fixtures. Every external service (Groq, Tavily, OpenWeather, LangSmith, the
embedding model download) is replaced by an in-process fake: tests need no credentials
and no network."""
from __future__ import annotations

import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as config_module  # noqa: E402
from app import create_app  # noqa: E402
from config import Settings  # noqa: E402
from database import Database  # noqa: E402
from rag import HashEmbedder, RagService  # noqa: E402

FAKE_GROQ_KEY = "gsk_" + "a1B2c3D4e5F6g7H8i9J0" * 2
FAKE_TAVILY_KEY = "tvly-" + "Zy9Xw8Vu7Ts6Rq5Po4"
FAKE_WEATHER_KEY = "0123456789abcdef0123456789abcdef"
SECRET_VALUES = (FAKE_GROQ_KEY, FAKE_TAVILY_KEY, FAKE_WEATHER_KEY)


# --------------------------------------------------------------------------- fake LLM
class ScriptedChat(BaseChatModel):
    """Chat model that replays scripted ``AIMessage`` objects (shared across instances,
    because the agent builds a fresh model object for every step)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    script: Any
    seen: Any

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # tool schemas are irrelevant to the fake
        return self

    def _next(self, messages) -> AIMessage:
        self.seen.append(list(messages))
        item = self.script.popleft() if self.script else AIMessage(content="(script exhausted)")
        if isinstance(item, Exception):
            raise item
        return item

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._next(messages)
        if msg.tool_calls:
            chunks = [{"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i, "type": "tool_call_chunk"}
                      for i, tc in enumerate(msg.tool_calls)]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks,
                                                             usage_metadata=msg.usage_metadata))
            return
        pieces = re.findall(r"\S+\s*", msg.content) or [""]
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=piece, usage_metadata=msg.usage_metadata if last else None))


class Script:
    """Handle on the scripted conversation: push replies, inspect what the model saw."""

    def __init__(self) -> None:
        self.queue: deque = deque()
        self.seen: list = []

    def reply(self, text: str, usage: tuple[int, int] | None = None) -> "Script":
        usage_md = None
        if usage:
            usage_md = {"input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": sum(usage)}
        self.queue.append(AIMessage(content=text, usage_metadata=usage_md))
        return self

    def call(self, name: str, args: dict, call_id: str | None = None) -> "Script":
        self.queue.append(AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": call_id or f"call_{len(self.queue)}_{name}", "type": "tool_call"}]))
        return self

    def fail(self, exc: Exception) -> "Script":
        self.queue.append(exc)
        return self

    def factory(self) -> Callable[[str, float], ScriptedChat]:
        return lambda model, temperature: ScriptedChat(script=self.queue, seen=self.seen)


# -------------------------------------------------------------------------- settings
def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        app_env="test",
        groq_api_key=FAKE_GROQ_KEY,
        tavily_api_key=FAKE_TAVILY_KEY,
        openweather_api_key=FAKE_WEATHER_KEY,
        database_url=f"sqlite:///{tmp_path / 'data' / 'akgpt.db'}",
        checkpoint_db_path=str(tmp_path / "data" / "checkpoints.sqlite"),
        faiss_index_path=str(tmp_path / "faiss_db"),
        upload_dir=str(tmp_path / "uploads"),
        embedding_provider="hash",
        rate_limit_per_minute=10_000,
        log_level="WARNING",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Never let a developer's real environment or .env leak into the tests."""
    for var in ("GROQ_API_KEY", "TAVILY_API_KEY", "OPENWEATHER_API_KEY", "LANGSMITH_API_KEY",
                "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "DATABASE_URL", "FAISS_INDEX_PATH",
                "UPLOAD_DIR", "CHUNK_SIZE", "CHUNK_OVERLAP", "APP_ENV", "ALLOWED_ORIGINS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(ROOT)
    config_module.reset_settings_cache()
    yield
    config_module.reset_settings_cache()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def script() -> Script:
    return Script()


# ------------------------------------------------------------------ external HTTP mock
class HttpMock:
    """Programmable httpx transport for Tavily / OpenWeather."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.handler: Callable[[httpx.Request], httpx.Response] = lambda req: httpx.Response(500, json={})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def respond(self, status: int = 200, payload: Any = None, *, text: str | None = None) -> None:
        if text is not None:
            self.handler = lambda req: httpx.Response(status, text=text)
        else:
            self.handler = lambda req: httpx.Response(status, json=payload if payload is not None else {})

    def raise_(self, exc: Exception) -> None:
        def handler(req):
            raise exc
        self.handler = handler

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


@pytest.fixture
def http_mock() -> HttpMock:
    return HttpMock()


# ----------------------------------------------------------------------- app + clients
@pytest.fixture
def make_client(tmp_path, script, http_mock):
    """Factory that returns a started TestClient (lifespan executed). Extra settings can
    be passed as keyword arguments; all clients are closed after the test."""
    clients: list[TestClient] = []

    def _make(**overrides: Any) -> TestClient:
        s = make_settings(tmp_path, **overrides)
        app = create_app(s, model_factory=script.factory(), transport=http_mock.transport)
        client = TestClient(app, base_url="http://testserver")
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


@pytest.fixture
def other_client(client) -> TestClient:
    """A second browser session talking to the SAME app (separate cookie jar)."""
    other = TestClient(client.app, base_url="http://testserver")
    return other


# ------------------------------------------------------------------- service fixtures
@pytest.fixture
def db(settings) -> Database:
    database = Database(settings.database_url)
    database.init_schema()
    yield database
    database.dispose()


@pytest.fixture
def rag(settings, db) -> RagService:
    service = RagService(settings, db, embedder=HashEmbedder())
    service.initialise()
    return service


@pytest.fixture
def scope_a(db) -> str:
    return "A" * 43


@pytest.fixture
def scope_b(db) -> str:
    return "B" * 43


# ----------------------------------------------------------------------------- helpers
def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``[(event, data), ...]`` (comments / keep-alives skipped)."""
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        name, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((name, json.loads(data) if data else {}))
    return events


def minimal_pdf(pages: list[str]) -> bytes:
    """Build a tiny valid text PDF (one text line per page) without extra dependencies."""
    objs: list[bytes] = []
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_id = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                     f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>").encode())
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def make_docx(paragraphs: list[str]) -> bytes:
    import io

    import docx

    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


class FakeUpload:
    """Minimal stand-in for Starlette's UploadFile (async ``read`` + ``filename``)."""

    def __init__(self, filename: str, data: bytes) -> None:
        import io

        self.filename = filename
        self._buf = io.BytesIO(data)

    async def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)


async def ingest_bytes(rag: RagService, settings: Settings, scope: str, thread: str, filename: str, data: bytes):
    """Stage + ingest raw bytes the same way the upload route does."""
    from rag import stage_upload
    import asyncio

    staged = await stage_upload(FakeUpload(filename, data), settings)
    return await asyncio.to_thread(rag.ingest, scope, thread, staged)

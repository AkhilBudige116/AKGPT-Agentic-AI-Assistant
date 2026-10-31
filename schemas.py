"""Pydantic request/response schemas. Requests reject unknown fields."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from security import THREAD_ID_RE

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


# ------------------------------------------------------------------ requests
class ChatRequest(StrictModel):
    message: str = Field(..., min_length=1, max_length=20000)
    thread_id: str | None = None
    model: str | None = Field(None, max_length=100)
    temperature: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1, le=20)
    response_style: Literal["concise", "balanced", "detailed"] | None = None
    client_message_id: str | None = Field(None, pattern=r"^[0-9a-fA-F-]{36}$")
    # regenerate: drop the last assistant answer and answer the last user message again.
    # edit: replace the last user message with ``message`` and answer again.
    mode: Literal["send", "regenerate", "edit"] = "send"

    @field_validator("thread_id")
    @classmethod
    def _thread(cls, v: str | None) -> str | None:
        if v is not None and not THREAD_ID_RE.match(v):
            raise ValueError("thread_id must be a UUID")
        return v

    @field_validator("message")
    @classmethod
    def _message(cls, v: str) -> str:
        v = _CONTROL.sub("", v)
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class RenameRequest(StrictModel):
    title: str = Field(..., min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        v = _CONTROL.sub("", v).strip()
        if not v:
            raise ValueError("title must not be blank")
        return v


class CreateConversationRequest(StrictModel):
    title: str | None = Field(None, max_length=200)


class SettingsUpdate(StrictModel):
    """Only these user-level preferences can be changed - never secrets or server config."""

    model: str | None = Field(None, max_length=100)
    theme: Literal["system", "light", "dark"] | None = None
    temperature: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1, le=20)
    response_style: Literal["concise", "balanced", "detailed"] | None = None


class MemoryUpdate(StrictModel):
    content: str = Field(..., min_length=1, max_length=1000)


# ----------------------------------------------------------------- responses
class MessageOut(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    meta: dict[str, Any] | None = None


class ConversationOut(BaseModel):
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]
    documents: list["DocumentOut"] = []


class ConversationList(BaseModel):
    conversations: list[ConversationOut]


class DocumentOut(BaseModel):
    document_id: str
    thread_id: str
    filename: str
    content_type: str
    file_size: int
    indexing_status: str
    chunk_count: int
    page_count: int | None = None
    error: str | None = None
    created_at: datetime
    duplicate: bool = False


class DocumentList(BaseModel):
    documents: list[DocumentOut]


class MemoryOut(BaseModel):
    memory_id: str
    content: str
    created_at: datetime
    updated_at: datetime


class MemoryList(BaseModel):
    memories: list[MemoryOut]


class SettingsOut(BaseModel):
    model: str
    theme: str
    temperature: float
    top_k: int
    response_style: str
    available_models: list[str]
    max_upload_mb: int
    supported_extensions: list[str]
    integrations: dict[str, bool]
    deployment_mode: str = "single-user prototype (anonymous browser session, no accounts)"


class ModelsOut(BaseModel):
    default: str
    models: list[str]
    provider: str


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    vector_index: bool
    llm_configured: bool
    version: str


class ErrorOut(BaseModel):
    error: str
    code: str
    request_id: str | None = None


# ------------------------------------------------------------------ mappers
def conversation_out(c) -> ConversationOut:
    return ConversationOut(thread_id=c.thread_id, title=c.title,
                           created_at=_utc(c.created_at), updated_at=_utc(c.updated_at))


def message_out(m) -> MessageOut:
    import json

    meta = None
    if m.meta_json:
        try:
            parsed = json.loads(m.meta_json)
            meta = parsed if isinstance(parsed, dict) else {"items": parsed}
        except ValueError:
            meta = None
    return MessageOut(message_id=m.message_id, role=m.role, content=m.content, created_at=_utc(m.created_at),
                      model=m.model, prompt_tokens=m.prompt_tokens, completion_tokens=m.completion_tokens,
                      latency_ms=m.latency_ms, meta=meta)


def document_out(d, duplicate: bool = False) -> DocumentOut:
    return DocumentOut(document_id=d.document_id, thread_id=d.thread_id, filename=d.filename,
                       content_type=d.content_type, file_size=d.file_size, indexing_status=d.indexing_status,
                       chunk_count=d.chunk_count, page_count=d.page_count, error=d.error,
                       created_at=_utc(d.created_at), duplicate=duplicate)


def memory_out(m) -> MemoryOut:
    return MemoryOut(memory_id=m.memory_id, content=m.content, created_at=_utc(m.created_at),
                     updated_at=_utc(m.updated_at))


ConversationDetail.model_rebuild()

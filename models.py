"""SQLAlchemy ORM models for AKGPT.

Ownership model: AKGPT has no user accounts. Every browser gets an anonymous,
random ``scope_id`` (stored in an HttpOnly cookie). Every conversation,
document, memory and preference row is owned by exactly one scope, and every
query in ``database.py`` filters by it.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    thread_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    scope_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True,
        order_by="Message.seq",
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True,
    )

    __table_args__ = (Index("ix_conversations_scope_updated", "scope_id", "updated_at"),)


class Message(Base):
    __tablename__ = "messages"

    # ``seq`` gives a strict, gap-tolerant insertion order independent of clock resolution.
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, default=new_id)
    thread_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.thread_id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON-encoded list of citation/tool-status objects shown in the UI (never secrets).
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        CheckConstraint("role IN ('user','assistant','tool','system')", name="ck_messages_role"),
        Index("ix_messages_thread_seq", "thread_id", "seq"),
    )


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    scope_id: Mapped[str] = mapped_column(String(64), nullable=False)
    thread_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.thread_id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)  # display name only
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)  # relative to UPLOAD_DIR
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    indexing_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "indexing_status IN ('pending','indexing','indexed','failed')", name="ck_documents_status"
        ),
        UniqueConstraint("thread_id", "sha256", name="uq_documents_thread_hash"),
        Index("ix_documents_scope", "scope_id"),
        Index("ix_documents_thread", "thread_id"),
    )


class Chunk(Base):
    """Docstore for the FAISS index: chunk text + metadata + access scope.

    ``chunk_id`` is the integer id under which the chunk's vector is stored in
    FAISS (IndexIDMap2), so a vector hit maps to exactly one row. Keeping the
    docstore in SQLite (instead of a pickled LangChain docstore) means nothing
    is ever unpickled from disk, and access scope is enforced with SQL.
    """

    __tablename__ = "chunks"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    scope_id: Mapped[str] = mapped_column(String(64), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(36), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document", "document_id"),
        Index("ix_chunks_scope_thread", "scope_id", "thread_id"),
    )


class Memory(Base):
    __tablename__ = "memories"

    memory_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    scope_id: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    __table_args__ = (Index("ix_memories_scope", "scope_id", "updated_at"),)


class Preference(Base):
    __tablename__ = "preferences"

    scope_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    theme: Mapped[str] = mapped_column(String(10), nullable=False, default="system")
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.3)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_style: Mapped[str] = mapped_column(String(12), nullable=False, default="balanced")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint("theme IN ('system','light','dark')", name="ck_pref_theme"),
        CheckConstraint("response_style IN ('concise','balanced','detailed')", name="ck_pref_style"),
        CheckConstraint("temperature >= 0 AND temperature <= 2", name="ck_pref_temperature"),
    )

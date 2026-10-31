"""SQLite persistence layer.

* One engine per ``Database`` instance, one short-lived session per operation
  (no connection sharing between threads/requests).
* Every operation runs in a transaction that commits on success and rolls back
  on error.
* Every read/update/delete is filtered by ``scope_id`` so one browser session
  can never touch another's data.
* Foreign keys are enforced (``PRAGMA foreign_keys=ON``) and WAL mode is used.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import create_engine, delete, event, func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from exceptions import ConflictError, NotFoundError
from models import Base, Conversation, Document, Memory, Message, Preference, new_id, utcnow

log = logging.getLogger(__name__)

DEFAULT_TITLE = "New chat"
TITLE_MAX = 60


def make_title(text: str) -> str:
    one_line = " ".join(text.split())
    if not one_line:
        return DEFAULT_TITLE
    return one_line if len(one_line) <= TITLE_MAX else one_line[: TITLE_MAX - 1].rstrip() + "…"


class Database:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        path = database_url.removeprefix("sqlite:///")
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 30},
            future=True,
        )
        self._configure_sqlite(self.engine)
        self._sessionmaker = sessionmaker(self.engine, expire_on_commit=False, future=True)

    @staticmethod
    def _configure_sqlite(engine: Engine) -> None:
        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn: sqlite3.Connection, _record: Any) -> None:  # pragma: no cover
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

    # ------------------------------------------------------------ lifecycle
    def init_schema(self) -> None:
        """Create tables if they do not exist (idempotent)."""
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()

    def ping(self) -> bool:
        try:
            with self.session() as s:
                s.execute(select(1))
            return True
        except Exception:
            log.exception("database ping failed")
            return False

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commit on success, rollback on any error."""
        s = self._sessionmaker()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # --------------------------------------------------------- conversations
    def create_conversation(self, scope_id: str, thread_id: str | None = None,
                            title: str = DEFAULT_TITLE) -> Conversation:
        with self.session() as s:
            conv = Conversation(thread_id=thread_id or new_id(), scope_id=scope_id,
                                title=make_title(title) if title != DEFAULT_TITLE else DEFAULT_TITLE)
            s.add(conv)
            try:
                s.flush()
            except Exception as exc:  # primary-key clash => someone else's thread id
                raise ConflictError("Conversation identifier already in use.") from exc
            return conv

    def get_conversation(self, scope_id: str, thread_id: str) -> Conversation:
        with self.session() as s:
            conv = s.scalar(select(Conversation).where(
                Conversation.thread_id == thread_id, Conversation.scope_id == scope_id))
            if conv is None:
                # Same error whether it does not exist or belongs to someone else.
                raise NotFoundError("Conversation not found.")
            return conv

    def conversation_exists(self, scope_id: str, thread_id: str) -> bool:
        with self.session() as s:
            return s.scalar(select(func.count()).select_from(Conversation).where(
                Conversation.thread_id == thread_id, Conversation.scope_id == scope_id)) == 1

    def thread_id_taken(self, thread_id: str) -> bool:
        with self.session() as s:
            return s.scalar(select(func.count()).select_from(Conversation).where(
                Conversation.thread_id == thread_id)) == 1

    def list_conversations(self, scope_id: str, limit: int = 200) -> list[Conversation]:
        with self.session() as s:
            return list(s.scalars(
                select(Conversation).where(Conversation.scope_id == scope_id)
                .order_by(Conversation.updated_at.desc()).limit(limit)))

    def rename_conversation(self, scope_id: str, thread_id: str, title: str) -> Conversation:
        with self.session() as s:
            conv = self._owned_conversation(s, scope_id, thread_id)
            conv.title = make_title(title)
            conv.updated_at = utcnow()
            s.flush()
            return conv

    def delete_conversation(self, scope_id: str, thread_id: str) -> list[Document]:
        """Delete a conversation, its messages and document rows (FK cascade).

        Returns the (detached) document rows so the caller can remove the
        uploaded files and vectors. Raises NotFoundError if not owned.
        """
        with self.session() as s:
            conv = self._owned_conversation(s, scope_id, thread_id)
            docs = list(s.scalars(select(Document).where(Document.thread_id == thread_id)))
            s.delete(conv)
            return docs

    def clear_messages(self, scope_id: str, thread_id: str) -> int:
        with self.session() as s:
            self._owned_conversation(s, scope_id, thread_id)
            res = s.execute(delete(Message).where(Message.thread_id == thread_id))
            s.execute(update(Conversation).where(Conversation.thread_id == thread_id)
                      .values(title=DEFAULT_TITLE, updated_at=utcnow()))
            return res.rowcount or 0

    def delete_all_conversations(self, scope_id: str) -> list[tuple[str, list[Document]]]:
        with self.session() as s:
            convs = list(s.scalars(select(Conversation).where(Conversation.scope_id == scope_id)))
            out = []
            for conv in convs:
                docs = list(s.scalars(select(Document).where(Document.thread_id == conv.thread_id)))
                out.append((conv.thread_id, docs))
                s.delete(conv)
            return out

    @staticmethod
    def _owned_conversation(s: Session, scope_id: str, thread_id: str) -> Conversation:
        conv = s.scalar(select(Conversation).where(
            Conversation.thread_id == thread_id, Conversation.scope_id == scope_id))
        if conv is None:
            raise NotFoundError("Conversation not found.")
        return conv

    # -------------------------------------------------------------- messages
    def add_message(self, scope_id: str, thread_id: str, role: str, content: str, *,
                    model: str | None = None, prompt_tokens: int | None = None,
                    completion_tokens: int | None = None, latency_ms: int | None = None,
                    meta: Any | None = None, message_id: str | None = None) -> Message:
        """Persist one message and bump the conversation's ``updated_at``.

        Idempotent on ``message_id``: re-sending the same id returns the
        existing row instead of creating a duplicate.
        """
        with self.session() as s:
            conv = self._owned_conversation(s, scope_id, thread_id)
            if message_id:
                existing = s.scalar(select(Message).where(Message.message_id == message_id))
                if existing is not None:
                    if existing.thread_id != thread_id:  # id belongs to another conversation
                        raise ConflictError("Message identifier already in use.")
                    return existing
            msg = Message(
                message_id=message_id or new_id(), thread_id=thread_id, role=role, content=content,
                model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                latency_ms=latency_ms, meta_json=json.dumps(meta, ensure_ascii=False) if meta else None,
            )
            s.add(msg)
            conv.updated_at = utcnow()
            # Auto-title from the first user message.
            if role == "user" and conv.title == DEFAULT_TITLE:
                conv.title = make_title(content)
            s.flush()
            return msg

    def message_exists(self, scope_id: str, thread_id: str, message_id: str) -> bool:
        with self.session() as s:
            self._owned_conversation(s, scope_id, thread_id)
            return s.scalar(select(func.count()).select_from(Message).where(
                Message.thread_id == thread_id, Message.message_id == message_id)) == 1

    def list_messages(self, scope_id: str, thread_id: str, limit: int | None = None,
                      roles: tuple[str, ...] | None = None) -> list[Message]:
        with self.session() as s:
            self._owned_conversation(s, scope_id, thread_id)
            stmt = select(Message).where(Message.thread_id == thread_id)
            if roles:
                stmt = stmt.where(Message.role.in_(roles))
            if limit:
                rows = list(s.scalars(stmt.order_by(Message.seq.desc()).limit(limit)))
                rows.reverse()
                return rows
            return list(s.scalars(stmt.order_by(Message.seq.asc())))

    def count_messages(self, scope_id: str, thread_id: str) -> int:
        with self.session() as s:
            self._owned_conversation(s, scope_id, thread_id)
            return s.scalar(select(func.count()).select_from(Message)
                            .where(Message.thread_id == thread_id)) or 0

    def delete_message(self, scope_id: str, thread_id: str, message_id: str) -> bool:
        with self.session() as s:
            self._owned_conversation(s, scope_id, thread_id)
            res = s.execute(delete(Message).where(Message.thread_id == thread_id,
                                                  Message.message_id == message_id))
            return bool(res.rowcount)

    def delete_messages_from(self, scope_id: str, thread_id: str, message_id: str,
                             inclusive: bool = True) -> int:
        """Delete ``message_id`` (optionally) and everything after it (used by
        regenerate / edit-and-resubmit)."""
        with self.session() as s:
            self._owned_conversation(s, scope_id, thread_id)
            target = s.scalar(select(Message).where(Message.thread_id == thread_id,
                                                    Message.message_id == message_id))
            if target is None:
                raise NotFoundError("Message not found.")
            cond = Message.seq >= target.seq if inclusive else Message.seq > target.seq
            res = s.execute(delete(Message).where(Message.thread_id == thread_id, cond))
            return res.rowcount or 0

    def search_messages(self, scope_id: str, thread_id: str, query: str, limit: int = 8) -> list[Message]:
        """Case-insensitive substring search within one owned conversation."""
        needle = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self.session() as s:
            self._owned_conversation(s, scope_id, thread_id)
            return list(s.scalars(
                select(Message).where(Message.thread_id == thread_id,
                                      Message.role.in_(("user", "assistant")),
                                      Message.content.like(needle, escape="\\"))
                .order_by(Message.seq.desc()).limit(limit)))

    # ------------------------------------------------------------- documents
    def add_document(self, scope_id: str, thread_id: str, *, document_id: str, filename: str,
                     storage_path: str, content_type: str, file_size: int, sha256: str) -> Document:
        with self.session() as s:
            self._owned_conversation(s, scope_id, thread_id)
            dup = s.scalar(select(Document).where(Document.thread_id == thread_id,
                                                  Document.sha256 == sha256))
            if dup is not None:
                raise ConflictError("This document has already been uploaded to this conversation.")
            doc = Document(document_id=document_id, scope_id=scope_id, thread_id=thread_id,
                           filename=filename, storage_path=storage_path, content_type=content_type,
                           file_size=file_size, sha256=sha256, indexing_status="indexing")
            s.add(doc)
            s.flush()
            return doc

    def update_document_status(self, document_id: str, status: str, *, chunk_count: int | None = None,
                               page_count: int | None = None, error: str | None = None) -> None:
        with self.session() as s:
            doc = s.get(Document, document_id)
            if doc is None:
                return
            doc.indexing_status = status
            if chunk_count is not None:
                doc.chunk_count = chunk_count
            if page_count is not None:
                doc.page_count = page_count
            doc.error = error[:300] if error else None

    def get_document(self, scope_id: str, document_id: str) -> Document:
        with self.session() as s:
            doc = s.scalar(select(Document).where(Document.document_id == document_id,
                                                  Document.scope_id == scope_id))
            if doc is None:
                raise NotFoundError("Document not found.")
            return doc

    def list_documents(self, scope_id: str, thread_id: str | None = None) -> list[Document]:
        with self.session() as s:
            stmt = select(Document).where(Document.scope_id == scope_id)
            if thread_id:
                stmt = stmt.where(Document.thread_id == thread_id)
            return list(s.scalars(stmt.order_by(Document.created_at.desc())))

    def list_all_documents(self) -> list[Document]:
        """Unscoped listing; only for maintenance scripts (index rebuild)."""
        with self.session() as s:
            return list(s.scalars(select(Document).order_by(Document.created_at.asc())))

    def delete_document(self, scope_id: str, document_id: str) -> Document:
        with self.session() as s:
            doc = s.scalar(select(Document).where(Document.document_id == document_id,
                                                  Document.scope_id == scope_id))
            if doc is None:
                raise NotFoundError("Document not found.")
            s.delete(doc)
            return doc

    # -------------------------------------------------------------- memories
    def add_memory(self, scope_id: str, content: str) -> Memory:
        with self.session() as s:
            existing = s.scalar(select(Memory).where(Memory.scope_id == scope_id,
                                                     func.lower(Memory.content) == content.lower()))
            if existing is not None:
                existing.updated_at = utcnow()
                s.flush()
                return existing
            mem = Memory(scope_id=scope_id, content=content)
            s.add(mem)
            s.flush()
            return mem

    def list_memories(self, scope_id: str, limit: int = 200) -> list[Memory]:
        with self.session() as s:
            return list(s.scalars(select(Memory).where(Memory.scope_id == scope_id)
                                  .order_by(Memory.updated_at.desc()).limit(limit)))

    def search_memories(self, scope_id: str, query: str, limit: int = 10) -> list[Memory]:
        """Keyword search; falls back to the most recent memories for generic queries."""
        words = [w for w in (query or "").lower().replace("?", " ").split() if len(w) > 2]
        memories = self.list_memories(scope_id)
        if not words:
            return memories[:limit]
        scored = []
        for m in memories:
            text = m.content.lower()
            score = sum(1 for w in words if w in text)
            if score:
                scored.append((score, m))
        scored.sort(key=lambda t: (t[0], t[1].updated_at), reverse=True)  # best match, then most recent
        return [m for _, m in scored[:limit]]

    def update_memory(self, scope_id: str, memory_id: str, content: str) -> Memory:
        with self.session() as s:
            mem = s.scalar(select(Memory).where(Memory.memory_id == memory_id,
                                                Memory.scope_id == scope_id))
            if mem is None:
                raise NotFoundError("Memory not found.")
            mem.content = content
            mem.updated_at = utcnow()
            s.flush()
            return mem

    def delete_memory(self, scope_id: str, memory_id: str) -> bool:
        with self.session() as s:
            res = s.execute(delete(Memory).where(Memory.memory_id == memory_id,
                                                 Memory.scope_id == scope_id))
            return bool(res.rowcount)

    def delete_all_memories(self, scope_id: str) -> int:
        with self.session() as s:
            res = s.execute(delete(Memory).where(Memory.scope_id == scope_id))
            return res.rowcount or 0

    # ----------------------------------------------------------- preferences
    def get_preferences(self, scope_id: str) -> Preference | None:
        with self.session() as s:
            return s.get(Preference, scope_id)

    def save_preferences(self, scope_id: str, values: dict[str, Any]) -> Preference:
        allowed = {"model", "theme", "temperature", "top_k", "response_style"}
        with self.session() as s:
            pref = s.get(Preference, scope_id)
            if pref is None:
                pref = Preference(scope_id=scope_id)
                s.add(pref)
            for key, value in values.items():
                if key in allowed:
                    setattr(pref, key, value)
            pref.updated_at = datetime.now(timezone.utc)
            s.flush()
            return pref


# ---------------------------------------------------------------------------
_db: Database | None = None


def init_database(database_url: str) -> Database:
    global _db
    if _db is not None:
        _db.dispose()
    _db = Database(database_url)
    _db.init_schema()
    return _db


def get_database() -> Database:
    if _db is None:
        raise RuntimeError("Database has not been initialised")
    return _db

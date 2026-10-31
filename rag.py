"""Persistent FAISS retrieval pipeline.

Design
------
* **Vectors** live in a local FAISS ``IndexIDMap2(IndexFlatIP)`` over L2-normalised
  embeddings, so inner product == cosine similarity (the configured metric).
* **Docstore** (chunk text, page, document, owner scope, conversation) lives in the
  SQLite ``chunks`` table. A FAISS hit's integer id is the ``chunk_id`` primary key.
  Nothing is ever unpickled; the only artefacts read from disk are a FAISS index file
  (checked against a SHA-256 recorded in ``index_meta.json``) and a small JSON file.
* **Access control** is enforced in SQL after the vector search: every hit must match
  the caller's ``scope_id`` *and* the conversation's ``thread_id``. The search
  over-fetches until enough in-scope hits are found, so other users' vectors can never
  crowd out (or leak into) a result.
* **Concurrency**: one process-wide lock serialises index mutation, persistence and
  search; embedding (the slow part) happens outside the lock. Index files are replaced
  atomically (write temp + ``os.replace``).
* **Rebuild**: because chunk text is kept in SQLite, the index can be rebuilt at any
  time without the original uploads (``scripts/rebuild_faiss.py``).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import threading
import uuid
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import func, select

from config import Settings
from database import Database
from exceptions import (
    ConflictError,
    IndexIncompatibleError,
    InvalidDocumentError,
    PayloadTooLargeError,
    UnsupportedFileError,
    ValidationFailed,
)
from models import Chunk, Document
from security import ALLOWED_EXTENSIONS, resolve_inside, safe_extension, sanitize_filename

log = logging.getLogger(__name__)

INDEX_FORMAT_VERSION = 1
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".py": "text/x-python",
}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".py"}
_DOCX_MAX_UNCOMPRESSED = 200 * 1024 * 1024  # zip-bomb guard
_ID_BATCH = 500
_EMBED_BATCH = 64


# =========================================================================== embeddings
class Embedder(Protocol):
    provider: str
    name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (len(texts), dim)."""


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashEmbedder:
    """Deterministic, dependency-free, offline lexical embedder (signed feature hashing).

    Not semantic - it exists so the whole pipeline runs and is testable without
    downloading a model. Select it explicitly with ``EMBEDDING_PROVIDER=hash``.
    """

    provider = "hash"
    _token = re.compile(r"[A-Za-z0-9_]+")

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self.name = f"hash-{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [t.lower() for t in self._token.findall(text)]
            feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for f in feats:
                h = zlib.crc32(f.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self._dim] += sign
        out = np.sign(out) * np.sqrt(np.abs(out))  # sub-linear term frequency
        return _l2_normalise(out)


class SentenceTransformerEmbedder:
    """Local sentence-transformers model (no API key). Loaded lazily on first use."""

    provider = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer  # heavy import, lazy

                log.info("loading embedding model %s", self.name)
                self._model = SentenceTransformer(self.name)
            return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False,
                            convert_to_numpy=True)
        return _l2_normalise(vecs)


class CrossEncoderReranker:
    """Optional reranker, only used when ``RERANKER_MODEL`` is configured."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def rerank(self, query: str, hits: list["Hit"]) -> list["Hit"]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.name)
        scores = self._model.predict([(query, h.text) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda t: float(t[1]), reverse=True)
        return [h for h, _ in ranked]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedder(settings.embedding_dimension)
    return SentenceTransformerEmbedder(settings.embedding_model)


# ============================================================================ extraction
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalise_text(text: str) -> str:
    """Tidy whitespace while preserving paragraph / code structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


Page = tuple[int | None, str]


def _read_text_file(path: Path, max_chars: int) -> str:
    # Never read more than we could use: bounded read protects memory.
    with open(path, "rb") as fh:
        raw = fh.read(max_chars * 4 + 1)
    if b"\x00" in raw[:8192]:
        raise InvalidDocumentError("This file looks binary, not text.")
    text = raw.decode("utf-8-sig", errors="replace")
    if text and text.count("�") / max(len(text), 1) > 0.05:
        raise InvalidDocumentError("This file is not valid UTF-8 text.")
    return text[:max_chars]


def _extract_pdf(path: Path, settings: Settings) -> tuple[list[Page], int]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise InvalidDocumentError("This PDF is password-protected.")
        total = len(reader.pages)
        if total > settings.max_pdf_pages:
            raise InvalidDocumentError(f"PDF has {total} pages; the limit is {settings.max_pdf_pages}.")
        pages: list[Page] = []
        chars = 0
        for i, page in enumerate(reader.pages, start=1):
            text = normalise_text(page.extract_text() or "")
            if text:
                pages.append((i, text))
                chars += len(text)
                if chars > settings.max_extracted_chars:
                    raise InvalidDocumentError("Document text is too large to index.")
        return pages, total
    except InvalidDocumentError:
        raise
    except (PyPdfError, ValueError, KeyError, AttributeError, OSError, RecursionError) as exc:
        raise InvalidDocumentError("This PDF could not be read (it may be corrupt).") from exc


def _extract_docx(path: Path, settings: Settings) -> tuple[list[Page], int | None]:
    import docx
    from docx.opc.exceptions import PackageNotFoundError

    try:
        with zipfile.ZipFile(path) as zf:
            if "word/document.xml" not in zf.namelist():
                raise InvalidDocumentError("This is not a valid DOCX file.")
            if sum(i.file_size for i in zf.infolist()) > _DOCX_MAX_UNCOMPRESSED:
                raise InvalidDocumentError("DOCX expands to an unreasonable size.")
        document = docx.Document(str(path))
    except InvalidDocumentError:
        raise
    except (zipfile.BadZipFile, PackageNotFoundError, KeyError, ValueError, OSError) as exc:
        raise InvalidDocumentError("This DOCX could not be read (it may be corrupt).") from exc

    parts: list[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    text = normalise_text("\n".join(parts))[: settings.max_extracted_chars]
    return ([(None, text)] if text else []), None


def _extract_csv(path: Path, settings: Settings) -> list[Page]:
    text = _read_text_file(path, settings.max_extracted_chars)
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise InvalidDocumentError("This CSV could not be parsed.") from exc
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    lines = [f"Columns: {', '.join(header)}"]
    for r in body:
        lines.append("; ".join(f"{h}: {v}" for h, v in zip(header, r) if v.strip()))
    return [(None, normalise_text("\n".join(lines)))]


def extract_pages(path: Path, ext: str, settings: Settings) -> tuple[list[Page], int | None]:
    """Extract ``[(page_number | None, text)]`` and the page count (when known)."""
    if ext == ".pdf":
        return _extract_pdf(path, settings)
    if ext == ".docx":
        return _extract_docx(path, settings)
    if ext == ".csv":
        pages = _extract_csv(path, settings)
        return pages, None
    if ext in TEXT_EXTENSIONS:
        text = normalise_text(_read_text_file(path, settings.max_extracted_chars))
        return ([(None, text)] if text else []), None
    raise UnsupportedFileError("Unsupported file type.")


def verify_signature(path: Path, ext: str) -> None:
    """Check magic bytes so a renamed binary cannot masquerade as a document."""
    with open(path, "rb") as fh:
        head = fh.read(8192)
    if not head:
        raise InvalidDocumentError("The file is empty.")
    if ext == ".pdf" and b"%PDF-" not in head[:1024]:
        raise InvalidDocumentError("File content does not look like a PDF.")
    if ext == ".docx" and not head.startswith(b"PK\x03\x04"):
        raise InvalidDocumentError("File content does not look like a DOCX.")
    if ext in TEXT_EXTENSIONS and (b"\x00" in head or head.startswith((b"%PDF-", b"PK\x03\x04", b"MZ", b"\x7fELF"))):
        raise InvalidDocumentError("File content does not look like plain text.")


# ================================================================================ upload
@dataclass
class StagedUpload:
    path: Path
    size: int
    sha256: str
    ext: str
    display_name: str

    def discard(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            log.warning("could not remove staged upload")


async def stage_upload(upload, settings: Settings) -> StagedUpload:
    """Stream an ``UploadFile`` to a private temp file with a hard size cap.

    The on-disk name is a fresh UUID; the client-supplied name is only kept as a
    sanitised display label, so path traversal through filenames is impossible.
    """
    display = sanitize_filename(getattr(upload, "filename", None), "document")
    ext = safe_extension(display)
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(
            "Unsupported file type. Upload PDF, DOCX, TXT, MD, CSV or PY files."
        )
    base = settings.upload_path
    base.mkdir(parents=True, exist_ok=True)
    target = resolve_inside(base, f".staging-{uuid.uuid4().hex}{ext}")
    digest = hashlib.sha256()
    size = 0
    try:
        with open(target, "xb") as out:
            while True:
                block = await upload.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > settings.max_upload_bytes:
                    raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_size_mb} MB limit.")
                digest.update(block)
                out.write(block)
        if size == 0:
            raise InvalidDocumentError("The file is empty.")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return StagedUpload(path=target, size=size, sha256=digest.hexdigest(), ext=ext, display_name=display)


# ================================================================================ search
@dataclass
class Hit:
    chunk_id: int
    document_id: str
    filename: str
    page: int | None
    chunk_index: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.filename}, page {self.page}]" if self.page else f"[{self.filename}, section {self.chunk_index + 1}]"


@dataclass
class IngestResult:
    document: Document
    duplicate: bool = False
    chunks: int = 0
    seconds: float = 0.0


@dataclass
class IndexStatus:
    loaded: bool = False
    compatible: bool = True
    ntotal: int = 0
    detail: str = ""
    extra: dict = field(default_factory=dict)


# =========================================================================== RAG service
class RagService:
    def __init__(self, settings: Settings, db: Database, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.db = db
        self.embedder = embedder or build_embedder(settings)
        self.reranker = CrossEncoderReranker(settings.reranker_model) if settings.reranker_model else None
        self._lock = threading.RLock()
        self._index: faiss.Index | None = None
        self._incompatible: str | None = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.index_file = settings.faiss_path / "index.faiss"
        self.meta_file = settings.faiss_path / "index_meta.json"

    # -------------------------------------------------------------- lifecycle
    def initialise(self) -> IndexStatus:
        """Load the persisted index (or create an empty one) and reconcile with SQLite."""
        self.settings.faiss_path.mkdir(parents=True, exist_ok=True)
        self.settings.upload_path.mkdir(parents=True, exist_ok=True)
        self._cleanup_staging()
        with self._lock:
            self._incompatible = None
            try:
                loaded = self._try_load()
                if loaded is None:
                    if self._chunk_count() > 0:
                        log.warning("FAISS index missing or invalid; rebuilding from stored chunks")
                        self._rebuild_locked()
                    else:
                        self._index = self._new_index()
                        self._persist_locked()
                else:
                    self._index = loaded
                    self._reconcile_locked()
            except (OSError, ImportError, RuntimeError, ValueError) as exc:
                # Most commonly: the embedding model could not be downloaded/loaded (offline, blocked, typo).
                self._index = None
                self._incompatible = (
                    f"The embedding model '{self.settings.embedding_model}' could not be loaded "
                    f"({type(exc).__name__}). Check EMBEDDING_MODEL and network access to the model hub, "
                    "pre-download the model, or set EMBEDDING_PROVIDER=hash for offline use, then restart."
                )
                log.error(self._incompatible)
        return self.status()

    def _cleanup_staging(self) -> None:
        for stale in self.settings.upload_path.glob(".staging-*"):
            try:
                stale.unlink()
            except OSError:  # pragma: no cover
                pass

    def status(self) -> IndexStatus:
        with self._lock:
            if self._incompatible:
                return IndexStatus(loaded=self._index is not None, compatible=False, detail=self._incompatible)
            return IndexStatus(loaded=self._index is not None, ntotal=self._index.ntotal if self._index else 0)

    def _new_index(self) -> faiss.Index:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedder.dim))

    def _try_load(self) -> faiss.Index | None:
        if not (self.index_file.exists() and self.meta_file.exists()):
            return None
        try:
            meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if meta.get("format") != INDEX_FORMAT_VERSION:
            return None
        if meta.get("provider") != self.embedder.provider or meta.get("model") != self.embedder.name:
            self._incompatible = (
                f"The FAISS index was built with embedding '{meta.get('provider')}:{meta.get('model')}' but the "
                f"configuration now uses '{self.embedder.provider}:{self.embedder.name}'. "
                "Run `python scripts/rebuild_faiss.py` to re-embed all documents."
            )
            log.error(self._incompatible)
            return faiss.IndexIDMap2(faiss.IndexFlatIP(int(meta.get("dimension", 1)) or 1))
        # Integrity check before handing the file to FAISS.
        if _sha256_file(self.index_file) != meta.get("index_sha256"):
            log.error("FAISS index checksum mismatch; ignoring file")
            return None
        try:
            index = faiss.read_index(str(self.index_file))
        except Exception:
            log.error("FAISS index could not be read")
            return None
        if index.d != int(meta.get("dimension", -1)):
            return None
        return index

    def _persist_locked(self) -> None:
        assert self._index is not None
        self.settings.faiss_path.mkdir(parents=True, exist_ok=True)
        tmp_index = self.index_file.with_suffix(".faiss.tmp")
        faiss.write_index(self._index, str(tmp_index))
        os.replace(tmp_index, self.index_file)
        meta = {
            "format": INDEX_FORMAT_VERSION,
            "provider": self.embedder.provider,
            "model": self.embedder.name,
            "dimension": int(self._index.d),
            "metric": "cosine (inner product on L2-normalised vectors)",
            "ntotal": int(self._index.ntotal),
            "index_sha256": _sha256_file(self.index_file),
        }
        tmp_meta = self.meta_file.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp_meta, self.meta_file)

    def _chunk_count(self) -> int:
        with self.db.session() as s:
            return s.scalar(select(func.count()).select_from(Chunk)) or 0

    def _index_ids(self) -> np.ndarray:
        assert self._index is not None
        if self._index.ntotal == 0:
            return np.array([], dtype=np.int64)
        return faiss.vector_to_array(self._index.id_map).astype(np.int64)

    def _reconcile_locked(self) -> None:
        """Make FAISS and SQLite agree (e.g. after a crash between the two writes)."""
        if self._incompatible:
            return
        with self.db.session() as s:
            db_ids = set(s.scalars(select(Chunk.chunk_id)))
        idx_ids = set(int(i) for i in self._index_ids())
        extra = idx_ids - db_ids
        missing = db_ids - idx_ids
        changed = False
        if extra:
            self._index.remove_ids(np.array(sorted(extra), dtype=np.int64))
            changed = True
            log.warning("removed %d orphan vectors from FAISS", len(extra))
        if missing:
            log.warning("re-embedding %d chunks missing from FAISS", len(missing))
            self._add_rows_by_ids_locked(sorted(missing))
            changed = True
        if changed:
            self._persist_locked()

    def _add_rows_by_ids_locked(self, ids: list[int]) -> None:
        for i in range(0, len(ids), _ID_BATCH):
            batch = ids[i : i + _ID_BATCH]
            with self.db.session() as s:
                rows = list(s.execute(select(Chunk.chunk_id, Chunk.text).where(Chunk.chunk_id.in_(batch))))
            if not rows:
                continue
            vecs = self._embed_batched([r.text for r in rows])
            self._index.add_with_ids(vecs, np.array([r.chunk_id for r in rows], dtype=np.int64))

    def _rebuild_locked(self) -> int:
        self._incompatible = None
        self._index = self._new_index()
        with self.db.session() as s:
            all_ids = list(s.scalars(select(Chunk.chunk_id).order_by(Chunk.chunk_id)))
        self._add_rows_by_ids_locked(all_ids)
        self._persist_locked()
        return len(all_ids)

    def rebuild(self) -> int:
        """Re-embed every stored chunk with the *current* embedder and replace the index."""
        with self._lock:
            return self._rebuild_locked()

    def _require_usable(self) -> None:
        if self._incompatible:
            raise IndexIncompatibleError(self._incompatible)
        if self._index is None:
            raise IndexIncompatibleError("The document index is not initialised.")

    # --------------------------------------------------------------- embedding
    def _embed_batched(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedder.dim), dtype=np.float32)
        parts = [self.embedder.embed(texts[i : i + _EMBED_BATCH]) for i in range(0, len(texts), _EMBED_BATCH)]
        return np.ascontiguousarray(np.vstack(parts), dtype=np.float32)

    # --------------------------------------------------------------- ingestion
    def ingest(self, scope_id: str, thread_id: str, staged: StagedUpload) -> IngestResult:
        """Validate, store, extract, chunk, embed and index one staged upload (blocking)."""
        import time

        started = time.perf_counter()
        try:
            with self._lock:
                self._require_usable()
            verify_signature(staged.path, staged.ext)

            existing = [d for d in self.db.list_documents(scope_id, thread_id) if d.sha256 == staged.sha256]
            if existing:
                doc = existing[0]
                if doc.indexing_status == "indexed":
                    staged.discard()
                    return IngestResult(document=doc, duplicate=True, chunks=doc.chunk_count)
                self.delete_document(scope_id, doc.document_id)  # retry a failed / stuck upload

            document_id = str(uuid.uuid4())
            stored_name = f"{document_id}{staged.ext}"
            final_path = resolve_inside(self.settings.upload_path, stored_name)
            os.replace(staged.path, final_path)
            try:
                doc = self.db.add_document(
                    scope_id, thread_id, document_id=document_id, filename=staged.display_name,
                    storage_path=stored_name, content_type=CONTENT_TYPES[staged.ext],
                    file_size=staged.size, sha256=staged.sha256,
                )
            except ConflictError:
                final_path.unlink(missing_ok=True)
                raise

            try:
                pages, page_count = extract_pages(final_path, staged.ext, self.settings)
                chunk_texts: list[tuple[int | None, int, str]] = []
                for page_no, text in pages:
                    for piece in self._splitter.split_text(text):
                        if piece.strip():
                            chunk_texts.append((page_no, len(chunk_texts), piece))
                if not chunk_texts:
                    raise InvalidDocumentError("No text could be extracted from this file.")
                vectors = self._embed_batched([c[2] for c in chunk_texts])
                self._store_chunks(scope_id, thread_id, document_id, chunk_texts, vectors)
            except Exception as exc:
                message = exc.message if isinstance(exc, (InvalidDocumentError, IndexIncompatibleError)) else "Indexing failed."
                if not isinstance(exc, (InvalidDocumentError, IndexIncompatibleError)):
                    log.exception("indexing failed for document %s", document_id)
                self.db.update_document_status(document_id, "failed", error=message)
                final_path.unlink(missing_ok=True)
                raise

            self.db.update_document_status(document_id, "indexed", chunk_count=len(chunk_texts), page_count=page_count)
            return IngestResult(document=self.db.get_document(scope_id, document_id), chunks=len(chunk_texts),
                                seconds=round(time.perf_counter() - started, 3))
        finally:
            staged.discard()  # no-op once moved

    def _store_chunks(self, scope_id: str, thread_id: str, document_id: str,
                      chunk_texts: list[tuple[int | None, int, str]], vectors: np.ndarray) -> None:
        if vectors.shape[1] != self._index.d:
            raise IndexIncompatibleError("Embedding dimension does not match the index; rebuild the index.")
        added_ids: np.ndarray | None = None
        with self.db.session() as s:
            rows = [Chunk(document_id=document_id, scope_id=scope_id, thread_id=thread_id,
                          chunk_index=idx, page=page, text=text) for page, idx, text in chunk_texts]
            s.add_all(rows)
            s.flush()  # assigns chunk_id primary keys == FAISS ids
            ids = np.array([r.chunk_id for r in rows], dtype=np.int64)
            with self._lock:
                try:
                    self._index.add_with_ids(vectors, ids)
                    added_ids = ids
                    self._persist_locked()
                except Exception:
                    if added_ids is not None:
                        self._index.remove_ids(added_ids)
                    raise
        # DB committed on context exit; if commit failed the vectors are orphans that
        # _reconcile_locked() removes on next start.

    # ---------------------------------------------------------------- deletion
    def delete_document(self, scope_id: str, document_id: str) -> None:
        doc = self.db.get_document(scope_id, document_id)  # raises NotFound when not owned
        with self.db.session() as s:
            ids = list(s.scalars(select(Chunk.chunk_id).where(Chunk.document_id == document_id,
                                                              Chunk.scope_id == scope_id)))
        with self._lock:
            if self._index is not None and ids and not self._incompatible:
                self._index.remove_ids(np.array(ids, dtype=np.int64))
                self._persist_locked()
        self.db.delete_document(scope_id, document_id)  # cascades chunk rows
        try:
            resolve_inside(self.settings.upload_path, doc.storage_path).unlink(missing_ok=True)
        except (ValidationFailed, OSError):  # pragma: no cover
            log.warning("could not remove stored file for document %s", document_id)

    def delete_documents_for_thread(self, scope_id: str, thread_id: str) -> int:
        docs = self.db.list_documents(scope_id, thread_id)
        for d in docs:
            self.delete_document(scope_id, d.document_id)
        return len(docs)

    # --------------------------------------------------------------- retrieval
    def has_documents(self, scope_id: str, thread_id: str) -> bool:
        with self.db.session() as s:
            return bool(s.scalar(select(func.count()).select_from(Chunk).where(
                Chunk.scope_id == scope_id, Chunk.thread_id == thread_id)))

    def retrieve(self, scope_id: str, thread_id: str, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.settings.top_k
        if not query.strip() or not self.has_documents(scope_id, thread_id):
            return []
        with self._lock:
            self._require_usable()
        qvec = self.embedder.embed([query])
        hits: list[Hit] = []
        with self._lock:
            ntotal = self._index.ntotal
            if ntotal == 0:
                return []
            fetch = min(max(top_k * 4, 20), ntotal)
            while True:
                scores, ids = self._index.search(qvec, fetch)
                hits = self._resolve(scope_id, thread_id, ids[0], scores[0])
                if len(hits) >= top_k or fetch >= ntotal:
                    break
                fetch = min(fetch * 4, ntotal)
        hits = [h for h in hits if h.score >= self.settings.rag_min_score]
        if self.reranker and len(hits) > 1:
            try:
                hits = self.reranker.rerank(query, hits)
            except Exception:
                log.warning("reranker failed; keeping vector order")
        return hits[:top_k]

    def _resolve(self, scope_id: str, thread_id: str, ids: Iterable[int], scores: Iterable[float]) -> list[Hit]:
        pairs = [(int(i), float(sc)) for i, sc in zip(ids, scores) if int(i) != -1]
        score_by_id = dict(pairs)
        found: dict[int, tuple] = {}
        with self.db.session() as s:
            for i in range(0, len(pairs), _ID_BATCH):
                batch = [p[0] for p in pairs[i : i + _ID_BATCH]]
                stmt = (
                    select(Chunk, Document.filename)
                    .join(Document, Document.document_id == Chunk.document_id)
                    .where(Chunk.chunk_id.in_(batch), Chunk.scope_id == scope_id, Chunk.thread_id == thread_id,
                           Document.scope_id == scope_id, Document.indexing_status == "indexed")
                )
                for chunk, filename in s.execute(stmt):
                    found[chunk.chunk_id] = (chunk, filename)
        out: list[Hit] = []
        for cid, _ in pairs:  # preserve FAISS ranking
            if cid in found:
                chunk, filename = found[cid]
                out.append(Hit(chunk_id=cid, document_id=chunk.document_id, filename=filename, page=chunk.page,
                               chunk_index=chunk.chunk_index, text=chunk.text, score=score_by_id[cid]))
        return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

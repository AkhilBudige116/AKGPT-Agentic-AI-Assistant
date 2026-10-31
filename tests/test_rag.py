"""FAISS pipeline: ingestion for every file type, persistence/reload, deletion,
isolation, integrity checks, rebuild and concurrency. Uses the offline hash embedder."""
from __future__ import annotations

import asyncio
import json

import pytest

from conftest import ingest_bytes, make_docx, make_settings, minimal_pdf
from database import Database
from exceptions import IndexIncompatibleError, InvalidDocumentError, PayloadTooLargeError, UnsupportedFileError
from rag import HashEmbedder, RagService, normalise_text, stage_upload
from conftest import FakeUpload

A, B = "A" * 43, "B" * 43


@pytest.fixture
def thread(db):
    return db.create_conversation(A).thread_id


def new_rag(settings, db, embedder=None) -> RagService:
    svc = RagService(settings, db, embedder=embedder or HashEmbedder())
    svc.initialise()
    return svc


# ------------------------------------------------------------------------ init / persistence
def test_empty_index_initialises_and_persists(settings, db):
    svc = new_rag(settings, db)
    st = svc.status()
    assert st.loaded and st.compatible and st.ntotal == 0
    assert (settings.faiss_path / "index.faiss").exists() and (settings.faiss_path / "index_meta.json").exists()
    assert svc.retrieve(A, "x" * 8, "anything") == []  # searching an empty index is safe


async def test_index_persists_and_reloads(settings, db, rag, thread):
    res = await ingest_bytes(rag, settings, A, thread, "notes.txt",
                             b"FAISS stores dense vectors for fast similarity search over documents. " * 10)
    assert res.document.indexing_status == "indexed" and res.chunks >= 1
    before = rag.status().ntotal
    reloaded = new_rag(settings, db)  # fresh service, same files -> loaded from disk, not rebuilt
    assert reloaded.status().ntotal == before
    hits = reloaded.retrieve(A, thread, "dense vectors similarity search")
    assert hits and hits[0].filename == "notes.txt"
    meta = json.loads((settings.faiss_path / "index_meta.json").read_text())
    assert meta["dimension"] == 384 and meta["provider"] == "hash" and "cosine" in meta["metric"]


async def test_retrieval_ranks_relevant_chunk_first_with_citation(settings, db, rag, thread):
    text = (b"Apples are red fruit grown in orchards across cold regions.\n\n" * 3 +
            b"The mitochondria is the powerhouse of the cell and produces ATP energy.\n\n" * 3 +
            b"Python is a programming language with dynamic typing and indentation.\n\n" * 3)
    await ingest_bytes(settings=make_settings(settings.upload_path.parent, chunk_size=150, chunk_overlap=20,
                                              embedding_provider="hash",
                                              database_url=settings.database_url,
                                              faiss_index_path=str(settings.faiss_path),
                                              upload_dir=str(settings.upload_path)),
                       rag=rag, scope=A, thread=thread, filename="mix.txt", data=text)
    hits = rag.retrieve(A, thread, "mitochondria ATP powerhouse", top_k=3)
    assert hits and "mitochondria" in hits[0].text
    assert hits[0].score > 0 and hits == sorted(hits, key=lambda h: -h.score)
    assert hits[0].citation.startswith("[mix.txt, section ")


# ------------------------------------------------------------------------ file types
async def test_pdf_pages_and_page_citations(settings, db, rag, thread):
    pdf = minimal_pdf(["Page one talks about volcanoes and magma chambers.",
                       "Page two explains glaciers and ice sheets in Antarctica."])
    res = await ingest_bytes(rag, settings, A, thread, "geo.pdf", pdf)
    assert res.document.page_count == 2
    hits = rag.retrieve(A, thread, "glaciers ice sheets Antarctica")
    assert hits[0].page == 2 and hits[0].citation == "[geo.pdf, page 2]"


async def test_docx_ingestion(settings, db, rag, thread):
    data = make_docx(["Project Orion deadline is the fifteenth of March.", "Budget approved by the finance board."])
    res = await ingest_bytes(rag, settings, A, thread, "plan.docx", data)
    assert res.document.indexing_status == "indexed"
    assert rag.retrieve(A, thread, "Orion deadline March")[0].filename == "plan.docx"


@pytest.mark.parametrize("name,data,probe", [
    ("readme.md", b"# Title\n\nSome *markdown* about gradient descent optimisation.\n", "gradient descent"),
    ("script.py", b"def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n", "fibonacci"),
    ("data.csv", b"name,city\nAda,London\nGrace,Arlington\n", "Grace Arlington"),
])
async def test_text_like_formats(settings, db, rag, thread, name, data, probe):
    await ingest_bytes(rag, settings, A, thread, name, data)
    hits = rag.retrieve(A, thread, probe)
    assert hits and hits[0].filename == name


def test_normalise_text_keeps_structure():
    out = normalise_text("a\r\n\r\n\r\n\r\nb\x00c  \t d")
    assert "\r" not in out and "\x00" not in out and "\n\n\n" not in out and "a" in out and "b" in out


# ------------------------------------------------------------------------ invalid input
async def test_invalid_documents_are_rejected_and_leave_nothing_behind(settings, db, rag, thread):
    cases = [("a.exe", b"MZ....", UnsupportedFileError), ("a.txt", b"", InvalidDocumentError),
             ("a.pdf", b"plain text pretending", InvalidDocumentError),
             ("a.docx", b"plain text pretending", InvalidDocumentError),
             ("a.txt", b"\x00\x01\x02binary" * 20, InvalidDocumentError),
             ("a.txt", b"   \n\n   ", InvalidDocumentError),
             ("a.pdf", b"%PDF-1.4\ntruncated garbage", InvalidDocumentError)]
    for name, data, exc in cases:
        with pytest.raises(exc):
            await ingest_bytes(rag, settings, A, thread, name, data)
    assert rag.status().ntotal == 0 and not rag.has_documents(A, thread)  # nothing searchable
    docs = db.list_documents(A, thread)  # rejected-after-validation files stay visible as "failed" (with the reason)
    assert docs and all(d.indexing_status == "failed" and d.error and d.chunk_count == 0 for d in docs)
    assert [p.name for p in settings.upload_path.iterdir() if not p.name.startswith(".")] == []  # no stored files
    assert list(settings.upload_path.glob(".staging-*")) == []  # staged temp files cleaned up


async def test_failed_upload_can_be_retried(settings, db, rag, thread):
    with pytest.raises(InvalidDocumentError):
        await ingest_bytes(rag, settings, A, thread, "a.pdf", b"%PDF-1.4\ntruncated garbage")
    assert [d.indexing_status for d in db.list_documents(A, thread)] == ["failed"]
    good = await ingest_bytes(rag, settings, A, thread, "ok.txt", b"a perfectly valid document about lighthouses " * 5)
    assert good.document.indexing_status == "indexed"


async def test_size_limit_enforced_while_streaming(tmp_path):
    s = make_settings(tmp_path, max_upload_size_mb=1)
    with pytest.raises(PayloadTooLargeError):
        await stage_upload(FakeUpload("big.txt", b"x" * (1024 * 1024 + 1)), s)
    assert list(s.upload_path.glob(".staging-*")) == []


async def test_traversal_filenames_never_touch_other_paths(settings, db, rag, thread, tmp_path):
    for name in ("../../evil.txt", "..\\..\\evil.txt", "/etc/cron.d/evil.txt", "a/../../b.txt"):
        res = await ingest_bytes(rag, settings, A, thread, name, name.encode() + b" some searchable body text here")
        stored = settings.upload_path / res.document.storage_path
        assert stored.resolve().parent == settings.upload_path.resolve()
        assert "/" not in res.document.filename and "\\" not in res.document.filename
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path.parent / "evil.txt").exists()


async def test_duplicate_upload_is_detected(settings, db, rag, thread):
    first = await ingest_bytes(rag, settings, A, thread, "same.txt", b"identical content for dedupe testing " * 5)
    second = await ingest_bytes(rag, settings, A, thread, "renamed.txt", b"identical content for dedupe testing " * 5)
    assert second.duplicate is True and second.document.document_id == first.document.document_id
    assert rag.status().ntotal == first.chunks  # nothing indexed twice


# ------------------------------------------------------------------------ deletion
async def test_document_deletion_removes_vectors_chunks_and_file(settings, db, rag, thread):
    res = await ingest_bytes(rag, settings, A, thread, "gone.txt", b"ephemeral content about zeppelins and airships " * 8)
    stored = settings.upload_path / res.document.storage_path
    assert stored.exists() and rag.status().ntotal > 0
    rag.delete_document(A, res.document.document_id)
    assert rag.status().ntotal == 0 and not stored.exists()
    assert rag.retrieve(A, thread, "zeppelins airships") == []
    assert not rag.has_documents(A, thread)
    reloaded = new_rag(settings, db)
    assert reloaded.status().ntotal == 0  # deletion was persisted


async def test_only_the_deleted_documents_vectors_are_removed(settings, db, rag, thread):
    keep = await ingest_bytes(rag, settings, A, thread, "keep.txt", b"keep this note about telescopes and lenses " * 6)
    drop = await ingest_bytes(rag, settings, A, thread, "drop.txt", b"drop this note about submarines and sonar " * 6)
    rag.delete_document(A, drop.document.document_id)
    assert rag.status().ntotal == keep.chunks
    assert rag.retrieve(A, thread, "telescopes lenses")[0].filename == "keep.txt"


async def test_foreign_scope_cannot_delete(settings, db, rag, thread):
    from exceptions import NotFoundError

    res = await ingest_bytes(rag, settings, A, thread, "mine.txt", b"my private notes about orchids " * 6)
    with pytest.raises(NotFoundError):
        rag.delete_document(B, res.document.document_id)
    assert rag.status().ntotal == res.chunks


async def test_delete_documents_for_thread(settings, db, rag, thread):
    await ingest_bytes(rag, settings, A, thread, "one.txt", b"first document about rivers and lakes " * 6)
    await ingest_bytes(rag, settings, A, thread, "two.txt", b"second document about deserts and dunes " * 6)
    assert rag.delete_documents_for_thread(A, thread) == 2 and rag.status().ntotal == 0


# ------------------------------------------------------------------------ isolation
async def test_retrieval_respects_scope_and_thread_boundaries(settings, db, rag, thread):
    other_thread = db.create_conversation(A).thread_id
    b_thread = db.create_conversation(B).thread_id
    await ingest_bytes(rag, settings, A, thread, "a.txt", b"the vault combination is seven seven nine " * 6)
    await ingest_bytes(rag, settings, B, b_thread, "b.txt", b"the vault combination is one two three " * 6)
    q = "vault combination"
    assert {h.filename for h in rag.retrieve(A, thread, q)} == {"a.txt"}
    assert {h.filename for h in rag.retrieve(B, b_thread, q)} == {"b.txt"}
    assert rag.retrieve(A, other_thread, q) == []          # same owner, different conversation
    assert rag.retrieve(B, thread, q) == []                 # someone else's thread id is useless
    assert rag.retrieve(A, b_thread, q) == []


async def test_top_k_is_respected(settings, db, rag, thread):
    s2 = make_settings(settings.upload_path.parent, chunk_size=100, chunk_overlap=10, embedding_provider="hash",
                       database_url=settings.database_url, faiss_index_path=str(settings.faiss_path),
                       upload_dir=str(settings.upload_path))
    small = RagService(s2, db, embedder=HashEmbedder())
    small.initialise()
    body = "\n\n".join(f"Paragraph {i} discusses topic alpha with detail number {i}." for i in range(30)).encode()
    await ingest_bytes(small, s2, A, thread, "many.txt", body)
    assert len(small.retrieve(A, thread, "topic alpha detail", top_k=3)) == 3
    assert len(small.retrieve(A, thread, "topic alpha detail", top_k=7)) == 7


# ------------------------------------------------------------------------ integrity / compat
async def test_embedding_model_change_is_detected_and_rebuild_fixes_it(settings, db, rag, thread):
    await ingest_bytes(rag, settings, A, thread, "doc.txt", b"orbital mechanics and satellite launch windows " * 6)
    different = new_rag(settings, db, embedder=HashEmbedder(dim=128))  # same files, different embedding config
    st = different.status()
    assert st.compatible is False and "rebuild_faiss.py" in st.detail
    with pytest.raises(IndexIncompatibleError):
        different.retrieve(A, thread, "satellite launch")
    assert different.rebuild() > 0  # re-embeds from the chunks stored in SQLite
    assert different.status().compatible is True
    assert different.retrieve(A, thread, "satellite launch windows")


async def test_corrupted_index_file_is_ignored_and_rebuilt_from_sqlite(settings, db, rag, thread):
    res = await ingest_bytes(rag, settings, A, thread, "doc.txt", b"tidal energy and wave turbines at sea " * 6)
    (settings.faiss_path / "index.faiss").write_bytes(b"corrupted!!")
    svc = new_rag(settings, db)
    assert svc.status().ntotal == res.chunks
    assert svc.retrieve(A, thread, "wave turbines tidal")


async def test_missing_index_files_are_rebuilt(settings, db, rag, thread):
    res = await ingest_bytes(rag, settings, A, thread, "doc.txt", b"solar panels and inverters for homes " * 6)
    for f in settings.faiss_path.glob("index*"):
        f.unlink()
    svc = new_rag(settings, db)
    assert svc.status().ntotal == res.chunks


async def test_reconcile_removes_orphan_vectors_after_crash(settings, db, rag, thread):
    from sqlalchemy import delete
    from models import Chunk

    await ingest_bytes(rag, settings, A, thread, "doc.txt", b"coral reefs and marine biodiversity " * 6)
    with db.session() as s:  # simulate a crash that lost the SQL rows but kept the vectors
        s.execute(delete(Chunk))
    svc = new_rag(settings, db)
    assert svc.status().ntotal == 0


# ------------------------------------------------------------------------ concurrency
async def test_concurrent_uploads_keep_index_and_database_consistent(settings, db, rag, thread):
    docs = [(f"doc{i}.txt", (f"unique topic number {i} about subject{i} " * 20).encode()) for i in range(8)]
    results = await asyncio.gather(*(ingest_bytes(rag, settings, A, thread, n, d) for n, d in docs))
    assert sum(r.chunks for r in results) == rag.status().ntotal
    for i in range(8):
        hit = rag.retrieve(A, thread, f"subject{i}", top_k=1)[0]
        assert hit.filename == f"doc{i}.txt"
    assert new_rag(settings, db).status().ntotal == rag.status().ntotal  # persisted state is consistent too


async def test_search_while_uploading_does_not_error(settings, db, rag, thread):
    await ingest_bytes(rag, settings, A, thread, "base.txt", b"baseline document about weather patterns " * 6)

    async def searcher():
        for _ in range(20):
            await asyncio.to_thread(rag.retrieve, A, thread, "weather patterns")
            await asyncio.sleep(0)

    uploads = [ingest_bytes(rag, settings, A, thread, f"u{i}.txt", f"extra upload {i} about clouds ".encode() * 30)
               for i in range(5)]
    await asyncio.gather(searcher(), *uploads)
    assert rag.status().ntotal > 0

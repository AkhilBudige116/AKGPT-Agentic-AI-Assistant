"""SQLite persistence: CRUD, scope isolation, cascades, transactions."""
from __future__ import annotations

import pytest
from sqlalchemy import text

from database import DEFAULT_TITLE, Database, make_title
from exceptions import ConflictError, NotFoundError


def test_schema_init_is_idempotent_and_ping(db):
    db.init_schema()
    db.init_schema()
    assert db.ping() is True


def test_foreign_keys_and_wal_enabled(db):
    with db.session() as s:
        assert s.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert s.execute(text("PRAGMA journal_mode")).scalar() == "wal"


def test_conversation_crud(db, scope_a):
    conv = db.create_conversation(scope_a)
    assert conv.title == DEFAULT_TITLE
    assert db.get_conversation(scope_a, conv.thread_id).thread_id == conv.thread_id
    db.rename_conversation(scope_a, conv.thread_id, "  Planning   notes  ")
    assert db.get_conversation(scope_a, conv.thread_id).title == "Planning notes"
    db.delete_conversation(scope_a, conv.thread_id)
    with pytest.raises(NotFoundError):
        db.get_conversation(scope_a, conv.thread_id)


def test_long_titles_are_truncated():
    assert len(make_title("word " * 100)) <= 80
    assert make_title("   ") == DEFAULT_TITLE


def test_messages_ordered_and_auto_title(db, scope_a):
    conv = db.create_conversation(scope_a)
    db.add_message(scope_a, conv.thread_id, "user", "What is FAISS?")
    db.add_message(scope_a, conv.thread_id, "assistant", "A vector search library.", model="m",
                   prompt_tokens=3, completion_tokens=5, latency_ms=12, meta={"tools": []})
    msgs = db.list_messages(scope_a, conv.thread_id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].prompt_tokens == 3 and msgs[1].latency_ms == 12
    assert db.get_conversation(scope_a, conv.thread_id).title == "What is FAISS?"
    assert db.count_messages(scope_a, conv.thread_id) == 2
    assert [m.content for m in db.list_messages(scope_a, conv.thread_id, limit=1)] == ["A vector search library."]


def test_add_message_is_idempotent_on_message_id(db, scope_a):
    conv = db.create_conversation(scope_a)
    first = db.add_message(scope_a, conv.thread_id, "user", "hello", message_id="11111111-1111-1111-1111-111111111111")
    again = db.add_message(scope_a, conv.thread_id, "user", "hello", message_id="11111111-1111-1111-1111-111111111111")
    assert first.message_id == again.message_id
    assert db.count_messages(scope_a, conv.thread_id) == 1


def test_message_id_cannot_collide_across_conversations(db, scope_a):
    c1, c2 = db.create_conversation(scope_a), db.create_conversation(scope_a)
    db.add_message(scope_a, c1.thread_id, "user", "x", message_id="22222222-2222-2222-2222-222222222222")
    with pytest.raises(ConflictError):
        db.add_message(scope_a, c2.thread_id, "user", "x", message_id="22222222-2222-2222-2222-222222222222")


def test_delete_messages_from_for_regenerate(db, scope_a):
    conv = db.create_conversation(scope_a)
    m1 = db.add_message(scope_a, conv.thread_id, "user", "q1")
    db.add_message(scope_a, conv.thread_id, "assistant", "a1")
    m3 = db.add_message(scope_a, conv.thread_id, "user", "q2")
    db.add_message(scope_a, conv.thread_id, "assistant", "a2")
    assert db.delete_messages_from(scope_a, conv.thread_id, m3.message_id) == 2
    assert [m.content for m in db.list_messages(scope_a, conv.thread_id)] == ["q1", "a1"]
    assert db.delete_messages_from(scope_a, conv.thread_id, m1.message_id, inclusive=False) == 1
    with pytest.raises(NotFoundError):
        db.delete_messages_from(scope_a, conv.thread_id, "missing")


def test_search_messages_escapes_wildcards(db, scope_a):
    conv = db.create_conversation(scope_a)
    db.add_message(scope_a, conv.thread_id, "user", "discount is 50% today")
    db.add_message(scope_a, conv.thread_id, "user", "nothing relevant")
    assert len(db.search_messages(scope_a, conv.thread_id, "50%")) == 1
    assert len(db.search_messages(scope_a, conv.thread_id, "%")) == 1  # '%' is literal, only the first matches
    assert len(db.search_messages(scope_a, conv.thread_id, "_")) == 0  # '_' is literal, not a wildcard


def test_scope_isolation_on_every_read_and_write(db, scope_a, scope_b):
    conv = db.create_conversation(scope_a)
    db.add_message(scope_a, conv.thread_id, "user", "mine")
    for call in (
        lambda: db.get_conversation(scope_b, conv.thread_id),
        lambda: db.rename_conversation(scope_b, conv.thread_id, "x"),
        lambda: db.delete_conversation(scope_b, conv.thread_id),
        lambda: db.list_messages(scope_b, conv.thread_id),
        lambda: db.add_message(scope_b, conv.thread_id, "user", "intruder"),
        lambda: db.clear_messages(scope_b, conv.thread_id),
        lambda: db.search_messages(scope_b, conv.thread_id, "mine"),
        lambda: db.message_exists(scope_b, conv.thread_id, "x"),
    ):
        with pytest.raises(NotFoundError):
            call()
    assert db.list_conversations(scope_b) == []
    assert db.count_messages(scope_a, conv.thread_id) == 1


def test_thread_id_clash_with_other_scope_is_conflict(db, scope_a, scope_b):
    conv = db.create_conversation(scope_a)
    with pytest.raises(ConflictError):
        db.create_conversation(scope_b, thread_id=conv.thread_id)


def test_transaction_rolls_back_on_error(db, scope_a):
    with pytest.raises(RuntimeError):
        with db.session() as s:
            from models import Conversation

            s.add(Conversation(thread_id="33333333-3333-3333-3333-333333333333", scope_id=scope_a, title="ghost"))
            s.flush()
            raise RuntimeError("fail after flush")
    assert db.list_conversations(scope_a) == []


def test_failed_message_write_leaves_no_partial_state(db, scope_a):
    conv = db.create_conversation(scope_a)
    with pytest.raises(NotFoundError):
        db.add_message(scope_a, "44444444-4444-4444-4444-444444444444", "user", "orphan")
    assert db.count_messages(scope_a, conv.thread_id) == 0


def test_documents_cascade_and_duplicate_guard(db, scope_a):
    conv = db.create_conversation(scope_a)
    kw = dict(filename="a.txt", storage_path="x.txt", content_type="text/plain", file_size=3, sha256="abc")
    db.add_document(scope_a, conv.thread_id, document_id="d1", **kw)
    with pytest.raises(ConflictError):
        db.add_document(scope_a, conv.thread_id, document_id="d2", **kw)
    db.update_document_status("d1", "indexed", chunk_count=4, page_count=2)
    doc = db.get_document(scope_a, "d1")
    assert doc.indexing_status == "indexed" and doc.chunk_count == 4 and doc.page_count == 2
    docs = db.delete_conversation(scope_a, conv.thread_id)
    assert [d.document_id for d in docs] == ["d1"]
    with pytest.raises(NotFoundError):
        db.get_document(scope_a, "d1")


def test_document_isolation(db, scope_a, scope_b):
    conv = db.create_conversation(scope_a)
    db.add_document(scope_a, conv.thread_id, document_id="d1", filename="a.txt", storage_path="x",
                    content_type="text/plain", file_size=1, sha256="h")
    with pytest.raises(NotFoundError):
        db.get_document(scope_b, "d1")
    with pytest.raises(NotFoundError):
        db.delete_document(scope_b, "d1")
    assert db.list_documents(scope_b) == []


def test_memory_crud_scoped_and_deduplicated(db, scope_a, scope_b):
    m = db.add_memory(scope_a, "Likes tea")
    assert db.add_memory(scope_a, "likes TEA").memory_id == m.memory_id  # case-insensitive dedupe
    db.add_memory(scope_a, "Lives in Hyderabad")
    assert {x.content for x in db.search_memories(scope_a, "tea")} == {"Likes tea"}
    assert db.list_memories(scope_b) == []
    with pytest.raises(NotFoundError):
        db.update_memory(scope_b, m.memory_id, "hijack")
    assert db.delete_memory(scope_b, m.memory_id) is False
    assert db.update_memory(scope_a, m.memory_id, "Likes coffee").content == "Likes coffee"
    assert db.delete_memory(scope_a, m.memory_id) is True
    assert db.delete_all_memories(scope_a) == 1


def test_preferences_only_accept_known_fields(db, scope_a):
    assert db.get_preferences(scope_a) is None
    db.save_preferences(scope_a, {"theme": "dark", "top_k": 7, "api_key": "should-be-ignored"})
    pref = db.get_preferences(scope_a)
    assert pref.theme == "dark" and pref.top_k == 7 and not hasattr(pref, "api_key")


def test_data_survives_reopen(settings):
    first = Database(settings.database_url)
    first.init_schema()
    conv = first.create_conversation("S" * 43, title="persisted")
    first.dispose()
    second = Database(settings.database_url)
    assert second.get_conversation("S" * 43, conv.thread_id).title == "persisted"
    second.dispose()


def test_clear_messages_resets_title(db, scope_a):
    conv = db.create_conversation(scope_a)
    db.add_message(scope_a, conv.thread_id, "user", "hello")
    assert db.clear_messages(scope_a, conv.thread_id) == 1
    assert db.get_conversation(scope_a, conv.thread_id).title == DEFAULT_TITLE

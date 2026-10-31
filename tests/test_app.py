"""HTTP API: startup, validation, conversations, chat (JSON + SSE), uploads, settings."""
from __future__ import annotations

import uuid

import pytest

from conftest import FAKE_GROQ_KEY, SECRET_VALUES, parse_sse

TXT = b"The capital of Atlantis is Poseidonis. Its harbour holds nine hundred ships.\n" * 20


def stream(client, **body):
    r = client.post("/api/chat/stream", json=body)
    return r, parse_sse(r.text)


# ------------------------------------------------------------------ startup / basics
def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "AKGPT" in r.text
    assert "/static/js/app.js" in r.text
    assert 'id="voice-input"' in r.text and 'aria-label="Start voice input"' in r.text
    assert client.get("/static/css/style.css").status_code == 200


def test_health_ok_and_does_not_leak(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["database"] and body["vector_index"] and body["llm_configured"]
    assert not any(secret in r.text for secret in SECRET_VALUES)


def test_health_degraded_without_groq_key(make_client):
    c = make_client(groq_api_key=None)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded" and r.json()["llm_configured"] is False


def test_models_endpoint(client):
    body = client.get("/api/models").json()
    assert body["default"] == "openai/gpt-oss-120b"
    assert body["default"] in body["models"] and "qwen/qwen3.6-27b" in body["models"]
    assert body["provider"] == "groq"


def test_session_cookie_and_security_headers(client):
    r = client.get("/api/models")
    assert "akgpt_session" in r.headers["set-cookie"] and "HttpOnly" in r.headers["set-cookie"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-request-id"]


def test_openapi_docs_disabled_in_production(make_client):
    c = make_client(app_env="production")
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404


# ---------------------------------------------------------------------- validation
@pytest.mark.parametrize("payload", [
    {},                                        # missing message
    {"message": ""},                           # empty
    {"message": "   "},                        # blank
    {"message": "hi", "unexpected": 1},        # unknown field
    {"message": "hi", "thread_id": "../../etc/passwd"},
    {"message": "hi", "temperature": 7},
    {"message": "hi", "top_k": 0},
    {"message": "hi", "mode": "nuke"},
])
def test_chat_request_validation(client, payload):
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"


def test_unknown_model_rejected(client):
    r = client.post("/api/chat", json={"message": "hi", "model": "gpt-9000"})
    assert r.status_code == 422


def test_invalid_json_and_oversized_body(client):
    assert client.post("/api/chat", content=b"{nope", headers={"content-type": "application/json"}).status_code == 422
    big = {"message": "x" * 300_000}
    assert client.post("/api/chat", json=big).status_code == 413


def test_unknown_route_returns_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["code"] == "not_found"


# ------------------------------------------------------------------------ chat JSON
def test_chat_roundtrip_persists_messages(client, script):
    script.reply("Hello from AKGPT!", usage=(11, 5))
    r = client.post("/api/chat", json={"message": "Say hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["message"]["content"] == "Hello from AKGPT!"
    assert body["usage"]["input_tokens"] == 11 and body["usage"]["output_tokens"] == 5
    assert body["latency_ms"] >= 0

    detail = client.get(f"/api/conversations/{body['thread_id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["content"] == "Say hello"
    assert detail["title"] == "Say hello"  # auto-titled from the first user message
    # the model saw the system prompt + the user message
    assert script.seen[0][0].type == "system" and script.seen[0][-1].content == "Say hello"


def test_chat_continues_same_thread_with_checkpoint_memory(client, script):
    script.reply("Nice to meet you, Akhil.").reply("Your name is Akhil.")
    first = client.post("/api/chat", json={"message": "My name is Akhil"}).json()
    client.post("/api/chat", json={"message": "What is my name?", "thread_id": first["thread_id"]})
    second_call_messages = script.seen[1]
    texts = [str(m.content) for m in second_call_messages]
    assert "My name is Akhil" in texts and "Nice to meet you, Akhil." in texts  # restored from the checkpoint
    detail = client.get(f"/api/conversations/{first['thread_id']}").json()
    assert len(detail["messages"]) == 4


def test_missing_groq_key_gives_actionable_error_and_no_conversation(make_client):
    c = make_client(groq_api_key=None)
    r = c.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 503
    assert r.json()["code"] == "service_not_configured" and "GROQ_API_KEY" in r.json()["error"]
    assert c.get("/api/conversations").json()["conversations"] == []


def test_llm_failure_is_safe_and_recoverable(client, script):
    script.fail(RuntimeError(f"boom with key {FAKE_GROQ_KEY}"))
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 502
    assert FAKE_GROQ_KEY not in r.text and "Traceback" not in r.text
    script.reply("recovered")
    thread = client.get("/api/conversations").json()["conversations"][0]["thread_id"]
    ok = client.post("/api/chat", json={"message": "again", "thread_id": thread})
    assert ok.status_code == 200 and ok.json()["message"]["content"] == "recovered"


def test_duplicate_client_message_id_is_rejected(client, script):
    script.reply("one").reply("two")
    mid = str(uuid.uuid4())
    first = client.post("/api/chat", json={"message": "hi", "client_message_id": mid}).json()
    r = client.post("/api/chat", json={"message": "hi", "client_message_id": mid, "thread_id": first["thread_id"]})
    assert r.status_code == 409
    detail = client.get(f"/api/conversations/{first['thread_id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]  # no duplicate persisted


def test_regenerate_replaces_last_answer(client, script):
    script.reply("first answer").reply("second answer")
    first = client.post("/api/chat", json={"message": "question"}).json()
    r = client.post("/api/chat", json={"message": "question", "thread_id": first["thread_id"], "mode": "regenerate"})
    assert r.json()["message"]["content"] == "second answer"
    msgs = client.get(f"/api/conversations/{first['thread_id']}").json()["messages"]
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "question"), ("assistant", "second answer")]
    # the regenerated turn must not show the model the discarded answer
    assert "first answer" not in [str(m.content) for m in script.seen[1]]


def test_edit_and_resubmit(client, script):
    script.reply("answer to typo").reply("answer to fixed")
    first = client.post("/api/chat", json={"message": "waht is 2+2"}).json()
    client.post("/api/chat", json={"message": "what is 2+2", "thread_id": first["thread_id"], "mode": "edit"})
    msgs = client.get(f"/api/conversations/{first['thread_id']}").json()["messages"]
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "what is 2+2"), ("assistant", "answer to fixed")]


# ------------------------------------------------------------------------ SSE stream
def test_stream_event_sequence_and_single_persistence(client, script):
    script.reply("Streaming works fine today", usage=(7, 4))
    r, events = stream(client, message="stream please")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    names = [n for n, _ in events]
    assert names[0] == "meta" and names[-1] == "done"
    assert names.count("done") == 1 and "delta" in names
    assert "".join(d["text"] for n, d in events if n == "delta") == "Streaming works fine today"
    done = events[-1][1]
    assert done["content"] == "Streaming works fine today" and done["usage"]["output_tokens"] == 4
    thread = events[0][1]["thread_id"]
    msgs = client.get(f"/api/conversations/{thread}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]  # exactly one assistant row
    assert msgs[1]["message_id"] == done["message_id"]


def test_stream_with_tool_call_reports_status_and_hides_tool_output(client, script):
    script.call("calculator", {"expression": "12*(3+4)"}).reply("The answer is 84.")
    r, events = stream(client, message="what is 12*(3+4)?")
    names = [n for n, _ in events]
    assert "tool_start" in names and "tool_end" in names and names[-1] == "done"
    tool_end = next(d for n, d in events if n == "tool_end")
    assert tool_end["name"] == "calculator" and tool_end["ok"] is True
    done = events[-1][1]
    assert done["content"] == "The answer is 84." and done["tools"] == [{"name": "calculator", "ok": True}]
    assert '"result"' not in "".join(d.get("text", "") for n, d in events if n == "delta")  # raw JSON never streamed
    thread = events[0][1]["thread_id"]
    msgs = client.get(f"/api/conversations/{thread}").json()["messages"]
    assert all(m["role"] in ("user", "assistant") for m in msgs)  # tool messages are not user-facing


def test_stream_error_event_is_safe_and_terminates(client, script):
    script.fail(RuntimeError(f"upstream exploded {FAKE_GROQ_KEY}"))
    r, events = stream(client, message="hi")
    names = [n for n, _ in events]
    assert names[0] == "meta" and "error" in names and names[-1] == "done"
    err = next(d for n, d in events if n == "error")
    assert err["retryable"] is True and FAKE_GROQ_KEY not in r.text and "Traceback" not in r.text
    assert events[-1][1]["status"] == "error"


def test_stream_validation_errors_use_http_status(client, make_client):
    assert client.post("/api/chat/stream", json={"message": "hi", "thread_id": str(uuid.uuid4())}).status_code == 404
    c = make_client(groq_api_key=None)
    assert c.post("/api/chat/stream", json={"message": "hi"}).status_code == 503


# ------------------------------------------------------------------- conversations
def test_conversation_lifecycle(client, script):
    created = client.post("/api/conversations", json={})
    assert created.status_code == 201
    tid = created.json()["thread_id"]
    assert client.get("/api/conversations").json()["conversations"][0]["thread_id"] == tid

    renamed = client.patch(f"/api/conversations/{tid}", json={"title": "Trip planning"})
    assert renamed.status_code == 200 and renamed.json()["title"] == "Trip planning"
    assert client.patch(f"/api/conversations/{tid}", json={"title": "  "}).status_code == 422
    assert client.patch(f"/api/conversations/{tid}", json={"title": "x", "extra": 1}).status_code == 422

    script.reply("Sure.")
    client.post("/api/chat", json={"message": "plan a trip", "thread_id": tid})
    assert client.delete(f"/api/conversations/{tid}").status_code == 204
    assert client.get(f"/api/conversations/{tid}").status_code == 404
    assert client.delete(f"/api/conversations/{tid}").status_code == 404
    assert client.get("/api/conversations").json()["conversations"] == []


def test_conversations_ordered_by_recent_activity(client, script):
    a = client.post("/api/conversations", json={"title": "older"}).json()["thread_id"]
    b = client.post("/api/conversations", json={"title": "newer"}).json()["thread_id"]
    script.reply("ok")
    client.post("/api/chat", json={"message": "bump", "thread_id": a})
    ids = [c["thread_id"] for c in client.get("/api/conversations").json()["conversations"]]
    assert ids == [a, b]


def test_clear_and_export(client, script):
    script.reply("Hi there")
    r = client.post("/api/chat", json={"message": "hello"}).json()
    tid = r["thread_id"]
    md = client.get(f"/api/conversations/{tid}/export?format=md")
    assert md.status_code == 200 and "## You" in md.text and "## AKGPT" in md.text and "Hi there" in md.text
    assert "attachment" in md.headers["content-disposition"]
    txt = client.get(f"/api/conversations/{tid}/export?format=txt")
    assert txt.status_code == 200 and "AKGPT:" in txt.text
    assert client.get(f"/api/conversations/{tid}/export?format=exe").status_code == 422
    assert client.delete(f"/api/conversations/{tid}/messages").json() == {"cleared": 2}
    assert client.get(f"/api/conversations/{tid}").json()["messages"] == []


def test_clear_resets_agent_memory(client, script):
    script.reply("noted").reply("fresh start")
    tid = client.post("/api/chat", json={"message": "secret word is pineapple"}).json()["thread_id"]
    client.delete(f"/api/conversations/{tid}/messages")
    client.post("/api/chat", json={"message": "what was the word?", "thread_id": tid})
    assert "secret word is pineapple" not in [str(m.content) for m in script.seen[1]]


# ------------------------------------------------------------------------- isolation
def test_other_session_cannot_touch_my_conversation(client, other_client, script):
    script.reply("private reply")
    tid = client.post("/api/chat", json={"message": "private"}).json()["thread_id"]
    assert other_client.get(f"/api/conversations/{tid}").status_code == 404
    assert other_client.patch(f"/api/conversations/{tid}", json={"title": "hijack"}).status_code == 404
    assert other_client.delete(f"/api/conversations/{tid}").status_code == 404
    assert other_client.get(f"/api/conversations/{tid}/export").status_code == 404
    assert other_client.delete(f"/api/conversations/{tid}/messages").status_code == 404
    assert other_client.post("/api/chat", json={"message": "hi", "thread_id": tid}).status_code == 404
    assert other_client.get("/api/conversations").json()["conversations"] == []
    assert client.get(f"/api/conversations/{tid}").json()["title"] == "private"  # untouched


def test_forged_cookie_does_not_grant_access(client, other_client, script):
    script.reply("x")
    tid = client.post("/api/chat", json={"message": "mine"}).json()["thread_id"]
    other_client.cookies.set("akgpt_session", "A" * 43)  # well-formed but different token
    assert other_client.get(f"/api/conversations/{tid}").status_code == 404
    other_client.cookies.set("akgpt_session", "not-a-valid-token")  # malformed -> fresh session issued
    assert other_client.get("/api/conversations").json()["conversations"] == []


# ---------------------------------------------------------------------------- uploads
def upload(client, name="notes.txt", data=TXT, thread_id=None, ctype="text/plain"):
    form = {"thread_id": thread_id} if thread_id else {}
    return client.post("/api/upload", files={"file": (name, data, ctype)}, data=form)


def test_upload_index_list_and_delete(client):
    r = upload(client)
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["filename"] == "notes.txt" and doc["indexing_status"] == "indexed" and doc["chunk_count"] >= 1
    listed = client.get("/api/documents").json()["documents"]
    assert [d["document_id"] for d in listed] == [doc["document_id"]]
    assert client.get("/health").json()["vector_index"] is True
    assert client.delete(f"/api/documents/{doc['document_id']}").status_code == 204
    assert client.get("/api/documents").json()["documents"] == []
    assert client.delete(f"/api/documents/{doc['document_id']}").status_code == 404


def test_upload_into_existing_conversation_and_duplicate(client):
    tid = client.post("/api/conversations", json={}).json()["thread_id"]
    first = upload(client, thread_id=tid)
    again = upload(client, thread_id=tid)
    assert first.status_code == 201 and again.status_code == 200
    assert again.json()["duplicate"] is True and again.json()["document_id"] == first.json()["document_id"]
    assert len(client.get(f"/api/conversations/{tid}").json()["documents"]) == 1


def test_upload_to_someone_elses_conversation_is_refused(client, other_client):
    tid = client.post("/api/conversations", json={}).json()["thread_id"]
    assert upload(other_client, thread_id=tid).status_code == 404


def test_documents_are_isolated_between_sessions(client, other_client):
    doc = upload(client).json()
    assert other_client.get("/api/documents").json()["documents"] == []
    assert other_client.delete(f"/api/documents/{doc['document_id']}").status_code == 404
    assert len(client.get("/api/documents").json()["documents"]) == 1


def test_deleting_conversation_removes_documents_and_vectors(client):
    tid = client.post("/api/conversations", json={}).json()["thread_id"]
    upload(client, thread_id=tid)
    assert client.app.state.rag.status().ntotal > 0
    client.delete(f"/api/conversations/{tid}")
    assert client.app.state.rag.status().ntotal == 0
    assert client.get("/api/documents").json()["documents"] == []


@pytest.mark.parametrize("name,data,status", [
    ("virus.exe", b"MZ\x90\x00 binary", 415),
    ("script.sh", b"#!/bin/sh\nrm -rf /", 415),
    ("noext", b"hello", 415),
    ("fake.txt", b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 64, 422),     # binary renamed to .txt
    ("fake.pdf", b"this is not a pdf at all", 422),
    ("fake.docx", b"this is not a zip", 422),
    ("empty.txt", b"", 422),
])
def test_unsafe_or_invalid_uploads_are_rejected(client, name, data, status):
    r = upload(client, name=name, data=data)
    assert r.status_code == status, r.text
    assert client.get("/api/documents").json()["documents"] == []
    assert client.app.state.rag.status().ntotal == 0


def test_upload_filename_cannot_escape_upload_dir(client, settings, tmp_path):
    r = upload(client, name="../../../../tmp/evil.txt", data=TXT)
    assert r.status_code == 201
    assert r.json()["filename"] == "evil.txt"  # directory components dropped
    stored = list((tmp_path / "uploads").iterdir())
    assert len(stored) == 1 and stored[0].parent == tmp_path / "uploads"
    assert stored[0].name != "evil.txt"  # internal name is a generated id
    assert not (tmp_path.parent / "evil.txt").exists()


def test_upload_size_limit(make_client):
    c = make_client(max_upload_size_mb=1)
    r = upload(c, data=b"a " * (1024 * 1024))  # ~2 MB
    assert r.status_code == 413
    assert c.get("/api/documents").json()["documents"] == []


# ---------------------------------------------------------------------------- settings
def test_settings_roundtrip_and_validation(client):
    s = client.get("/api/settings").json()
    assert s["model"] == "openai/gpt-oss-120b" and s["theme"] == "system" and s["top_k"] == 5
    assert s["integrations"] == {"groq": True, "tavily": True, "openweather": True, "langsmith": False}
    updated = client.post("/api/settings", json={"theme": "dark", "temperature": 0.7, "top_k": 3,
                                                   "response_style": "concise"}).json()
    assert updated["theme"] == "dark" and updated["top_k"] == 3 and updated["response_style"] == "concise"
    assert client.get("/api/settings").json()["temperature"] == 0.7  # persisted
    assert client.post("/api/settings", json={"model": "not-a-model"}).status_code == 422
    assert client.post("/api/settings", json={"theme": "neon"}).status_code == 422
    assert not any(secret in str(updated) for secret in SECRET_VALUES)


@pytest.mark.parametrize("field", ["groq_api_key", "database_url", "upload_dir", "tavily_api_key", "app_env"])
def test_settings_endpoint_rejects_config_and_secrets(client, field):
    r = client.post("/api/settings", json={field: "evil"})
    assert r.status_code == 422


def test_settings_are_per_session(client, other_client):
    client.post("/api/settings", json={"theme": "dark"})
    assert other_client.get("/api/settings").json()["theme"] == "system"


def test_saved_preferences_are_used_by_chat(client, script):
    client.post("/api/settings", json={"model": "openai/gpt-oss-20b", "response_style": "concise"})
    script.reply("short")
    r = client.post("/api/chat/stream", json={"message": "hi"})
    assert parse_sse(r.text)[0][1]["model"] == "openai/gpt-oss-20b"
    assert "Be brief" in script.seen[0][0].content


def test_memories_endpoints(client, script):
    script.call("remember_information", {"content": "Prefers Python over Java"}).reply("Saved.")
    client.post("/api/chat", json={"message": "remember I prefer Python over Java"})
    mems = client.get("/api/memories").json()["memories"]
    assert [m["content"] for m in mems] == ["Prefers Python over Java"]
    mid = mems[0]["memory_id"]
    assert client.patch(f"/api/memories/{mid}", json={"content": "Prefers Python"}).status_code == 200
    assert client.patch(f"/api/memories/{mid}", json={"content": "password is hunter2hunter2"}).status_code == 422
    assert client.delete(f"/api/memories/{mid}").status_code == 204
    assert client.delete(f"/api/memories/{mid}").status_code == 404


def test_rate_limit_returns_429_with_retry_after(make_client, script):
    c = make_client(rate_limit_per_minute=2)
    for _ in range(2):
        script.reply("ok")
        assert c.post("/api/chat", json={"message": "hi"}).status_code == 200
    r = c.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 429 and "retry-after" in r.headers and r.json()["code"] == "rate_limited"
    assert c.get("/api/conversations").status_code == 200  # reads are not limited

"""Security controls: filenames/paths, redaction, sensitive-data detection, rate limiting,
CSRF/origin checks, config validation, secret handling and safe errors."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from conftest import FAKE_GROQ_KEY, FAKE_TAVILY_KEY, FAKE_WEATHER_KEY, SECRET_VALUES, make_settings
from config import Settings, get_settings, reset_settings_cache
from exceptions import ConfigurationError, RateLimitedError, ValidationFailed
from security import (
    RateLimiter,
    is_valid_scope_id,
    looks_sensitive,
    redact,
    resolve_inside,
    safe_extension,
    sanitize_filename,
    validate_thread_id,
)


# ---------------------------------------------------------------- filenames / paths
@pytest.mark.parametrize("raw,expected", [
    ("report.pdf", "report.pdf"),
    ("../../etc/passwd", "passwd"),
    ("..\\..\\windows\\system32\\cmd.exe", "cmd.exe"),
    ("/abs/path/file.txt", "file.txt"),
    ("na<me>:with|bad*chars?.txt", "na_me__with_bad_chars_.txt"),
    ("evil\x00.txt", "evil.txt"),
    ("   ...   ", "document"),
    ("", "document"),
    (None, "document"),
    ("..", "document"),
])
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_limits_length_and_keeps_extension():
    out = sanitize_filename("a" * 400 + ".pdf")
    assert len(out) <= 150 and out.endswith(".pdf")


def test_safe_extension_is_lowercase_and_basename_only():
    assert safe_extension("Doc.PDF") == ".pdf"
    assert safe_extension("../x.docx") == ".docx"
    assert safe_extension("archive.tar.gz") == ".gz"
    assert safe_extension("noext") == ""


def test_resolve_inside_blocks_traversal_and_symlink_escape(tmp_path):
    base = tmp_path / "uploads"
    base.mkdir()
    assert resolve_inside(base, "file.txt") == (base / "file.txt").resolve()
    for evil in ("../escape.txt", "../../etc/passwd", "sub/../../escape"):
        with pytest.raises(ValidationFailed):
            resolve_inside(base, evil)
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "link").symlink_to(outside)
    with pytest.raises(ValidationFailed):
        resolve_inside(base, "link", "stolen.txt")


def test_identifier_validation():
    assert validate_thread_id("123e4567-e89b-42d3-a456-426614174000")
    for bad in ("", "../../x", "123", "123e4567-e89b-42d3-a456-42661417400z", "A" * 36, None, 5):
        with pytest.raises(ValidationFailed):
            validate_thread_id(bad)  # type: ignore[arg-type]
    assert is_valid_scope_id("a" * 43) and not is_valid_scope_id("short") and not is_valid_scope_id(None)


# ---------------------------------------------------------------- redaction / detection
@pytest.mark.parametrize("text", [
    f"https://api.openweathermap.org/data/2.5/weather?q=Paris&appid={FAKE_WEATHER_KEY}",
    f"Authorization: Bearer {FAKE_GROQ_KEY}",
    f"key={FAKE_GROQ_KEY}",
    f"{FAKE_TAVILY_KEY} leaked",
    "token=abcdefghijklmnop",
])
def test_redact_removes_credentials(text):
    out = redact(text)
    assert not any(s in out for s in SECRET_VALUES) and "abcdefghijklmnop" not in out
    assert "[REDACTED]" in out


def test_redact_leaves_normal_text_alone():
    assert redact("The weather in Paris is sunny") == "The weather in Paris is sunny"


@pytest.mark.parametrize("text,expected", [
    ("my password is correcthorse", True),
    ("api_key = abcd1234", True),
    ("-----BEGIN PRIVATE KEY-----", True),
    ("AKIAABCDEFGHIJKLMNOP", True),
    ("ghp_" + "a" * 30, True),
    ("card: 4111 1111 1111 1111", True),
    ("order number 1234 5678 9012 3456", False),  # fails the Luhn check
    ("I like hiking and Python", False),
    ("My birthday is in May", False),
])
def test_looks_sensitive(text, expected):
    assert looks_sensitive(text) is expected


# ---------------------------------------------------------------- rate limiter
def test_rate_limiter_blocks_then_recovers():
    rl = RateLimiter(limit=3, window_seconds=0.2)
    for _ in range(3):
        rl.check("k")
    with pytest.raises(RateLimitedError) as info:
        rl.check("k")
    assert int(info.value.headers["Retry-After"]) >= 1
    rl.check("other-key")  # independent per key
    time.sleep(0.25)
    rl.check("k")


# ---------------------------------------------------------------- HTTP-level controls
def test_cross_origin_state_changes_are_blocked(client):
    r = client.post("/api/conversations", json={}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and r.json()["code"] == "forbidden"
    r = client.post("/api/conversations", json={}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert client.get("/api/conversations").json()["conversations"] == []  # nothing was created


def test_same_origin_and_allowed_origin_pass(client):
    assert client.post("/api/conversations", json={}, headers={"Origin": "http://testserver"}).status_code == 201
    assert client.post("/api/conversations", json={}, headers={"Origin": "http://localhost:8080"}).status_code == 201
    assert client.post("/api/conversations", json={}).status_code == 201  # non-browser clients (no Origin)


def test_cors_preflight_only_for_configured_origins(client):
    ok = client.options("/api/chat", headers={"Origin": "http://localhost:8080",
                                              "Access-Control-Request-Method": "POST"})
    bad = client.options("/api/chat", headers={"Origin": "http://evil.example",
                                               "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8080"
    assert "access-control-allow-origin" not in bad.headers


def test_secrets_never_appear_in_any_response(client, script):
    script.reply("hello")
    client.post("/api/chat", json={"message": "hi"})
    paths = ["/", "/health", "/api/models", "/api/settings", "/api/conversations", "/api/documents",
             "/api/memories", "/api/nope"]
    for p in paths:
        r = client.get(p)
        assert not any(s in r.text for s in SECRET_VALUES), p
        assert not any(s in str(r.headers) for s in SECRET_VALUES), p


def test_static_assets_contain_no_secrets_or_keys():
    root = Path(__file__).resolve().parent.parent
    for f in list((root / "static").rglob("*.js")) + list((root / "templates").glob("*.html")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        assert "gsk_" not in text and "tvly-" not in text and "GROQ_API_KEY" not in text, f


def test_unhandled_errors_return_generic_500_without_traceback(client):
    app = client.app

    @app.get("/boom")
    async def boom():  # pragma: no cover - exercised through the client
        raise RuntimeError(f"db password hunter2 key {FAKE_GROQ_KEY}")

    from fastapi.testclient import TestClient

    safe = TestClient(app, raise_server_exceptions=False)
    r = safe.get("/boom")
    assert r.status_code == 500 and r.json()["error"] == "Internal server error."
    assert "hunter2" not in r.text and FAKE_GROQ_KEY not in r.text and "Traceback" not in r.text
    assert r.json()["request_id"]


def test_request_id_is_echoed_or_generated(client):
    assert client.get("/health", headers={"X-Request-ID": "abc123def"}).headers["x-request-id"] == "abc123def"
    generated = client.get("/health", headers={"X-Request-ID": "bad id with spaces!"}).headers["x-request-id"]
    assert generated != "bad id with spaces!" and generated.isalnum()


def test_session_cookie_flags(make_client):
    c = make_client(cookie_secure=True)
    cookie = c.get("/api/models").headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie.replace("Lax", "lax") and "Secure" in cookie


def test_raw_db_never_stores_the_bearer_cookie(client, script):
    script.reply("x")
    client.post("/api/chat", json={"message": "hi"})
    token = client.cookies.get("akgpt_session")
    import sqlite3

    path = client.app.state.settings.sqlite_path
    dump = "\n".join(sqlite3.connect(path).iterdump())
    assert token and token not in dump  # only a hash of it is used as the scope id


# ---------------------------------------------------------------- configuration
def test_settings_repr_and_str_hide_secrets(settings):
    for rendered in (repr(settings), str(settings), f"{settings}"):
        assert not any(s in rendered for s in SECRET_VALUES)


def test_secret_fields_are_secretstr(settings):
    assert settings.groq_api_key.get_secret_value() == FAKE_GROQ_KEY
    assert FAKE_GROQ_KEY not in repr(settings.groq_api_key)
    assert settings.key_status() == {"groq": True, "tavily": True, "openweather": True, "langsmith": False}


def test_blank_keys_are_treated_as_missing(tmp_path):
    s = make_settings(tmp_path, groq_api_key="  ", tavily_api_key="", openweather_api_key=None)
    assert s.key_status() == {"groq": False, "tavily": False, "openweather": False, "langsmith": False}
    warnings = " ".join(s.startup_warnings())
    assert "GROQ_API_KEY" in warnings and "TAVILY_API_KEY" in warnings and "OPENWEATHER_API_KEY" in warnings


def test_app_starts_without_any_optional_key(make_client):
    c = make_client(groq_api_key=None, tavily_api_key=None, openweather_api_key=None)
    assert c.get("/").status_code == 200 and c.get("/health").status_code == 200


def test_invalid_configuration_gives_actionable_error(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "900")
    reset_settings_cache()
    with pytest.raises(ConfigurationError) as info:
        get_settings()
    assert "CHUNK_OVERLAP" in info.value.message
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    monkeypatch.setenv("APP_ENV", "staging")
    reset_settings_cache()
    with pytest.raises(ConfigurationError) as info:
        get_settings()
    assert "app_env" in info.value.message


def test_non_sqlite_database_url_is_rejected():
    with pytest.raises(Exception) as info:
        Settings(_env_file=None, database_url="postgresql://u:p@h/db")
    assert "sqlite" in str(info.value).lower()


def test_wildcard_cors_forbidden_in_production(tmp_path):
    s = make_settings(tmp_path, app_env="production", allowed_origins="*")
    with pytest.raises(ConfigurationError):
        s.cors_origins
    assert make_settings(tmp_path, allowed_origins="http://a.test/, http://b.test").cors_origins == [
        "http://a.test", "http://b.test"]


def test_production_without_secure_cookie_warns(tmp_path):
    s = make_settings(tmp_path, app_env="production", cookie_secure=False)
    assert any("COOKIE_SECURE" in w for w in s.startup_warnings())


def test_env_example_contains_placeholders_only():
    root = Path(__file__).resolve().parent.parent
    for line in (root / ".env.example").read_text().splitlines():
        if line.startswith(("GROQ_API_KEY", "TAVILY_API_KEY", "OPENWEATHER_API_KEY", "LANGSMITH_API_KEY")):
            assert line.split("=", 1)[1].strip() == "", line


def test_docker_context_excludes_secrets_and_runtime_data():
    root = Path(__file__).resolve().parent.parent
    ignore = (root / ".dockerignore").read_text()
    for needle in (".env", ".git", "__pycache__", ".venv", ".pytest_cache", "data/*", "uploads/*", "faiss_db/*"):
        assert needle in ignore, needle
    dockerfile = (root / "Dockerfile").read_text()
    assert "USER akgpt" in dockerfile and "HEALTHCHECK" in dockerfile and "--reload" not in dockerfile
    assert "COPY .env" not in dockerfile and "EXPOSE 8080" in dockerfile

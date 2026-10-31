"""Agent tools with mocked Tavily / OpenWeather: success paths, every failure mode,
missing keys, argument validation, scope isolation, calculator safety."""
from __future__ import annotations

import json

import httpx
import pytest

import tools
from conftest import FAKE_TAVILY_KEY, FAKE_WEATHER_KEY, SECRET_VALUES, ingest_bytes
from rag import HashEmbedder, RagService

CONFIG = {"configurable": {"scope_id": "S" * 43, "thread_id": "T" * 8}}


@pytest.fixture
def wired(settings, db, rag, http_mock):
    tools.configure_tools(settings, db, rag, transport=http_mock.transport)
    return settings


@pytest.fixture
def conv(db):
    c = db.create_conversation("S" * 43)
    return {"configurable": {"scope_id": "S" * 43, "thread_id": c.thread_id}}


async def call(name: str, args: dict, config=None) -> dict:
    out = await tools.TOOL_REGISTRY[name].ainvoke(args, config or CONFIG)
    return json.loads(out)


# --------------------------------------------------------------------------- registry
def test_registry_is_allowlisted_and_has_no_code_execution():
    names = {t.name for t in tools.get_tools()}
    assert names == {"web_search", "get_weather", "calculator", "search_uploaded_documents",
                     "remember_information", "recall_memory", "search_conversation_history",
                     "update_memory", "forget_memory"}
    assert not names & {"python", "shell", "bash", "exec", "eval"}


async def test_extra_or_invalid_arguments_return_error_not_exception(wired):
    out = await tools.TOOL_REGISTRY["calculator"].ainvoke({"expression": "1+1", "evil": "x"}, CONFIG)
    assert "validation error" in out.lower()  # unknown argument -> text result, the graph keeps running
    out = await tools.TOOL_REGISTRY["web_search"].ainvoke({"query": "x" * 1000}, CONFIG)
    assert "validation error" in out.lower()
    out = await tools.TOOL_REGISTRY["calculator"].ainvoke({}, CONFIG)
    assert "validation error" in out.lower()


# --------------------------------------------------------------------------- calculator
@pytest.mark.parametrize("expr,expected", [
    ("2+2", 4), ("2*(3+4)**2", 98), ("10/4", 2.5), ("7//2", 3), ("7%4", 3), ("-3+5", 2),
    ("sqrt(144)", 12), ("math.sqrt(16)", 4), ("round(3.14159, 2)", 3.14), ("max(1, 9, 3)", 9),
    ("factorial(5)", 120), ("log10(1000)", 3), ("pi*2", 6.283185307179), ("12 × 3", 36),
])
async def test_calculator_values(wired, expr, expected):
    data = await call("calculator", {"expression": expr})
    assert data["ok"] is True and data["result"] == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__[0].__subclasses__()",
    "[x for x in range(10)]",
    "lambda: 1",
    "eval('1+1')",
    "exec('x=1')",
    "os.getcwd()",
    "math.__dict__",
    "'a' * 5",
    "True + 1",
    "1 if 1 else 2",
    "abs.__call__(1)",
    "9**9**9",
    "10**100000",
    "factorial(100000)",
    "1/0",
    "sqrt(-1)",
    "2 ^ 3",
    "x + 1",
    "1;2",
    "1+" * 100 + "1",
])
async def test_calculator_rejects_unsafe_or_invalid_input(wired, expr):
    data = await call("calculator", {"expression": expr})
    assert data["ok"] is False and data["code"] == "invalid_expression"
    assert "pwned" not in json.dumps(data)


def test_deeply_parenthesised_but_valid_expression_still_works():
    assert tools.safe_eval("(" * 100 + "1" + ")" * 100) == 1


def test_safe_eval_never_uses_eval_or_exec():
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(tools.safe_eval)))
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"eval", "exec", "compile", "__import__", "getattr", "open"}


# --------------------------------------------------------------------------- web search
TAVILY_OK = {"results": [
    {"title": "AKGPT news", "url": "https://example.com/a", "content": "Alpha snippet", "score": 0.9},
    {"title": "Bad scheme", "url": "javascript:alert(1)", "content": "nope"},
    {"title": "Second", "url": "https://example.org/b", "content": "Beta snippet"},
]}


async def test_web_search_success_filters_bad_urls_and_cites(wired, http_mock):
    http_mock.respond(200, TAVILY_OK)
    data = await call("web_search", {"query": "akgpt latest", "max_results": 3})
    assert data["ok"] is True and data["untrusted_content"] is True
    assert [r["url"] for r in data["results"]] == ["https://example.com/a", "https://example.org/b"]
    assert data["sources"][0] == {"title": "AKGPT news", "url": "https://example.com/a"}
    req = http_mock.requests[0]
    assert req.headers["authorization"] == f"Bearer {FAKE_TAVILY_KEY}"  # key only in the outbound header
    assert json.loads(req.content)["max_results"] == 3
    assert FAKE_TAVILY_KEY not in json.dumps(data)


async def test_web_search_empty_results(wired, http_mock):
    http_mock.respond(200, {"results": []})
    data = await call("web_search", {"query": "zzzz qqqq"})
    assert data["ok"] is True and data["results"] == [] and "No results" in data["note"]


@pytest.mark.parametrize("status,fragment", [
    (401, "API key"), (403, "API key"), (429, "rate limited"), (432, "rate limited"), (500, "HTTP 500"),
])
async def test_web_search_http_errors(wired, http_mock, status, fragment):
    http_mock.respond(status, {"detail": "x"})
    data = await call("web_search", {"query": "anything"})
    assert data["ok"] is False and fragment in data["error"]
    assert FAKE_TAVILY_KEY not in json.dumps(data)


async def test_web_search_timeout_network_and_garbage(wired, http_mock):
    http_mock.raise_(httpx.ReadTimeout("slow"))
    assert (await call("web_search", {"query": "anything"}))["code"] == "timeout"
    http_mock.raise_(httpx.ConnectError("down"))
    assert (await call("web_search", {"query": "anything"}))["code"] == "upstream_error"
    http_mock.respond(200, text="<html>not json</html>")
    data = await call("web_search", {"query": "anything"})
    assert data["ok"] is False and "unreadable" in data["error"]


async def test_web_search_without_key(settings, db, rag, http_mock, tmp_path):
    from conftest import make_settings

    s = make_settings(tmp_path, tavily_api_key=None)
    tools.configure_tools(s, db, rag, transport=http_mock.transport)
    data = await call("web_search", {"query": "anything"})
    assert data["ok"] is False and "TAVILY_API_KEY" in data["error"]
    assert http_mock.requests == []  # never even attempted a request


# --------------------------------------------------------------------------- weather
OWM_OK = {"name": "Hyderabad", "sys": {"country": "IN"}, "coord": {"lat": 17.38, "lon": 78.47}, "dt": 1_761_800_000,
          "main": {"temp": 29.4, "feels_like": 31.0, "humidity": 62, "pressure": 1009},
          "wind": {"speed": 3.2, "deg": 240}, "clouds": {"all": 40}, "weather": [{"description": "scattered clouds"}]}


async def test_weather_success_metric(wired, http_mock):
    http_mock.respond(200, OWM_OK)
    data = await call("get_weather", {"city": "Hyderabad", "country_code": "in"})
    assert data["ok"] is True and data["temperature"] == 29.4 and data["feels_like"] == 31.0
    assert data["humidity"] == 62 and data["wind_speed"] == 3.2 and data["description"] == "scattered clouds"
    assert data["units"]["temperature"] == "°C" and data["units"]["wind_speed"] == "m/s"
    assert data["location"] == {"name": "Hyderabad", "country": "IN", "lat": 17.38, "lon": 78.47}
    assert data["observed_at_utc"].startswith("2025-")
    assert "Current conditions only" in data["note"]  # never pretends to be a forecast
    req = http_mock.requests[0]
    assert req.url.params["q"] == "Hyderabad,IN" and req.url.params["units"] == "metric"
    assert FAKE_WEATHER_KEY not in json.dumps(data)  # appid only on the outbound request


async def test_weather_units_from_run_config(wired, http_mock):
    http_mock.respond(200, OWM_OK)
    cfg = {"configurable": {**CONFIG["configurable"], "weather_units": "imperial"}}
    data = await call("get_weather", {"city": "Austin"}, cfg)
    assert data["units"]["temperature"] == "°F" and data["units"]["wind_speed"] == "mph"
    assert http_mock.requests[0].url.params["units"] == "imperial"


@pytest.mark.parametrize("status,fragment", [(404, "not found"), (401, "API key"), (429, "rate limit"), (500, "HTTP 500")])
async def test_weather_http_errors(wired, http_mock, status, fragment):
    http_mock.respond(status, {"cod": status})
    data = await call("get_weather", {"city": "Nowhereville"})
    assert data["ok"] is False and fragment in data["error"]
    assert FAKE_WEATHER_KEY not in json.dumps(data)


async def test_weather_malformed_timeout_and_validation(wired, http_mock):
    http_mock.respond(200, {"unexpected": True})
    assert "unexpected response" in (await call("get_weather", {"city": "X"}))["error"]
    http_mock.raise_(httpx.ConnectTimeout("slow"))
    assert (await call("get_weather", {"city": "X"}))["code"] == "timeout"
    out = await tools.TOOL_REGISTRY["get_weather"].ainvoke({"city": "X", "country_code": "India"}, CONFIG)
    assert "validation error" in out.lower()  # 2-letter code enforced by the schema


async def test_weather_without_key(db, rag, http_mock, tmp_path):
    from conftest import make_settings

    s = make_settings(tmp_path, openweather_api_key=None)
    tools.configure_tools(s, db, rag, transport=http_mock.transport)
    data = await call("get_weather", {"city": "Paris"})
    assert data["ok"] is False and "OPENWEATHER_API_KEY" in data["error"] and http_mock.requests == []


# --------------------------------------------------------------------------- memory
async def test_memory_remember_recall_update_forget(wired, db, conv):
    saved = await call("remember_information", {"content": "User prefers dark mode"}, conv)
    assert saved["ok"] is True
    recalled = await call("recall_memory", {"query": "dark"}, conv)
    assert [m["content"] for m in recalled["memories"]] == ["User prefers dark mode"]
    mid = saved["memory_id"]
    assert (await call("update_memory", {"memory_id": mid, "content": "User prefers light mode"}, conv))["ok"]
    assert (await call("recall_memory", {}, conv))["memories"][0]["content"] == "User prefers light mode"
    assert (await call("forget_memory", {"memory_id": mid}, conv))["ok"]
    assert (await call("recall_memory", {}, conv))["memories"] == []
    assert (await call("forget_memory", {"memory_id": mid}, conv))["ok"] is False


@pytest.mark.parametrize("secret", [
    "my password is hunter2hunter2", "api key: gsk_abcdefghijklmnop1234", "card 4111 1111 1111 1111",
    "-----BEGIN RSA PRIVATE KEY-----",
])
async def test_memory_refuses_secrets(wired, db, conv, secret):
    data = await call("remember_information", {"content": secret}, conv)
    assert data["ok"] is False and "not saved" in data["error"]
    assert db.list_memories("S" * 43) == []


async def test_memory_is_scoped_per_session(wired, db, conv):
    await call("remember_information", {"content": "Alice likes tea"}, conv)
    other = {"configurable": {"scope_id": "O" * 43, "thread_id": "x" * 8}}
    assert (await call("recall_memory", {}, other))["memories"] == []
    mid = db.list_memories("S" * 43)[0].memory_id
    assert (await call("forget_memory", {"memory_id": mid}, other))["ok"] is False  # cannot delete others' memory
    assert len(db.list_memories("S" * 43)) == 1


async def test_tools_need_conversation_context(wired):
    data = await call("recall_memory", {}, {"configurable": {}})
    assert data["ok"] is False and "context" in data["error"]


async def test_conversation_history_search_is_thread_scoped(wired, db):
    a, b = db.create_conversation("S" * 43), db.create_conversation("S" * 43)
    db.add_message("S" * 43, a.thread_id, "user", "We talked about quantum computing")
    db.add_message("S" * 43, b.thread_id, "user", "We talked about gardening")
    cfg_a = {"configurable": {"scope_id": "S" * 43, "thread_id": a.thread_id}}
    hits = await call("search_conversation_history", {"query": "quantum"}, cfg_a)
    assert len(hits["matches"]) == 1 and "persistent memory" in hits["note"].lower() or "history" in hits["note"].lower()
    assert (await call("search_conversation_history", {"query": "gardening"}, cfg_a))["matches"] == []


# --------------------------------------------------------------------------- documents
async def test_document_search_returns_cited_chunks(wired, settings, db, rag, conv):
    cfg = conv["configurable"]
    data = (b"Quarterly revenue grew twelve percent thanks to the new Hyderabad office. " * 5)
    await ingest_bytes(rag, settings, cfg["scope_id"], cfg["thread_id"], "report.txt", data)
    out = await call("search_uploaded_documents", {"query": "quarterly revenue growth"}, conv)
    assert out["ok"] is True and out["chunks"], out
    assert out["chunks"][0]["citation"].startswith("[report.txt")
    assert out["untrusted_content"] is True and "untrusted" in out["note"].lower()


async def test_document_search_with_no_documents_says_so(wired, conv):
    out = await call("search_uploaded_documents", {"query": "anything at all"}, conv)
    assert out["ok"] is True and out["chunks"] == [] and "invent" in out["note"].lower()


async def test_document_search_cannot_cross_sessions_or_threads(wired, settings, db, rag, conv):
    cfg = conv["configurable"]
    await ingest_bytes(rag, settings, cfg["scope_id"], cfg["thread_id"], "secret.txt",
                       b"The launch code is swordfish and the vault is under the library. " * 4)
    other_scope = {"configurable": {"scope_id": "O" * 43, "thread_id": cfg["thread_id"]}}
    assert (await call("search_uploaded_documents", {"query": "launch code swordfish"}, other_scope))["chunks"] == []
    other_thread = {"configurable": {"scope_id": cfg["scope_id"], "thread_id": "elsewhere"}}
    assert (await call("search_uploaded_documents", {"query": "launch code swordfish"}, other_thread))["chunks"] == []


async def test_tool_output_is_bounded(wired, http_mock):
    big = {"results": [{"title": f"t{i}", "url": f"https://e.com/{i}", "content": "x" * 600} for i in range(10)]}
    http_mock.respond(200, big)
    out = await tools.TOOL_REGISTRY["web_search"].ainvoke({"query": "lots", "max_results": 10}, CONFIG)
    assert len(out) <= tools.MAX_TOOL_OUTPUT_CHARS + 200


async def test_no_secret_ever_appears_in_any_tool_output(wired, http_mock):
    http_mock.respond(500, {"error": f"leak {FAKE_TAVILY_KEY} {FAKE_WEATHER_KEY}"})
    for name, args in (("web_search", {"query": "abc"}), ("get_weather", {"city": "Paris"})):
        out = await tools.TOOL_REGISTRY[name].ainvoke(args, CONFIG)
        assert not any(s in out for s in SECRET_VALUES)

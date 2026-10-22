"""HTTP API tests. The security layer is exercised first because it protects shell access."""

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from kestrel.app import App
from kestrel.llm import HashingEmbedder, ScriptedChat, text_reply, tool_reply
from kestrel.web.runs import Approvals, RunManager
from kestrel.web.server import create_web_app

TOKEN = "test-token"
HOST = "127.0.0.1:8765"
AUTH = {"x-kestrel-token": TOKEN, "host": HOST}


def sse_events(text: str) -> list[dict]:
    return [json.loads(chunk[6:]) for chunk in text.split("\n\n") if chunk.startswith("data: ")]


@pytest.fixture
async def make_web(settings):
    apps = []

    async def factory(script, *, approval="auto"):
        settings.approval = approval
        approvals = Approvals(timeout=5)
        app = await App.create(settings, llm=ScriptedChat(script), embedder=HashingEmbedder(), approver=approvals.request, use_mcp=False)
        apps.append(app)
        web = create_web_app(app, approvals, TOKEN, allowed_hosts={HOST, "localhost:8765"})
        return app, approvals, web

    yield factory
    for app in apps:
        await app.aclose()


def client(web) -> TestClient:
    return TestClient(web, base_url=f"http://{HOST}")


# ---- security ---------------------------------------------------------------

async def test_api_requires_the_token(make_web):
    _, _, web = await make_web([])
    c = client(web)
    assert c.get("/api/state").status_code == 401
    assert c.get("/api/state", headers={"x-kestrel-token": "wrong"}).status_code == 401
    assert c.get("/api/state", headers={"x-kestrel-token": TOKEN}).status_code == 200


async def test_unexpected_host_header_is_refused(make_web):
    _, _, web = await make_web([])
    c = client(web)
    assert c.get("/api/state", headers={"x-kestrel-token": TOKEN, "host": "evil.example:8765"}).status_code == 403
    assert c.get("/", headers={"host": "evil.example"}).status_code == 403  # DNS-rebinding defence covers static pages too


async def test_cross_origin_writes_are_refused(make_web):
    _, _, web = await make_web([])
    c = client(web)
    bad = c.post("/api/sessions", headers={**AUTH, "origin": "http://evil.example"}, json={})
    assert bad.status_code == 403
    ok = c.post("/api/sessions", headers={**AUTH, "origin": f"http://{HOST}"}, json={})
    assert ok.status_code == 201


async def test_security_headers_and_no_store(make_web):
    _, _, web = await make_web([])
    c = client(web)
    page = c.get("/", headers={"host": HOST})
    assert page.status_code == 200 and "Kestrel" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"] and page.headers["x-content-type-options"] == "nosniff"
    assert c.get("/api/state", headers=AUTH).headers["cache-control"] == "no-store"
    assert "'unsafe-inline'" not in page.headers["content-security-policy"]


async def test_static_assets_are_served_and_page_has_no_inline_code(make_web):
    _, _, web = await make_web([])
    c = client(web)
    html = c.get("/", headers={"host": HOST}).text
    assert "<script src=\"/app.js\">" in html and "onclick=" not in html and " style=" not in html
    assert c.get("/app.js", headers={"host": HOST}).status_code == 200
    assert c.get("/style.css", headers={"host": HOST}).status_code == 200


# ---- sessions & chat --------------------------------------------------------

async def test_chat_streams_tokens_tool_events_and_a_done_event(make_web, settings):
    app, _, web = await make_web([tool_reply("list_dir", {"path": "."}), text_reply("The workspace is empty.")])
    c = client(web)
    sid = c.post("/api/sessions", headers=AUTH, json={}).json()["id"]
    response = c.post(f"/api/sessions/{sid}/messages", headers=AUTH, json={"text": "what's in the workspace?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    types = [e["type"] for e in events]
    assert types[:2] == ["tool_call", "tool_result"] and "token" in types and types[-1] == "done"
    call, result = events[0], events[1]
    assert call["id"] == result["id"] and call["name"] == "list_dir" and not result["is_error"]
    done = events[-1]
    assert done["status"] == "completed" and done["final_text"] == "The workspace is empty." and done["steps"] == 2


async def test_messages_endpoint_projects_the_log_for_display(make_web):
    _, _, web = await make_web([tool_reply("list_dir", {"path": "."}), text_reply("done")])
    c = client(web)
    sid = c.post("/api/sessions", headers=AUTH, json={}).json()["id"]
    c.post(f"/api/sessions/{sid}/messages", headers=AUTH, json={"text": "hi"})
    data = c.get(f"/api/sessions/{sid}/messages", headers=AUTH).json()
    roles = [m["role"] for m in data["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"] and data["running"] is False
    assistant = data["messages"][1]
    assert assistant["tool_calls"][0]["name"] == "list_dir" and data["messages"][2]["tool_call_id"] == assistant["tool_calls"][0]["id"]
    listed = c.get("/api/sessions", headers=AUTH).json()
    assert listed[0]["id"] == sid and listed[0]["title"] == "hi" and listed[0]["messages"] == 4


async def test_input_validation(make_web):
    _, _, web = await make_web([])
    c = client(web)
    sid = c.post("/api/sessions", headers=AUTH, json={}).json()["id"]
    assert c.post(f"/api/sessions/{sid}/messages", headers=AUTH, json={"text": "   "}).status_code == 400
    assert c.post(f"/api/sessions/{sid}/messages", headers=AUTH, json={}).status_code == 400
    assert c.post(f"/api/sessions/{sid}/messages", headers=AUTH, content="not json").status_code == 400
    assert c.post(f"/api/sessions/{sid}/messages", headers=AUTH, json={"text": "x" * 60_000}).status_code == 413
    assert c.post("/api/approval-mode", headers=AUTH, json={"mode": "yolo"}).status_code == 400
    assert c.post("/api/approvals/nope", headers=AUTH, json={"allow": True}).status_code == 404
    assert c.post("/api/approvals/nope", headers=AUTH, json={"allow": "yes"}).status_code == 400


async def test_a_failed_run_reports_done_with_the_error(make_web):
    from kestrel.common import LLMError

    def boom(messages):
        raise LLMError("provider down")

    _, _, web = await make_web([boom])
    c = client(web)
    sid = c.post("/api/sessions", headers=AUTH, json={}).json()["id"]
    events = sse_events(c.post(f"/api/sessions/{sid}/messages", headers=AUTH, json={"text": "hi"}).text)
    assert events[-1]["type"] == "done" and events[-1]["status"] == "failed" and "provider down" in events[-1]["error"]


async def test_changing_the_approval_mode_applies_to_the_live_policy(make_web):
    app, _, web = await make_web([])
    c = client(web)
    assert c.post("/api/approval-mode", headers=AUTH, json={"mode": "readonly"}).json() == {"approval": "readonly"}
    assert app.policy.mode == "readonly" and c.get("/api/state", headers=AUTH).json()["approval"] == "readonly"


# ---- approvals (exercised below the HTTP layer so the stream can stay open) ---

async def test_approval_round_trip_allow_and_deny(make_web, settings):
    app, approvals, _ = await make_web([
        tool_reply("write_file", {"path": "a.txt", "content": "v1"}), tool_reply("write_file", {"path": "b.txt", "content": "v2"}), text_reply("finished"),
    ], approval="ask")
    runs = RunManager(app)
    seen = []
    decisions = iter([True, False])
    async for event in runs.stream("s", "write two files"):
        seen.append(event)
        if event["type"] == "approval_request":
            assert approvals.pending()[0]["tool"] == "write_file" and event["risk"] == "write"
            assert approvals.resolve(event["id"], next(decisions))
    assert (settings.workspace / "a.txt").read_text() == "v1"
    assert not (settings.workspace / "b.txt").exists()
    results = [e for e in seen if e["type"] == "tool_result"]
    assert "Wrote" in results[0]["text"] and results[1]["text"].startswith("Denied")
    assert seen[-1]["type"] == "done" and seen[-1]["status"] == "completed"
    assert approvals.pending() == []


async def test_approval_times_out_to_deny(make_web, settings):
    app, approvals, _ = await make_web([tool_reply("write_file", {"path": "a.txt", "content": "x"}), text_reply("ok")], approval="ask")
    approvals.timeout = 0.2
    events = [e async for e in RunManager(app).stream("s", "write")]
    assert any(e["type"] == "approval_expired" for e in events)
    assert not (settings.workspace / "a.txt").exists()


async def test_risky_actions_are_denied_when_no_browser_is_attached(make_web):
    app, approvals, _ = await make_web([], approval="ask")
    tool = app.registry.get("write_file")
    assert await approvals.request(tool, {"path": "x", "content": "y"}) is False   # no active run stream


async def test_stop_cancels_a_running_turn(make_web):
    class Hang:
        model = "hang"

        async def chat(self, messages, tools=None, **kwargs):
            await asyncio.sleep(30)

    app, _, _ = await make_web([])
    app.agent.llm = Hang()
    runs = RunManager(app)
    seen = []

    async def consume():
        async for event in runs.stream("s", "hello"):
            seen.append(event)

    consumer = asyncio.create_task(consume())
    for _ in range(500):
        if runs.is_running("s"):
            break
        await asyncio.sleep(0.02)
    assert runs.is_running("s")
    with pytest.raises(Exception):
        async for _ in runs.stream("s", "second"):   # a second concurrent run is refused
            pass
    assert runs.stop("s") is True
    await asyncio.wait_for(consumer, 15)
    assert seen[-1]["type"] == "cancelled" and not runs.is_running("s")


# ---- memory, trace, state, jobs -----------------------------------------------

async def test_memory_endpoints(make_web, settings):
    app, _, web = await make_web([])
    c = client(web)
    assert c.post("/api/memory/facts", headers=AUTH, json={"fact": "User likes tea"}).json()["facts"] == ["User likes tea"]
    assert c.post("/api/memory/facts", headers=AUTH, json={"fact": "x" * 500}).status_code == 400
    memory = c.get("/api/memory", headers=AUTH).json()
    assert memory["facts"] == ["User likes tea"] and memory["persona"] == ""   # the persona file is created on the first run
    assert c.request("DELETE", "/api/memory/facts", headers=AUTH, json={"text": "tea"}).json()["removed"] == 0   # UI deletes exact facts only
    assert c.request("DELETE", "/api/memory/facts", headers=AUTH, json={"text": "User likes tea"}).json()["removed"] == 1
    assert c.request("DELETE", "/api/memory/facts", headers=AUTH, json={"text": "a"}).status_code == 400
    assert c.put("/api/memory/persona", headers=AUTH, json={"text": "You are terse."}).status_code == 200
    assert settings.persona_path.read_text().strip() == "You are terse."
    assert c.put("/api/memory/persona", headers=AUTH, json={"text": "  "}).status_code == 400


async def test_memory_search_endpoint(make_web):
    app, _, web = await make_web([])
    await app.episodic.learn("old", 1, "my dog is called Mochi", "lovely")
    c = client(web)
    hits = c.get("/api/memory/search", headers=AUTH, params={"q": "dog name"}).json()["hits"]
    assert hits and "Mochi" in hits[0]["text"] and {"score", "dense", "sparse", "age"} <= set(hits[0])
    assert c.get("/api/memory/search", headers=AUTH, params={"q": ""}).json() == {"hits": []}


async def test_trace_state_and_jobs_endpoints(make_web):
    app, _, web = await make_web([tool_reply("list_dir", {"path": "."}), text_reply("done")])
    c = client(web)
    sid = c.post("/api/sessions", headers=AUTH, json={}).json()["id"]
    c.post(f"/api/sessions/{sid}/messages", headers=AUTH, json={"text": "go"})
    trace = c.get(f"/api/sessions/{sid}/trace", headers=AUTH).json()
    assert trace["summary"]["llm_calls"] == 2 and trace["summary"]["tool_calls"] == 1 and trace["summary"]["by_tool"] == {"list_dir": 1}
    assert {e["kind"] for e in trace["events"]} >= {"llm", "tool"}
    state = c.get("/api/state", headers=AUTH).json()
    names = {t["name"] for t in state["tools"]}
    assert {"read_file", "shell", "spawn_subagent", "load_skill"} <= names
    assert next(t for t in state["tools"] if t["name"] == "shell")["risk"] == "dangerous"
    assert "api_key" not in json.dumps(state).lower()
    assert c.get("/api/jobs", headers=AUTH).json() == []
    assert c.post("/api/jobs/nope/cancel", headers=AUTH, json={}).status_code == 409


async def test_closing_the_stream_cancels_the_run(make_web):
    """A browser disconnect must not leave an orphaned turn running in the background."""

    class Hang:
        model = "hang"

        async def chat(self, messages, tools=None, **kwargs):
            await asyncio.sleep(30)

    app, _, _ = await make_web([])
    app.agent.llm = Hang()
    runs = RunManager(app)
    stream = runs.stream("s", "hello")
    reader = asyncio.create_task(stream.__anext__())
    for _ in range(500):
        if runs.is_running("s"):
            break
        await asyncio.sleep(0.02)
    assert runs.is_running("s")
    reader.cancel()                      # what the server does when the client goes away
    with pytest.raises(asyncio.CancelledError):
        await reader
    await stream.aclose()
    for _ in range(500):
        if not runs.is_running("s"):
            break
        await asyncio.sleep(0.02)
    assert not runs.is_running("s")

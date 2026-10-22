import json

from kestrel.agent import UNKNOWN_OUTCOME, open_calls
from kestrel.common import ChatResponse, ToolCall, Usage
from kestrel.llm import text_reply, tool_reply
from kestrel.tools import Tool, ToolRegistry
from pydantic import BaseModel


def roles(app, session):
    return [m["role"] for _, m in app.store.load_context(session)[2]]


async def test_plain_answer(make_app):
    app = await make_app([text_reply("hello there")])
    result = await app.agent.run("s", "hi")
    assert (result.status, result.final_text, result.steps) == ("completed", "hello there", 1)
    assert roles(app, "s") == ["user", "assistant"]


async def test_tool_loop_reads_a_file_then_answers(make_app, settings):
    app = await make_app([tool_reply("read_file", {"path": "notes.txt"}), text_reply("it says: hello")])
    (settings.workspace / "notes.txt").write_text("hello")
    result = await app.agent.run("s", "what is in notes.txt?")
    assert result.status == "completed" and result.tool_calls == ["read_file"] and result.steps == 2
    assert roles(app, "s") == ["user", "assistant", "tool", "assistant"]
    tool_message = app.store.load_context("s")[2][2][1]
    assert "hello" in tool_message["content"]
    # the second model request must have seen the tool result
    second_request = app.llm.requests[1]["messages"]
    assert second_request[-1]["role"] == "tool"


async def test_invalid_tool_arguments_are_fed_back_so_the_model_can_correct(make_app, settings):
    app = await make_app([tool_reply("read_file", {"path": 123}), tool_reply("read_file", {"path": "a.txt"}), text_reply("done")])
    (settings.workspace / "a.txt").write_text("A")
    result = await app.agent.run("s", "read it")
    rows = app.store.load_context("s")[2]
    first_result = rows[2][1]["content"]
    assert result.status == "completed" and "Invalid arguments" in first_result


async def test_message_is_committed_before_its_tools_run(make_app):
    app = await make_app([tool_reply("probe", {}), text_reply("ok")])
    observed = {}

    class Empty(BaseModel):
        pass

    async def probe(args, ctx):
        observed["roles"] = [m["role"] for _, m in ctx.store.load_context(ctx.session_id)[2]]
        return "fine"

    app.registry.register(Tool.from_model("probe", "p", Empty, probe))
    await app.agent.run("s", "go")
    assert observed["roles"] == ["user", "assistant"]  # the assistant message was already in the log


async def test_step_limit_stops_a_runaway_loop(make_app):
    loop = [tool_reply("list_dir", {"path": "."}) for _ in range(10)]
    app = await make_app(loop, max_steps=3)
    app.agent.max_steps = 3
    result = await app.agent.run("s", "loop forever")
    assert result.status == "step_limit" and result.steps == 3
    assert "step limit" in result.final_text


async def test_empty_responses_are_retried_then_recover(make_app):
    empty = ChatResponse(content="", usage=Usage(1, 0))
    app = await make_app([empty, empty, text_reply("finally")])
    result = await app.agent.run("s", "hi")
    assert result.status == "completed" and result.final_text == "finally"
    # the retry request carries a nudge, and the empty replies were never stored
    assert any("previous reply was empty" in str(m.get("content")) for m in app.llm.requests[1]["messages"])
    assert roles(app, "s") == ["user", "assistant"]


async def test_persistent_empty_responses_fail_cleanly(make_app):
    empty = ChatResponse(content="")
    app = await make_app([empty, empty, empty, empty])
    result = await app.agent.run("s", "hi")
    assert result.status == "failed" and "empty" in result.error


async def test_length_truncation_never_executes_tool_calls(make_app, settings):
    truncated = ChatResponse(content="partial", tool_calls=[ToolCall("c1", "write_file", '{"path": "x.txt", "content": "ha')], finish_reason="length")
    app = await make_app([truncated])
    result = await app.agent.run("s", "write")
    assert result.status == "failed" and not (settings.workspace / "x.txt").exists()
    assert "length" in result.error


async def test_llm_failure_is_reported_not_raised(make_app):
    from kestrel.common import LLMError

    def boom(messages):
        raise LLMError("provider down")

    app = await make_app([boom])
    result = await app.agent.run("s", "hi")
    assert result.status == "failed" and "provider down" in result.error


async def test_parallel_safe_calls_run_concurrently_and_are_stored_in_order(make_app, settings):
    import asyncio

    class Empty(BaseModel):
        pass

    active, peak = 0, 0

    async def slow(args, ctx):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return "done"

    two_calls = ChatResponse(content="", finish_reason="tool_calls", tool_calls=[ToolCall("a", "slow", "{}"), ToolCall("b", "slow", "{}")])
    app = await make_app([two_calls, text_reply("ok")])
    app.registry.register(Tool.from_model("slow", "s", Empty, slow, parallel_safe=True, idempotent=True))
    await app.agent.run("s", "go")
    assert peak == 2
    tool_ids = [m["tool_call_id"] for _, m in app.store.load_context("s")[2] if m["role"] == "tool"]
    assert tool_ids == ["a", "b"]


async def test_non_parallel_calls_run_one_at_a_time(make_app):
    import asyncio

    class Empty(BaseModel):
        pass

    active, peak = 0, 0

    async def slow(args, ctx):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.03)
        active -= 1
        return "done"

    two = ChatResponse(content="", finish_reason="tool_calls", tool_calls=[ToolCall("a", "slow", "{}"), ToolCall("b", "slow", "{}")])
    app = await make_app([two, text_reply("ok")])
    app.registry.register(Tool.from_model("slow", "s", Empty, slow))
    await app.agent.run("s", "go")
    assert peak == 1


# ---- crash recovery -------------------------------------------------------

def crash_log(app, session, calls):
    """Simulate a crash after the assistant message was committed but before any tool result."""
    app.store.create_session(session_id=session)
    app.store.append(session, {"role": "user", "content": "do it"})
    app.store.append(session, {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}} for cid, name, args in calls]})


async def test_open_calls_are_computed_from_the_log(make_app):
    app = await make_app([])
    crash_log(app, "s", [("a", "read_file", {"path": "x"}), ("b", "shell", {"command": "ls"})])
    app.store.append("s", {"role": "tool", "tool_call_id": "a", "content": "result"})
    assert [c.id for c in open_calls(app.store.rows("s"))] == ["b"]


async def test_recovery_reruns_idempotent_tools_and_flags_unknown_outcomes(make_app, settings):
    app = await make_app([text_reply("recovered")])
    (settings.workspace / "x.txt").write_text("content")
    crash_log(app, "s", [("a", "read_file", {"path": "x.txt"}), ("b", "shell", {"command": "echo side-effect"})])
    result = await app.agent.run("s", None)  # resume without a new user message
    assert result.status == "completed"
    results = {m["tool_call_id"]: m["content"] for _, m in app.store.load_context("s")[2] if m["role"] == "tool"}
    assert "content" in results["a"]               # idempotent: safely re-run
    assert results["b"] == UNKNOWN_OUTCOME         # not idempotent: never blindly repeated
    events = [e["name"] for e in app.store.events("s") if e["kind"] == "recovery"]
    assert sorted(events) == ["retry_idempotent", "unknown_outcome"]


async def test_resume_does_not_duplicate_the_user_message(make_app):
    app = await make_app([text_reply("ok")])
    crash_log(app, "s", [("a", "list_dir", {"path": "."})])
    await app.agent.run("s", None)
    assert roles(app, "s").count("user") == 1


async def test_sessions_are_isolated_and_serialised(make_app):
    import asyncio

    app = await make_app([text_reply("one"), text_reply("two")])
    a, b = await asyncio.gather(app.agent.run("s1", "x"), app.agent.run("s2", "y"))
    assert {a.final_text, b.final_text} == {"one", "two"}
    assert roles(app, "s1") == ["user", "assistant"] and roles(app, "s2") == ["user", "assistant"]


def valid_for_providers(messages) -> bool:
    """Every tool message must directly follow the assistant message that requested it (or another tool message of the same batch)."""
    previous = None
    for message in messages:
        if message["role"] == "tool" and not (previous and (previous.get("tool_calls") or previous["role"] == "tool")):
            return False
        previous = message
    return True


async def test_new_message_after_a_crash_keeps_tool_results_next_to_their_call(make_app, settings):
    """Regression: results used to be appended after the new user message, which providers reject with HTTP 400."""
    app = await make_app([text_reply("ok, moving on")])
    (settings.workspace / "x.txt").write_text("content")
    crash_log(app, "s", [("a", "read_file", {"path": "x.txt"})])
    await app.agent.run("s", "never mind, tell me a joke")           # a NEW message, not a bare resume
    sent = app.llm.requests[0]["messages"][1:]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "user"]
    assert valid_for_providers(sent)
    assert [b["role"] for _, k, b in app.store.rows("s")] == ["user", "assistant", "tool", "user", "assistant"]

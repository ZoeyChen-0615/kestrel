import asyncio

import pytest

from kestrel.app import App
from kestrel.common import ChatResponse
from kestrel.llm import HashingEmbedder, ScriptedChat, text_reply, tool_reply
from kestrel.subagents import NAMESPACE


class RoutedChat:
    """Routes requests to the child or parent script by looking at the system prompt."""

    model = "routed"

    def __init__(self, parent, child):
        self.parent, self.child = ScriptedChat(parent), child

    async def chat(self, messages, tools=None, **kwargs):
        is_child = "focused sub-agent" in messages[0]["content"]
        target = self.child if is_child else self.parent
        return await target.chat(messages, tools, **kwargs)


class SlowChat:
    model = "slow"

    def __init__(self):
        self.started = asyncio.Event()

    async def chat(self, messages, tools=None, **kwargs):
        self.started.set()
        await asyncio.sleep(30)
        return text_reply("never")


@pytest.fixture
async def build(settings):
    apps = []

    async def factory(llm, **overrides):
        for key, value in overrides.items():
            setattr(settings, key, value)
        app = await App.create(settings, llm=llm, embedder=HashingEmbedder(), use_mcp=False)
        apps.append(app)
        return app

    yield factory
    for app in apps:
        await app.aclose()


async def test_synchronous_subagent_returns_its_report_to_the_parent(build):
    llm = RoutedChat(
        parent=[tool_reply("spawn_subagent", {"task": "list the workspace", "profile": "research"}), text_reply("parent final")],
        child=ScriptedChat([text_reply("child report: nothing found")]),
    )
    app = await build(llm)
    result = await app.agent.run("p", "delegate this")
    assert result.status == "completed" and result.final_text == "parent final"
    tool_message = [m for _, m in app.store.load_context("p")[2] if m["role"] == "tool"][0]["content"]
    assert "child report: nothing found" in tool_message and "completed" in tool_message
    child = app.store.list_sessions(kind="subagent")[0]
    assert child["id"].startswith("sub-")
    (_, _, job), = app.store.kv_list(NAMESPACE)
    assert job["status"] == "completed" and job["parent"] == "p"


async def test_children_get_restricted_tools_and_cannot_spawn(build):
    child = ScriptedChat([text_reply("ok")])
    llm = RoutedChat([tool_reply("spawn_subagent", {"task": "just look", "profile": "research"}), text_reply("done")], child)
    app = await build(llm)
    await app.agent.run("p", "go")
    names = {t["function"]["name"] for t in child.requests[0]["tools"]}
    assert "read_file" in names and "recall_memory" in names
    assert not names & {"spawn_subagent", "write_file", "shell", "remember"}


async def test_scripting_profile_has_files_and_shell_only(build):
    child = ScriptedChat([text_reply("ok")])
    llm = RoutedChat([tool_reply("spawn_subagent", {"task": "write a script", "profile": "scripting"}), text_reply("done")], child)
    app = await build(llm)
    await app.agent.run("p", "go")
    assert {t["function"]["name"] for t in child.requests[0]["tools"]} == {"read_file", "list_dir", "write_file", "shell"}


async def test_capacity_limit_refuses_extra_spawns(build):
    llm = RoutedChat([tool_reply("spawn_subagent", {"task": "work", "profile": "research"}), text_reply("done")], ScriptedChat([]))
    app = await build(llm, subagent_max_active=0)
    await app.agent.run("p", "go")
    tool_message = [m for _, m in app.store.load_context("p")[2] if m["role"] == "tool"][0]["content"]
    assert "capacity reached" in tool_message


async def test_background_job_is_reported_to_the_parent_exactly_once(build):
    parent_requests = []

    def second_turn(messages):
        parent_requests.append(messages[0]["content"])
        return text_reply("noted")

    llm = RoutedChat(
        parent=[tool_reply("spawn_subagent", {"task": "slow research", "profile": "research", "background": True}),
                text_reply("started it"), second_turn, second_turn],
        child=ScriptedChat([text_reply("BACKGROUND RESULT")]),
    )
    app = await build(llm)
    first = await app.agent.run("p", "start")
    assert first.final_text == "started it"
    await asyncio.gather(*app.subagents._tasks.values())            # let the background child finish
    await app.agent.run("p", "any news?")
    assert "BACKGROUND RESULT" in parent_requests[0]                   # delivered on the next turn
    await app.agent.run("p", "and now?")
    assert "BACKGROUND RESULT" not in parent_requests[1]               # ...but only once


async def test_cancelling_a_background_job_records_a_terminal_state(build):
    slow = SlowChat()
    llm = RoutedChat([tool_reply("spawn_subagent", {"task": "never ends", "profile": "research", "background": True}), text_reply("ok")], slow)
    app = await build(llm)
    await app.agent.run("p", "start")
    await asyncio.wait_for(slow.started.wait(), 15)
    job_id = app.store.kv_list(NAMESPACE)[0][0]
    assert not app.subagents.cancel(job_id).is_error
    await asyncio.gather(*app.subagents._tasks.values(), return_exceptions=True)
    assert app.store.kv_get(NAMESPACE, job_id)[1]["status"] == "cancelled"
    assert app.subagents.cancel(job_id).is_error                        # already finished


async def test_jobs_left_running_by_a_dead_process_are_marked_failed(build):
    app = await build(RoutedChat([], ScriptedChat([])))
    app.store.kv_put(NAMESPACE, "ghost", {"job_id": "ghost", "parent": "p", "status": "running", "result": "", "error": "",
                                           "profile": "research", "label": "", "delivered": False}, expected_version=None)
    assert app.subagents.recover() == 1
    job = app.store.kv_get(NAMESPACE, "ghost")[1]
    assert job["status"] == "failed" and "restart" in job["error"]
    assert "interrupted by a restart" in app.subagents.collect_notices("p")[0]


async def test_a_failing_child_is_reported_as_failed(build):
    empty = ChatResponse(content="")
    llm = RoutedChat([tool_reply("spawn_subagent", {"task": "doomed", "profile": "research"}), text_reply("handled")],
                     ScriptedChat([empty, empty, empty, empty]))
    app = await build(llm)
    result = await app.agent.run("p", "go")
    tool_message = [m for _, m in app.store.load_context("p")[2] if m["role"] == "tool"][0]["content"]
    assert result.status == "completed" and "failed" in tool_message

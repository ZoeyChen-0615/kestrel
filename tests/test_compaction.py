import pytest

from kestrel.common import LLMError
from kestrel.config import Settings
from kestrel.context import Compactor, estimate_tokens, render_transcript
from kestrel.llm import ScriptedChat, text_reply
from kestrel.memory import LongTermMemory
from kestrel.store import Store
from kestrel.tracing import Tracer


def fill(store, session, turns, size=300):
    store.create_session(session_id=session)
    for i in range(turns):
        store.append(session, {"role": "user", "content": f"question {i} " + "q" * size})
        store.append(session, {"role": "assistant", "content": f"answer {i} " + "a" * size})


@pytest.fixture
def env(tmp_path):
    store = Store(":memory:")
    settings = Settings(home=tmp_path, context_window=1000)
    return store, settings, Tracer(store)


async def test_below_threshold_nothing_happens(env):
    store, settings, tracer = env
    fill(store, "s", 1, size=50)
    compactor = Compactor(ScriptedChat([]), store, settings, tracer)
    assert await compactor.maybe_compact("s", 0) is False


async def test_compaction_appends_a_summary_and_keeps_every_original_row(env):
    store, settings, tracer = env
    fill(store, "s", 10)
    before = len(store.rows("s"))
    compactor = Compactor(ScriptedChat([text_reply("SUMMARY OF EARLY TURNS")]), store, settings, tracer)
    assert await compactor.maybe_compact("s", 0) is True
    summary, covers, messages = store.load_context("s")
    assert summary == "SUMMARY OF EARLY TURNS" and covers > 0
    assert messages[0][1]["role"] == "user"                       # cut lands on a turn boundary
    assert len(store.rows("s")) == before + 1                      # only the summary row was added
    assert [kind for _, kind, _ in store.rows("s")].count("summary") == 1


async def test_cut_never_splits_a_tool_call_from_its_results(env):
    store, settings, tracer = env
    store.create_session(session_id="s")
    for i in range(8):
        store.append("s", {"role": "user", "content": f"q{i} " + "x" * 200})
        store.append("s", {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "t", "arguments": "{}"}}]})
        # Large tool result: the keep-tail budget is reached *inside* a turn, so a naive cut would split the pair.
        store.append("s", {"role": "tool", "tool_call_id": f"c{i}", "content": "r" * 600})
        store.append("s", {"role": "assistant", "content": "done " + "y" * 200})
    compactor = Compactor(ScriptedChat([text_reply("sum")]), store, settings, tracer)
    assert await compactor.maybe_compact("s", 0)
    kept = [m for _, m in store.load_context("s")[2]]
    assert kept[0]["role"] == "user"
    seen_calls = {c["id"] for m in kept for c in m.get("tool_calls", [])}
    assert {m["tool_call_id"] for m in kept if m["role"] == "tool"} <= seen_calls


async def test_second_compaction_builds_on_the_first_summary(env):
    store, settings, tracer = env
    fill(store, "s", 10)
    llm = ScriptedChat([text_reply("first summary"), text_reply("second summary")])
    compactor = Compactor(llm, store, settings, tracer)
    assert await compactor.maybe_compact("s", 0)
    fill_more = [store.append("s", {"role": "user" if i % 2 == 0 else "assistant", "content": "z" * 300}) for i in range(24)]
    assert fill_more and await compactor.maybe_compact("s", 0)
    assert store.load_context("s")[0] == "second summary"
    assert "first summary" in llm.requests[1]["messages"][1]["content"]


async def test_summarisation_failure_is_survivable(env):
    store, settings, tracer = env
    fill(store, "s", 10)

    def fail(messages):
        raise LLMError("down")

    compactor = Compactor(ScriptedChat([fail]), store, settings, tracer)
    assert await compactor.maybe_compact("s", 0) is False
    assert store.load_context("s")[0] is None


async def test_no_safe_cut_point_is_reported_not_forced(env):
    store, settings, tracer = env
    store.create_session(session_id="s")
    store.append("s", {"role": "user", "content": "x" * 5000})  # one giant turn: nothing earlier to summarise
    compactor = Compactor(ScriptedChat([]), store, settings, tracer)
    assert await compactor.maybe_compact("s", 0) is False
    assert any(e["name"] == "skipped" for e in store.events("s"))


async def test_compaction_mines_durable_facts_into_long_term_memory(env, tmp_path):
    store, settings, tracer = env
    fill(store, "s", 10)
    longterm = LongTermMemory(tmp_path / "MEMORY.md")
    llm = ScriptedChat([text_reply("summary"), text_reply('[{"op":"add","text":"User prefers concise answers"}]')])
    await Compactor(llm, store, settings, tracer, longterm).maybe_compact("s", 0)
    assert longterm.facts() == ["User prefers concise answers"]
    assert "question 0" in llm.requests[1]["messages"][1]["content"]   # extraction saw only user text
    assert "answer 0" not in llm.requests[1]["messages"][1]["content"]


def test_token_estimate_and_transcript_rendering():
    assert estimate_tokens("a" * 300) == 101
    rows = [(1, {"role": "user", "content": "hi"}),
            (2, {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "t", "arguments": "{}"}}]}),
            (3, {"role": "tool", "content": "x" * 5000})]
    text = render_transcript(rows)
    assert "user: hi" in text and "t({})" in text and len(text) < 2000

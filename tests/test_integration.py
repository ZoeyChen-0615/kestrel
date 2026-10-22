"""End-to-end behaviour across modules, driven by a scripted model."""

from kestrel.app import App
from kestrel.llm import HashingEmbedder, ScriptedChat, text_reply, tool_reply


def system_prompt(llm: ScriptedChat, index: int) -> str:
    return llm.requests[index]["messages"][0]["content"]


async def test_exchanges_are_recalled_in_a_later_session(make_app):
    app = await make_app([text_reply("Mochi is a lovely name."), text_reply("Your dog is called Mochi.")])
    await app.agent.run("a", "my dog is called Mochi and he loves pumpkin")
    await app.agent.run("b", "what is my dog called?")
    assert "Mochi" in system_prompt(app.llm, 1) and "past conversations" in system_prompt(app.llm, 1)


async def test_recall_is_skipped_for_exchanges_still_in_the_current_context(make_app):
    app = await make_app([text_reply("Mochi is a lovely name."), text_reply("Yes.")])
    await app.agent.run("a", "my dog is called Mochi")
    await app.agent.run("a", "what is my dog called?")
    assert "past conversations" not in system_prompt(app.llm, 1)   # the first exchange is already in the messages


async def test_a_failing_embedder_degrades_to_no_recall(make_app):
    app = await make_app([text_reply("one"), text_reply("two")])

    async def broken(texts):
        raise RuntimeError("embedding service down")

    app.embedder.embed = broken
    assert (await app.agent.run("a", "hello")).status == "completed"
    assert (await app.agent.run("b", "hello again")).status == "completed"


async def test_remember_tool_writes_the_file_and_it_reaches_the_next_prompt(make_app, settings):
    app = await make_app([tool_reply("remember", {"fact": "User prefers metric units"}), text_reply("Saved."), text_reply("ok")])
    await app.agent.run("a", "please remember that I prefer metric units")
    assert settings.long_term_path.read_text().count("User prefers metric units") == 1
    await app.agent.run("b", "convert 5 miles")
    assert "User prefers metric units" in system_prompt(app.llm, 2)


async def test_skill_flow_index_then_load(make_app, settings):
    app = await make_app([tool_reply("load_skill", {"name": "dog-notes"}), text_reply("answered")])
    skill = settings.workspace / "skills" / "dog-notes"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: dog-notes\ndescription: Care notes for the dog\n---\nRead references/care.md.")
    await app.agent.run("s", "how do I look after the dog?")
    assert "dog-notes: Care notes for the dog" in system_prompt(app.llm, 0)
    loaded = [m for _, m in app.store.load_context("s")[2] if m["role"] == "tool"][0]["content"]
    assert "Read references/care.md." in loaded and str(skill) in loaded


async def test_persona_file_edits_apply_on_the_next_run(make_app, settings):
    app = await make_app([text_reply("a"), text_reply("b")])
    await app.agent.run("s", "hi")
    settings.persona_path.write_text("You are a pirate.")
    await app.agent.run("s", "hi again")
    assert "pirate" not in system_prompt(app.llm, 0) and "You are a pirate." in system_prompt(app.llm, 1)


async def test_approval_flow_blocks_then_allows_writes(settings):
    decisions = iter([False, True])
    asked = []

    async def approver(tool, args):
        asked.append(tool.name)
        return next(decisions)

    settings.approval = "ask"
    llm = ScriptedChat([
        tool_reply("write_file", {"path": "out.txt", "content": "v1"}), tool_reply("write_file", {"path": "out.txt", "content": "v2"}), text_reply("done"),
    ])
    app = await App.create(settings, llm=llm, embedder=HashingEmbedder(), approver=approver, use_mcp=False)
    try:
        await app.agent.run("s", "write it")
        results = [m["content"] for _, m in app.store.load_context("s")[2] if m["role"] == "tool"]
        assert results[0].startswith("Denied") and "declined" in results[0]
        assert "Wrote" in results[1] and (settings.workspace / "out.txt").read_text() == "v2"
        assert asked == ["write_file", "write_file"]
    finally:
        await app.aclose()


async def test_workspace_escape_is_refused_end_to_end(make_app):
    app = await make_app([tool_reply("read_file", {"path": "../../etc/passwd"}), text_reply("sorry")])
    await app.agent.run("s", "read passwd")
    result = [m for _, m in app.store.load_context("s")[2] if m["role"] == "tool"][0]["content"]
    assert "outside the workspace" in result


async def test_shell_runs_with_a_scrubbed_environment_and_times_out(make_app, monkeypatch):
    monkeypatch.setenv("SOME_SECRET_TOKEN", "hunter2")
    app = await make_app([
        tool_reply("shell", {"command": "echo token=$SOME_SECRET_TOKEN; echo cwd=$(pwd)"}),
        tool_reply("shell", {"command": "sleep 5", "timeout_s": 1}),
        text_reply("done"),
    ])
    await app.agent.run("s", "run things")
    outputs = [m["content"] for _, m in app.store.load_context("s")[2] if m["role"] == "tool"]
    assert "hunter2" not in outputs[0] and "token=" in outputs[0] and "workspace" in outputs[0]
    assert "timed out" in outputs[1]


async def test_trace_records_llm_and_tool_events(make_app):
    from kestrel.tracing import summarise

    app = await make_app([tool_reply("list_dir", {"path": "."}), text_reply("done")])
    await app.agent.run("s", "look")
    summary = summarise(app.store.events("s"), "s")
    assert summary.llm_calls == 2 and summary.tool_calls == 1 and summary.by_tool["list_dir"] == 1
    assert summary.usage.total > 0


async def test_skill_resources_outside_the_workspace_are_readable_but_not_writable(make_app, settings):
    app = await make_app([])
    skill = settings.home / "skills" / "cat-notes"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\ndescription: cat\n---\nRead references/care.md")
    (skill / "references" / "care.md").write_text("Pixel eats salmon at 6pm.")
    app.agent.llm.__init__([
        tool_reply("read_file", {"path": str(skill / "references" / "care.md")}),
        tool_reply("list_dir", {"path": str(skill)}),
        tool_reply("read_file", {"path": "/etc/hosts"}),
        tool_reply("write_file", {"path": str(skill / "references" / "care.md"), "content": "hacked"}),
        text_reply("done"),
    ])
    await app.agent.run("s", "check the skill")
    results = [m["content"] for _, m in app.store.load_context("s")[2] if m["role"] == "tool"]
    assert "Pixel eats salmon" in results[0]
    assert "SKILL.md" in results[1] and "references/" in results[1]
    assert "outside the workspace" in results[2]                       # unrelated absolute paths are still refused
    assert "outside the workspace" in results[3]                       # writing into a skill directory is refused
    assert (skill / "references" / "care.md").read_text() == "Pixel eats salmon at 6pm."


async def test_memory_tools_update_atomically_and_report_what_they_changed(make_app, settings):
    """`replaces` swaps one fact in place; every memory tool echoes what it removed and what remains,
    so the model can notice a mistake instead of assuming success."""
    app = await make_app([
        tool_reply("remember", {"fact": "Name is Priya"}),
        tool_reply("remember", {"fact": "Allergic to peanuts"}),
        tool_reply("remember", {"fact": "Lives in Chicago"}),
        tool_reply("remember", {"fact": "Lives in Boston (moved from Chicago)", "replaces": "Lives in Chicago"}),
        tool_reply("forget", {"text": "Priya"}),
        tool_reply("forget", {"text": "Boston"}),
        text_reply("done"),
    ])
    await app.agent.run("s", "memory exercise")
    results = [m["content"] for _, m in app.store.load_context("s")[2] if m["role"] == "tool"]
    assert "Updated." in results[3] and "- Lives in Boston (moved from Chicago)" in results[3] and "Lives in Chicago\n" not in results[3]
    assert results[4].startswith("Removed: Name is Priya") and "Allergic to peanuts" in results[4]   # the tool shows what was removed + what remains
    assert results[5].startswith("Removed: Lives in Boston")
    assert app.longterm.facts() == ["Allergic to peanuts"]


async def test_forget_refuses_ambiguous_matches_and_lists_candidates(make_app):
    app = await make_app([
        tool_reply("remember", {"fact": "Likes green tea"}), tool_reply("remember", {"fact": "Likes black tea"}),
        tool_reply("forget", {"text": "tea"}), tool_reply("forget", {"text": "whisky"}), text_reply("done"),
    ])
    await app.agent.run("s", "x")
    results = [m for _, m in app.store.load_context("s")[2] if m["role"] == "tool"]
    assert "matches 2 facts" in results[2]["content"] and "green tea" in results[2]["content"] and "black tea" in results[2]["content"]
    assert "Nothing matched" in results[3]["content"] and "Current memory" in results[3]["content"]
    assert len(app.longterm.facts()) == 2                               # nothing was deleted


async def test_remember_with_ambiguous_replaces_changes_nothing(make_app):
    app = await make_app([
        tool_reply("remember", {"fact": "Likes green tea"}), tool_reply("remember", {"fact": "Likes black tea"}),
        tool_reply("remember", {"fact": "Likes coffee", "replaces": "tea"}), text_reply("done"),
    ])
    await app.agent.run("s", "x")
    last = [m for _, m in app.store.load_context("s")[2] if m["role"] == "tool"][2]["content"]
    assert "matches several facts" in last and app.longterm.facts() == ["Likes green tea", "Likes black tea"]


async def test_system_prompt_tells_the_model_when_to_remember(make_app):
    app = await make_app([text_reply("ok")])
    await app.agent.run("s", "hi")
    prompt = app.llm.requests[0]["messages"][0]["content"]
    assert "call `remember` BEFORE replying" in prompt and "Never say you have saved" in prompt

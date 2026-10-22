import json
from dataclasses import replace

from kestrel.app import App
from kestrel.evals import evaluate_checks, load_suite, run_suite
from kestrel.common import RunResult, Usage
from kestrel.llm import HashingEmbedder, ScriptedChat, text_reply, tool_reply


def result(text="", status="completed", steps=1):
    return RunResult("s", status, text, steps, Usage(1, 1))


def test_checks_report_each_failure(tmp_path):
    failures = evaluate_checks(
        [{"type": "contains", "value": "Hello"}, {"type": "regex", "value": r"\d+"}, {"type": "tool_called", "name": "shell"},
         {"type": "tool_not_called", "name": "read_file"}, {"type": "max_steps", "value": 1}, {"type": "file_contains", "path": "a.txt", "value": "x"},
         {"type": "not_contains", "value": "bye"}],
        result("bye", steps=3), ["read_file"], tmp_path)
    assert len(failures) == 7


def test_all_checks_pass(tmp_path):
    (tmp_path / "a.txt").write_text("xyz")
    ok = evaluate_checks(
        [{"type": "contains", "value": "hello"}, {"type": "regex", "value": r"\d+"}, {"type": "tool_called", "name": "read_file"},
         {"type": "max_steps", "value": 2}, {"type": "file_contains", "path": "a.txt", "value": "xy"}],
        result("Hello 42", steps=2), ["read_file"], tmp_path)
    assert ok == []


def test_a_non_completed_run_fails_unless_expected(tmp_path):
    assert evaluate_checks([], result(status="step_limit"), [], tmp_path)
    assert evaluate_checks([], result(status="step_limit"), [], tmp_path, expect_status="step_limit") == []


def test_suite_loading_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"cases": [{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}]}))
    try:
        load_suite(path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected ValueError")


async def test_suite_runs_cases_in_isolated_workspaces(settings):
    scripts = {
        "reads": [tool_reply("read_file", {"path": "notes.txt"}), text_reply("The note says: purple elephant")],
        "wrong": [text_reply("I have no idea")],
    }
    cases = [
        {"id": "reads", "prompt": "what does notes.txt say?", "files": {"notes.txt": "purple elephant"},
         "checks": [{"type": "contains", "value": "purple elephant"}, {"type": "tool_called", "name": "read_file"}]},
        {"id": "wrong", "prompt": "what does notes.txt say?", "checks": [{"type": "contains", "value": "purple elephant"}]},
    ]
    order = iter(["reads", "wrong"])

    async def make_app(tmp):
        return await App.create(replace(settings, home=tmp, workspace=None), llm=ScriptedChat(scripts[next(order)]),
                                embedder=HashingEmbedder(), use_mcp=False)

    suite = await run_suite(cases, make_app)
    assert [r.passed for r in suite.results] == [True, False]
    assert suite.pass_rate == 0.5 and "purple elephant" in suite.results[1].failures[0]
    assert "PASS" in suite.format() and "FAIL" in suite.format()
    assert suite.to_dict()["cases"] == 2


async def test_multi_turn_cases_accumulate_tool_usage(settings):
    cases = [{"id": "two-turns", "turns": ["list files", "thanks"], "checks": [{"type": "tool_called", "name": "list_dir"}, {"type": "contains", "value": "welcome"}]}]

    async def make_app(tmp):
        script = [tool_reply("list_dir", {"path": "."}), text_reply("listed"), text_reply("you're welcome")]
        return await App.create(replace(settings, home=tmp, workspace=None), llm=ScriptedChat(script), embedder=HashingEmbedder(), use_mcp=False)

    suite = await run_suite(cases, make_app)
    assert suite.results[0].passed and suite.results[0].steps == 3

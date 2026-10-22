import pytest

from kestrel.cli import build_parser, main
from kestrel.store import Store
from kestrel.tracing import format_timeline


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # so no stray .env is picked up


def test_parser_accepts_the_documented_commands():
    parser = build_parser()
    assert parser.parse_args(["chat", "--approval", "auto", "--no-stream"]).no_stream
    assert parser.parse_args(["run", "hello", "--session", "x"]).prompt == "hello"
    assert parser.parse_args(["eval", "suite.json", "--json", "out.json"]).json == "out.json"
    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--approval", "bogus"])


def test_missing_api_key_gives_a_clear_error(capsys):
    assert main(["run", "hi"]) == 2
    assert "KESTREL_API_KEY" in capsys.readouterr().err


def test_info_runs_without_a_key(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "MISSING" in out and "offline hashing" in out


def test_sessions_and_trace_read_the_database(tmp_path, capsys):
    store = Store(tmp_path / "home" / "state.db")
    store.create_session(session_id="abc", title="my chat")
    store.add_event("abc", "llm", "m", 120.0, {"prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"})
    store.add_event("abc", "tool", "read_file", 8.0, {"is_error": False, "preview": "hello"})
    store.close()
    assert main(["sessions"]) == 0 and "my chat" in capsys.readouterr().out
    assert main(["trace", "abc"]) == 0
    out = capsys.readouterr().out
    assert "read_file" in out and "llm calls 1" in out and "tokens in 10 out 5" in out


def test_trace_with_no_sessions_is_a_clean_failure(capsys):
    assert main(["trace"]) == 1


def test_timeline_marks_errors():
    events = [{"ts": 1.0, "kind": "tool", "name": "shell", "duration_ms": 5.0, "data": {"is_error": True, "error": "boom"}}]
    assert "ERROR" in format_timeline(events) and "boom" in format_timeline(events)

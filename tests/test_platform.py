"""Platform-specific behaviour, tested on every OS by passing the platform in explicitly."""

import asyncio
import io
import os
import signal
import sys

import pytest

from kestrel import cli
from kestrel.mcp_client import resolve_command
from kestrel.tools.builtin import kill_process_tree, process_group_options


class FakeProcess:
    pid = 4242


async def test_windows_termination_walks_the_tree_with_taskkill_and_never_calls_killpg(monkeypatch):
    calls = []

    class Killer:
        async def wait(self):
            return 0

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        return Killer()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(os, "killpg", lambda *a: pytest.fail("os.killpg does not exist on Windows"), raising=False)
    await kill_process_tree(FakeProcess(), platform="win32")
    assert calls == [("taskkill", "/F", "/T", "/PID", "4242")]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL and process groups are POSIX concepts")
async def test_posix_termination_kills_the_whole_process_group(monkeypatch):
    sent = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: sent.append((pid, sig)), raising=False)
    await kill_process_tree(FakeProcess(), platform="linux")
    assert sent == [(4242, signal.SIGKILL)]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL and process groups are POSIX concepts")
async def test_posix_termination_tolerates_an_already_dead_process(monkeypatch):
    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", gone, raising=False)
    await kill_process_tree(FakeProcess(), platform="linux")      # must not raise


def test_process_group_options_are_posix_only():
    assert process_group_options("win32") == {}
    assert process_group_options("linux") == {"start_new_session": True}
    assert process_group_options("darwin") == {"start_new_session": True}


def test_resolve_command_finds_cmd_launchers_on_windows_only(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: r"C:\Program Files\nodejs\npx.CMD" if name == "npx" else None)
    assert resolve_command("npx", platform="win32") == r"C:\Program Files\nodejs\npx.CMD"
    assert resolve_command("not-installed", platform="win32") == "not-installed"      # unknown names pass through
    assert resolve_command("npx", platform="linux") == "npx"                          # untouched elsewhere
    assert resolve_command("npx", platform="darwin") == "npx"


def test_cli_output_survives_a_legacy_windows_code_page(monkeypatch):
    """A cp1252 console cannot encode emoji or Chinese; the CLI must switch to UTF-8 instead of crashing."""
    raw_out, raw_err = io.BytesIO(), io.BytesIO()
    out = io.TextIOWrapper(raw_out, encoding="cp1252", errors="strict", write_through=True)
    err = io.TextIOWrapper(raw_err, encoding="cp1252", errors="strict", write_through=True)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    with pytest.raises(UnicodeEncodeError):
        print("🎃 中文")                      # proves the premise: the legacy code page really fails
    cli._use_utf8_output()
    print("🎃 中文")
    out.flush()
    assert "🎃 中文" in raw_out.getvalue().decode("utf-8")

"""Small cross-platform helpers shared by the tests."""

import shlex
import subprocess
import sys


def py_command(code: str) -> str:
    """A shell command line that runs ``code`` with the current Python, valid for both cmd.exe and POSIX shells."""
    argv = [sys.executable, "-c", code]
    return subprocess.list2cmdline(argv) if sys.platform == "win32" else shlex.join(argv)

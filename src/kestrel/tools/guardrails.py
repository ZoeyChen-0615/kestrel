"""Approval policy for risky tools.

Three modes:

* ``readonly`` - only ``read`` tools may run.
* ``ask``      - ``write`` and ``dangerous`` tools need an approver callback to say yes.
* ``auto``     - everything runs, except commands matching the shell deny-list.

The deny-list is a seat belt, not a sandbox: it catches obviously destructive
commands but cannot make arbitrary shell execution safe.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..config import ApprovalMode
from .registry import Tool

Approver = Callable[[Tool, dict[str, Any]], Awaitable[bool]]

SHELL_DENY_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(-[a-zA-Z]+\s+)*(/|~|\$HOME|\*)(\s|$)",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+[^|;]*\bof=/dev/",
    r":\(\)\s*\{.*\};\s*:",
    r"\b(curl|wget)\b[^|;]*\|\s*(sudo\s+)?(ba|z|da)?sh\b",
    r"\bchmod\s+-R\s+[0-7]{3,4}\s+/(\s|$)",
    r">\s*/dev/(sd|nvme|disk)",
    # Only when used as a command (start of line, after ; & | ( or sudo), not as a word inside an argument.
    r"(?:^|[;&|(]\s*|\bsudo\s+)(shutdown|reboot|halt|poweroff)\b",
    # Windows (cmd.exe and PowerShell)
    r"(?i)\b(rd|rmdir)\s+(/[sq]\s+)+[a-z]:\\?(\s|$)",
    r"(?i)\bdel\s+(/[fsq]\s+)+[a-z]:\\",
    r"(?i)\bformat\s+[a-z]:",
    r"(?i)remove-item\b[^|;]*-recurse[^|;]*\s[a-z]:\\?(\s|$)",
    r"(?i)\b(iwr|invoke-webrequest|irm|invoke-restmethod|curl)\b[^|;]*\|\s*(iex|invoke-expression)\b",
    r"(?i)(?:^|[;&|(]\s*)(shutdown|restart-computer|stop-computer)\b",
]
_COMPILED = [re.compile(p) for p in SHELL_DENY_PATTERNS]


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


def shell_command_blocked(command: str) -> str | None:
    for pattern in _COMPILED:
        if pattern.search(command):
            return f"command matches blocked pattern {pattern.pattern!r}"
    return None


class Policy:
    def __init__(self, mode: ApprovalMode = "ask", approver: Approver | None = None) -> None:
        self.mode = mode
        self.approver = approver

    async def check(self, tool: Tool, args: dict[str, Any]) -> Decision:
        if tool.risk == "read":
            return Decision(True)
        if self.mode == "readonly":
            return Decision(False, f"{tool.name} is a {tool.risk} tool and the agent is in read-only mode")
        if tool.name == "shell":
            reason = shell_command_blocked(str(args.get("command", "")))
            if reason:
                return Decision(False, reason)
        if self.mode == "auto":
            return Decision(True)
        if self.approver is None:
            return Decision(False, f"{tool.name} needs approval but no interactive approver is available")
        if await self.approver(tool, args):
            return Decision(True)
        return Decision(False, "the user declined this action")

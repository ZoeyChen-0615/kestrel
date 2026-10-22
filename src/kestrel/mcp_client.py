"""MCP client: connects to stdio MCP servers and exposes their tools through the registry.

Connections live inside one dedicated background task. The MCP SDK's transports
use task-group scopes that must be entered and exited by the same task, so
owning them in a single long-lived task keeps start-up and shutdown reliable no
matter which task calls ``aclose``.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools.registry import Tool, ToolContext, ToolRegistry, ToolResult

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]")


@dataclass
class ServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # "read" = tools only read data (no approval needed); "write" = may have side effects.
    risk: str = "write"
    connect_timeout: float = 30.0


def load_config(path: Path) -> list[ServerConfig]:
    """Parse ``{"servers": {"name": {"command": ..., "args": [...], "env": {...}, "risk": "read"}}}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    servers = []
    for name, raw in (data.get("servers") or {}).items():
        if raw.get("risk", "write") not in ("read", "write"):
            raise ValueError(f"server {name!r}: risk must be 'read' or 'write'")
        servers.append(ServerConfig(
            name=name, command=raw["command"], args=list(raw.get("args", [])), env=dict(raw.get("env", {})),
            risk=raw.get("risk", "write"), connect_timeout=float(raw.get("connect_timeout", 30)),
        ))
    return servers


def resolve_command(command: str, platform: str | None = None) -> str:
    """On Windows, launchers such as ``npx`` are ``npx.cmd`` files that CreateProcess cannot find by bare name.

    Resolve the real path via PATH/PATHEXT there; elsewhere the name is used unchanged.
    """
    if (platform or sys.platform) != "win32":
        return command
    return shutil.which(command) or command


def tool_name(server: str, tool: str) -> str:
    return _SAFE_NAME.sub("_", f"{server}__{tool}")[:64]


class McpManager:
    def __init__(self, servers: list[ServerConfig]) -> None:
        self._servers = servers
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._ready: asyncio.Future[dict[str, Any]] | None = None
        self._registry: ToolRegistry | None = None
        self.registered: list[str] = []

    async def start(self, registry: ToolRegistry) -> dict[str, str]:
        """Connect to every server and register its tools. Returns ``{server: 'ok' | error message}``.

        One broken server never prevents the others (or the agent) from starting.
        """
        self._registry = registry
        self._ready = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._run(), name="mcp-manager")
        return await self._ready

    async def aclose(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        if self._registry is not None:
            for name in self.registered:
                self._registry.unregister(name)
        self.registered.clear()

    async def _run(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        report: dict[str, str] = {}
        async with AsyncExitStack() as stack:
            for config in self._servers:
                try:
                    params = StdioServerParameters(command=resolve_command(config.command), args=config.args, env=config.env or None)
                    async with asyncio.timeout(config.connect_timeout):
                        read, write = await stack.enter_async_context(stdio_client(params))
                        session = await stack.enter_async_context(ClientSession(read, write))
                        await session.initialize()
                        listing = await session.list_tools()
                    for tool in listing.tools:
                        self._register(config, session, tool)
                    report[config.name] = "ok"
                except Exception as exc:  # isolate per-server failures
                    report[config.name] = f"{type(exc).__name__}: {exc}"
            assert self._ready is not None
            self._ready.set_result(report)
            await self._stop.wait()

    def _register(self, config: ServerConfig, session: Any, tool: Any) -> None:
        assert self._registry is not None
        name = tool_name(config.name, tool.name)
        remote_name = tool.name

        async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            result = await session.call_tool(remote_name, args)
            text = "\n".join(getattr(part, "text", "") for part in result.content if getattr(part, "type", "") == "text")
            return ToolResult(text or "(no text content)", bool(result.isError))

        self._registry.register(Tool.from_json_schema(
            name, f"[MCP:{config.name}] {tool.description or tool.name}", dict(tool.inputSchema or {}), handler,
            risk=config.risk,  # type: ignore[arg-type]
            idempotent=config.risk == "read", untrusted_output=True,
        ))
        self.registered.append(name)

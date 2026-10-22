import sys
from pathlib import Path

import pytest

from kestrel.common import ToolCall
from kestrel.config import Settings
from kestrel.mcp_client import McpManager, ServerConfig, load_config, tool_name
from kestrel.store import Store
from kestrel.tools import ToolContext, ToolRegistry

SERVER = str(Path(__file__).parent / "fixtures" / "echo_server.py")


@pytest.fixture
def ctx(tmp_path):
    return ToolContext("s", tmp_path, Settings(home=tmp_path), Store(":memory:"))


def call(name, args):
    import json

    return ToolCall("c", name, json.dumps(args))


async def test_tools_are_discovered_registered_and_callable(ctx):
    registry = ToolRegistry()
    manager = McpManager([ServerConfig("echo", sys.executable, [SERVER], risk="read")])
    try:
        assert await manager.start(registry) == {"echo": "ok"}
        assert {"echo__shout", "echo__add"} <= set(registry.names())
        tool = registry.get("echo__shout")
        assert tool.risk == "read" and tool.untrusted_output and tool.description.startswith("[MCP:echo]")
        result = await registry.execute(call("echo__shout", {"text": "hello"}), ctx)
        assert not result.is_error and "HELLO" in result.text and result.text.startswith("<untrusted_content")
        total = await registry.execute(call("echo__add", {"a": 2, "b": 3}), ctx)
        assert "5" in total.text
    finally:
        await manager.aclose()


async def test_arguments_are_validated_against_the_servers_json_schema(ctx):
    registry = ToolRegistry()
    manager = McpManager([ServerConfig("echo", sys.executable, [SERVER])])
    try:
        await manager.start(registry)
        bad = await registry.execute(call("echo__add", {"a": "two", "b": 3}), ctx)
        assert bad.is_error and "Invalid arguments" in bad.text
        missing = await registry.execute(call("echo__add", {"a": 1}), ctx)
        assert missing.is_error
    finally:
        await manager.aclose()


async def test_one_broken_server_does_not_block_the_others(ctx):
    registry = ToolRegistry()
    manager = McpManager([
        ServerConfig("broken", "definitely-not-a-real-binary-xyz", [], connect_timeout=5),
        ServerConfig("echo", sys.executable, [SERVER]),
    ])
    try:
        report = await manager.start(registry)
        assert report["echo"] == "ok" and report["broken"] != "ok"
        assert "echo__shout" in registry.names()
    finally:
        await manager.aclose()


async def test_close_unregisters_the_tools():
    registry = ToolRegistry()
    manager = McpManager([ServerConfig("echo", sys.executable, [SERVER])])
    await manager.start(registry)
    await manager.aclose()
    assert not [n for n in registry.names() if n.startswith("echo__")]


def test_config_loading_and_name_sanitising(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text('{"servers": {"time": {"command": "uvx", "args": ["mcp-server-time"], "risk": "read"}}}')
    (server,) = load_config(path)
    assert (server.name, server.command, server.args, server.risk) == ("time", "uvx", ["mcp-server-time"], "read")
    assert tool_name("my server", "get.time") == "my_server__get_time"
    path.write_text('{"servers": {"x": {"command": "c", "risk": "banana"}}}')
    with pytest.raises(ValueError):
        load_config(path)

import asyncio
import json

import pytest
from pydantic import BaseModel, Field

from kestrel.common import ToolCall
from kestrel.config import Settings
from kestrel.store import Store
from kestrel.tools import Policy, Tool, ToolContext, ToolRegistry, ToolResult


class AddArgs(BaseModel):
    a: int
    b: int = Field(default=0, ge=0)


@pytest.fixture
def ctx(tmp_path):
    settings = Settings(home=tmp_path, max_tool_output_chars=50, tool_timeout=0.3)
    return ToolContext("s1", tmp_path, settings, Store(":memory:"))


def call(name, args):
    raw = args if isinstance(args, str) else json.dumps(args)
    return ToolCall("c1", name, raw)


async def add(args, ctx):
    return str(args["a"] + args["b"])


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(Tool.from_model("add", "Add numbers.", AddArgs, add))
    return r


async def test_valid_call_runs(registry, ctx):
    result = await registry.execute(call("add", {"a": 2, "b": 3}), ctx)
    assert (result.text, result.is_error) == ("5", False)


@pytest.mark.parametrize("args, fragment", [
    ({"b": 1}, "a"),                      # missing required field
    ({"a": "2"}, "a"),                    # strict mode: no silent str->int coercion
    ({"a": 1, "zzz": 1}, "zzz"),          # unknown field rejected
    ({"a": 1, "b": -1}, "b"),             # constraint violation
])
async def test_invalid_arguments_become_readable_errors(registry, ctx, args, fragment):
    result = await registry.execute(call("add", args), ctx)
    assert result.is_error and "Invalid arguments for add" in result.text and fragment in result.text


async def test_bad_json_and_non_object_and_unknown_tool(registry, ctx):
    assert "not valid JSON" in (await registry.execute(call("add", "{oops"), ctx)).text
    assert "must be a JSON object" in (await registry.execute(call("add", "[1]"), ctx)).text
    unknown = await registry.execute(call("nope", {}), ctx)
    assert unknown.is_error and "Available tools: add" in unknown.text


async def test_schema_is_generated_from_the_model(registry):
    schema = registry.schemas()[0]["function"]
    assert schema["name"] == "add"
    assert schema["parameters"]["required"] == ["a"]
    assert schema["parameters"]["additionalProperties"] is False
    assert "title" not in json.dumps(schema["parameters"]["properties"]["a"])


async def test_output_truncation_timeout_and_exceptions(ctx):
    r = ToolRegistry()

    async def long(args, c):
        return "x" * 500

    async def slow(args, c):
        await asyncio.sleep(5)

    async def boom(args, c):
        raise RuntimeError("kaput")

    for name, handler in (("long", long), ("slow", slow), ("boom", boom)):
        r.register(Tool.from_json_schema(name, "d", {"type": "object", "properties": {}}, handler))
    assert "output truncated" in (await r.execute(call("long", {}), ctx)).text
    timed_out = await r.execute(call("slow", {}), ctx)
    assert timed_out.is_error and "timed out" in timed_out.text
    failed = await r.execute(call("boom", {}), ctx)
    assert failed.is_error and "RuntimeError: kaput" in failed.text


async def test_json_schema_tools_validate_and_label_untrusted_output(ctx):
    r = ToolRegistry()

    async def echo(args, c):
        return ToolResult(args["text"])

    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    r.register(Tool.from_json_schema("echo", "d", schema, echo, untrusted_output=True))
    bad = await r.execute(call("echo", {"text": 5}), ctx)
    assert bad.is_error and "text" in bad.text
    ok = await r.execute(call("echo", {"text": "hi"}), ctx)
    assert ok.text.startswith("<untrusted_content") and "hi" in ok.text


async def test_policy_denial_is_a_result_not_an_exception(registry, ctx):
    registry.get("add").risk = "write"
    result = await registry.execute(call("add", {"a": 1}), ctx, Policy("readonly"))
    assert result.is_error and result.text.startswith("Denied")


def test_duplicate_registration_is_rejected(registry):
    with pytest.raises(ValueError):
        registry.register(Tool.from_model("add", "again", AddArgs, add))

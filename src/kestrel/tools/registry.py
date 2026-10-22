"""Tool definitions, schema generation, argument validation and safe execution.

The model's arguments are untrusted input. Every call goes through the same
pipeline: look the tool up, parse JSON, validate against the tool's schema,
ask the guardrail policy, run with a timeout, then truncate and label the
output. Failures at any stage become an error *result* the model can read and
correct, instead of an exception that would end the run.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jsonschema
from pydantic import BaseModel, ValidationError

from ..common import ToolCall

if TYPE_CHECKING:
    from ..config import Settings
    from ..store import Store
    from .guardrails import Policy

Risk = Literal["read", "write", "dangerous"]


class ArgumentError(Exception):
    """Arguments failed validation; the message is shown to the model."""


@dataclass
class ToolResult:
    text: str
    is_error: bool = False


@dataclass
class ToolContext:
    """Everything a tool handler may need about the current call."""

    session_id: str
    workspace: Path
    settings: Settings
    store: Store
    services: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[dict[str, Any], ToolContext], Awaitable["str | ToolResult"]]
Validator = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    validator: Validator
    handler: Handler
    # Safe to run again if an earlier attempt's outcome is unknown (used by crash recovery).
    idempotent: bool = False
    # Safe to run concurrently with other parallel-safe calls in the same batch.
    parallel_safe: bool = False
    risk: Risk = "read"
    # Output comes from outside the trust boundary (web pages, MCP servers) and gets labelled.
    untrusted_output: bool = False
    timeout: float | None = None

    def to_openai(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.schema}}

    @classmethod
    def from_model(cls, name: str, description: str, model: type[BaseModel], handler: Handler, **options: Any) -> Tool:
        """Build a tool whose schema and validation both come from one pydantic model."""

        known = set(model.model_fields)

        def validate(raw: dict[str, Any]) -> dict[str, Any]:
            # pydantic ignores unknown keys by default; the advertised schema forbids them, so enforce that here.
            unknown = sorted(set(raw) - known)
            if unknown:
                raise ArgumentError(f"unknown argument(s): {', '.join(unknown)}")
            return model.model_validate(raw, strict=True).model_dump()

        return cls(name, description, _clean_schema(model.model_json_schema()), validate, handler, **options)

    @classmethod
    def from_json_schema(cls, name: str, description: str, schema: dict[str, Any], handler: Handler, **options: Any) -> Tool:
        """Build a tool from a raw JSON Schema (used for tools discovered over MCP)."""
        schema = schema or {"type": "object", "properties": {}}
        validator = jsonschema.Draft202012Validator(schema)

        def validate(raw: dict[str, Any]) -> dict[str, Any]:
            errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.path))
            if errors:
                raise ArgumentError("; ".join(f"{'.'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5]))
            return raw

        return cls(name, description, schema, validate, handler, **options)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic's cosmetic ``title`` keys to keep the prompt small."""

    def strip(node: Any, is_properties: bool = False) -> Any:
        if isinstance(node, dict):
            return {k: strip(v, k == "properties") for k, v in node.items() if is_properties or k != "title"}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    cleaned = strip(schema)
    cleaned.setdefault("additionalProperties", False)
    return cleaned


def format_validation_error(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in error.errors()[:5])


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in self.names()]

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai() for tool in self.all()]

    def subset(self, predicate: Callable[[Tool], bool]) -> ToolRegistry:
        child = ToolRegistry()
        for tool in self.all():
            if predicate(tool):
                child.register(tool)
        return child

    def tool(self, name: str, description: str, model: type[BaseModel], **options: Any) -> Callable[[Handler], Handler]:
        """Decorator form of :meth:`register` for pydantic-modelled tools."""

        def decorator(handler: Handler) -> Handler:
            self.register(Tool.from_model(name, description, model, handler, **options))
            return handler

        return decorator

    async def execute(self, call: ToolCall, ctx: ToolContext, policy: Policy | None = None) -> ToolResult:
        """Run one model-requested call. Never raises for tool-level failures."""
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(f"Unknown tool '{call.name}'. Available tools: {', '.join(self.names()) or '(none)'}.", True)
        try:
            raw = json.loads(call.arguments) if call.arguments.strip() else {}
        except ValueError as exc:
            return ToolResult(f"Arguments are not valid JSON: {exc}. Send a single JSON object.", True)
        if not isinstance(raw, dict):
            return ToolResult("Arguments must be a JSON object.", True)
        try:
            args = tool.validator(raw)
        except ValidationError as exc:
            return ToolResult(f"Invalid arguments for {tool.name}: {format_validation_error(exc)}", True)
        except ArgumentError as exc:
            return ToolResult(f"Invalid arguments for {tool.name}: {exc}", True)
        if policy is not None:
            decision = await policy.check(tool, args)
            if not decision.allowed:
                return ToolResult(f"Denied: {decision.reason}", True)
        timeout = tool.timeout if tool.timeout is not None else ctx.settings.tool_timeout
        try:
            outcome = await asyncio.wait_for(tool.handler(args, ctx), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResult(f"{tool.name} timed out after {timeout:.0f}s. The action may or may not have completed.", True)
        except asyncio.CancelledError:
            raise
        except ArgumentError as exc:
            return ToolResult(f"Invalid arguments for {tool.name}: {exc}", True)
        except Exception as exc:  # tool bugs and I/O errors become results, not crashes
            return ToolResult(f"{tool.name} failed: {type(exc).__name__}: {exc}", True)
        result = outcome if isinstance(outcome, ToolResult) else ToolResult(str(outcome))
        return self._finalise(tool, result, ctx.settings.max_tool_output_chars)

    @staticmethod
    def _finalise(tool: Tool, result: ToolResult, limit: int) -> ToolResult:
        text = result.text
        if len(text) > limit:
            text = text[:limit] + f"\n[output truncated: {len(result.text) - limit} more characters]"
        if tool.untrusted_output and not result.is_error:
            text = f'<untrusted_content source="{tool.name}">\n{text}\n</untrusted_content>'
        return ToolResult(text, result.is_error)


def timed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000

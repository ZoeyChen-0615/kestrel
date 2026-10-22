"""A deterministic chat model that replays scripted responses.

Used by the test-suite and by offline evals: it makes the whole agent loop
testable without network access or API keys.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from typing import Any

from ..common import ChatResponse, TokenCallback, ToolCall, Usage

_ids = itertools.count(1)

Script = list[ChatResponse | Callable[[list[dict[str, Any]]], ChatResponse]]


def text_reply(text: str, *, finish_reason: str = "stop") -> ChatResponse:
    return ChatResponse(content=text, finish_reason=finish_reason, usage=Usage(10, max(1, len(text) // 4)))


def tool_reply(name: str, arguments: dict[str, Any] | str, *, call_id: str | None = None, content: str = "") -> ChatResponse:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ChatResponse(
        content=content,
        tool_calls=[ToolCall(call_id or f"call_{next(_ids)}", name, raw)],
        finish_reason="tool_calls",
        usage=Usage(10, 5),
    )


class ScriptedChat:
    """Returns the next scripted response on every call; records each request."""

    model = "scripted"

    def __init__(self, script: Script) -> None:
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return len(self._script)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_token: TokenCallback | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.requests.append({"messages": messages, "tools": tools, "max_tokens": max_tokens})
        if not self._script:
            raise AssertionError("ScriptedChat ran out of scripted responses")
        item = self._script.pop(0)
        response = item(messages) if callable(item) else item
        if on_token and response.content:
            on_token(response.content)
        return response

"""Shared value types and exceptions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class KestrelError(Exception):
    """Base class for all errors raised by this package."""


class LLMError(KestrelError):
    """A model-provider call failed.

    ``retryable`` tells the retry loop whether another attempt can help
    (rate limits, 5xx, transport errors) or not (bad request, auth).
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


class ConflictError(KestrelError):
    """An optimistic-concurrency check failed (stale head or stale version)."""


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model.

    ``arguments`` is the raw JSON string exactly as the model produced it;
    it is validated later, never trusted.
    """

    id: str
    name: str
    arguments: str


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    reasoning: str = ""
    model: str = ""
    latency_ms: float = 0.0


TokenCallback = Callable[[str], None]


class ChatModel(Protocol):
    """Anything that can answer a chat request, optionally streaming tokens."""

    model: str

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_token: TokenCallback | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse: ...


class Embedder(Protocol):
    """Turns texts into fixed-size vectors."""

    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


RunStatus = Literal["completed", "step_limit", "failed"]


@dataclass
class RunResult:
    session_id: str
    status: RunStatus
    final_text: str
    steps: int
    usage: Usage
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None

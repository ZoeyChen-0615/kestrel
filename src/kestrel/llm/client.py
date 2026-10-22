"""OpenAI-compatible chat client built directly on httpx.

Features: exponential backoff with jitter for transient failures (honouring
``Retry-After``), SSE streaming, and tolerant parsing of tool-call deltas.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from ..common import ChatResponse, LLMError, TokenCallback, ToolCall, Usage

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

Sleep = Callable[[float], Awaitable[None]]


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _backoff_delay(attempt: int, base: float, cap: float, retry_after: float | None) -> float:
    """Server hint wins; otherwise exponential backoff with +/-50% jitter."""
    if retry_after is not None:
        return min(retry_after, cap)
    return min(cap, base * (2**attempt)) * (0.5 + random.random() / 2)


def _error_from_response(status: int, body: str, retry_after: str | None) -> LLMError:
    snippet = body.strip().replace("\n", " ")[:300]
    return LLMError(
        f"provider returned HTTP {status}: {snippet}",
        retryable=status in RETRYABLE_STATUS,
        status=status,
        retry_after=_parse_retry_after(retry_after),
    )


async def post_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    max_retries: int,
    backoff_base: float = 1.0,
    backoff_cap: float = 30.0,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, Any]:
    """POST JSON and return the decoded body, retrying only transient failures."""
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code >= 400:
                raise _error_from_response(
                    response.status_code, response.text, response.headers.get("retry-after")
                )
            return response.json()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            error = LLMError(f"transport error: {type(exc).__name__}: {exc}", retryable=True)
        except ValueError as exc:  # invalid JSON in a 2xx body
            error = LLMError(f"provider returned invalid JSON: {exc}", retryable=True)
        except LLMError as exc:
            error = exc
        if not error.retryable or attempt == max_retries:
            raise error
        await sleep(_backoff_delay(attempt, backoff_base, backoff_cap, error.retry_after))
    raise AssertionError("unreachable")  # pragma: no cover


def _normalise_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall]:
    calls: list[ToolCall] = []
    seen: set[str] = set()
    for item in raw or []:
        function = item.get("function") or {}
        call_id = item.get("id") or ""
        if not call_id or call_id in seen:  # some providers omit or repeat ids
            call_id = f"call_{uuid.uuid4().hex[:12]}"
        seen.add(call_id)
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})
        calls.append(ToolCall(call_id, function.get("name") or "", arguments or "{}"))
    return calls


def _usage(raw: dict[str, Any] | None) -> Usage:
    raw = raw or {}
    return Usage(int(raw.get("prompt_tokens") or 0), int(raw.get("completion_tokens") or 0))


class OpenAICompatChat:
    """Chat completions against any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        max_tokens_param: str = "max_tokens",
        client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._max_tokens_param = max_tokens_param
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._sleep = sleep

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_token: TokenCallback | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens:
            payload[self._max_tokens_param] = max_tokens
        started = time.perf_counter()
        if on_token is None:
            data = await post_json_with_retry(
                self._client, self._url, payload, self._headers,
                max_retries=self._max_retries, backoff_base=self._backoff_base,
                backoff_cap=self._backoff_cap, sleep=self._sleep,
            )
            response = self._parse_full(data)
        else:
            response = await self._stream_with_retry(payload, on_token)
        response.latency_ms = (time.perf_counter() - started) * 1000
        return response

    def _parse_full(self, data: dict[str, Any]) -> ChatResponse:
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from exc
        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=_normalise_tool_calls(message.get("tool_calls")),
            finish_reason=choice.get("finish_reason") or "stop",
            usage=_usage(data.get("usage")),
            reasoning=message.get("reasoning_content") or "",
            model=data.get("model") or self.model,
        )

    async def _stream_with_retry(self, payload: dict[str, Any], on_token: TokenCallback) -> ChatResponse:
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        for attempt in range(self._max_retries + 1):
            emitted = False

            def forward(text: str) -> None:
                nonlocal emitted
                emitted = True
                on_token(text)

            try:
                return await self._stream_once(payload, forward)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                error = LLMError(f"transport error: {type(exc).__name__}: {exc}", retryable=True)
            except LLMError as exc:
                error = exc
            # Never retry after tokens reached the caller: that would duplicate output.
            if emitted or not error.retryable or attempt == self._max_retries:
                raise error
            await self._sleep(_backoff_delay(attempt, self._backoff_base, self._backoff_cap, error.retry_after))
        raise AssertionError("unreachable")  # pragma: no cover

    async def _stream_once(self, payload: dict[str, Any], on_token: TokenCallback) -> ChatResponse:
        content: list[str] = []
        reasoning: list[str] = []
        pending: dict[int, dict[str, Any]] = {}
        finish = "stop"
        usage = Usage()
        model = self.model
        async with self._client.stream("POST", self._url, json=payload, headers=self._headers) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                raise _error_from_response(response.status_code, body, response.headers.get("retry-after"))
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue  # tolerate keep-alive or malformed fragments
                model = chunk.get("model") or model
                if chunk.get("usage"):
                    usage = _usage(chunk["usage"])
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                        on_token(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    for part in delta.get("tool_calls") or []:
                        slot = pending.setdefault(part.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        if part.get("id"):
                            slot["id"] = part["id"]
                        function = part.get("function") or {}
                        if function.get("name"):
                            slot["name"] += function["name"]
                        if function.get("arguments"):
                            slot["arguments"] += function["arguments"]
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        raw_calls = [
            {"id": slot["id"], "function": {"name": slot["name"], "arguments": slot["arguments"]}}
            for _, slot in sorted(pending.items())
        ]
        return ChatResponse(
            content="".join(content),
            tool_calls=_normalise_tool_calls(raw_calls),
            finish_reason=finish,
            usage=usage,
            reasoning="".join(reasoning),
            model=model,
        )

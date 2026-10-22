"""Lightweight tracing: every model call, tool call and recovery action becomes an event row."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .common import Usage
from .store import Store


@dataclass
class Span:
    """Mutable bag of data attached to the event written when the span closes."""

    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def set(self, **values: Any) -> None:
        self.data.update(values)


class Tracer:
    def __init__(self, store: Store) -> None:
        self._store = store

    @contextmanager
    def span(self, kind: str, name: str, session_id: str | None = None, **data: Any) -> Iterator[Span]:
        span = Span(dict(data))
        started = time.perf_counter()
        try:
            yield span
        except BaseException as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration = (time.perf_counter() - started) * 1000
            payload = dict(span.data)
            if span.error:
                payload["error"] = span.error
            self._store.add_event(session_id, kind, name, duration, payload)

    def event(self, kind: str, name: str, session_id: str | None = None, **data: Any) -> None:
        self._store.add_event(session_id, kind, name, 0.0, data)


@dataclass
class TraceSummary:
    session_id: str
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)
    by_tool: Counter = field(default_factory=Counter)
    recoveries: Counter = field(default_factory=Counter)

    def cost(self, price_in_per_mtok: float, price_out_per_mtok: float) -> float:
        return (self.usage.prompt_tokens * price_in_per_mtok + self.usage.completion_tokens * price_out_per_mtok) / 1_000_000


def summarise(events: list[dict[str, Any]], session_id: str) -> TraceSummary:
    summary = TraceSummary(session_id)
    for event in events:
        data = event["data"]
        if event["kind"] == "llm":
            summary.llm_calls += 1
            summary.llm_ms += event["duration_ms"]
            summary.usage += Usage(int(data.get("prompt_tokens", 0)), int(data.get("completion_tokens", 0)))
        elif event["kind"] == "tool":
            summary.tool_calls += 1
            summary.tool_ms += event["duration_ms"]
            summary.by_tool[event["name"]] += 1
            if data.get("is_error"):
                summary.tool_errors += 1
        elif event["kind"] == "recovery":
            summary.recoveries[event["name"]] += 1
    return summary


def format_timeline(events: list[dict[str, Any]]) -> str:
    """Human-readable timeline of one session."""
    if not events:
        return "(no events)"
    t0 = events[0]["ts"]
    lines = []
    for event in events:
        data = event["data"]
        detail = []
        if event["kind"] == "llm":
            detail.append(f"in={data.get('prompt_tokens', 0)} out={data.get('completion_tokens', 0)} finish={data.get('finish_reason', '?')}")
            if data.get("tool_calls"):
                detail.append(f"tools={data['tool_calls']}")
        elif event["kind"] == "tool":
            detail.append("ERROR" if data.get("is_error") else "ok")
            if data.get("preview"):
                detail.append(str(data["preview"])[:70].replace("\n", " "))
        else:
            detail.extend(f"{k}={v}" for k, v in list(data.items())[:4])
        if data.get("error"):
            detail.append(f"!! {data['error']}"[:100])
        lines.append(f"+{event['ts'] - t0:7.2f}s  {event['kind']:<9} {event['name']:<22} {event['duration_ms']:8.0f}ms  {' '.join(detail)}")
    return "\n".join(lines)

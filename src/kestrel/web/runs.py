"""Run management for the web interface.

* ``RunManager`` runs one agent turn per session in a background task and exposes it
  as an async stream of JSON-serialisable events (tokens, tool calls, approvals...).
* ``Approvals`` turns the policy's "may I run this?" question into an event the browser
  can answer. If no browser is attached to the run, the answer is "no".
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from typing import Any

from ..app import App
from ..tools import Tool

Sink = Callable[[dict[str, Any]], None]

# The event sink of the run that is currently executing. Tasks inherit it, so tool calls made by
# sub-agents started during the run can still reach the same browser stream.
current_sink: ContextVar[Sink | None] = ContextVar("kestrel_current_sink", default=None)

_END = object()
UI_TEXT_LIMIT = 6000


class RunBusy(Exception):
    """A run is already in progress for this session."""


class Approvals:
    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = timeout
        self._pending: dict[str, dict[str, Any]] = {}

    async def request(self, tool: Tool, args: dict[str, Any]) -> bool:
        """Policy approver: ask the attached browser and wait for the human."""
        sink = current_sink.get()
        if sink is None:
            return False  # nobody is watching, so a risky action cannot be approved
        approval_id = uuid.uuid4().hex[:10]
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = {"id": approval_id, "tool": tool.name, "risk": tool.risk, "args": args,
                                      "created": time.time(), "future": future}
        sink({"type": "approval_request", "id": approval_id, "tool": tool.name, "risk": tool.risk, "args": args})
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            sink({"type": "approval_expired", "id": approval_id})
            return False
        finally:
            self._pending.pop(approval_id, None)

    def resolve(self, approval_id: str, allow: bool) -> bool:
        entry = self._pending.get(approval_id)
        if entry is None or entry["future"].done():
            return False
        entry["future"].set_result(allow)
        return True

    def pending(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in e.items() if k != "future"} for e in self._pending.values()]


def _clip(text: str) -> str:
    return text if len(text) <= UI_TEXT_LIMIT else text[:UI_TEXT_LIMIT] + f"\n[{len(text) - UI_TEXT_LIMIT} more characters not shown]"


class RunManager:
    def __init__(self, app: App) -> None:
        self.app = app
        self._active: dict[str, asyncio.Task[None]] = {}

    def is_running(self, session_id: str) -> bool:
        task = self._active.get(session_id)
        return task is not None and not task.done()

    def stop(self, session_id: str) -> bool:
        task = self._active.get(session_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def stream(self, session_id: str, message: str) -> AsyncIterator[dict[str, Any]]:
        """Run one turn and yield its events. Closing the stream (browser disconnect) cancels the run."""
        if self.is_running(session_id):
            raise RunBusy(session_id)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sink: Sink = queue.put_nowait
        reset = current_sink.set(sink)
        try:
            task = asyncio.create_task(self._run(session_id, message, sink), name=f"run-{session_id}")
        finally:
            current_sink.reset(reset)
        self._active[session_id] = task
        try:
            while True:
                event = await queue.get()
                if event is _END:
                    return
                yield event
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=5)
            self._active.pop(session_id, None)

    async def _run(self, session_id: str, message: str, sink: Sink) -> None:
        def on_event(kind: str, data: dict[str, Any]) -> None:
            if kind == "tool_result":
                data = {**data, "text": _clip(data["text"])}
            sink({"type": kind, **data})

        try:
            result = await self.app.agent.run(session_id, message, on_token=lambda t: sink({"type": "token", "text": t}), on_event=on_event)
            sink({"type": "done", "status": result.status, "final_text": result.final_text, "steps": result.steps,
                  "tokens": result.usage.total, "error": result.error})
        except asyncio.CancelledError:
            sink({"type": "cancelled"})
            raise
        except Exception as exc:
            sink({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            sink({"type": "done", "status": "failed", "final_text": "", "steps": 0, "tokens": 0, "error": str(exc)})
        finally:
            sink(_END)

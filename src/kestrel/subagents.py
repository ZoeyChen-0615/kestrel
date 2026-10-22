"""Sub-agents: a parent agent delegates a self-contained task to a child agent.

Design choices:

* The child is a normal ``Agent`` with its own session, a *restricted* tool set
  chosen by profile, a smaller step budget, and no ability to spawn further agents.
* Concurrency is bounded; excess spawns are refused with a clear message.
* Background jobs are tracked in the versioned key/value table, so their state
  survives restarts. A job that was running when the process died is marked
  ``failed`` on recovery rather than silently disappearing.
* Finished background jobs are reported to the parent on its next run.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from .agent import Agent
from .common import ConflictError
from .store import Store
from .tools import Tool, ToolContext, ToolRegistry, ToolResult

NAMESPACE = "subagent"
SUBAGENT_TOOLS = {"spawn_subagent", "subagent_status", "subagent_cancel"}
RESULT_LIMIT = 6000

CHILD_PERSONA = """You are a focused sub-agent working for another agent. Complete only the task you were given.
Finish with a short report: what you did, what you found, and anything you could not do. Do not ask questions."""

PROFILES: dict[str, Callable[[Tool], bool]] = {
    # Read-only investigation.
    "research": lambda t: t.risk == "read" and t.name not in SUBAGENT_TOOLS,
    # Files and shell only.
    "scripting": lambda t: t.name in {"read_file", "list_dir", "write_file", "shell"},
    # Everything the parent has, except further delegation.
    "general": lambda t: t.name not in SUBAGENT_TOOLS,
}


class SpawnArgs(BaseModel):
    task: str = Field(min_length=3, description="Complete, self-contained description of the task, with all needed context.")
    profile: Literal["research", "scripting", "general"] = Field(default="research", description="Tool access: research=read-only, scripting=files+shell, general=all tools.")
    background: bool = Field(default=False, description="Return immediately with a job id instead of waiting for the result.")
    label: str = Field(default="", max_length=60, description="Short name for the job.")


class JobArgs(BaseModel):
    job_id: str = Field(description="Job id returned by spawn_subagent.")


class SubagentManager:
    def __init__(self, store: Store, build_child: Callable[[str, str], Agent], *, max_active: int = 3) -> None:
        self._store = store
        self._build_child = build_child  # (profile, child_session_id) -> Agent
        self._max_active = max_active
        self._tasks: dict[str, asyncio.Task[None]] = {}

    # ---- lifecycle ----------------------------------------------------

    def recover(self) -> int:
        """Mark jobs left 'running' by a previous process as failed (explicit terminal state)."""
        fixed = 0
        for key, version, job in self._store.kv_list(NAMESPACE):
            if job["status"] == "running" and key not in self._tasks:
                job.update(status="failed", error="interrupted by a restart", finished=time.time())
                try:
                    self._store.kv_put(NAMESPACE, key, job, expected_version=version)
                    fixed += 1
                except ConflictError:
                    pass
        return fixed

    def active_count(self) -> int:
        return sum(1 for task in self._tasks.values() if not task.done())

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    # ---- operations -----------------------------------------------------

    async def spawn(self, parent_session: str, args: dict[str, Any]) -> ToolResult:
        if self.active_count() >= self._max_active:
            return ToolResult(f"Sub-agent capacity reached ({self._max_active} active). Wait for one to finish or do the work yourself.", True)
        job_id = uuid.uuid4().hex[:10]
        child_session = f"sub-{job_id}"
        self._store.create_session(session_id=child_session, kind="subagent", parent_id=parent_session, title=args["label"] or args["task"][:60])
        job = {"job_id": job_id, "parent": parent_session, "session": child_session, "profile": args["profile"],
               "label": args["label"], "task": args["task"], "status": "running", "result": "", "error": "",
               "delivered": not args["background"], "created": time.time(), "finished": None}
        self._store.kv_put(NAMESPACE, job_id, job, expected_version=None)
        task = asyncio.create_task(self._run(job_id, args["task"], args["profile"], child_session), name=f"subagent-{job_id}")
        self._tasks[job_id] = task
        if args["background"]:
            return ToolResult(f"Started background sub-agent {job_id}. Its result will be reported on your next turn; check with subagent_status.")
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()  # the parent was cancelled: do not leave an orphan running
            raise
        job = self._store.kv_get(NAMESPACE, job_id)[1]  # type: ignore[index]
        return self._describe(job, full=True)

    async def _run(self, job_id: str, task_text: str, profile: str, child_session: str) -> None:
        status, result, error = "completed", "", ""
        try:
            outcome = await self._build_child(profile, child_session).run(child_session, task_text)
            result, error = outcome.final_text, outcome.error or ""
            if outcome.status != "completed":
                status = "failed"
        except asyncio.CancelledError:
            status, error = "cancelled", "cancelled"
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        self._finish(job_id, status, result[:RESULT_LIMIT], error)

    def _finish(self, job_id: str, status: str, result: str, error: str) -> None:
        for _ in range(5):  # compare-and-swap retry loop
            found = self._store.kv_get(NAMESPACE, job_id)
            if found is None:
                return
            version, job = found
            job.update(status=status, result=result, error=error, finished=time.time())
            try:
                self._store.kv_put(NAMESPACE, job_id, job, expected_version=version)
                return
            except ConflictError:
                continue

    def status(self, job_id: str) -> ToolResult:
        found = self._store.kv_get(NAMESPACE, job_id)
        if found is None:
            return ToolResult(f"No such job: {job_id}", True)
        return self._describe(found[1], full=True)

    def cancel(self, job_id: str) -> ToolResult:
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return ToolResult(f"Job {job_id} is not running.", True)
        task.cancel()
        return ToolResult(f"Cancellation requested for {job_id}.")

    def collect_notices(self, parent_session: str) -> list[str]:
        """Finished background jobs of ``parent_session`` that have not been reported yet (reported once)."""
        notices = []
        for key, version, job in self._store.kv_list(NAMESPACE):
            if job["parent"] != parent_session or job["delivered"] or job["status"] == "running":
                continue
            job["delivered"] = True
            try:
                self._store.kv_put(NAMESPACE, key, job, expected_version=version)  # CAS: exactly-once delivery
            except ConflictError:
                continue
            notices.append("Background sub-agent finished. " + self._describe(job, full=True).text)
        return notices

    @staticmethod
    def _describe(job: dict[str, Any], *, full: bool) -> ToolResult:
        head = f"job {job['job_id']} ({job['profile']}{', ' + job['label'] if job['label'] else ''}): {job['status']}"
        if job["status"] == "running":
            return ToolResult(head)
        body = job["result"] if job["status"] == "completed" else f"error: {job['error']}"
        return ToolResult(f"{head}\n{body}" if full else head, job["status"] != "completed")


def register_subagent_tools(registry: ToolRegistry, manager: SubagentManager) -> None:
    async def spawn(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return await manager.spawn(ctx.session_id, args)

    async def status(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return manager.status(args["job_id"])

    async def cancel(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return manager.cancel(args["job_id"])

    registry.register(Tool.from_model(
        "spawn_subagent", "Delegate a self-contained task to a sub-agent with restricted tools. Use for independent, "
        "parallelisable or exploratory work; the sub-agent cannot see this conversation, so include all context.",
        SpawnArgs, spawn, risk="read", timeout=900.0))
    registry.register(Tool.from_model("subagent_status", "Check a sub-agent job.", JobArgs, status, idempotent=True, parallel_safe=True))
    registry.register(Tool.from_model("subagent_cancel", "Cancel a running sub-agent job.", JobArgs, cancel, risk="write"))

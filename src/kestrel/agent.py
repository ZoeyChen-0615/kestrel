"""The agent loop: reason, act, observe, repeat.

Ordering rules that make the loop recoverable:

1. Every message is committed to the log *before* anything it triggers runs.
   An assistant message with tool calls is stored first, then the tools run.
2. Tool results are stored as they finish, so a crash leaves a log from which
   the exact set of unanswered calls can be recomputed.
3. On the next run, unanswered calls are settled first: idempotent tools are
   re-run, everything else gets an explicit "outcome unknown" result so the
   model checks state instead of blindly repeating a side effect.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from .common import ChatModel, LLMError, RunResult, ToolCall, Usage
from .config import Settings
from .context import Compactor, PromptParts, build_system_prompt, ensure_persona, estimate_tokens
from .memory import EpisodicMemory, LongTermMemory
from .skills import SkillCatalog
from .store import Store
from .tools import Policy, ToolContext, ToolRegistry, ToolResult
from .tracing import Tracer

EventCallback = Callable[[str, dict[str, Any]], None]

UNKNOWN_OUTCOME = (
    "Outcome unknown: the run was interrupted before this call's result was recorded. "
    "The action may or may not have taken effect. Inspect the current state before retrying."
)


def open_calls(rows: list[tuple[int, str, dict[str, Any]]]) -> list[ToolCall]:
    """Tool calls in the latest assistant message that have no recorded result."""
    messages = [body for _, kind, body in rows if kind == "message"]
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") != "assistant":
            continue
        calls = messages[index].get("tool_calls") or []
        answered = {m.get("tool_call_id") for m in messages[index + 1 :] if m.get("role") == "tool"}
        return [
            ToolCall(c["id"], c["function"]["name"], c["function"]["arguments"])
            for c in calls
            if c["id"] not in answered
        ]
    return []


class Agent:
    def __init__(
        self,
        llm: ChatModel,
        store: Store,
        tools: ToolRegistry,
        policy: Policy,
        settings: Settings,
        tracer: Tracer,
        *,
        episodic: EpisodicMemory | None = None,
        longterm: LongTermMemory | None = None,
        skills: SkillCatalog | None = None,
        compactor: Compactor | None = None,
        services: dict[str, Any] | None = None,
        persona: str | None = None,
        learn: bool = True,
        max_steps: int | None = None,
        notices: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.tools = tools
        self.policy = policy
        self.settings = settings
        self.tracer = tracer
        self.episodic = episodic
        self.longterm = longterm
        self.skills = skills
        self.compactor = compactor
        self.services = services or {}
        self.persona = persona
        self.learn = learn
        self.max_steps = max_steps or settings.max_steps
        self._notices = notices
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------ public

    async def run(
        self,
        session_id: str,
        user_text: str | None,
        *,
        on_token: Callable[[str], None] | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        """Run one user turn to completion. ``user_text=None`` resumes an interrupted session."""
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:  # one run per session at a time keeps the log linear
            return await self._run_locked(session_id, user_text, on_token, on_event)

    # ----------------------------------------------------------------- internals

    async def _run_locked(self, session_id: str, user_text: str | None, on_token: Callable[[str], None] | None,
                          on_event: EventCallback | None) -> RunResult:
        emit = on_event or (lambda kind, data: None)
        self.store.create_session(session_id=session_id)
        ctx = ToolContext(session_id, self.settings.workspace, self.settings, self.store, self.services)  # type: ignore[arg-type]
        # Settle calls left unanswered by a crash BEFORE appending the new user message: providers require every
        # tool result to directly follow the assistant message that requested it.
        await self._settle_open_calls(session_id, ctx, emit)
        user_seq = self._record_user_message(session_id, user_text)
        if user_text is None:
            user_text = self._last_user_text(session_id)
        steps, usage, used_tools = 0, Usage(), []
        empty_retries = 0

        persona = self.persona or ensure_persona(self.settings.persona_path)
        long_term = self.longterm.read() if self.longterm else ""
        skills_index = self.skills.index_text() if self.skills else ""
        recalled = await self._recall(session_id, user_text or "")
        notices = self._notices(session_id) if self._notices else []
        schemas = self.tools.schemas()

        while True:
            await self._settle_open_calls(session_id, ctx, emit)
            if steps >= self.max_steps:
                note = f"I stopped because I reached the step limit ({self.max_steps} model calls). Ask me to continue if you want me to keep going."
                self.store.append(session_id, {"role": "assistant", "content": note})
                return RunResult(session_id, "step_limit", note, steps, usage, used_tools)

            summary, _, rows = self.store.load_context(session_id)
            parts = PromptParts(persona, long_term, skills_index, summary, recalled, notices, self.settings.workspace)  # type: ignore[arg-type]
            if self.compactor:
                fixed = estimate_tokens(build_system_prompt(PromptParts(persona, long_term, skills_index, None, recalled, notices, self.settings.workspace)), self.settings.chars_per_token) + estimate_tokens(schemas, self.settings.chars_per_token)  # type: ignore[arg-type]
                if await self.compactor.maybe_compact(session_id, fixed):
                    emit("compaction", {})
                    summary, _, rows = self.store.load_context(session_id)
                    parts.summary = summary
            messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(parts)}]
            messages += [m for _, m in rows]
            if empty_retries:
                messages.append({"role": "system", "content": "Your previous reply was empty. Reply with text or a tool call."})

            try:
                with self.tracer.span("llm", self.llm.model, session_id, step=steps + 1) as span:
                    response = await self.llm.chat(messages, schemas or None, on_token=on_token, max_tokens=self.settings.max_output_tokens)
                    span.set(prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
                             finish_reason=response.finish_reason, tool_calls=[c.name for c in response.tool_calls])
            except LLMError as exc:
                emit("error", {"message": str(exc)})
                return RunResult(session_id, "failed", "", steps, usage, used_tools, error=str(exc))
            steps += 1
            usage += response.usage

            if response.finish_reason == "length":
                # Truncated output may contain half-written tool arguments: never execute them.
                partial = response.content.strip() or "(output was cut off by the length limit)"
                self.store.append(session_id, {"role": "assistant", "content": partial})
                return RunResult(session_id, "failed", partial, steps, usage, used_tools, error="output hit the length limit; tool calls were not executed")

            if not response.content.strip() and not response.tool_calls:
                empty_retries += 1
                self.tracer.event("recovery", "empty_response", session_id, attempt=empty_retries)
                if empty_retries > self.settings.empty_retries:
                    return RunResult(session_id, "failed", "", steps, usage, used_tools, error="model returned empty responses repeatedly")
                continue
            empty_retries = 0

            message: dict[str, Any] = {"role": "assistant", "content": response.content}
            if response.tool_calls:
                message["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in response.tool_calls
                ]
            self.store.append(session_id, message)  # commit first, execute second
            if not response.tool_calls:
                await self._learn(session_id, user_seq, user_text, response.content)
                return RunResult(session_id, "completed", response.content, steps, usage, used_tools)
            used_tools += [c.name for c in response.tool_calls]
            await self._execute(session_id, response.tool_calls, ctx, emit)

    def _record_user_message(self, session_id: str, user_text: str | None) -> int:
        if user_text is None:
            return 0
        self.store.set_title(session_id, user_text.strip()[:60])
        return self.store.append(session_id, {"role": "user", "content": user_text})

    def _last_user_text(self, session_id: str) -> str | None:
        for _, kind, body in reversed(self.store.rows(session_id)):
            if kind == "message" and body.get("role") == "user":
                return body.get("content")
        return None

    async def _recall(self, session_id: str, query: str) -> list[str]:
        """Auto-retrieve relevant past exchanges. Failure degrades to 'no recall', never to a crash."""
        if self.episodic is None or not query.strip():
            return []
        _, covers, _ = self.store.load_context(session_id)
        try:
            with self.tracer.span("memory", "recall", session_id) as span:
                # Exchanges after the last summary are still in the prompt; recalling them would be redundant.
                hits = await self.episodic.search(query, k=self.settings.recall_k, exclude=lambda s, seq: s == session_id and seq > covers)
                span.set(hits=len(hits))
        except Exception:
            return []
        return [f"({h.age_label()}) {h.text}" for h in hits]

    async def _learn(self, session_id: str, user_seq: int, user_text: str | None, answer: str) -> None:
        if not (self.learn and self.episodic and user_seq and user_text and answer.strip()):
            return
        try:
            with self.tracer.span("memory", "learn", session_id):
                await self.episodic.learn(session_id, user_seq, user_text, answer)
        except Exception:
            pass  # memory is best effort; the answer has already been delivered

    async def _settle_open_calls(self, session_id: str, ctx: ToolContext, emit: EventCallback) -> None:
        for call in open_calls(self.store.rows(session_id)):
            tool = self.tools.get(call.name)
            if tool is not None and tool.idempotent:
                self.tracer.event("recovery", "retry_idempotent", session_id, tool=call.name)
                result = await self._run_one(session_id, call, ctx, emit)
            else:
                self.tracer.event("recovery", "unknown_outcome", session_id, tool=call.name)
                result = ToolResult(UNKNOWN_OUTCOME, True)
            self._record_result(session_id, call, result)

    async def _execute(self, session_id: str, calls: list[ToolCall], ctx: ToolContext, emit: EventCallback) -> None:
        """Run calls in order; consecutive parallel-safe calls overlap, results are stored in call order."""
        index = 0
        while index < len(calls):
            tool = self.tools.get(calls[index].name)
            if tool is not None and tool.parallel_safe:
                end = index
                while end < len(calls) and (t := self.tools.get(calls[end].name)) is not None and t.parallel_safe:
                    end += 1
                batch = calls[index:end]
                gate = asyncio.Semaphore(self.settings.max_parallel_tools)

                async def guarded(call: ToolCall) -> ToolResult:
                    async with gate:
                        return await self._run_one(session_id, call, ctx, emit)

                results = await asyncio.gather(*(guarded(c) for c in batch))
                for call, result in zip(batch, results):
                    self._record_result(session_id, call, result)
                index = end
            else:
                result = await self._run_one(session_id, calls[index], ctx, emit)
                self._record_result(session_id, calls[index], result)
                index += 1

    async def _run_one(self, session_id: str, call: ToolCall, ctx: ToolContext, emit: EventCallback) -> ToolResult:
        emit("tool_call", {"id": call.id, "name": call.name, "arguments": call.arguments})
        with self.tracer.span("tool", call.name, session_id) as span:
            result = await self.tools.execute(call, ctx, self.policy)
            span.set(is_error=result.is_error, preview=result.text[:200])
        emit("tool_result", {"id": call.id, "name": call.name, "is_error": result.is_error, "text": result.text})
        return result

    def _record_result(self, session_id: str, call: ToolCall, result: ToolResult) -> None:
        self.store.append(session_id, {"role": "tool", "tool_call_id": call.id, "content": result.text})

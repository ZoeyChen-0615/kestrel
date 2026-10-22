"""Built-in tools: workspace files, shell, memory and skills."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .registry import ArgumentError, Tool, ToolContext, ToolRegistry, ToolResult

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def resolve_in_workspace(workspace: Path, raw: str) -> Path:
    """Resolve ``raw`` and refuse anything that escapes the workspace (including via symlinks)."""
    root = workspace.resolve()
    candidate = Path(raw).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if path != root and root not in path.parents:
        raise ArgumentError(f"path '{raw}' is outside the workspace")
    return path


def resolve_readable(ctx: ToolContext, raw: str) -> Path:
    """Workspace paths, plus absolute paths inside skill directories (so a skill can ship ``references/``).

    Skill directories are read-only: ``write_file`` still uses :func:`resolve_in_workspace`.
    """
    try:
        return resolve_in_workspace(ctx.workspace, raw)
    except ArgumentError as workspace_error:
        skills = ctx.services.get("skills")
        if skills is not None and Path(raw).expanduser().is_absolute():
            for root in skills.read_roots():
                try:
                    return resolve_in_workspace(root, raw)
                except ArgumentError:
                    continue
        raise workspace_error


def scrubbed_env() -> dict[str, str]:
    """Environment for subprocesses with anything that looks like a credential removed."""
    return {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in _SECRET_MARKERS)}


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path, relative to the workspace.")
    offset: int = Field(default=0, ge=0, description="Zero-based line to start from.")
    limit: int = Field(default=400, ge=1, le=2000, description="Maximum number of lines to return.")


class WriteFileArgs(BaseModel):
    path: str = Field(description="File path, relative to the workspace. Parent directories are created.")
    content: str = Field(description="Full new content of the file.")


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Directory path, relative to the workspace.")


class ShellArgs(BaseModel):
    command: str = Field(description="Shell command to run, with the workspace as the working directory.")
    timeout_s: int = Field(default=60, ge=1, le=600, description="Kill the command after this many seconds.")


class RecallArgs(BaseModel):
    query: str = Field(description="What to look for in past conversations.")
    k: int = Field(default=5, ge=1, le=10, description="Maximum number of results.")


class RememberArgs(BaseModel):
    fact: str = Field(min_length=3, max_length=200, description="ONE short, durable fact about the user or a lasting preference. Never combine unrelated facts in one call.")
    replaces: str | None = Field(default=None, min_length=3, description="If this fact updates an earlier one, a distinctive substring of the OLD fact; it is swapped atomically.")


class ForgetArgs(BaseModel):
    text: str = Field(min_length=3, description="Distinctive substring of the one remembered fact to delete. Refused if it matches several.")


class LoadSkillArgs(BaseModel):
    name: str = Field(description="Skill name from the skills index.")


async def read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = resolve_readable(ctx, args["path"])
    if not path.is_file():
        raise ArgumentError(f"'{args['path']}' is not a file")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start, stop = args["offset"], args["offset"] + args["limit"]
    numbered = [f"{i + 1:>6}\t{line}" for i, line in enumerate(lines[start:stop], start=start)]
    footer = f"\n[showing lines {start + 1}-{min(stop, len(lines))} of {len(lines)}]" if len(lines) > stop or start else ""
    return "\n".join(numbered) + footer if numbered else "(empty file or offset past end)"


async def write_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = resolve_in_workspace(ctx.workspace, args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(args["content"], encoding="utf-8")
    os.replace(temp, path)  # atomic: readers never see a half-written file
    return f"Wrote {len(args['content'].encode('utf-8'))} bytes to {path.relative_to(ctx.workspace.resolve())}"


async def list_dir(args: dict[str, Any], ctx: ToolContext) -> str:
    path = resolve_readable(ctx, args["path"])
    if not path.is_dir():
        raise ArgumentError(f"'{args['path']}' is not a directory")
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    return "\n".join(f"{p.name}{'/' if p.is_dir() else ''}" for p in entries[:500]) or "(empty directory)"


async def shell(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    process = await asyncio.create_subprocess_shell(
        args["command"], cwd=ctx.workspace, env=scrubbed_env(),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # own process group so a timeout can kill children too
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=args["timeout_s"])
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        return ToolResult(f"Command timed out after {args['timeout_s']}s and was killed.", True)
    text = output.decode("utf-8", errors="replace")
    return ToolResult(f"exit code: {process.returncode}\n{text}", process.returncode != 0)


async def recall_memory(args: dict[str, Any], ctx: ToolContext) -> str:
    episodic = ctx.services.get("episodic")
    if episodic is None:
        return "Episodic memory is not enabled."
    hits = await episodic.search(args["query"], k=args["k"])
    if not hits:
        return "No relevant past conversations found."
    return "\n\n".join(f"[{h.id}] ({h.age_label()}, score {h.score:.2f})\n{h.text}" for h in hits)


def _memory_listing(longterm: Any) -> str:
    facts = longterm.facts()
    return "Current memory:\n" + ("\n".join(f"- {f}" for f in facts) if facts else "(empty)")


async def remember(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    longterm = ctx.services["longterm"]
    if args.get("replaces"):
        status = longterm.replace(args["replaces"], args["fact"])
        if status == "ambiguous":
            matches = "\n".join(f"- {f}" for f in longterm.matching(args["replaces"]))
            return ToolResult(f"Not changed: 'replaces' matches several facts. Use a more distinctive substring.\n{matches}", True)
        message = {"replaced": "Updated.", "added": "No earlier fact matched; added as a new fact.", "duplicate": "Already remembered."}.get(status, status)
    else:
        status = longterm.add(args["fact"])
        message = {"added": "Remembered.", "duplicate": "Already remembered."}.get(status, status)
    return ToolResult(f"{message}\n{_memory_listing(longterm)}", status not in ("added", "duplicate", "replaced"))


async def forget(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    longterm = ctx.services["longterm"]
    matches = longterm.matching(args["text"])
    if not matches:
        return ToolResult(f"Nothing matched '{args['text']}'.\n{_memory_listing(longterm)}", True)
    if len(matches) > 1:
        listed = "\n".join(f"- {f}" for f in matches)
        return ToolResult(f"Not removed: '{args['text']}' matches {len(matches)} facts. Use a more distinctive substring.\n{listed}", True)
    longterm.remove(matches[0], exact=True)
    return ToolResult(f"Removed: {matches[0]}\n{_memory_listing(longterm)}")


async def load_skill(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    skills = ctx.services["skills"]
    return skills.load(args["name"])


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register(Tool.from_model(
        "read_file", "Read a text file from the workspace with line numbers.", ReadFileArgs, read_file,
        idempotent=True, parallel_safe=True, risk="read"))
    registry.register(Tool.from_model(
        "list_dir", "List the entries of a workspace directory.", ListDirArgs, list_dir,
        idempotent=True, parallel_safe=True, risk="read"))
    registry.register(Tool.from_model(
        "write_file", "Create or overwrite a file in the workspace.", WriteFileArgs, write_file,
        idempotent=True, risk="write"))
    registry.register(Tool.from_model(
        "shell", "Run a shell command in the workspace and return its exit code and combined output.", ShellArgs, shell,
        risk="dangerous", timeout=620.0))
    registry.register(Tool.from_model(
        "recall_memory", "Search past conversations by meaning and keywords.", RecallArgs, recall_memory,
        idempotent=True, parallel_safe=True, risk="read"))
    registry.register(Tool.from_model(
        "remember", "Save ONE durable fact about the user to long-term memory (name, location, allergies, lasting preferences). "
        "Use replaces=<substring of the old fact> when a fact changes. Call it once per fact.",
        RememberArgs, remember, idempotent=True, risk="write"))
    registry.register(Tool.from_model(
        "forget", "Delete one fact from long-term memory (only when the user asks to forget it, or it is simply wrong).", ForgetArgs, forget, idempotent=True, risk="write"))
    registry.register(Tool.from_model(
        "load_skill", "Load the full instructions of a skill listed in the skills index.", LoadSkillArgs, load_skill,
        idempotent=True, parallel_safe=True, risk="read"))

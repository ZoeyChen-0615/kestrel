"""Skills: folders containing a ``SKILL.md`` that the model loads on demand.

Only a one-line index (name + description) lives in the system prompt. The
full instructions are fetched with the ``load_skill`` tool, together with the
skill's directory so relative resources (``references/...``) can be read.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .tools.registry import ToolResult

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    directory: Path
    source: str
    missing: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return not self.missing


def parse_skill(path: Path, source: str) -> Skill | None:
    """Parse one ``SKILL.md``; returns None when the file is unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta: dict[str, Any] = {}
    body = text
    match = _FRONTMATTER.match(text)
    if match:
        try:
            loaded = yaml.safe_load(match.group(1)) or {}
            meta = loaded if isinstance(loaded, dict) else {}
        except yaml.YAMLError:
            meta = {}
        body = match.group(2)
    requires = meta.get("requires") or {}
    missing = [f"bin:{b}" for b in requires.get("bins", []) if shutil.which(str(b)) is None]
    missing += [f"env:{e}" for e in requires.get("env", []) if not os.environ.get(str(e))]
    name = str(meta.get("name") or path.parent.name)
    return Skill(name, str(meta.get("description") or "").strip(), body.strip(), path.parent, source, missing)


class SkillCatalog:
    """Scans skill directories in priority order: later directories override earlier ones."""

    def __init__(self, roots: list[tuple[str, Path]]) -> None:
        self._roots = roots

    def read_roots(self) -> list[Path]:
        """Existing skill directories; tools may read (never write) files inside them."""
        return [root.resolve() for _, root in self._roots if root.is_dir()]

    def scan(self) -> dict[str, Skill]:
        found: dict[str, Skill] = {}
        for source, root in self._roots:
            if not root.is_dir():
                continue
            for entry in sorted(root.iterdir()):
                skill_file = entry / "SKILL.md"
                if entry.is_dir() and skill_file.is_file():
                    skill = parse_skill(skill_file, source)
                    if skill and skill.body:
                        found[skill.name] = skill
        return found

    def index_text(self) -> str:
        skills = self.scan()
        if not skills:
            return ""
        lines = []
        for skill in sorted(skills.values(), key=lambda s: s.name):
            note = "" if skill.available else f" (unavailable: missing {', '.join(skill.missing)})"
            lines.append(f"- {skill.name}: {skill.description}{note}")
        return "\n".join(lines)

    def load(self, name: str) -> ToolResult:
        skills = self.scan()
        skill = skills.get(name)
        if skill is None:
            return ToolResult(f"Unknown skill '{name}'. Available: {', '.join(sorted(skills)) or '(none)'}.", True)
        if not skill.available:
            return ToolResult(f"Skill '{name}' is unavailable: missing {', '.join(skill.missing)}. Tell the user instead of improvising.", True)
        return ToolResult(f"Skill: {skill.name}\nBase directory: {skill.directory}\n\n{skill.body}")

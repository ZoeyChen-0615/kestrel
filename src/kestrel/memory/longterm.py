"""Long-term memory: a small, human-readable Markdown file of curated facts.

The file is injected into every prompt, so it is kept short (a character
quota) and edited conservatively: facts are one-liners, duplicates are
ignored, and when the quota is exceeded the oldest facts are dropped first.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..common import ChatModel

MAX_FACT_CHARS = 200

EXTRACTION_PROMPT = """You maintain a user's long-term memory file.
From the conversation excerpt below, extract durable facts worth remembering across future sessions:
stable personal facts, preferences, ongoing projects, and explicit requests to remember something.
Only use facts the USER stated. Ignore small talk, one-off tasks and anything already listed under "Existing memory".
Return ONLY a JSON array of operations, each one of:
  {"op": "add", "text": "<one short sentence>"}
  {"op": "remove", "text": "<substring of an existing fact that is now wrong>"}
Return [] if nothing qualifies. At most 5 operations."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class LongTermMemory:
    def __init__(self, path: Path, *, max_chars: int = 3000) -> None:
        self.path = path
        self.max_chars = max_chars

    def facts(self) -> list[str]:
        if not self.path.is_file():
            return []
        return [line[2:].strip() for line in self.path.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]

    def read(self) -> str:
        facts = self.facts()
        return "\n".join(f"- {f}" for f in facts)

    def _write(self, facts: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = "# Long-term memory\n\n" + "\n".join(f"- {f}" for f in facts) + ("\n" if facts else "")
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(body, encoding="utf-8")
        os.replace(temp, self.path)  # atomic replace

    def add(self, fact: str) -> str:
        fact = " ".join(fact.split())
        if not fact or len(fact) > MAX_FACT_CHARS:
            return f"fact must be 1-{MAX_FACT_CHARS} characters"
        facts = self.facts()
        if _norm(fact) in {_norm(f) for f in facts}:
            return "duplicate"
        facts.append(fact)
        while len("\n".join(facts)) > self.max_chars and len(facts) > 1:
            facts.pop(0)  # quota exceeded: forget the oldest
        self._write(facts)
        return "added"

    def matching(self, fragment: str) -> list[str]:
        """Facts containing ``fragment`` (case- and whitespace-insensitive)."""
        needle = _norm(fragment)
        return [f for f in self.facts() if needle in _norm(f)]

    def remove(self, fragment: str, *, exact: bool = False) -> int:
        """Remove every fact containing ``fragment`` (or equal to it when ``exact``). Returns how many were removed."""
        needle = _norm(fragment)
        facts = self.facts()
        kept = [f for f in facts if (_norm(f) != needle if exact else needle not in _norm(f))]
        if len(kept) != len(facts):
            self._write(kept)
        return len(facts) - len(kept)

    def replace(self, old_fragment: str, new_fact: str) -> str:
        """Atomically swap the single fact containing ``old_fragment`` for ``new_fact``.

        Returns ``replaced``, ``added`` (nothing matched, so the new fact was simply added),
        ``ambiguous`` (several facts matched; nothing changed) or a validation message.
        """
        new_fact = " ".join(new_fact.split())
        if not new_fact or len(new_fact) > MAX_FACT_CHARS:
            return f"fact must be 1-{MAX_FACT_CHARS} characters"
        matches = self.matching(old_fragment)
        if len(matches) > 1:
            return "ambiguous"
        if not matches:
            return "added" if self.add(new_fact) == "added" else "duplicate"
        facts = self.facts()
        facts[facts.index(matches[0])] = new_fact
        self._write(facts)
        return "replaced"

    def apply_operations(self, operations: list[dict[str, Any]]) -> int:
        """Apply validated add/remove operations; returns how many changed the file."""
        changed = 0
        for operation in operations[:5]:
            if not isinstance(operation, dict) or not isinstance(operation.get("text"), str):
                continue
            if operation.get("op") == "add" and self.add(operation["text"]) == "added":
                changed += 1
            elif operation.get("op") == "remove":
                changed += 1 if self.remove(operation["text"]) else 0
        return changed

    async def extract_from(self, llm: ChatModel, transcript: str) -> int:
        """Ask the model for memory operations over ``transcript`` and apply them."""
        existing = self.read() or "(empty)"
        response = await llm.chat(
            [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Existing memory:\n{existing}\n\nConversation excerpt:\n{transcript}"},
            ],
            max_tokens=500,
        )
        return self.apply_operations(parse_operations(response.content))


def parse_operations(text: str) -> list[dict[str, Any]]:
    """Pull a JSON array out of a model reply, tolerating code fences and prose."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

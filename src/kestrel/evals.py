"""A small evaluation harness.

A suite is a JSON file of cases. Each case runs in a fresh, throw-away
workspace and database, then declarative checks decide pass/fail. Reports
include the numbers that matter when comparing models or prompts: pass rate,
steps, tokens and latency.

Case format::

    {"id": "read-file", "prompt": "What is in notes.txt?",
     "files": {"notes.txt": "hello"},
     "checks": [{"type": "contains", "value": "hello"}, {"type": "tool_called", "name": "read_file"}]}

``"turns": [...]`` can replace ``"prompt"`` for multi-turn cases (checks apply to the last reply).
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .app import App
from .common import RunResult, Usage


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    status: str
    steps: int
    tokens: int
    seconds: float
    failures: list[str] = field(default_factory=list)
    final_text: str = ""


@dataclass
class SuiteResult:
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def to_dict(self) -> dict[str, Any]:
        n = len(self.results) or 1
        return {
            "pass_rate": self.pass_rate,
            "cases": len(self.results),
            "avg_steps": sum(r.steps for r in self.results) / n,
            "total_tokens": sum(r.tokens for r in self.results),
            "avg_seconds": sum(r.seconds for r in self.results) / n,
            "results": [r.__dict__ for r in self.results],
        }

    def format(self) -> str:
        lines = [f"{'case':<28} {'result':<6} {'status':<11} {'steps':>5} {'tokens':>7} {'sec':>6}"]
        for r in self.results:
            lines.append(f"{r.case_id:<28} {'PASS' if r.passed else 'FAIL':<6} {r.status:<11} {r.steps:>5} {r.tokens:>7} {r.seconds:>6.1f}")
            lines += [f"    - {failure}" for failure in r.failures]
        d = self.to_dict()
        lines.append(f"\npass rate {d['pass_rate']:.0%} ({sum(r.passed for r in self.results)}/{d['cases']})  "
                     f"avg steps {d['avg_steps']:.1f}  tokens {d['total_tokens']}  avg {d['avg_seconds']:.1f}s")
        return "\n".join(lines)


def evaluate_checks(checks: list[dict[str, Any]], result: RunResult, used_tools: list[str], workspace: Path,
                    expect_status: str = "completed") -> list[str]:
    """Return a list of human-readable failures (empty means every check passed)."""
    failures = []
    if result.status != expect_status:
        failures.append(f"status was {result.status!r}, expected {expect_status!r}" + (f" ({result.error})" if result.error else ""))
    text = result.final_text
    for check in checks:
        kind = check["type"]
        if kind == "contains" and check["value"].lower() not in text.lower():
            failures.append(f"reply does not contain {check['value']!r}")
        elif kind == "not_contains" and check["value"].lower() in text.lower():
            failures.append(f"reply unexpectedly contains {check['value']!r}")
        elif kind == "regex" and not re.search(check["value"], text, re.IGNORECASE | re.DOTALL):
            failures.append(f"reply does not match /{check['value']}/")
        elif kind == "tool_called" and check["name"] not in used_tools:
            failures.append(f"tool {check['name']!r} was not called (used: {sorted(set(used_tools)) or 'none'})")
        elif kind == "tool_not_called" and check["name"] in used_tools:
            failures.append(f"tool {check['name']!r} should not have been called")
        elif kind == "max_steps" and result.steps > check["value"]:
            failures.append(f"took {result.steps} steps, limit {check['value']}")
        elif kind == "file_contains":
            path = workspace / check["path"]
            if not path.is_file():
                failures.append(f"file {check['path']!r} does not exist")
            elif check["value"] not in path.read_text(encoding="utf-8", errors="replace"):
                failures.append(f"file {check['path']!r} does not contain {check['value']!r}")
    return failures


async def run_case(case: dict[str, Any], make_app: Callable[[Path], Awaitable[App]]) -> CaseResult:
    turns = case.get("turns") or [case["prompt"]]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="kestrel-eval-") as tmp:
        app = await make_app(Path(tmp))
        try:
            workspace = app.settings.workspace
            assert workspace is not None
            for name, content in (case.get("files") or {}).items():
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            session = app.store.create_session(session_id=f"eval-{case['id']}")
            used: list[str] = []
            usage, steps, last = Usage(), 0, None
            for turn in turns:
                last = await app.agent.run(session, turn)
                used += last.tool_calls
                usage += last.usage
                steps += last.steps
            assert last is not None
            last.steps = steps
            failures = evaluate_checks(case.get("checks", []), last, used, workspace, case.get("expect_status", "completed"))
            return CaseResult(case["id"], not failures, last.status, steps, usage.total, time.perf_counter() - started, failures, last.final_text)
        finally:
            await app.aclose()


async def run_suite(cases: list[dict[str, Any]], make_app: Callable[[Path], Awaitable[App]]) -> SuiteResult:
    return SuiteResult([await run_case(case, make_app) for case in cases])


def load_suite(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids in suite")
    return cases

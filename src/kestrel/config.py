"""Runtime settings, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ApprovalMode = Literal["ask", "auto", "readonly"]


def load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` lines into ``os.environ`` without overriding existing values."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


@dataclass
class Settings:
    """All tunables in one place. Every field has a safe default."""

    home: Path = field(default_factory=lambda: Path.home() / ".kestrel")
    workspace: Path | None = None

    # Chat model (any OpenAI-compatible endpoint).
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = "deepseek-chat"
    # Some newer OpenAI models reject "max_tokens" and want "max_completion_tokens".
    max_tokens_param: str = "max_tokens"

    # Embeddings (any OpenAI-compatible /embeddings endpoint). Empty key => offline hashing embedder.
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key: str = ""
    embed_model: str = "text-embedding-3-small"

    # Loop limits.
    max_steps: int = 40
    max_output_tokens: int = 4096
    empty_retries: int = 2
    llm_max_retries: int = 5
    llm_timeout: float = 120.0

    # Context management.
    context_window: int = 64000
    compact_ratio: float = 0.70
    keep_tail_ratio: float = 0.25
    chars_per_token: float = 3.0
    long_term_max_chars: int = 3000

    # Tools and guardrails.
    approval: ApprovalMode = "ask"
    tool_timeout: float = 60.0
    max_tool_output_chars: int = 20000
    max_parallel_tools: int = 4

    # Sub-agents.
    subagent_max_active: int = 3
    subagent_max_steps: int = 15

    # Retrieval.
    recall_k: int = 4
    recall_min_dense: float = 0.25

    # Optional cost estimate (USD per million tokens). Zero disables the estimate.
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0

    # Telegram channel (optional). The allow-list is mandatory: usernames (without @) and/or numeric user ids.
    telegram_token: str = ""
    telegram_allowed: list[str] = field(default_factory=list)

    mcp_config: Path | None = None
    extra_skill_dirs: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.home = Path(self.home).expanduser()
        if self.workspace is None:
            self.workspace = self.home / "workspace"
        self.workspace = Path(self.workspace).expanduser()

    @property
    def db_path(self) -> Path:
        return self.home / "state.db"

    @property
    def memory_dir(self) -> Path:
        return self.workspace / "memory"  # type: ignore[operator]

    @property
    def persona_path(self) -> Path:
        return self.memory_dir / "PERSONA.md"

    @property
    def long_term_path(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    def ensure_dirs(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, dotenv: Path | None = Path(".env")) -> Settings:
        if dotenv is not None:
            load_dotenv(dotenv)
        home = Path(_env("KESTREL_HOME", str(Path.home() / ".kestrel")))
        workspace = _env("KESTREL_WORKSPACE") or None
        mcp = _env("KESTREL_MCP_CONFIG") or None
        approval = _env("KESTREL_APPROVAL", "ask")
        if approval not in ("ask", "auto", "readonly"):
            raise ValueError(f"KESTREL_APPROVAL must be ask|auto|readonly, got {approval!r}")
        return cls(
            home=home,
            workspace=Path(workspace) if workspace else None,
            base_url=_env("KESTREL_BASE_URL", "https://api.deepseek.com"),
            api_key=_env("KESTREL_API_KEY"),
            model=_env("KESTREL_MODEL", "deepseek-chat"),
            max_tokens_param=_env("KESTREL_MAX_TOKENS_PARAM", "max_tokens"),
            embed_base_url=_env("KESTREL_EMBED_BASE_URL", "https://api.openai.com/v1"),
            embed_api_key=_env("KESTREL_EMBED_API_KEY"),
            embed_model=_env("KESTREL_EMBED_MODEL", "text-embedding-3-small"),
            max_steps=_env_int("KESTREL_MAX_STEPS", 40),
            context_window=_env_int("KESTREL_CONTEXT_WINDOW", 64000),
            approval=approval,  # type: ignore[arg-type]
            price_in_per_mtok=_env_float("KESTREL_PRICE_IN_PER_MTOK", 0.0),
            price_out_per_mtok=_env_float("KESTREL_PRICE_OUT_PER_MTOK", 0.0),
            mcp_config=Path(mcp) if mcp else None,
            telegram_token=_env("KESTREL_TELEGRAM_TOKEN"),
            telegram_allowed=[x.strip().lstrip("@") for x in _env("KESTREL_TELEGRAM_ALLOWED").split(",") if x.strip()],
        )

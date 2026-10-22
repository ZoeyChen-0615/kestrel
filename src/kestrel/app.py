"""Composition root: builds every component once and wires them together."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent import Agent
from .common import ChatModel, Embedder, KestrelError
from .config import Settings
from .context import Compactor
from .llm import HashingEmbedder, OpenAICompatChat, OpenAICompatEmbedder
from .mcp_client import McpManager, load_config
from .memory import EpisodicMemory, LongTermMemory
from .skills import SkillCatalog
from .store import Store
from .subagents import CHILD_PERSONA, PROFILES, SubagentManager, register_subagent_tools
from .tools import Policy, ToolRegistry
from .tools.builtin import register_builtin_tools
from .tools.guardrails import Approver
from .tracing import Tracer


class ConfigError(KestrelError):
    pass


def _bundled_skills_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "skills"


class App:
    """Owns the lifetime of the store, model clients, MCP connections and sub-agents."""

    def __init__(self, settings: Settings, store: Store, llm: ChatModel, embedder: Embedder, registry: ToolRegistry,
                 policy: Policy, tracer: Tracer, episodic: EpisodicMemory, longterm: LongTermMemory,
                 skills: SkillCatalog, compactor: Compactor, subagents: SubagentManager, mcp: McpManager | None,
                 mcp_report: dict[str, str]) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.registry = registry
        self.policy = policy
        self.tracer = tracer
        self.episodic = episodic
        self.longterm = longterm
        self.skills = skills
        self.compactor = compactor
        self.subagents = subagents
        self.mcp = mcp
        self.mcp_report = mcp_report
        self.agent = Agent(
            llm, store, registry, policy, settings, tracer,
            episodic=episodic, longterm=longterm, skills=skills, compactor=compactor,
            services=self._services(), notices=subagents.collect_notices,
        )

    def _services(self) -> dict[str, Any]:
        return {"episodic": self.episodic, "longterm": self.longterm, "skills": self.skills}

    @classmethod
    async def create(
        cls,
        settings: Settings,
        *,
        llm: ChatModel | None = None,
        embedder: Embedder | None = None,
        approver: Approver | None = None,
        use_mcp: bool = True,
    ) -> App:
        settings.ensure_dirs()
        if llm is None:
            if not settings.api_key:
                raise ConfigError("KESTREL_API_KEY is not set. Copy .env.example to .env and fill it in.")
            llm = OpenAICompatChat(
                settings.base_url, settings.api_key, settings.model, timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries, max_tokens_param=settings.max_tokens_param,
            )
        if embedder is None:
            embedder = (OpenAICompatEmbedder(settings.embed_base_url, settings.embed_api_key, settings.embed_model)
                        if settings.embed_api_key else HashingEmbedder())

        store = Store(settings.db_path)
        tracer = Tracer(store)
        registry = ToolRegistry()
        register_builtin_tools(registry)

        mcp: McpManager | None = None
        mcp_report: dict[str, str] = {}
        if use_mcp and settings.mcp_config is not None:
            mcp = McpManager(load_config(settings.mcp_config))
            mcp_report = await mcp.start(registry)

        roots = [("bundled", _bundled_skills_dir()), ("user", settings.home / "skills"),
                 ("workspace", settings.workspace / "skills"),  # type: ignore[operator]
                 *[("extra", d) for d in settings.extra_skill_dirs]]
        skills = SkillCatalog(roots)
        longterm = LongTermMemory(settings.long_term_path, max_chars=settings.long_term_max_chars)
        episodic = EpisodicMemory(store, embedder, min_dense=settings.recall_min_dense)
        compactor = Compactor(llm, store, settings, tracer, longterm)
        policy = Policy(settings.approval, approver)

        # Sub-agents: children get a restricted copy of the registry and cannot spawn further agents.
        child_compactor = Compactor(llm, store, settings, tracer, None)
        services = {"episodic": episodic, "longterm": longterm, "skills": skills}

        def build_child(profile: str, session_id: str) -> Agent:
            child_tools = registry.subset(PROFILES[profile])
            return Agent(llm, store, child_tools, policy, settings, tracer, skills=skills, compactor=child_compactor,
                         services=services, persona=CHILD_PERSONA, learn=False, max_steps=settings.subagent_max_steps)

        subagents = SubagentManager(store, build_child, max_active=settings.subagent_max_active)
        subagents.recover()
        register_subagent_tools(registry, subagents)
        return cls(settings, store, llm, embedder, registry, policy, tracer, episodic, longterm, skills,
                   compactor, subagents, mcp, mcp_report)

    async def aclose(self) -> None:
        await self.subagents.aclose()
        if self.mcp is not None:
            await self.mcp.aclose()
        for client in (self.llm, self.embedder):
            closer = getattr(client, "aclose", None)
            if closer is not None:
                await closer()
        self.store.close()

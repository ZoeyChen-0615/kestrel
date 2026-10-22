from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.app import App
from kestrel.config import Settings
from kestrel.llm import HashingEmbedder, ScriptedChat


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "home", approval="auto", context_window=4000, empty_retries=2, max_steps=12)


@pytest.fixture
async def make_app(settings: Settings):
    """Factory: build an App around a scripted model; every app is closed after the test."""
    created: list[App] = []

    async def factory(script, **overrides) -> App:
        for key, value in overrides.items():
            setattr(settings, key, value)
        app = await App.create(settings, llm=ScriptedChat(script), embedder=HashingEmbedder(), use_mcp=False)
        created.append(app)
        return app

    yield factory
    for app in created:
        await app.aclose()

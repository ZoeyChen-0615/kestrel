"""Run the full agent loop with no API key: a scripted model stands in for the LLM.

    python examples/offline_demo.py

It shows tool calls, validation feedback, memory, and the trace output.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from kestrel.app import App
from kestrel.config import Settings
from kestrel.llm import HashingEmbedder, ScriptedChat, text_reply, tool_reply
from kestrel.tracing import format_timeline, summarise


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(home=Path(tmp), approval="auto")
        script = [
            tool_reply("write_file", {"path": "plan.txt", "content": "1. read\n2. summarise\n"}),
            tool_reply("read_file", {"path": "plan.txt", "offset": "not-a-number"}),   # invalid: the model sees the error...
            tool_reply("read_file", {"path": "plan.txt"}),                            # ...and corrects itself
            text_reply("I wrote plan.txt and read it back: a two-step plan."),
        ]
        app = await App.create(settings, llm=ScriptedChat(script), embedder=HashingEmbedder(), use_mcp=False)
        try:
            session = app.store.create_session()
            result = await app.agent.run(
                session, "Write a two-step plan to plan.txt and read it back.",
                on_event=lambda kind, data: print(f"  [{kind}] {data.get('name', '')} {str(data.get('text', data.get('arguments', '')))[:70]!r}")
                if kind in ("tool_call", "tool_result") else None,
            )
            print(f"\nstatus={result.status} steps={result.steps}\nreply: {result.final_text}\n")
            events = app.store.events(session)
            print(format_timeline(events))
            s = summarise(events, session)
            print(f"\nllm calls={s.llm_calls} tool calls={s.tool_calls} tool errors={s.tool_errors}")
        finally:
            await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())

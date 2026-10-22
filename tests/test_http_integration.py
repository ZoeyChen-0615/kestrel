"""The real HTTP client wired into the real agent loop (only the network is faked)."""

import json

import httpx
import pytest

from kestrel.app import App
from kestrel.llm import HashingEmbedder, OpenAICompatChat


def fake_provider(stream: bool):
    """Replies with a tool call first, then with a final answer once a tool result is present."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        done = any(m["role"] == "tool" for m in body["messages"])
        if stream:
            if not done:
                chunks = [
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "list_dir", "arguments": '{"path":'}}]}}]},
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "."}'}}]}, "finish_reason": "tool_calls"}]},
                ]
            else:
                chunks = [{"choices": [{"delta": {"content": "all "}}]}, {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}]
            chunks.append({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
            text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=text)
        message = ({"role": "assistant", "content": "all done"} if done else
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "."}'}}]})
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop" if done else "tool_calls"}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    return handler


@pytest.mark.parametrize("stream", [False, True])
async def test_full_run_over_the_http_client(settings, stream):
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_provider(stream)))
    llm = OpenAICompatChat("http://provider/v1", "key", "fake-model", client=client)
    app = await App.create(settings, llm=llm, embedder=HashingEmbedder(), use_mcp=False)
    try:
        tokens = []
        result = await app.agent.run("s", "list files", on_token=tokens.append if stream else None)
        assert (result.status, result.final_text, result.steps) == ("completed", "all done", 2)
        assert result.usage.total == 20 and result.tool_calls == ["list_dir"]
        assert ("".join(tokens) == "all done") if stream else not tokens
    finally:
        await app.aclose()

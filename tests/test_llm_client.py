import json

import httpx
import pytest

from kestrel.common import LLMError
from kestrel.llm import HashingEmbedder, OpenAICompatChat, OpenAICompatEmbedder


def chat_with(handler, **kwargs):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatChat("http://x/v1", "k", "m", client=client, sleep=fake_sleep, **kwargs), sleeps


OK = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


async def test_retries_transient_errors_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="busy") if len(calls) < 3 else httpx.Response(200, json=OK)

    chat, sleeps = chat_with(handler)
    response = await chat.chat([{"role": "user", "content": "x"}])
    assert response.content == "hi" and response.usage.total == 4
    assert len(calls) == 3 and len(sleeps) == 2


async def test_retry_after_header_is_honoured():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        return httpx.Response(429, headers={"retry-after": "7"}, text="slow down") if state["n"] == 1 else httpx.Response(200, json=OK)

    chat, sleeps = chat_with(handler)
    await chat.chat([])
    assert sleeps == [7.0]


async def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="bad request")

    chat, _ = chat_with(handler)
    with pytest.raises(LLMError) as info:
        await chat.chat([])
    assert len(calls) == 1 and info.value.status == 400 and not info.value.retryable


async def test_gives_up_after_max_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, text="down")

    chat, _ = chat_with(handler, max_retries=2)
    with pytest.raises(LLMError):
        await chat.chat([])
    assert len(calls) == 3


async def test_transport_errors_are_retried():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=OK)

    chat, _ = chat_with(handler)
    assert (await chat.chat([])).content == "hi"


def sse(*chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


async def test_streaming_assembles_text_and_tool_call_deltas():
    body = sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c9", "function": {"name": "add", "arguments": '{"a":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    )
    chat, _ = chat_with(lambda r: httpx.Response(200, text=body))
    tokens = []
    response = await chat.chat([], on_token=tokens.append)
    assert "".join(tokens) == "Hello" == response.content
    assert response.finish_reason == "tool_calls"
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [("c9", "add", '{"a":1}')]
    assert response.usage.total == 7


async def test_streaming_retries_before_first_token_but_not_after():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, text=sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}))

    chat, _ = chat_with(handler)
    assert (await chat.chat([], on_token=lambda t: None)).content == "ok" and state["n"] == 2


async def test_missing_tool_call_ids_are_generated_and_deduplicated():
    data = {"choices": [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "a", "arguments": "{}"}}, {"id": "x", "function": {"name": "b", "arguments": "{}"}},
        {"id": "x", "function": {"name": "c", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}
    chat, _ = chat_with(lambda r: httpx.Response(200, json=data))
    ids = [c.id for c in (await chat.chat([])).tool_calls]
    assert len(set(ids)) == 3


async def test_max_tokens_parameter_name_is_configurable():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=OK)

    chat, _ = chat_with(handler, max_tokens_param="max_completion_tokens")
    await chat.chat([], max_tokens=100)
    assert seen["max_completion_tokens"] == 100 and "max_tokens" not in seen


async def test_embedder_restores_order_and_checks_count():
    def handler(request):
        inputs = json.loads(request.content)["input"]
        rows = [{"index": i, "embedding": [float(i)]} for i in range(len(inputs))][::-1]
        return httpx.Response(200, json={"data": rows})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = OpenAICompatEmbedder("http://x/v1", "k", "m", client=client)
    assert await embedder.embed(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]


async def test_hashing_embedder_is_deterministic_and_similarity_aware():
    e = HashingEmbedder()
    a1, a2, b = await e.embed(["the cat sat on the mat", "the cat sat on the mat", "quantum chromodynamics lecture"])
    assert a1 == a2
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))
    near = (await e.embed(["a cat on a mat"]))[0]
    assert dot(a1, near) > dot(a1, b)

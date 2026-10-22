"""Telegram channel tests against a fake Bot API (no network)."""

import asyncio
import json
import logging

import httpx
import pytest

from kestrel.app import App, ConfigError
from kestrel.channels.telegram import TelegramChannel, TelegramConflict, TelegramError, routed_approver, split_message
from kestrel.common import ChatResponse
from kestrel.llm import HashingEmbedder, ScriptedChat, text_reply, tool_reply

TOKEN = "123456:SECRET-TOKEN-VALUE"


class FakeTelegram:
    """Just enough of the Bot API: queued updates in, recorded calls out."""

    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.answers: list[dict] = []
        self.actions: list[dict] = []
        self.failures: dict[str, list[httpx.Response]] = {}
        self._next = 100
        self._message_id = 0

    def push(self, **fields) -> int:
        self._next += 1
        self.updates.append({"update_id": self._next, **fields})
        return self._next

    def message(self, text="hello", *, user_id=7, username="owner", chat_id=7, chat_type="private", **extra) -> int:
        body = {"message_id": 1, "chat": {"id": chat_id, "type": chat_type}, "from": {"id": user_id, "username": username}, **extra}
        if text is not None:
            body["text"] = text
        return self.push(message=body)

    def callback(self, data, *, user_id=7, username="owner", chat_id=7, query_id="q1") -> int:
        return self.push(callback_query={"id": query_id, "data": data, "from": {"id": user_id, "username": username},
                                         "message": {"chat": {"id": chat_id}, "message_id": 5}})

    def texts(self) -> list[str]:
        return [m["text"] for m in self.sent]

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.startswith(f"/bot{TOKEN}/"), "token must be in the URL path"
        method = request.url.path.rsplit("/", 1)[1]
        payload = json.loads(request.content or b"{}")
        if self.failures.get(method):
            return self.failures[method].pop(0)
        result: object = True
        if method == "getUpdates":
            result = [u for u in self.updates if u["update_id"] >= payload.get("offset", 0)]
        elif method == "getMe":
            result = {"username": "kestrel_test_bot"}
        elif method == "sendMessage":
            self._message_id += 1
            self.sent.append(payload)
            result = {"message_id": self._message_id}
        elif method == "editMessageText":
            self.edits.append(payload)
        elif method == "answerCallbackQuery":
            self.answers.append(payload)
        elif method == "sendChatAction":
            self.actions.append(payload)
        return httpx.Response(200, json={"ok": True, "result": result})


@pytest.fixture
async def tg(settings):
    """Factory returning (fake, channel, app); everything is closed afterwards."""
    made: list[tuple[TelegramChannel, App]] = []

    async def factory(script, *, allowed=("owner",), approval="auto", approval_timeout=120.0, llm=None):
        settings.approval = approval
        fake = FakeTelegram()
        holder: dict = {}
        app = await App.create(settings, llm=llm or ScriptedChat(script), embedder=HashingEmbedder(), use_mcp=False,
                               approver=routed_approver(None, lambda: holder.get("tg")))
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
        channel = TelegramChannel(app, TOKEN, list(allowed), client=client, poll_timeout=0, approval_timeout=approval_timeout,
                                  retry_base=0.01, conflict_wait=0.01, sleep=lambda s: asyncio.sleep(0))   # no real-time waiting in tests
        holder["tg"] = channel
        made.append((channel, app))
        return fake, channel, app

    yield factory
    for channel, app in made:
        await channel.aclose()
        await app.aclose()


async def pump(channel: TelegramChannel) -> None:
    await channel.poll_once()
    await channel.wait_idle()


# ---- basics -------------------------------------------------------------------

async def test_authorised_message_gets_a_reply_in_its_own_session(tg):
    fake, channel, app = await tg([text_reply("Hello from Kestrel")])
    fake.message("hi there")
    await pump(channel)
    assert fake.texts() == ["Hello from Kestrel"] and fake.sent[0]["chat_id"] == 7
    sessions = {s["id"]: s for s in app.store.list_sessions()}
    assert "tg:7" in sessions and sessions["tg:7"]["title"].startswith("Telegram")
    assert [m["role"] for _, m in app.store.load_context("tg:7")[2]] == ["user", "assistant"]


async def test_conversation_continues_in_the_same_session(tg):
    fake, channel, app = await tg([text_reply("one"), text_reply("two")])
    fake.message("first")
    fake.message("second")
    await pump(channel)
    assert fake.texts() == ["one", "two"]
    assert len(app.store.load_context("tg:7")[2]) == 4


async def test_verify_reads_the_bot_username(tg):
    fake, channel, _ = await tg([])
    assert await channel.verify() == "kestrel_test_bot"


# ---- access control -------------------------------------------------------------

async def test_unknown_users_and_groups_get_no_answer_and_no_agent_run(tg, caplog):
    fake, channel, app = await tg([])               # an empty script would raise if the model were ever called
    caplog.set_level(logging.INFO, logger="kestrel.telegram")
    fake.message("hi", user_id=99, username="stranger", chat_id=99)
    fake.message("hi", chat_type="group", chat_id=-100)
    await pump(channel)
    assert fake.sent == [] and app.store.list_sessions() == []
    assert "unauthorised" in caplog.text and "id=99" in caplog.text
    assert "non-private" in caplog.text


@pytest.mark.parametrize("allowed, user_id, username, ok", [
    (["Owner"], 1, "owner", True),          # usernames are case-insensitive
    (["@owner"], 1, "OWNER", True),         # a leading @ is tolerated
    (["424242"], 424242, "", True),         # numeric ids work without a username
    (["owner"], 5, "someone_else", False),
    (["424242"], 5, "424242", False),       # a username that merely looks like an id does not match
])
async def test_allow_list_matching(tg, allowed, user_id, username, ok):
    _, channel, _ = await tg([], allowed=allowed)
    assert channel.is_allowed({"id": user_id, "username": username}) is ok


async def test_an_empty_allow_list_refuses_to_start(settings):
    app = await App.create(settings, llm=ScriptedChat([]), embedder=HashingEmbedder(), use_mcp=False)
    try:
        with pytest.raises(ConfigError, match="ALLOWED"):
            TelegramChannel(app, TOKEN, [])
        with pytest.raises(ConfigError, match="TOKEN"):
            TelegramChannel(app, "", ["owner"])
    finally:
        await app.aclose()


# ---- commands & message kinds -----------------------------------------------------

async def test_new_starts_a_separate_conversation(tg):
    fake, channel, app = await tg([text_reply("a1"), text_reply("b1")])
    fake.message("first topic")
    await pump(channel)
    fake.message("/new")
    fake.message("second topic")
    await pump(channel)
    assert "Started a new conversation" in fake.texts()[1] and fake.texts()[2] == "b1"
    ids = {s["id"] for s in app.store.list_sessions()}
    assert "tg:7" in ids and len(ids) == 2
    new_id = next(i for i in ids if i != "tg:7")
    assert len(app.store.load_context("tg:7")[2]) == 2 and len(app.store.load_context(new_id)[2]) == 2


async def test_help_unknown_command_and_non_text(tg):
    fake, channel, _ = await tg([])
    fake.message("/start")
    fake.message("/help@kestrel_test_bot")
    fake.message("/frobnicate")
    fake.message(None, photo=[{"file_id": "x"}])
    await pump(channel)
    assert "/new" in fake.texts()[0] and "/new" in fake.texts()[1]
    assert fake.texts()[2].startswith("Unknown command") and "text messages" in fake.texts()[3]


async def test_long_replies_are_split_under_telegrams_limit(tg):
    long_reply = "\n\n".join(f"paragraph {i} " + "x" * 600 for i in range(20))
    fake, channel, _ = await tg([text_reply(long_reply)])
    fake.message("write a lot")
    await pump(channel)
    assert len(fake.sent) > 1 and all(len(t) <= 4000 for t in fake.texts())
    assert "".join(fake.texts()).replace("\n", "").replace(" ", "") == long_reply.replace("\n", "").replace(" ", "")


def test_split_message_edge_cases():
    assert split_message("short") == ["short"]
    assert split_message("   ") == ["(empty reply)"]
    chunks = split_message("a" * 9000)
    assert [len(c) for c in chunks] == [4000, 4000, 1000]


async def test_failed_runs_are_reported_to_the_user(tg):
    from kestrel.common import LLMError

    def boom(messages):
        raise LLMError("provider down")

    fake, channel, _ = await tg([boom])
    fake.message("hi")
    await pump(channel)
    assert "couldn't finish" in fake.texts()[0] and "provider down" in fake.texts()[0]


async def test_stop_cancels_a_running_turn(tg):
    class Hang:
        model = "hang"

        async def chat(self, messages, tools=None, **kwargs):
            await asyncio.sleep(30)

    fake, channel, _ = await tg([], llm=Hang())
    fake.message("do something slow")
    await channel.poll_once()
    for _ in range(500):
        if channel._running:
            break
        await asyncio.sleep(0.02)
    assert channel._running
    fake.message("/stop")
    await pump(channel)
    assert fake.texts() == ["Stopped."] and not channel._running
    fake.message("/stop")
    await pump(channel)
    assert fake.texts()[-1] == "Nothing is running."


async def test_typing_indicator_is_sent_while_working(tg):
    class Slow:                      # a real model takes seconds; a scripted one returns instantly
        model = "slow"

        async def chat(self, messages, tools=None, **kwargs):
            await asyncio.sleep(0.2)
            return text_reply("done")

    fake, channel, _ = await tg([], llm=Slow())
    fake.message("hi")
    await pump(channel)
    assert fake.actions and fake.actions[0]["action"] == "typing"


# ---- approvals via inline buttons ---------------------------------------------------

async def wait_for(predicate, timeout=15.0):
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached in time")


async def test_risky_tool_asks_in_the_chat_and_runs_after_allow(tg, settings):
    fake, channel, _ = await tg([tool_reply("write_file", {"path": "a.txt", "content": "v1"}), text_reply("written")], approval="ask")
    fake.message("create a.txt")
    await channel.poll_once()
    await wait_for(lambda: any("reply_markup" in m for m in fake.sent))
    prompt = next(m for m in fake.sent if "reply_markup" in m)
    assert "write_file" in prompt["text"] and "a.txt" in prompt["text"] and prompt["chat_id"] == 7
    buttons = prompt["reply_markup"]["inline_keyboard"][0]
    assert [b["text"] for b in buttons] == ["Allow", "Deny"]
    assert not (settings.workspace / "a.txt").exists()                       # nothing happens before the answer
    fake.callback(buttons[0]["callback_data"])
    await pump(channel)
    assert (settings.workspace / "a.txt").read_text() == "v1" and fake.texts()[-1] == "written"
    assert fake.answers[0]["text"] == "Allowed" and fake.edits[0]["text"] == "Allowed."


async def test_deny_button_blocks_the_action(tg, settings):
    fake, channel, app = await tg([tool_reply("write_file", {"path": "b.txt", "content": "x"}), text_reply("understood")], approval="ask")
    fake.message("create b.txt")
    await channel.poll_once()
    await wait_for(lambda: any("reply_markup" in m for m in fake.sent))
    deny = next(m for m in fake.sent if "reply_markup" in m)["reply_markup"]["inline_keyboard"][0][1]["callback_data"]
    fake.callback(deny)
    await pump(channel)
    assert not (settings.workspace / "b.txt").exists()
    result = [m for _, m in app.store.load_context("tg:7")[2] if m["role"] == "tool"][0]["content"]
    assert result.startswith("Denied") and fake.answers[0]["text"] == "Denied"


async def test_unanswered_approval_expires_and_denies(tg, settings):
    fake, channel, _ = await tg([tool_reply("write_file", {"path": "c.txt", "content": "x"}), text_reply("ok")], approval="ask", approval_timeout=0.2)
    fake.message("create c.txt")
    await pump(channel)
    assert not (settings.workspace / "c.txt").exists()
    assert any("expired" in e["text"] for e in fake.edits)


async def test_approval_answers_from_other_people_or_chats_are_ignored(tg, settings):
    fake, channel, _ = await tg([tool_reply("write_file", {"path": "d.txt", "content": "x"}), text_reply("ok")], approval="ask", approval_timeout=0.6)
    fake.message("create d.txt")
    await channel.poll_once()
    await wait_for(lambda: any("reply_markup" in m for m in fake.sent))
    allow = next(m for m in fake.sent if "reply_markup" in m)["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    fake.callback(allow, user_id=99, username="stranger", query_id="q-stranger")      # not on the allow-list
    fake.callback(allow, chat_id=555, query_id="q-elsewhere")                         # allowed user, wrong chat
    await channel.poll_once()
    await channel.wait_idle()                                                         # times out -> denied
    assert not (settings.workspace / "d.txt").exists()


async def test_stale_or_malformed_callbacks_are_harmless(tg):
    fake, channel, _ = await tg([])
    fake.callback("ap:deadbeef00:1", query_id="old")
    fake.callback("garbage", query_id="g1")
    fake.callback("ap:too:many:parts", query_id="g2")
    await pump(channel)
    assert [a["callback_query_id"] for a in fake.answers] == ["old"] and fake.answers[0]["text"].startswith("This request")


async def test_routed_approver_falls_back_to_web_or_denies(tg):
    fake, channel, app = await tg([])
    tool = app.registry.get("write_file")
    seen = []

    async def web(t, a):
        seen.append(t.name)
        return True

    assert await routed_approver(web, lambda: channel)(tool, {}) is True and seen == ["write_file"]   # no Telegram chat in scope
    assert await routed_approver(None, lambda: channel)(tool, {}) is False


# ---- resilience -----------------------------------------------------------------------

async def test_offset_is_persisted_so_a_restart_never_replays_messages(tg, settings):
    fake, channel, app = await tg([text_reply("once")])
    fake.message("hi")
    await pump(channel)
    assert len(fake.sent) == 1
    # a "restarted" channel on the same database sees nothing new, although Telegram would still hold the update
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    reborn = TelegramChannel(app, TOKEN, ["owner"], client=client, poll_timeout=0)
    assert await reborn.poll_once() == 0 and len(fake.sent) == 1


async def test_transient_api_errors_are_retried_with_retry_after(tg):
    fake, channel, _ = await tg([text_reply("hello")])
    slept = []
    channel._sleep = lambda s: slept.append(s) or asyncio.sleep(0)
    fake.failures["sendMessage"] = [
        httpx.Response(429, json={"ok": False, "error_code": 429, "description": "slow down", "parameters": {"retry_after": 3}}),
        httpx.Response(502, text="bad gateway"),
    ]
    fake.message("hi")
    await pump(channel)
    assert fake.texts() == ["hello"] and slept[0] == 3


async def test_client_errors_are_not_retried_and_do_not_kill_the_channel(tg):
    fake, channel, _ = await tg([text_reply("a"), text_reply("b")])
    fake.failures["sendMessage"] = [httpx.Response(400, json={"ok": False, "error_code": 400, "description": "chat not found"})]
    fake.message("one")
    await pump(channel)
    fake.message("two")
    await pump(channel)
    assert fake.texts()[-1] == "b"


async def test_conflicting_poller_is_reported_clearly(tg):
    fake, channel, _ = await tg([])
    fake.failures["getUpdates"] = [httpx.Response(409, json={"ok": False, "error_code": 409, "description": "Conflict: terminated by other getUpdates request"})]
    with pytest.raises(TelegramConflict, match="another process"):
        await channel.poll_once()


async def test_the_polling_loop_survives_errors_and_keeps_serving(tg):
    fake, channel, _ = await tg([text_reply("still here")])
    fake.failures["getUpdates"] = [httpx.Response(500, text="oops"), httpx.Response(503, text="oops")]
    fake.message("hi")
    loop = asyncio.create_task(channel.run())
    await wait_for(lambda: fake.texts() == ["still here"])
    await channel.aclose()
    await asyncio.wait_for(loop, 15)


async def test_the_bot_token_never_leaks_into_errors_or_logs(tg, caplog):
    fake, channel, _ = await tg([])
    caplog.set_level(logging.DEBUG)
    fake.failures["sendMessage"] = [httpx.Response(400, json={"ok": False, "error_code": 400, "description": f"bad url /bot{TOKEN}/sendMessage"})]
    with pytest.raises(TelegramError) as info:
        await channel.send(7, "x")
    assert TOKEN not in str(info.value) and "***" in str(info.value)
    fake.failures["getUpdates"] = [httpx.Response(500, json={"ok": False, "error_code": 500, "description": f"oops {TOKEN}"})] * 2
    task = asyncio.create_task(channel.run())
    await asyncio.sleep(0.1)
    await channel.aclose()
    await asyncio.wait_for(task, 15)
    assert TOKEN not in caplog.text

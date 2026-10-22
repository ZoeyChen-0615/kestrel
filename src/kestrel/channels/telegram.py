"""Telegram channel: talk to the agent from a Telegram private chat.

Design notes
* Transport is the plain Bot API over httpx with long polling; no extra dependency.
* One chat = one session (``tg:<chat_id>``), stored in the same database as web sessions, so conversations are
  visible in the web UI and share long-term and episodic memory. ``/new`` starts a fresh session.
* Access control is deny-by-default: only users on the allow-list (numeric id or username) are served, only in
  private chats. Anyone else is ignored silently (and logged, so the owner can add them).
* When the policy needs a human decision for a risky tool, the bot asks in the chat with Allow / Deny buttons.
* The next update offset is persisted before an update is handled, so a restart never replays old messages
  (at-most-once handling; an interrupted run is recovered by the agent's normal crash recovery).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextvars import ContextVar
from typing import Any

import httpx

from ..app import App, ConfigError
from ..common import ConflictError, KestrelError
from ..tools import Tool

log = logging.getLogger("kestrel.telegram")


class _RedactToken(logging.Filter):
    """Replace the bot token in any log record that passes through this logger."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self._token in message:
            record.msg, record.args = message.replace(self._token, "***"), ()
        return True


def _protect_token_in_logs(token: str) -> None:
    """The Bot API puts the token in the URL path, and httpx logs request URLs at INFO.

    Keep the HTTP client libraries quiet below WARNING (so URLs are never logged) and redact as a second line of defence.
    """
    for name in ("httpx", "httpcore"):
        logger = logging.getLogger(name)
        if logger.getEffectiveLevel() < logging.WARNING:
            logger.setLevel(logging.WARNING)
        logger.addFilter(_RedactToken(token))

API = "https://api.telegram.org"
MAX_MESSAGE = 4000          # Telegram's hard limit is 4096; keep headroom
NAMESPACE = "telegram"

# The chat a run is serving. Tasks inherit it, so even a sub-agent that outlives the run can still ask this chat.
current_chat: ContextVar[int | None] = ContextVar("kestrel_telegram_chat", default=None)

HELP = ("I'm Kestrel. Send me a message and I'll work on it.\n\n"
        "/new  start a fresh conversation\n/stop  cancel what I'm doing\n/help  show this message")


class TelegramError(KestrelError):
    def __init__(self, message: str, *, code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


class TelegramConflict(TelegramError):
    """Another process is polling the same bot."""


def split_message(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """Split on paragraph, then line, then hard boundaries so every chunk fits Telegram's size limit."""
    text = text.strip() or "(empty reply)"
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n")
    chunks.append(text)
    return [c for c in chunks if c]


def parse_allowed(raw: list[str]) -> tuple[set[int], set[str]]:
    ids: set[int] = set()
    names: set[str] = set()
    for item in raw:
        item = item.strip().lstrip("@")
        if not item:
            continue
        if item.isdigit():
            ids.add(int(item))
        else:
            names.add(item.lower())
    return ids, names


class TelegramChannel:
    def __init__(
        self,
        app: App,
        token: str,
        allowed: list[str],
        *,
        base_url: str = API,
        client: httpx.AsyncClient | None = None,
        poll_timeout: int = 30,
        approval_timeout: float = 120.0,
        sleep=asyncio.sleep,
    ) -> None:
        if not token:
            raise ConfigError("KESTREL_TELEGRAM_TOKEN is not set")
        self._ids, self._names = parse_allowed(allowed)
        if not self._ids and not self._names:
            raise ConfigError("KESTREL_TELEGRAM_ALLOWED is empty. List the Telegram usernames or numeric ids that may use the bot; "
                              "an empty list would let anyone run commands as you.")
        self.app = app
        self._token = token
        _protect_token_in_logs(token)
        self._base = base_url.rstrip("/")
        self._poll_timeout = poll_timeout
        self._client = client or httpx.AsyncClient(timeout=poll_timeout + 15)
        self._owns_client = client is None
        self._approval_timeout = approval_timeout
        self._sleep = sleep
        self._stop = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._running: dict[int, asyncio.Task[None]] = {}
        self._pending: dict[str, tuple[int, asyncio.Future[bool], int | None]] = {}
        self.bot_username = ""

    # ------------------------------------------------------------------ Bot API

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "***")

    async def _call(self, method: str, payload: dict[str, Any] | None = None, *, retries: int = 3) -> Any:
        """Call a Bot API method; retry transient failures; never let the token reach an error message."""
        url = f"{self._base}/bot{self._token}/{method}"
        for attempt in range(retries + 1):
            try:
                response = await self._client.post(url, json=payload or {})
                body = response.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                error = TelegramError(f"transport error: {type(exc).__name__}")
            except ValueError:
                error = TelegramError(f"invalid JSON from Telegram (HTTP {response.status_code})", code=response.status_code)
            else:
                if body.get("ok"):
                    return body.get("result")
                code = int(body.get("error_code") or response.status_code)
                description = self._scrub(str(body.get("description", "")))
                params = body.get("parameters") or {}
                if code == 409:
                    raise TelegramConflict(f"Telegram says another process is polling this bot ({description})", code=409)
                error = TelegramError(f"{method} failed: {code} {description}", code=code, retry_after=params.get("retry_after"))
                if code not in (429, 500, 502, 503, 504):
                    raise error
            if attempt == retries:
                raise error
            await self._sleep(error.retry_after or min(2 ** attempt, 10))
        raise AssertionError("unreachable")  # pragma: no cover

    async def verify(self) -> str:
        """Check the token and remember the bot's username."""
        me = await self._call("getMe", retries=1)
        self.bot_username = me.get("username", "")
        return self.bot_username

    async def send(self, chat_id: int, text: str, **extra: Any) -> None:
        for chunk in split_message(text):
            await self._call("sendMessage", {"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True, **extra})

    # --------------------------------------------------------------- polling loop

    def _offset(self) -> int:
        found = self.app.store.kv_get(NAMESPACE, "offset")
        return found[1]["next"] if found else 0

    def _commit_offset(self, update_id: int) -> None:
        for _ in range(3):
            found = self.app.store.kv_get(NAMESPACE, "offset")
            try:
                self.app.store.kv_put(NAMESPACE, "offset", {"next": update_id + 1}, expected_version=found[0] if found else None)
                return
            except ConflictError:
                continue

    async def poll_once(self) -> int:
        """Fetch one batch of updates and dispatch them. Returns how many updates were received."""
        updates = await self._call("getUpdates", {"offset": self._offset(), "timeout": self._poll_timeout,
                                                  "allowed_updates": ["message", "callback_query"]}, retries=0)
        for update in updates:
            self._commit_offset(update["update_id"])   # persisted before handling: restarts never replay it
            if "callback_query" in update:
                await self._on_callback(update["callback_query"])
            elif "message" in update:
                self._spawn(self._on_message(update["message"]))
        return len(updates)

    def _spawn(self, coro) -> None:
        # Messages run as tasks so polling keeps going while a run waits for an approval button press.
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self) -> None:
        """Poll until :meth:`aclose`. Errors back off and retry; the loop itself never dies."""
        delay = 1.0
        while not self._stop.is_set():
            try:
                received = await self.poll_once()
                delay = 1.0
                if received == 0 and self._poll_timeout == 0:
                    await self._interruptible_sleep(0.2)   # short polling must not spin the CPU
            except asyncio.CancelledError:
                raise
            except TelegramConflict as exc:
                log.error("%s. Stop the other process or use a different bot. Retrying in 15s.", exc)
                await self._interruptible_sleep(15)
            except Exception as exc:
                log.warning("polling error: %s; retrying in %.0fs", self._scrub(str(exc)), delay)
                await self._interruptible_sleep(delay)
                delay = min(delay * 2, 60)

    async def _interruptible_sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def wait_idle(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self) -> None:
        self._stop.set()
        for future in [p[1] for p in self._pending.values()]:
            if not future.done():
                future.set_result(False)
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------- access

    def is_allowed(self, user: dict[str, Any] | None) -> bool:
        if not user:
            return False
        return user.get("id") in self._ids or str(user.get("username", "")).lower() in self._names

    # ----------------------------------------------------------------- sessions

    def session_for(self, chat_id: int, *, fresh: bool = False, username: str = "") -> str:
        key = f"chat:{chat_id}"
        found = self.app.store.kv_get(NAMESPACE, key)
        if found and not fresh:
            return found[1]["session"]
        session = f"tg:{chat_id}" if not found else f"tg:{chat_id}:{uuid.uuid4().hex[:6]}"
        self.app.store.create_session(session_id=session, title=f"Telegram {('@' + username) if username else chat_id}")
        self.app.store.kv_put(NAMESPACE, key, {"session": session}, expected_version=found[0] if found else None)
        return session

    # ----------------------------------------------------------------- messages

    async def _on_message(self, message: dict[str, Any]) -> None:
        chat, user = message.get("chat", {}), message.get("from")
        chat_id = chat.get("id")
        if chat.get("type") != "private":
            log.info("ignored a message from a non-private chat %s", chat_id)
            return
        if not self.is_allowed(user):
            log.warning("ignored unauthorised Telegram user id=%s username=%s (add to KESTREL_TELEGRAM_ALLOWED to allow)",
                        (user or {}).get("id"), (user or {}).get("username"))
            return
        text = message.get("text")
        username = (user or {}).get("username", "")
        try:
            if text is None:
                await self.send(chat_id, "I can only read text messages for now.")
            elif text.startswith("/"):
                await self._command(chat_id, text, username)
            else:
                await self._converse(chat_id, text, username)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("failed to handle a message: %s", self._scrub(f"{type(exc).__name__}: {exc}"))
            try:
                await self.send(chat_id, "Something went wrong on my side; please try again.")
            except Exception:
                pass

    async def _command(self, chat_id: int, text: str, username: str) -> None:
        command = text.split()[0].split("@")[0].lower()
        if command in ("/start", "/help"):
            await self.send(chat_id, HELP)
        elif command == "/new":
            self.session_for(chat_id, fresh=True, username=username)
            await self.send(chat_id, "Started a new conversation. Long-term memory carries over.")
        elif command == "/stop":
            task = self._running.get(chat_id)
            if task and not task.done():
                task.cancel()
                await self.send(chat_id, "Stopped.")
            else:
                await self.send(chat_id, "Nothing is running.")
        else:
            await self.send(chat_id, "Unknown command.\n\n" + HELP)

    async def _typing(self, chat_id: int) -> None:
        while True:
            try:
                await self._call("sendChatAction", {"chat_id": chat_id, "action": "typing"}, retries=0)
            except Exception:
                pass
            await asyncio.sleep(4)

    async def _converse(self, chat_id: int, text: str, username: str) -> None:
        session = self.session_for(chat_id, username=username)
        token = current_chat.set(chat_id)
        typing = asyncio.create_task(self._typing(chat_id))
        self._running[chat_id] = asyncio.current_task()  # type: ignore[assignment]
        try:
            result = await self.app.agent.run(session, text)
        finally:
            typing.cancel()
            self._running.pop(chat_id, None)
            current_chat.reset(token)
        if result.status == "completed" or result.final_text.strip():
            await self.send(chat_id, result.final_text)
        if result.status == "failed":
            await self.send(chat_id, f"I couldn't finish that: {self._scrub(result.error or 'unknown error')}")

    # ---------------------------------------------------------------- approvals

    async def request_approval(self, tool: Tool, args: dict[str, Any]) -> bool:
        """Policy approver: ask in the chat that triggered the run. Denies if there is no chat or nobody answers."""
        chat_id = current_chat.get()
        if chat_id is None:
            return False
        approval_id = uuid.uuid4().hex[:10]
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        shown = json.dumps(args, ensure_ascii=False, indent=1)
        shown = shown if len(shown) <= 900 else shown[:900] + "\n..."
        buttons = [[{"text": "Allow", "callback_data": f"ap:{approval_id}:1"}, {"text": "Deny", "callback_data": f"ap:{approval_id}:0"}]]
        sent = await self._call("sendMessage", {"chat_id": chat_id, "text": f"Approval needed: {tool.name} ({tool.risk})\n{shown}",
                                                "reply_markup": {"inline_keyboard": buttons}})
        self._pending[approval_id] = (chat_id, future, (sent or {}).get("message_id"))
        try:
            return await asyncio.wait_for(future, timeout=self._approval_timeout)
        except asyncio.TimeoutError:
            await self._finish_prompt(chat_id, (sent or {}).get("message_id"), f"Approval for {tool.name} expired (denied).")
            return False
        finally:
            self._pending.pop(approval_id, None)

    async def _on_callback(self, query: dict[str, Any]) -> None:
        data = str(query.get("data", ""))
        user = query.get("from")
        try:
            if not data.startswith("ap:") or data.count(":") != 2:
                return
            _, approval_id, flag = data.split(":")
            entry = self._pending.get(approval_id)
            if entry is None:
                await self._call("answerCallbackQuery", {"callback_query_id": query["id"], "text": "This request is no longer active."}, retries=0)
                return
            chat_id, future, message_id = entry
            # Only an allowed user, in the chat that asked, may decide.
            if not self.is_allowed(user) or (query.get("message") or {}).get("chat", {}).get("id") != chat_id:
                log.warning("ignored an approval answer from an unauthorised source")
                return
            allow = flag == "1"
            if not future.done():
                future.set_result(allow)
            await self._call("answerCallbackQuery", {"callback_query_id": query["id"], "text": "Allowed" if allow else "Denied"}, retries=0)
            await self._finish_prompt(chat_id, message_id, "Allowed." if allow else "Denied.")
        except Exception as exc:
            log.warning("callback handling failed: %s", self._scrub(str(exc)))

    async def _finish_prompt(self, chat_id: int, message_id: int | None, text: str) -> None:
        if message_id is None:
            return
        try:
            await self._call("editMessageText", {"chat_id": chat_id, "message_id": message_id, "text": text}, retries=0)
        except Exception:
            pass


def routed_approver(web_request, get_channel):
    """One policy approver for every front end: a run started from Telegram asks in that chat,
    anything else asks through ``web_request`` (or is denied when that is None)."""

    async def approve(tool: Tool, args: dict[str, Any]) -> bool:
        channel = get_channel()
        if channel is not None and current_chat.get() is not None:
            return await channel.request_approval(tool, args)
        return await web_request(tool, args) if web_request is not None else False

    return approve

"""Command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import __version__
from .app import App, ConfigError
from .common import KestrelError, RunResult
from .config import Settings
from .evals import load_suite, run_suite
from .store import Store
from .tools import Tool
from .tracing import format_timeline, summarise

DIM, RESET = "\033[2m", "\033[0m"


async def terminal_approver(tool: Tool, args: dict[str, Any]) -> bool:
    """Ask on the terminal before a risky tool runs."""
    shown = json.dumps(args, ensure_ascii=False)
    shown = shown if len(shown) < 300 else shown[:300] + "..."
    answer = await asyncio.to_thread(input, f"\n{DIM}Allow {tool.name} ({tool.risk}) with {shown}? [y/N] {RESET}")
    return answer.strip().lower() in {"y", "yes"}


def print_event(kind: str, data: dict[str, Any]) -> None:
    if kind == "tool_call":
        print(f"\n{DIM}-> {data['name']}({data['arguments'][:120]}){RESET}", file=sys.stderr)
    elif kind == "tool_result":
        flag = "error" if data["is_error"] else "ok"
        print(f"{DIM}<- {data['name']} [{flag}] {data['text'][:100].replace(chr(10), ' ')}{RESET}", file=sys.stderr)
    elif kind == "compaction":
        print(f"{DIM}(context compacted){RESET}", file=sys.stderr)
    elif kind == "error":
        print(f"\n[error] {data['message']}", file=sys.stderr)


def report(result: RunResult, *, streamed: bool) -> None:
    if not streamed and result.final_text:
        print(result.final_text)
    elif streamed:
        print()
    if result.status != "completed":
        print(f"[{result.status}] {result.error or ''}".rstrip(), file=sys.stderr)
    print(f"{DIM}steps={result.steps} tokens={result.usage.total}{RESET}", file=sys.stderr)


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "approval", None):
        settings.approval = args.approval
    if getattr(args, "mcp", None):
        settings.mcp_config = Path(args.mcp)
    if getattr(args, "model", None):
        settings.model = args.model
    return settings


async def cmd_chat(args: argparse.Namespace) -> int:
    app = await App.create(settings_from_args(args), approver=terminal_approver)
    session = args.session or app.store.create_session()
    stream = not args.no_stream
    try:
        for name, status in app.mcp_report.items():
            print(f"{DIM}mcp {name}: {status}{RESET}", file=sys.stderr)
        print(f"Kestrel {__version__} | model {app.llm.model} | session {session} | approval {app.settings.approval} | /help for commands")
        while True:
            try:
                line = (await asyncio.to_thread(input, "\nyou> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                return 0
            if line == "/help":
                print("/new  start a new session   /trace  show this session's trace   /exit  quit")
                continue
            if line == "/new":
                session = app.store.create_session()
                print(f"new session {session}")
                continue
            if line == "/trace":
                print(format_timeline(app.store.events(session)))
                continue
            print("kestrel> ", end="", flush=True) if stream else None
            result = await app.agent.run(session, line, on_token=(lambda t: print(t, end="", flush=True)) if stream else None, on_event=print_event)
            report(result, streamed=stream)
    finally:
        await app.aclose()


async def cmd_web(args: argparse.Namespace) -> int:
    import webbrowser

    import uvicorn

    from .web.runs import Approvals
    from .web.server import create_web_app

    from .channels.telegram import TelegramChannel, routed_approver

    settings = settings_from_args(args)
    approvals = Approvals()
    holder: dict[str, TelegramChannel] = {}
    app = await App.create(settings, approver=routed_approver(approvals.request, lambda: holder.get("tg")))
    telegram_task: asyncio.Task[None] | None = None
    if settings.telegram_token and not args.no_telegram:
        try:
            channel = TelegramChannel(app, settings.telegram_token, settings.telegram_allowed)
            username = await channel.verify()
            holder["tg"] = channel
            telegram_task = asyncio.create_task(channel.run(), name="telegram")
            print(f"Telegram: connected as @{username}; {len(settings.telegram_allowed)} allowed user(s)")
        except KestrelError as exc:
            print(f"Telegram disabled: {exc}", file=sys.stderr)
    token = args.token or os.environ.get("KESTREL_WEB_TOKEN") or secrets.token_urlsafe(24)
    names = {"127.0.0.1", "localhost", args.host, *args.allow_host}
    allowed = {f"{name}:{args.port}" if ":" not in name else name for name in names}
    web = create_web_app(app, approvals, token, allowed_hosts=allowed)
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown_host}:{args.port}/#token={token}"
    print(f"Kestrel web UI  (model {app.llm.model}, approval {app.settings.approval})")
    print(f"Open: {url}")
    print("The token after '#' is your access key; keep it private. Stop with Ctrl+C.")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: listening on a non-loopback address. Anyone who gets the token can run commands as you.", file=sys.stderr)
    if not args.no_open:
        webbrowser.open(url)
    server = uvicorn.Server(uvicorn.Config(web, host=args.host, port=args.port, log_level="warning"))
    try:
        await server.serve()
    finally:
        if "tg" in holder:
            await holder["tg"].aclose()
        if telegram_task is not None:
            telegram_task.cancel()
            await asyncio.gather(telegram_task, return_exceptions=True)
        await app.aclose()
    return 0


async def cmd_telegram(args: argparse.Namespace) -> int:
    from .channels.telegram import TelegramChannel, routed_approver

    settings = settings_from_args(args)
    if not settings.telegram_token:
        raise ConfigError("KESTREL_TELEGRAM_TOKEN is not set")
    holder: dict[str, TelegramChannel] = {}
    app = await App.create(settings, approver=routed_approver(None, lambda: holder.get("tg")))
    try:
        channel = TelegramChannel(app, settings.telegram_token, settings.telegram_allowed)
        holder["tg"] = channel
        username = await channel.verify()
        print(f"Telegram bot @{username} is running ({len(settings.telegram_allowed)} allowed user(s), approval {app.settings.approval}). Ctrl+C to stop.")
        await channel.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if "tg" in holder:
            await holder["tg"].aclose()
        await app.aclose()
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    app = await App.create(settings_from_args(args), approver=terminal_approver)
    try:
        session = args.session or app.store.create_session()
        result = await app.agent.run(session, args.prompt, on_event=print_event)
        report(result, streamed=False)
        return 0 if result.status == "completed" else 1
    finally:
        await app.aclose()


def cmd_sessions(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = Store(settings.db_path)
    for row in store.list_sessions(kind=None if args.all else "chat"):
        print(f"{row['id']:<14} {row['kind']:<9} {row['messages']:>4} msgs  {row['title'] or ''}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = Store(settings.db_path)
    session = args.session
    if session is None:
        sessions = store.list_sessions(limit=1, kind="chat")  # sub-agent sessions are traced by explicit id
        if not sessions:
            print("no sessions yet")
            return 1
        session = sessions[0]["id"]
    events = store.events(session)
    print(f"session {session}\n{format_timeline(events)}")
    summary = summarise(events, session)
    cost = summary.cost(settings.price_in_per_mtok, settings.price_out_per_mtok)
    print(f"\nllm calls {summary.llm_calls} ({summary.llm_ms / 1000:.1f}s) | tool calls {summary.tool_calls} "
          f"({summary.tool_errors} errors) | tokens in {summary.usage.prompt_tokens} out {summary.usage.completion_tokens}"
          + (f" | est. cost ${cost:.4f}" if cost else ""))
    if summary.by_tool:
        print("tools: " + ", ".join(f"{n} x{c}" for n, c in summary.by_tool.most_common()))
    if summary.recoveries:
        print("recoveries: " + ", ".join(f"{n} x{c}" for n, c in summary.recoveries.items()))
    return 0


async def cmd_eval(args: argparse.Namespace) -> int:
    base = settings_from_args(args)
    base.approval = "auto"  # evals must not block on a prompt

    async def make_app(tmp: Path) -> App:
        return await App.create(replace(base, home=tmp, workspace=None), use_mcp=False)

    suite = await run_suite(load_suite(Path(args.suite)), make_app)
    print(suite.format())
    if args.json:
        Path(args.json).write_text(json.dumps(suite.to_dict(), indent=2), encoding="utf-8")
    return 0 if suite.pass_rate == 1.0 else 1


def cmd_info(args: argparse.Namespace) -> int:
    s = Settings.from_env()
    print(f"home        {s.home}\nworkspace   {s.workspace}\nmodel       {s.model} @ {s.base_url} (key {'set' if s.api_key else 'MISSING'})")
    print(f"embeddings  {s.embed_model if s.embed_api_key else 'offline hashing fallback'}")
    print(f"approval    {s.approval}\nmcp config  {s.mcp_config or '(none)'}")
    if s.telegram_token:
        print(f"telegram    token set; {len(s.telegram_allowed)} allowed user(s)" + ("" if s.telegram_allowed else "  <- EMPTY: the channel will refuse to start"))
    else:
        print("telegram    not configured")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kestrel", description="A compact tool-using LLM agent.")
    parser.add_argument("--version", action="version", version=f"kestrel {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--approval", choices=["ask", "auto", "readonly"], help="tool approval mode (default from KESTREL_APPROVAL, else ask)")
        p.add_argument("--mcp", help="path to an MCP servers JSON file")
        p.add_argument("--model", help="override the chat model")

    chat = sub.add_parser("chat", help="interactive chat")
    common(chat)
    chat.add_argument("--session", help="resume this session id")
    chat.add_argument("--no-stream", action="store_true", help="disable token streaming")
    web = sub.add_parser("web", help="start the local web interface")
    common(web)
    web.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1; anything else exposes shell access)")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--token", help="fixed access token (default: random per start, or KESTREL_WEB_TOKEN)")
    web.add_argument("--allow-host", action="append", default=[], help="extra Host header name to accept (repeatable)")
    web.add_argument("--no-open", action="store_true", help="do not open a browser window")
    web.add_argument("--no-telegram", action="store_true", help="do not start the Telegram channel even if a token is configured")
    tg = sub.add_parser("telegram", help="run only the Telegram channel (no web UI)")
    common(tg)
    run = sub.add_parser("run", help="run one prompt and exit")
    common(run)
    run.add_argument("prompt")
    run.add_argument("--session")
    sessions = sub.add_parser("sessions", help="list sessions")
    sessions.add_argument("--all", action="store_true", help="include sub-agent sessions")
    trace = sub.add_parser("trace", help="show the trace of a session (default: latest)")
    trace.add_argument("session", nargs="?")
    ev = sub.add_parser("eval", help="run an evaluation suite")
    common(ev)
    ev.add_argument("suite")
    ev.add_argument("--json", help="write a JSON report to this path")
    sub.add_parser("info", help="show effective configuration")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "chat":
            return asyncio.run(cmd_chat(args))
        if args.command == "run":
            return asyncio.run(cmd_run(args))
        if args.command == "eval":
            return asyncio.run(cmd_eval(args))
        if args.command == "web":
            return asyncio.run(cmd_web(args))
        if args.command == "telegram":
            return asyncio.run(cmd_telegram(args))
        return {"sessions": cmd_sessions, "trace": cmd_trace, "info": cmd_info}[args.command](args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KestrelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

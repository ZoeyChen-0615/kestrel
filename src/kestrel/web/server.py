"""The web server: a small JSON API plus the static single-page UI.

Because the agent can run shell commands and write files, the server treats every
request as potentially hostile:

* every ``/api`` call needs the access token (sent in a header, never in a URL);
* the ``Host`` header must be one of the expected local names (blocks DNS rebinding);
* state-changing requests carrying an ``Origin`` header must come from the same origin (blocks CSRF);
* a strict Content-Security-Policy keeps the page from loading anything else.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..app import App
from ..subagents import NAMESPACE
from ..tracing import summarise
from .runs import Approvals, RunBusy, RunManager

STATIC_DIR = Path(__file__).parent / "static"
TOKEN_HEADER = "x-kestrel-token"
MAX_PERSONA_CHARS = 20_000
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, *, token: str, allowed_hosts: set[str]) -> None:
        super().__init__(app)
        self._token = token
        self._hosts = {h.lower() for h in allowed_hosts}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.headers.get("host", "").lower() not in self._hosts:
            return JSONResponse({"error": "host not allowed"}, status_code=403)
        is_api = request.url.path.startswith("/api/")
        if is_api:
            supplied = request.headers.get(TOKEN_HEADER, "")
            if not secrets.compare_digest(supplied.encode(), self._token.encode()):
                return JSONResponse({"error": "missing or invalid access token"}, status_code=401)
            origin = request.headers.get("origin")
            if request.method not in ("GET", "HEAD", "OPTIONS") and origin and urlparse(origin).netloc.lower() not in self._hosts:
                return JSONResponse({"error": "cross-origin request refused"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = CSP
        if is_api:
            response.headers["Cache-Control"] = "no-store"
        return response


def _bad(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _json_body(request: Request) -> dict[str, Any] | None:
    try:
        data = await request.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def project_messages(app: App, session_id: str) -> list[dict[str, Any]]:
    """The whole log in display form, including rows hidden from the model by compaction."""
    _, covers, _ = app.store.load_context(session_id)
    out: list[dict[str, Any]] = []
    for seq, kind, body in app.store.rows(session_id):
        if kind == "summary":
            out.append({"seq": seq, "role": "summary", "content": body["content"], "covers_through": body["covers_through"]})
            continue
        item: dict[str, Any] = {"seq": seq, "role": body.get("role", "?"), "content": body.get("content") or "", "compacted": seq <= covers}
        if body.get("tool_calls"):
            item["tool_calls"] = [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in body["tool_calls"]]
        if body.get("tool_call_id"):
            item["tool_call_id"] = body["tool_call_id"]
        out.append(item)
    return out


def create_web_app(app: App, approvals: Approvals, token: str, *, allowed_hosts: set[str]) -> Starlette:
    runs = RunManager(app)

    async def state(request: Request) -> Response:
        s = app.settings
        skills = app.skills.scan()
        return JSONResponse({
            "model": app.llm.model,
            "base_url": getattr(s, "base_url", ""),
            "embeddings": s.embed_model if s.embed_api_key else "offline hashing fallback",
            "approval": app.policy.mode,
            "workspace": str(s.workspace),
            "mcp": app.mcp_report,
            "tools": [{"name": t.name, "description": t.description[:300], "risk": t.risk, "idempotent": t.idempotent,
                       "parallel_safe": t.parallel_safe, "external": "__" in t.name and t.untrusted_output}
                      for t in app.registry.all()],
            "skills": [{"name": k.name, "description": k.description, "source": k.source, "available": k.available, "missing": k.missing}
                       for k in sorted(skills.values(), key=lambda k: k.name)],
            "price": {"in": s.price_in_per_mtok, "out": s.price_out_per_mtok},
        })

    async def set_approval(request: Request) -> Response:
        body = await _json_body(request)
        mode = (body or {}).get("mode")
        if mode not in ("ask", "auto", "readonly"):
            return _bad("mode must be ask, auto or readonly")
        app.policy.mode = mode
        app.settings.approval = mode
        return JSONResponse({"approval": mode})

    async def list_sessions(request: Request) -> Response:
        rows = app.store.list_sessions(limit=100)
        for row in rows:
            row["running"] = runs.is_running(row["id"])
        return JSONResponse(rows)

    async def create_session(request: Request) -> Response:
        return JSONResponse({"id": app.store.create_session()}, status_code=201)

    async def get_messages(request: Request) -> Response:
        session_id = request.path_params["session_id"]
        return JSONResponse({"messages": project_messages(app, session_id), "running": runs.is_running(session_id)})

    async def post_message(request: Request) -> Response:
        session_id = request.path_params["session_id"]
        body = await _json_body(request)
        text = (body or {}).get("text")
        if not isinstance(text, str) or not text.strip():
            return _bad("text is required")
        if len(text) > 50_000:
            return _bad("message too long", 413)
        if runs.is_running(session_id):
            return _bad("a run is already in progress for this session", 409)
        app.store.create_session(session_id=session_id)

        async def events() -> AsyncIterator[bytes]:
            try:
                async for event in runs.stream(session_id, text):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
            except RunBusy:
                yield b'data: {"type": "error", "message": "a run is already in progress for this session"}\n\n'

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    async def stop_run(request: Request) -> Response:
        return JSONResponse({"stopped": runs.stop(request.path_params["session_id"])})

    async def trace(request: Request) -> Response:
        session_id = request.path_params["session_id"]
        events = app.store.events(session_id)
        summary = summarise(events, session_id)
        cost = summary.cost(app.settings.price_in_per_mtok, app.settings.price_out_per_mtok)
        return JSONResponse({
            "events": events,
            "summary": {"llm_calls": summary.llm_calls, "tool_calls": summary.tool_calls, "tool_errors": summary.tool_errors,
                        "llm_ms": summary.llm_ms, "tool_ms": summary.tool_ms, "prompt_tokens": summary.usage.prompt_tokens,
                        "completion_tokens": summary.usage.completion_tokens, "cost": cost,
                        "by_tool": dict(summary.by_tool), "recoveries": dict(summary.recoveries)},
        })

    async def list_approvals(request: Request) -> Response:
        return JSONResponse(approvals.pending())

    async def decide_approval(request: Request) -> Response:
        body = await _json_body(request)
        if body is None or not isinstance(body.get("allow"), bool):
            return _bad("allow (boolean) is required")
        if not approvals.resolve(request.path_params["approval_id"], body["allow"]):
            return _bad("no such pending approval", 404)
        return JSONResponse({"ok": True})

    async def get_memory(request: Request) -> Response:
        persona = app.settings.persona_path
        return JSONResponse({"facts": app.longterm.facts(), "persona": persona.read_text(encoding="utf-8") if persona.is_file() else "",
                             "max_chars": app.longterm.max_chars, "used_chars": len("\n".join(app.longterm.facts()))})

    async def add_fact(request: Request) -> Response:
        body = await _json_body(request)
        status = app.longterm.add(str((body or {}).get("fact", "")))
        if status not in ("added", "duplicate"):
            return _bad(status)
        return JSONResponse({"status": status, "facts": app.longterm.facts()})

    async def remove_fact(request: Request) -> Response:
        body = await _json_body(request)
        text = str((body or {}).get("text", ""))
        if len(text.strip()) < 3:
            return _bad("text must be at least 3 characters")
        return JSONResponse({"removed": app.longterm.remove(text, exact=True), "facts": app.longterm.facts()})

    async def put_persona(request: Request) -> Response:
        body = await _json_body(request)
        text = (body or {}).get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_PERSONA_CHARS:
            return _bad(f"text must be 1-{MAX_PERSONA_CHARS} characters")
        path = app.settings.persona_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(text.strip() + "\n", encoding="utf-8")
        os.replace(temp, path)
        return JSONResponse({"ok": True})

    async def search_memory(request: Request) -> Response:
        query = request.query_params.get("q", "").strip()
        if not query:
            return JSONResponse({"hits": []})
        try:
            hits = await app.episodic.search(query, k=8)
        except Exception as exc:  # e.g. embedding service unreachable
            return _bad(f"search failed: {type(exc).__name__}: {exc}", 502)
        return JSONResponse({"hits": [{"id": h.id, "session_id": h.session_id, "text": h.text, "score": h.score, "dense": h.dense,
                                       "sparse": h.sparse, "age": h.age_label()} for h in hits]})

    async def list_jobs(request: Request) -> Response:
        jobs = [job for _, _, job in app.store.kv_list(NAMESPACE)]
        jobs.sort(key=lambda j: j.get("created") or 0, reverse=True)
        for job in jobs:
            job["result"] = (job.get("result") or "")[:2000]
        return JSONResponse(jobs)

    async def cancel_job(request: Request) -> Response:
        result = app.subagents.cancel(request.path_params["job_id"])
        return JSONResponse({"ok": not result.is_error, "message": result.text}, status_code=200 if not result.is_error else 409)

    routes = [
        Route("/api/state", state),
        Route("/api/approval-mode", set_approval, methods=["POST"]),
        Route("/api/sessions", list_sessions),
        Route("/api/sessions", create_session, methods=["POST"]),
        Route("/api/sessions/{session_id}/messages", get_messages),
        Route("/api/sessions/{session_id}/messages", post_message, methods=["POST"]),
        Route("/api/sessions/{session_id}/stop", stop_run, methods=["POST"]),
        Route("/api/sessions/{session_id}/trace", trace),
        Route("/api/approvals", list_approvals),
        Route("/api/approvals/{approval_id}", decide_approval, methods=["POST"]),
        Route("/api/memory", get_memory),
        Route("/api/memory/facts", add_fact, methods=["POST"]),
        Route("/api/memory/facts", remove_fact, methods=["DELETE"]),
        Route("/api/memory/persona", put_persona, methods=["PUT"]),
        Route("/api/memory/search", search_memory),
        Route("/api/jobs", list_jobs),
        Route("/api/jobs/{job_id}/cancel", cancel_job, methods=["POST"]),
        Mount("/", StaticFiles(directory=str(STATIC_DIR), html=True)),
    ]
    web = Starlette(routes=routes)
    web.add_middleware(SecurityMiddleware, token=token, allowed_hosts=allowed_hosts)
    return web

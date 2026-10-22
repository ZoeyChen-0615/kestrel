"""SQLite persistence.

The message log is append-only: rows are only ever inserted, never updated or
deleted, and sequence numbers are never reused. Context compaction is a
*projection* over the log (a summary row that hides older rows from the model),
so the original conversation always remains available.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import ConflictError

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL DEFAULT 'chat',
    parent_id TEXT,
    title TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,            -- 'message' | 'summary'
    body TEXT NOT NULL,            -- JSON
    PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS kv (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    version INTEGER NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    duration_ms REAL NOT NULL DEFAULT 0,
    data TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_session ON events (session_id, id);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    text TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    UNIQUE (session_id, user_seq)
);
"""


class Store:
    """Thread-safe wrapper around one SQLite connection."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one IMMEDIATE transaction so check-then-write is atomic."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ---- sessions -------------------------------------------------------

    def create_session(self, *, session_id: str | None = None, kind: str = "chat",
                       parent_id: str | None = None, title: str | None = None) -> str:
        session_id = session_id or uuid.uuid4().hex[:12]
        with self._tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO sessions (id, created_at, kind, parent_id, title) VALUES (?,?,?,?,?)",
                (session_id, time.time(), kind, parent_id, title),
            )
        return session_id

    def list_sessions(self, limit: int = 20, kind: str | None = "chat") -> list[dict[str, Any]]:
        sql = ("SELECT s.id, s.created_at, s.kind, s.title, "
               "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS messages "
               "FROM sessions s")
        args: list[Any] = []
        if kind:
            sql += " WHERE s.kind = ?"
            args.append(kind)
        sql += " ORDER BY s.created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args)]

    def set_title(self, session_id: str, title: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE sessions SET title = ? WHERE id = ? AND title IS NULL", (title, session_id))

    # ---- message log ----------------------------------------------------

    def head(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?", (session_id,)).fetchone()
        return int(row[0])

    def append(self, session_id: str, body: dict[str, Any], *, kind: str = "message",
               expected_head: int | None = None) -> int:
        """Append one row and return its sequence number.

        ``expected_head`` makes the append conditional: if another writer got
        in first, ``ConflictError`` is raised instead of interleaving.
        """
        with self._tx() as c:
            head = c.execute("SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
            if expected_head is not None and head != expected_head:
                raise ConflictError(f"session {session_id}: expected head {expected_head}, found {head}")
            seq = head + 1
            c.execute(
                "INSERT INTO messages (session_id, seq, ts, kind, body) VALUES (?,?,?,?,?)",
                (session_id, seq, time.time(), kind, json.dumps(body, ensure_ascii=False)),
            )
            return seq

    def rows(self, session_id: str, *, after_seq: int = 0) -> list[tuple[int, str, dict[str, Any]]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT seq, kind, body FROM messages WHERE session_id = ? AND seq > ? ORDER BY seq",
                (session_id, after_seq),
            )
            return [(r["seq"], r["kind"], json.loads(r["body"])) for r in cur]

    def load_context(self, session_id: str) -> tuple[str | None, int, list[tuple[int, dict[str, Any]]]]:
        """Project the log into what the model should see.

        Returns ``(summary_text, covers_through, messages)`` where ``messages``
        are the rows after the latest summary. Older rows stay in the log.
        """
        all_rows = self.rows(session_id)
        summary: str | None = None
        covers = 0
        for seq, kind, body in all_rows:
            if kind == "summary" and body.get("covers_through", 0) >= covers:
                summary = body["content"]
                covers = body["covers_through"]
        messages = [(seq, body) for seq, kind, body in all_rows if kind == "message" and seq > covers]
        return summary, covers, messages

    # ---- versioned key/value (compare-and-swap) -------------------------

    def kv_get(self, namespace: str, key: str) -> tuple[int, dict[str, Any]] | None:
        with self._lock:
            row = self._conn.execute("SELECT version, value FROM kv WHERE namespace = ? AND key = ?", (namespace, key)).fetchone()
        return (row["version"], json.loads(row["value"])) if row else None

    def kv_put(self, namespace: str, key: str, value: dict[str, Any], *, expected_version: int | None) -> int:
        """Write ``value``. ``expected_version=None`` means the key must not exist yet."""
        encoded = json.dumps(value, ensure_ascii=False)
        with self._tx() as c:
            row = c.execute("SELECT version FROM kv WHERE namespace = ? AND key = ?", (namespace, key)).fetchone()
            if row is None:
                if expected_version is not None:
                    raise ConflictError(f"{namespace}/{key}: expected version {expected_version}, key missing")
                c.execute("INSERT INTO kv (namespace, key, version, value) VALUES (?,?,?,?)", (namespace, key, 1, encoded))
                return 1
            if expected_version != row["version"]:
                raise ConflictError(f"{namespace}/{key}: expected version {expected_version}, found {row['version']}")
            c.execute("UPDATE kv SET version = ?, value = ? WHERE namespace = ? AND key = ?",
                      (row["version"] + 1, encoded, namespace, key))
            return row["version"] + 1

    def kv_list(self, namespace: str) -> list[tuple[str, int, dict[str, Any]]]:
        with self._lock:
            cur = self._conn.execute("SELECT key, version, value FROM kv WHERE namespace = ? ORDER BY key", (namespace,))
            return [(r["key"], r["version"], json.loads(r["value"])) for r in cur]

    # ---- events (traces) ------------------------------------------------

    def add_event(self, session_id: str | None, kind: str, name: str, duration_ms: float, data: dict[str, Any]) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO events (session_id, ts, kind, name, duration_ms, data) VALUES (?,?,?,?,?,?)",
                (session_id, time.time(), kind, name, duration_ms, json.dumps(data, ensure_ascii=False, default=str)),
            )

    def events(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM events WHERE session_id = ? ORDER BY id", (session_id,))
            out = []
            for r in cur:
                item = dict(r)
                item["data"] = json.loads(item["data"])
                out.append(item)
            return out

    # ---- episodic memory -------------------------------------------------

    def add_episode(self, session_id: str, user_seq: int, text: str, model: str, dim: int, vector: bytes) -> bool:
        with self._tx() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO episodes (session_id, user_seq, ts, text, model, dim, vector) VALUES (?,?,?,?,?,?,?)",
                (session_id, user_seq, time.time(), text, model, dim, vector),
            )
            return cur.rowcount > 0

    def episodes(self, *, model: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT id, session_id, user_seq, ts, text, model, dim, vector FROM episodes"
        args: tuple[Any, ...] = ()
        if model:
            sql += " WHERE model = ?"
            args = (model,)
        with self._lock:
            return list(self._conn.execute(sql, args))

    def delete_episode(self, episode_id: int) -> bool:
        with self._tx() as c:
            return c.execute("DELETE FROM episodes WHERE id = ?", (episode_id,)).rowcount > 0

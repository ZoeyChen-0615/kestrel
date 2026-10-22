"""Episodic memory: every finished exchange is embedded and stored; search is hybrid.

Retrieval fuses two rankings with Reciprocal Rank Fusion:

* dense  - cosine similarity between normalised embedding vectors,
* sparse - BM25 over the raw text (catches exact names and rare terms
  that embeddings blur).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..common import Embedder
from ..store import Store
from .bm25 import bm25_scores

RRF_K = 60
MAX_STORED_CHARS = 2000


@dataclass
class Hit:
    id: int
    session_id: str
    user_seq: int
    ts: float
    text: str
    score: float
    dense: float
    sparse: float

    def age_label(self) -> str:
        seconds = max(0, time.time() - self.ts)
        for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
            if seconds >= size:
                return f"{int(seconds // size)}{unit} ago"
        return "just now"


def _normalise(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array


class EpisodicMemory:
    def __init__(self, store: Store, embedder: Embedder, *, min_dense: float = 0.25) -> None:
        self._store = store
        self._embedder = embedder
        self._min_dense = min_dense

    async def learn(self, session_id: str, user_seq: int, user_text: str, assistant_text: str) -> bool:
        """Embed and store one exchange. Idempotent per (session, user message)."""
        text = f"User: {user_text.strip()}\nAssistant: {assistant_text.strip()}"[:MAX_STORED_CHARS]
        vector = _normalise((await self._embedder.embed([text]))[0])
        return self._store.add_episode(session_id, user_seq, text, self._embedder.model, int(vector.shape[0]), vector.tobytes())

    async def search(
        self,
        query: str,
        k: int = 5,
        *,
        exclude: Callable[[str, int], bool] | None = None,
    ) -> list[Hit]:
        """Return up to ``k`` fused hits. ``exclude(session, user_seq)`` hides episodes already in context."""
        rows = [r for r in self._store.episodes(model=self._embedder.model) if not (exclude and exclude(r["session_id"], r["user_seq"]))]
        if not rows or not query.strip():
            return []
        query_vector = _normalise((await self._embedder.embed([query]))[0])
        matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        dense = matrix @ query_vector
        sparse = np.asarray(bm25_scores(query, [r["text"] for r in rows]), dtype=np.float32)

        candidates = set(np.argsort(-dense)[: max(k * 3, 10)].tolist())
        candidates |= set(np.argsort(-sparse)[: max(k * 3, 10)].tolist())
        dense_rank = {int(i): rank for rank, i in enumerate(np.argsort(-dense))}
        sparse_rank = {int(i): rank for rank, i in enumerate(np.argsort(-sparse))}

        hits: list[Hit] = []
        for index in candidates:
            relevant = dense[index] >= self._min_dense or sparse[index] > 0
            if not relevant:
                continue
            fused = 1 / (RRF_K + dense_rank[index]) + (1 / (RRF_K + sparse_rank[index]) if sparse[index] > 0 else 0.0)
            row = rows[index]
            hits.append(Hit(row["id"], row["session_id"], row["user_seq"], row["ts"], row["text"],
                            float(fused), float(dense[index]), float(sparse[index])))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def forget(self, episode_id: int) -> bool:
        return self._store.delete_episode(episode_id)

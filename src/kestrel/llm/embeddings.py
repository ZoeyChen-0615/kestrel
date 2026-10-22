"""Embedding backends: a real OpenAI-compatible client and an offline hashing fallback."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any

import httpx

from ..common import LLMError
from .client import Sleep, post_json_with_retry

_WORD = re.compile(r"[a-z0-9_]+")
_CJK = re.compile(r"[一-鿿]")


class OpenAICompatEmbedder:
    """Calls ``POST {base_url}/embeddings`` on an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 4,
        batch_size: int = 64,
        client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._sleep = sleep

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            data = await post_json_with_retry(
                self._client, self._url, {"model": self.model, "input": batch}, self._headers,
                max_retries=self._max_retries, sleep=self._sleep,
            )
            rows: list[dict[str, Any]] = data.get("data") or []
            if len(rows) != len(batch):
                raise LLMError(f"embedding count mismatch: sent {len(batch)}, got {len(rows)}")
            vectors.extend(row["embedding"] for row in sorted(rows, key=lambda r: r.get("index", 0)))
        return vectors


class HashingEmbedder:
    """Deterministic feature-hashing embedder for offline use and tests.

    Quality is far below a trained model, but it needs no network, is stable
    across processes (it uses md5, not Python's salted ``hash``), and still
    places texts with shared words close together.
    """

    def __init__(self, dim: int = 256) -> None:
        self.model = f"hashing-{dim}"
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[str]:
        lowered = text.lower()
        words = _WORD.findall(lowered)
        features = list(words)
        features += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        cjk = _CJK.findall(lowered)
        features += cjk
        features += [a + b for a, b in zip(cjk, cjk[1:])]
        return features

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

"""EmbeddingProvider Protocol + concrete clients.

Two providers ship:

  - ``JinaHTTPEmbeddingClient`` — talks to a Jina-compatible HTTP server
    (any OpenAI-compatible /v1/embeddings endpoint also works). Used for
    the local Jina-embedding-v3-small server.
  - ``RandomEmbeddingClient`` — deterministic per-text random vectors,
    useful as a baseline / for tests when no real server is running.

``build_embedding_client(profile)`` is the factory: pick the right
client based on ``profile.provider``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from annotation_pipeline_skill.similarity.profiles import SimilarityProfile


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray  # shape (N, dim)
    model: str
    provider: str


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingResult: ...


class JinaHTTPEmbeddingClient:
    """OpenAI-compatible /v1/embeddings client. Tested against a local
    Jina embedding server but anything that speaks the OpenAI shape
    works.
    """

    def __init__(self, profile: SimilarityProfile):
        self.profile = profile
        if not profile.base_url:
            raise ValueError(f"profile {profile.name!r} needs base_url")
        self._http = httpx.Client(timeout=profile.timeout_seconds)
        self._url = profile.base_url.rstrip("/") + "/v1/embeddings"

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=np.zeros((0, 0), dtype=np.float32),
                model=self.profile.model, provider="jina_http",
            )
        headers = {"Content-Type": "application/json"}
        api_key = self.profile.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        all_vecs: list[list[float]] = []
        bs = max(1, self.profile.batch_size)
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            resp = self._http.post(
                self._url,
                json={"model": self.profile.model, "input": batch},
                headers=headers,
                timeout=self.profile.timeout_seconds,
            )
            resp.raise_for_status()
            payload = resp.json()
            for row in payload["data"]:
                all_vecs.append(row["embedding"])
        return EmbeddingResult(
            vectors=np.asarray(all_vecs, dtype=np.float32),
            model=self.profile.model, provider="jina_http",
        )

    def close(self) -> None:
        self._http.close()


class RandomEmbeddingClient:
    """Deterministic per-text random vectors. Seeded from a hash of the
    text so the same text always maps to the same vector — useful for
    unit tests, plumbing checks, and as a structural baseline."""

    def __init__(self, profile: SimilarityProfile):
        self.profile = profile
        self.dim = int(profile.dim or 128)

    def embed(self, texts: list[str]) -> EmbeddingResult:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(
                hashlib.sha256(t.encode("utf-8")).digest()[:4], "little",
            )
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self.dim).astype(np.float32)
        return EmbeddingResult(vectors=out, model=self.profile.model, provider="random")

    def close(self) -> None:
        return None


def build_embedding_client(profile: SimilarityProfile) -> EmbeddingProvider:
    if profile.provider == "jina_http":
        return JinaHTTPEmbeddingClient(profile)
    if profile.provider == "random":
        return RandomEmbeddingClient(profile)
    raise ValueError(f"unknown embedding provider: {profile.provider!r}")

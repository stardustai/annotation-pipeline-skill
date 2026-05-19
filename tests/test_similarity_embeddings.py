import numpy as np
import pytest

from annotation_pipeline_skill.similarity.embeddings import (
    EmbeddingResult,
    JinaHTTPEmbeddingClient,
    RandomEmbeddingClient,
    build_embedding_client,
)
from annotation_pipeline_skill.similarity.profiles import SimilarityProfile


def test_random_client_returns_deterministic_vectors():
    profile = SimilarityProfile(
        name="rand", provider="random", model="r-128", dim=128,
    )
    client = build_embedding_client(profile)
    out = client.embed(["a", "b", "a"])
    assert isinstance(out, EmbeddingResult)
    assert out.vectors.shape == (3, 128)
    # Same text → same vector (deterministic seed by hash).
    np.testing.assert_allclose(out.vectors[0], out.vectors[2])


def test_jina_http_url_handles_v1_already_in_base_url():
    """base_url may or may not already include the OpenAI /v1 prefix; in
    either case the final URL should land at .../v1/embeddings (not
    .../v1/v1/embeddings)."""
    p1 = SimilarityProfile(
        name="a", provider="jina_http", model="m", base_url="https://example.com",
    )
    p2 = SimilarityProfile(
        name="b", provider="jina_http", model="m", base_url="https://example.com/v1",
    )
    p3 = SimilarityProfile(
        name="c", provider="jina_http", model="m", base_url="https://example.com/v1/",
    )
    assert JinaHTTPEmbeddingClient(p1)._url == "https://example.com/v1/embeddings"
    assert JinaHTTPEmbeddingClient(p2)._url == "https://example.com/v1/embeddings"
    assert JinaHTTPEmbeddingClient(p3)._url == "https://example.com/v1/embeddings"


def test_jina_http_client_batches_and_returns_vectors(monkeypatch):
    profile = SimilarityProfile(
        name="jina_small", provider="jina_http",
        model="jina-embeddings-v3-small-en",
        base_url="http://127.0.0.1:8001", api_key="sk-test",
        batch_size=2,
    )
    client = build_embedding_client(profile)
    calls = []

    class FakeResp:
        def __init__(self, vectors):
            self.vectors = vectors
        def raise_for_status(self): pass
        def json(self):
            return {"data": [{"embedding": v} for v in self.vectors]}

    def fake_post(url, json, headers, timeout):
        calls.append(json["input"])
        return FakeResp([[float(len(t)), 0.0, 1.0] for t in json["input"]])

    monkeypatch.setattr(client._http, "post", fake_post)
    out = client.embed(["aa", "b", "ccc", "dddd"])
    # batch_size=2 → exactly 2 HTTP calls
    assert len(calls) == 2
    assert calls[0] == ["aa", "b"]
    assert calls[1] == ["ccc", "dddd"]
    assert out.vectors.shape == (4, 3)
    np.testing.assert_array_equal(out.vectors[:, 0], [2.0, 1.0, 3.0, 4.0])

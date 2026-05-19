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


class _FakeResp:
    def __init__(self, vectors):
        self.vectors = vectors
    def raise_for_status(self): pass
    def json(self):
        return {"data": [{"embedding": v} for v in self.vectors]}


def _make_fake_post(calls):
    def fake_post(url, json, headers, timeout):
        calls.append(json["input"])
        return _FakeResp([[float(len(t)), 0.0, 1.0] for t in json["input"]])
    return fake_post


def test_jina_http_client_sends_all_in_one_request_by_default(monkeypatch):
    """Default behaviour (batch_size=0/None): one HTTP POST regardless of
    input size — server is expected to do its own internal batching."""
    profile = SimilarityProfile(
        name="jina_small", provider="jina_http",
        model="m", base_url="http://x", api_key="sk-test",
        batch_size=0,
    )
    client = build_embedding_client(profile)
    calls: list[list[str]] = []
    monkeypatch.setattr(client._http, "post", _make_fake_post(calls))
    out = client.embed(["aa", "b", "ccc", "dddd"])
    # Exactly one HTTP call containing the whole input list.
    assert len(calls) == 1
    assert calls[0] == ["aa", "b", "ccc", "dddd"]
    assert out.vectors.shape == (4, 3)
    np.testing.assert_array_equal(out.vectors[:, 0], [2.0, 1.0, 3.0, 4.0])


def test_jina_http_client_chunks_when_batch_size_positive(monkeypatch):
    """Opt-in chunking: positive batch_size still works for servers that
    cap inputs per request."""
    profile = SimilarityProfile(
        name="jina_small", provider="jina_http",
        model="m", base_url="http://x", api_key="sk-test",
        batch_size=2,
    )
    client = build_embedding_client(profile)
    calls: list[list[str]] = []
    monkeypatch.setattr(client._http, "post", _make_fake_post(calls))
    out = client.embed(["aa", "b", "ccc", "dddd"])
    assert len(calls) == 2
    assert calls[0] == ["aa", "b"]
    assert calls[1] == ["ccc", "dddd"]
    assert out.vectors.shape == (4, 3)


def test_minhash_client_produces_signature_vectors():
    from annotation_pipeline_skill.similarity.embeddings import (
        MinHashSignatureClient,
    )
    profile = SimilarityProfile(
        name="mh", provider="minhash", model="minhash-w5-p128",
        shingle_size=5, num_perm=128,
    )
    client = build_embedding_client(profile)
    assert isinstance(client, MinHashSignatureClient)
    out = client.embed([
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",  # identical
        "completely different sentence about machine learning model",
    ])
    assert out.vectors.shape == (3, 128)
    assert out.provider == "minhash"
    # Identical texts produce identical signatures.
    np.testing.assert_array_equal(out.vectors[0], out.vectors[1])
    # Different texts produce mostly-different signatures (MinHash signature
    # agreement is a Jaccard estimator; for these unrelated inputs we expect
    # roughly all 128 positions to differ).
    diff = (out.vectors[0] != out.vectors[2]).sum()
    assert diff > 100, f"expected >100 differing positions, got {diff}"

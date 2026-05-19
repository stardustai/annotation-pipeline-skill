import pytest

from annotation_pipeline_skill.similarity.profiles import (
    SimilarityProfile,
    load_similarity_profiles,
)


def test_load_profiles_from_yaml(tmp_path):
    yaml_text = """
profiles:
  jina_small:
    provider: jina_http
    model: jina-embeddings-v3-small-en
    base_url: http://127.0.0.1:8001
    api_key: sk-local
    batch_size: 32
    max_tokens: 512
  random_baseline:
    provider: random
    model: random-128
    dim: 128
"""
    path = tmp_path / "similarity_profiles.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    profiles = load_similarity_profiles(path)
    assert "jina_small" in profiles
    assert profiles["jina_small"].provider == "jina_http"
    assert profiles["jina_small"].base_url == "http://127.0.0.1:8001"
    assert profiles["jina_small"].batch_size == 32
    assert profiles["random_baseline"].provider == "random"


def test_load_profiles_rejects_unknown_provider(tmp_path):
    yaml_text = """
profiles:
  bad:
    provider: not_a_real_provider
    model: x
"""
    path = tmp_path / "similarity_profiles.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown provider"):
        load_similarity_profiles(path)


def test_batch_size_zero_is_preserved(tmp_path):
    """``batch_size: 0`` is a meaningful 'no client-side batching' signal
    and must not silently fall back to the default 32."""
    yaml_text = """
profiles:
  no_batch:
    provider: jina_http
    model: m
    base_url: http://x
    batch_size: 0
"""
    path = tmp_path / "similarity_profiles.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    profiles = load_similarity_profiles(path)
    assert profiles["no_batch"].batch_size == 0


def test_profile_resolve_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "from-env")
    p = SimilarityProfile(
        name="jina_small", provider="jina_http", model="x",
        base_url="http://x", api_key=None, api_key_env="JINA_API_KEY",
    )
    assert p.resolve_api_key() == "from-env"

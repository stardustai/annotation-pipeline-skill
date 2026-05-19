"""Tests for RowEmbeddingCache (per-row embedding cache)."""
from __future__ import annotations

import numpy as np
import pytest

from annotation_pipeline_skill.similarity.embedding_cache import text_content_hash
from annotation_pipeline_skill.similarity.row_embedding_cache import (
    CachedRowEmbedding,
    RowEmbeddingCache,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _open_store(tmp_path):
    return SqliteStore.open(tmp_path / "proj")


def test_row_cache_get_put_round_trip(tmp_path):
    """Put 3 row embeddings for task t-001 (indices 0, 1, 2), get them all back."""
    store = _open_store(tmp_path)
    cache = RowEmbeddingCache(store)

    h0 = text_content_hash("row zero text")
    h1 = text_content_hash("row one text")
    h2 = text_content_hash("row two text")

    v0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v1 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    v2 = np.array([7.0, 8.0, 9.0], dtype=np.float32)

    cache.put_many(
        profile_name="test_profile",
        model="test-model",
        dim=3,
        entries=[
            ("t-001", 0, h0, v0),
            ("t-001", 1, h1, v1),
            ("t-001", 2, h2, v2),
        ],
    )

    specs = [("t-001", 0, h0), ("t-001", 1, h1), ("t-001", 2, h2)]
    got = cache.get_many(profile_name="test_profile", specs=specs)

    assert len(got) == 3
    assert ("t-001", 0) in got
    assert ("t-001", 1) in got
    assert ("t-001", 2) in got

    np.testing.assert_array_equal(got[("t-001", 0)].vector, v0)
    np.testing.assert_array_equal(got[("t-001", 1)].vector, v1)
    np.testing.assert_array_equal(got[("t-001", 2)].vector, v2)

    # Check dataclass fields
    entry = got[("t-001", 0)]
    assert isinstance(entry, CachedRowEmbedding)
    assert entry.task_id == "t-001"
    assert entry.row_index == 0
    assert entry.content_hash == h0


def test_row_cache_miss_on_stale_content_hash(tmp_path):
    """Put with hash=H, get with hash=H' → empty (stale cache miss)."""
    store = _open_store(tmp_path)
    cache = RowEmbeddingCache(store)

    old_hash = text_content_hash("original text")
    new_hash = text_content_hash("modified text")

    cache.put_many(
        profile_name="p",
        model="m",
        dim=4,
        entries=[("task-A", 0, old_hash, np.ones(4, dtype=np.float32))],
    )

    # Lookup with the NEW hash — should return nothing (text changed)
    got = cache.get_many(profile_name="p", specs=[("task-A", 0, new_hash)])
    assert got == {}

    # Lookup with the OLD hash — should still hit
    got2 = cache.get_many(profile_name="p", specs=[("task-A", 0, old_hash)])
    assert ("task-A", 0) in got2


def test_row_cache_distinguishes_rows_in_same_task(tmp_path):
    """Same task, two different row_indices, different vectors round-trip independently."""
    store = _open_store(tmp_path)
    cache = RowEmbeddingCache(store)

    h_row0 = text_content_hash("task row 0")
    h_row1 = text_content_hash("task row 1")

    vec_row0 = np.array([10.0, 20.0], dtype=np.float32)
    vec_row1 = np.array([30.0, 40.0], dtype=np.float32)

    cache.put_many(
        profile_name="pf",
        model="mod",
        dim=2,
        entries=[
            ("task-Z", 0, h_row0, vec_row0),
            ("task-Z", 1, h_row1, vec_row1),
        ],
    )

    got = cache.get_many(
        profile_name="pf",
        specs=[("task-Z", 0, h_row0), ("task-Z", 1, h_row1)],
    )

    assert len(got) == 2
    np.testing.assert_array_equal(got[("task-Z", 0)].vector, vec_row0)
    np.testing.assert_array_equal(got[("task-Z", 1)].vector, vec_row1)

    # Cross-fetch: ask for row 0 with row 1's hash → miss
    cross = cache.get_many(profile_name="pf", specs=[("task-Z", 0, h_row1)])
    assert cross == {}

    # Partial get — only row 0
    partial = cache.get_many(profile_name="pf", specs=[("task-Z", 0, h_row0)])
    assert len(partial) == 1
    assert ("task-Z", 0) in partial
    assert ("task-Z", 1) not in partial


def test_row_cache_get_many_empty_specs(tmp_path):
    """get_many with empty specs returns empty dict without error."""
    store = _open_store(tmp_path)
    cache = RowEmbeddingCache(store)
    got = cache.get_many(profile_name="p", specs=[])
    assert got == {}


def test_row_cache_put_many_empty_entries(tmp_path):
    """put_many with empty entries is a no-op."""
    store = _open_store(tmp_path)
    cache = RowEmbeddingCache(store)
    # Should not raise
    cache.put_many(profile_name="p", model="m", dim=8, entries=[])


def test_row_cache_upsert_updates_existing(tmp_path):
    """Putting the same (task_id, row_index, profile_name) twice updates the entry."""
    store = _open_store(tmp_path)
    cache = RowEmbeddingCache(store)

    h = text_content_hash("consistent text")
    v_old = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    v_new = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    cache.put_many(profile_name="q", model="m", dim=3, entries=[("t", 5, h, v_old)])
    cache.put_many(profile_name="q", model="m", dim=3, entries=[("t", 5, h, v_new)])

    got = cache.get_many(profile_name="q", specs=[("t", 5, h)])
    assert ("t", 5) in got
    np.testing.assert_array_equal(got[("t", 5)].vector, v_new)

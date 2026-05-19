"""Tests for TaskEmbeddingCache + cache-aware scan."""
import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.similarity.embedding_cache import (
    TaskEmbeddingCache, text_content_hash,
)
from annotation_pipeline_skill.similarity.profiles import SimilarityProfile
from annotation_pipeline_skill.services.distribution_service import DistributionService
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _make_task(task_id: str, text: str, status: TaskStatus = TaskStatus.ACCEPTED) -> Task:
    return Task(
        task_id=task_id, pipeline_id="proj",
        source_ref={"payload": {"rows": [{"row_index": 0, "input": text}]}},
        external_ref=None, modality="text",
        annotation_requirements={}, selected_annotator_id=None,
        status=status, current_attempt=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        active_run_id=None, next_retry_at=None,
        metadata={}, document_version_id=None,
    )


def _seed(tmp_path: Path, count: int = 5) -> SqliteStore:
    root = tmp_path / "proj"
    root.mkdir()
    store = SqliteStore.open(root)
    for i in range(count):
        t = _make_task(f"t-{i:03d}", f"This is task {i} with some unique text content.")
        store.save_task(t)
    return store


def test_cache_get_put_round_trip(tmp_path):
    store = _seed(tmp_path, count=3)
    cache = TaskEmbeddingCache(store)
    h0 = text_content_hash("hello")
    h1 = text_content_hash("world")
    v0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v1 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    cache.put_many(
        profile_name="p", model="m", dim=3,
        entries=[("t-000", h0, v0), ("t-001", h1, v1)],
    )
    got = cache.get_many(profile_name="p", task_specs=[("t-000", h0), ("t-001", h1)])
    assert len(got) == 2
    np.testing.assert_array_equal(got["t-000"].vector, v0)
    np.testing.assert_array_equal(got["t-001"].vector, v1)


def test_cache_miss_on_stale_content_hash(tmp_path):
    store = _seed(tmp_path, count=1)
    cache = TaskEmbeddingCache(store)
    cache.put_many(
        profile_name="p", model="m", dim=3,
        entries=[("t-000", "old-hash", np.zeros(3, dtype=np.float32))],
    )
    # Ask with a different content_hash → miss, NOT a stale hit
    got = cache.get_many(profile_name="p", task_specs=[("t-000", "new-hash")])
    assert got == {}


def test_scan_reuses_cached_vectors_on_second_run(tmp_path):
    """Second scan against same text should be near-instant — no re-embed."""
    store = _seed(tmp_path, count=5)
    profile = SimilarityProfile(
        name="rand", provider="random", model="r-32", dim=32,
    )
    svc = DistributionService(store, {"rand": profile})
    # First scan: 5 misses
    p1 = svc.scan(
        project_id="proj", profile_name="rand",
        min_cluster_size=2, umap_neighbors=3,
    )
    assert p1["params"]["embedding_cache"]["hits"] == 0
    assert p1["params"]["embedding_cache"]["misses"] == 5
    # Second scan with same text: 5 hits, 0 misses
    p2 = svc.scan(
        project_id="proj", profile_name="rand",
        min_cluster_size=2, umap_neighbors=3,
    )
    assert p2["params"]["embedding_cache"]["hits"] == 5
    assert p2["params"]["embedding_cache"]["misses"] == 0


def test_scan_reembeds_only_changed_tasks(tmp_path):
    store = _seed(tmp_path, count=5)
    profile = SimilarityProfile(name="rand", provider="random", model="r-32", dim=32)
    svc = DistributionService(store, {"rand": profile})
    svc.scan(project_id="proj", profile_name="rand", min_cluster_size=2, umap_neighbors=3)
    # Edit one task's input text → its content_hash changes
    t = store.load_task("t-002")
    t.source_ref = {"payload": {"rows": [{"row_index": 0, "input": "completely different content here"}]}}
    store.save_task(t)
    p3 = svc.scan(project_id="proj", profile_name="rand", min_cluster_size=2, umap_neighbors=3)
    assert p3["params"]["embedding_cache"]["hits"] == 4
    assert p3["params"]["embedding_cache"]["misses"] == 1

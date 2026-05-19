"""Tests for DistributionService.

Uses a real SqliteStore against tmp_path and the random_baseline profile
(no HTTP calls). Seeds tasks across multiple statuses to verify all
public methods.
"""
from __future__ import annotations

import pytest

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.services.distribution_service import DistributionService
from annotation_pipeline_skill.similarity.profiles import SimilarityProfile
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RANDOM_PROFILE = SimilarityProfile(
    name="random_baseline",
    provider="random",
    model="random-128",
    dim=128,
)

_PROFILES = {"random_baseline": _RANDOM_PROFILE}


def _make_task(
    task_id: str,
    pipeline_id: str,
    status: TaskStatus,
    text: str,
) -> Task:
    """Build a Task with a minimal source_ref payload containing one row."""
    task = Task.new(
        task_id=task_id,
        pipeline_id=pipeline_id,
        source_ref={
            "kind": "jsonl",
            "payload": {
                "rows": [{"row_index": 0, "input": text}],
            },
        },
    )
    task.status = status
    return task


def _seed_store(store: SqliteStore, pipeline_id: str) -> list[Task]:
    """Seed 8 tasks across multiple statuses and return them."""
    tasks = [
        _make_task("t-01", pipeline_id, TaskStatus.ACCEPTED, "Apple fruit sweet red"),
        _make_task("t-02", pipeline_id, TaskStatus.ACCEPTED, "Apple fruit sweet red variant"),
        _make_task("t-03", pipeline_id, TaskStatus.ACCEPTED, "Banana tropical yellow"),
        _make_task("t-04", pipeline_id, TaskStatus.ACCEPTED, "Banana tropical yellow fruit"),
        _make_task("t-05", pipeline_id, TaskStatus.ACCEPTED, "Cherry small red stone fruit"),
        _make_task("t-06", pipeline_id, TaskStatus.REJECTED, "Old duplicate entry"),
        _make_task("t-07", pipeline_id, TaskStatus.PENDING,  "Waiting to be processed"),
        _make_task("t-08", pipeline_id, TaskStatus.QC,       "Quality check in progress"),
    ]
    for t in tasks:
        store.save_task(t)
    return tasks


def _make_service(store: SqliteStore) -> DistributionService:
    return DistributionService(store, _PROFILES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scan_all_statuses_includes_every_task(tmp_path):
    """scan(statuses=None) should return coords for all seeded tasks."""
    store = SqliteStore.open(tmp_path)
    tasks = _seed_store(store, "proj-a")
    svc = _make_service(store)

    result = svc.scan(
        project_id="proj-a",
        profile_name="random_baseline",
        statuses=None,
        min_cluster_size=2,
        umap_neighbors=3,
    )

    assert result["task_count"] == len(tasks)
    assert len(result["coords"]) == len(tasks)
    assert "clusters" in result
    assert "params" in result

    # Verify required keys are present on each coord entry.
    for coord in result["coords"]:
        assert "task_id" in coord
        assert "x" in coord
        assert "y" in coord
        assert "status" in coord
        assert "text_preview" in coord
        assert "cluster_id" in coord  # may be None (noise)


def test_scan_status_filter_restricts_coords(tmp_path):
    """scan(statuses=['accepted']) must only include accepted tasks."""
    store = SqliteStore.open(tmp_path)
    _seed_store(store, "proj-b")
    svc = _make_service(store)

    result = svc.scan(
        project_id="proj-b",
        profile_name="random_baseline",
        statuses=["accepted"],
        min_cluster_size=2,
        umap_neighbors=3,
    )

    assert result["task_count"] == 5  # 5 accepted tasks
    assert all(c["status"] == "accepted" for c in result["coords"])
    assert result["params"]["statuses"] == ["accepted"]


def test_scan_writes_cache_and_cache_state_reflects_it(tmp_path):
    """get_cache_state should return cached=False before scan and cached=True after."""
    store = SqliteStore.open(tmp_path)
    _seed_store(store, "proj-c")
    svc = _make_service(store)

    pre = svc.get_cache_state(project_id="proj-c", profile_name="random_baseline")
    assert pre["cached"] is False
    assert pre["payload"] is None
    assert pre["stale"] is False

    svc.scan(
        project_id="proj-c",
        profile_name="random_baseline",
        statuses=None,
        min_cluster_size=2,
        umap_neighbors=3,
    )

    post = svc.get_cache_state(project_id="proj-c", profile_name="random_baseline")
    assert post["cached"] is True
    assert post["stale"] is False
    assert post["payload"] is not None
    assert post["cached_content_hash"] == post["current_content_hash"]


def test_cache_becomes_stale_after_task_status_change(tmp_path):
    """get_cache_state should report stale=True when the task set changes."""
    store = SqliteStore.open(tmp_path)
    tasks = _seed_store(store, "proj-d")
    svc = _make_service(store)

    svc.scan(
        project_id="proj-d",
        profile_name="random_baseline",
        statuses=None,
        min_cluster_size=2,
        umap_neighbors=3,
    )

    # Mutate one task's status (ACCEPTED → ARBITRATING is allowed).
    from annotation_pipeline_skill.core.transitions import transition_task
    accepted_task = next(t for t in tasks if t.status is TaskStatus.ACCEPTED)
    ev = transition_task(
        accepted_task,
        TaskStatus.ARBITRATING,
        actor="test",
        reason="staleness test",
        stage="test",
    )
    store.save_task(accepted_task)
    store.append_event(ev)

    state = svc.get_cache_state(project_id="proj-d", profile_name="random_baseline")
    assert state["stale"] is True
    assert state["cached_content_hash"] != state["current_content_hash"]


def test_reject_duplicates_moves_accepted_tasks(tmp_path):
    """reject_duplicates should transition ACCEPTED tasks to REJECTED."""
    store = SqliteStore.open(tmp_path)
    _seed_store(store, "proj-e")
    svc = _make_service(store)

    result = svc.reject_duplicates(
        project_id="proj-e",
        task_ids=["t-01", "t-02"],
        cluster_id="emb-0",
        representative_task_id="t-03",
        cluster_similarity=0.92,
        embedding_profile="random_baseline",
        embedding_model="random-128",
    )

    assert result["moved"] == 2
    assert result["skipped"] == 0
    assert result["skipped_task_ids"] == []

    # Verify the tasks are now REJECTED in the store.
    assert store.load_task("t-01").status is TaskStatus.REJECTED
    assert store.load_task("t-02").status is TaskStatus.REJECTED


def test_reject_duplicates_is_idempotent(tmp_path):
    """Calling reject_duplicates twice with the same IDs returns moved=0, skipped=N."""
    store = SqliteStore.open(tmp_path)
    _seed_store(store, "proj-f")
    svc = _make_service(store)

    svc.reject_duplicates(
        project_id="proj-f",
        task_ids=["t-01", "t-02"],
        cluster_id="emb-0",
        representative_task_id="t-03",
    )

    result = svc.reject_duplicates(
        project_id="proj-f",
        task_ids=["t-01", "t-02"],
        cluster_id="emb-0",
        representative_task_id="t-03",
    )

    assert result["moved"] == 0
    assert result["skipped"] == 2
    assert set(result["skipped_task_ids"]) == {"t-01", "t-02"}


def test_reject_duplicates_writes_audit_event_with_correct_stage(tmp_path):
    """An audit event with stage='similarity_dedup_embedding' must be written."""
    store = SqliteStore.open(tmp_path)
    _seed_store(store, "proj-g")
    svc = _make_service(store)

    svc.reject_duplicates(
        project_id="proj-g",
        task_ids=["t-03"],
        cluster_id="emb-1",
        representative_task_id="t-04",
        cluster_similarity=0.88,
        embedding_profile="random_baseline",
        embedding_model="random-128",
    )

    events = store.list_events("t-03")
    dedup_events = [e for e in events if e.stage == "similarity_dedup_embedding"]
    assert len(dedup_events) == 1
    ev = dedup_events[0]
    assert ev.next_status is TaskStatus.REJECTED
    assert ev.metadata["cluster_id"] == "emb-1"
    assert ev.metadata["representative_task_id"] == "t-04"
    assert ev.metadata["rejection_kind"] == "similarity_dedup_embedding"

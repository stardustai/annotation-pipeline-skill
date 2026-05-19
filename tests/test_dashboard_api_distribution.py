"""Smoke tests for /api/distribution GET + scan + reject endpoints.

Heavy correctness testing lives in test_distribution_service.py. These
tests verify the HTTP layer wiring: routing, response shape, and basic
error paths.
"""
from __future__ import annotations

import json
from pathlib import Path

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROFILES_YAML = (
    "profiles:\n"
    "  random_baseline:\n"
    "    provider: random\n"
    "    model: random-128\n"
    "    dim: 128\n"
)


def _seed_workspace_with_profiles(tmp_path: Path) -> tuple[SqliteStore, Path]:
    """Return (store, workspace_root) with a similarity_profiles.yaml that
    has only the random_baseline profile (no HTTP)."""
    workspace = tmp_path / "ws"
    project = workspace / "proj"
    project.mkdir(parents=True)
    store = SqliteStore.open(project)
    (workspace / "similarity_profiles.yaml").write_text(
        _PROFILES_YAML,
        encoding="utf-8",
    )
    return store, workspace


def _make_task(task_id: str, pipeline_id: str, text: str) -> Task:
    task = Task.new(
        task_id=task_id,
        pipeline_id=pipeline_id,
        source_ref={
            "kind": "jsonl",
            "payload": {"rows": [{"row_index": 0, "input": text}]},
        },
    )
    task.status = TaskStatus.ACCEPTED
    return task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_distribution_get_returns_empty_cache_with_available_profiles(tmp_path):
    """GET /api/distribution on a fresh store returns cached=False and lists profiles."""
    store, workspace = _seed_workspace_with_profiles(tmp_path)
    api = DashboardApi(store, workspace_root=workspace)

    status, headers, body = api.handle_get("/api/distribution?project=proj&profile=random_baseline")

    assert status == 200
    payload = json.loads(body.decode("utf-8"))
    assert payload["cached"] is False
    assert payload["payload"] is None
    assert payload["available_profiles"] == ["random_baseline"]
    # Staleness fields present
    assert "stale" in payload
    assert "current_content_hash" in payload


def test_distribution_scan_populates_cache_and_get_returns_payload(tmp_path):
    """POST /api/distribution/scan runs the pipeline; subsequent GET returns cached result."""
    store, workspace = _seed_workspace_with_profiles(tmp_path)

    # Seed enough tasks so UMAP doesn't hit the k >= N edge case.
    # UMAP clamps n_neighbors to max(1, n-1), and scipy's eigsh fails when
    # k >= N for very small matrices. 6 tasks with umap_neighbors=3 works.
    task_texts = [
        "apple sweet red fruit",
        "banana tropical yellow",
        "cherry stone small",
        "date palm sweet desert",
        "elderberry dark tart",
        "fig soft sweet drupe",
    ]
    for i, text in enumerate(task_texts):
        t = _make_task(f"t-{i:02d}", "proj", text)
        store.save_task(t)

    api = DashboardApi(store, workspace_root=workspace)
    scan_body = json.dumps({
        "profile": "random_baseline",
        "min_cluster_size": 2,
        "umap_neighbors": 3,
    }).encode()

    status, _headers, body = api.handle_post("/api/distribution/scan?project=proj", scan_body)

    assert status == 200
    result = json.loads(body.decode("utf-8"))
    assert result["cached"] is True
    assert result["payload"] is not None
    assert result["payload"]["task_count"] == 6
    assert len(result["payload"]["coords"]) == 6
    assert result["available_profiles"] == ["random_baseline"]

    # Now GET should return the cached payload.
    status2, _h2, body2 = api.handle_get("/api/distribution?project=proj&profile=random_baseline")
    assert status2 == 200
    get_result = json.loads(body2.decode("utf-8"))
    assert get_result["cached"] is True
    assert get_result["payload"] is not None
    assert get_result["payload"]["task_count"] == 6


def test_distribution_reject_returns_400_on_empty_task_ids(tmp_path):
    """POST /api/distribution/reject with empty task_ids list → 400."""
    store, workspace = _seed_workspace_with_profiles(tmp_path)
    api = DashboardApi(store, workspace_root=workspace)

    reject_body = json.dumps({"task_ids": []}).encode()
    status, _headers, body = api.handle_post(
        "/api/distribution/reject?project=proj", reject_body
    )

    assert status == 400
    result = json.loads(body.decode("utf-8"))
    assert result["error"] == "task_ids_required"

"""Smoke tests for /api/row-dedup GET + scan + mask + unmask endpoints.

Heavy correctness testing lives in test_row_dedup_service.py. These
tests verify HTTP layer wiring: routing, response shape, and basic
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
    "  MinHash:\n"
    "    provider: minhash\n"
    "    model: minhash-w3-p64\n"
    "    shingle_size: 3\n"
    "    num_perm: 64\n"
)


def _seed_workspace(tmp_path: Path) -> tuple[SqliteStore, Path]:
    workspace = tmp_path / "ws"
    project = workspace / "proj"
    project.mkdir(parents=True)
    store = SqliteStore.open(project)
    (workspace / "similarity_profiles.yaml").write_text(
        _PROFILES_YAML, encoding="utf-8",
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


def _seed_tasks(store: SqliteStore, pipeline_id: str, n: int = 4) -> list[Task]:
    template = "The equipment report for unit {n} shows a critical fault in the circuit breaker."
    tasks = []
    for i in range(n):
        t = _make_task(f"t-{i:02d}", pipeline_id, template.format(n=i))
        store.save_task(t)
        tasks.append(t)
    return tasks


# ---------------------------------------------------------------------------
# GET /api/row-dedup
# ---------------------------------------------------------------------------

def test_row_dedup_get_returns_empty_cache_with_available_profiles(tmp_path):
    """GET /api/row-dedup on a fresh store returns cached=False and lists profiles."""
    store, workspace = _seed_workspace(tmp_path)
    api = DashboardApi(store, workspace_root=workspace)

    status, _headers, body = api.handle_get(
        "/api/row-dedup?project=proj&profile=MinHash"
    )

    assert status == 200
    payload = json.loads(body.decode("utf-8"))
    assert payload["cached"] is False
    assert payload["payload"] is None
    assert "MinHash" in payload["available_profiles"]
    assert "stale" in payload
    assert "current_content_hash" in payload


def test_row_dedup_get_missing_project_returns_400(tmp_path):
    """GET /api/row-dedup without ?project= returns 400."""
    store, workspace = _seed_workspace(tmp_path)
    api = DashboardApi(store, workspace_root=workspace)

    status, _headers, body = api.handle_get("/api/row-dedup")
    assert status == 400
    result = json.loads(body.decode("utf-8"))
    assert result["error"] == "project_required"


# ---------------------------------------------------------------------------
# POST /api/row-dedup/scan
# ---------------------------------------------------------------------------

def test_row_dedup_scan_populates_cache_and_get_returns_payload(tmp_path):
    """POST /api/row-dedup/scan runs the pipeline; subsequent GET returns cached result."""
    store, workspace = _seed_workspace(tmp_path)
    _seed_tasks(store, "proj", n=4)

    api = DashboardApi(store, workspace_root=workspace)
    scan_body = json.dumps({
        "profile": "MinHash",
        "statuses": None,
        "jaccard_threshold": 0.3,
        "max_rows_per_task": 100,
    }).encode()

    status, _headers, body = api.handle_post(
        "/api/row-dedup/scan?project=proj", scan_body,
    )

    assert status == 200
    result = json.loads(body.decode("utf-8"))
    assert result["cached"] is True
    assert result["payload"] is not None
    assert result["payload"]["row_count"] == 4  # 4 tasks × 1 row
    assert result["payload"]["task_count"] == 4
    assert "clusters" in result["payload"]
    assert "MinHash" in result["available_profiles"]

    # GET should now return cached result
    status2, _h2, body2 = api.handle_get(
        "/api/row-dedup?project=proj&profile=MinHash"
    )
    assert status2 == 200
    get_result = json.loads(body2.decode("utf-8"))
    assert get_result["cached"] is True
    assert get_result["payload"] is not None
    assert get_result["payload"]["row_count"] == 4


def test_row_dedup_scan_missing_project_returns_400(tmp_path):
    store, workspace = _seed_workspace(tmp_path)
    api = DashboardApi(store, workspace_root=workspace)

    status, _headers, body = api.handle_post(
        "/api/row-dedup/scan", b'{"profile": "MinHash"}',
    )
    assert status == 400
    result = json.loads(body.decode("utf-8"))
    assert result["error"] == "project_required"


def test_row_dedup_scan_unknown_profile_returns_400(tmp_path):
    store, workspace = _seed_workspace(tmp_path)
    _seed_tasks(store, "proj", n=1)
    api = DashboardApi(store, workspace_root=workspace)

    status, _headers, body = api.handle_post(
        "/api/row-dedup/scan?project=proj",
        b'{"profile": "NonExistentProfile"}',
    )
    assert status == 400
    result = json.loads(body.decode("utf-8"))
    assert result["error"] == "unknown_profile"


# ---------------------------------------------------------------------------
# POST /api/row-dedup/mask + DELETE /api/row-dedup/mask
# ---------------------------------------------------------------------------

def test_row_dedup_mask_and_delete_round_trip(tmp_path):
    """POST mask applies masks; DELETE unmask removes them."""
    store, workspace = _seed_workspace(tmp_path)
    tasks = _seed_tasks(store, "proj", n=3)
    api = DashboardApi(store, workspace_root=workspace)

    members = [
        {"task_id": tasks[0].task_id, "row_index": 0},
        {"task_id": tasks[1].task_id, "row_index": 0},
        {"task_id": tasks[2].task_id, "row_index": 0},
    ]

    mask_body = json.dumps({
        "cluster_id": "row-0",
        "members": members,
        "cluster_similarity": 0.85,
        "embedding_profile": "MinHash",
        "embedding_model": "minhash-w3-p64",
        "actor": "operator",
    }).encode()

    status, _headers, body = api.handle_post(
        "/api/row-dedup/mask?project=proj", mask_body,
    )
    assert status == 200
    result = json.loads(body.decode("utf-8"))
    # 3 members → representative kept, 2 masked
    assert result["masked"] == 2
    assert result["skipped"] == 0

    # DELETE: remove the masks that were applied
    # (all non-representative members)
    delete_pairs = [
        {"task_id": tasks[1].task_id, "row_index": 0},
        {"task_id": tasks[2].task_id, "row_index": 0},
    ]
    delete_body = json.dumps({"pairs": delete_pairs}).encode()

    d_status, _dh, d_body = api.handle_delete(
        "/api/row-dedup/mask?project=proj", delete_body,
    )
    assert d_status == 200
    d_result = json.loads(d_body.decode("utf-8"))
    assert d_result["removed"] == 2


def test_row_dedup_mask_missing_members_returns_400(tmp_path):
    store, workspace = _seed_workspace(tmp_path)
    api = DashboardApi(store, workspace_root=workspace)

    status, _headers, body = api.handle_post(
        "/api/row-dedup/mask?project=proj",
        b'{"cluster_id": "row-0", "members": []}',
    )
    assert status == 400
    result = json.loads(body.decode("utf-8"))
    assert result["error"] == "members_required"


def test_row_dedup_delete_mask_missing_pairs_returns_400(tmp_path):
    store, workspace = _seed_workspace(tmp_path)
    api = DashboardApi(store, workspace_root=workspace)

    status, _headers, body = api.handle_delete(
        "/api/row-dedup/mask?project=proj",
        b'{"pairs": []}',
    )
    assert status == 400
    result = json.loads(body.decode("utf-8"))
    assert result["error"] == "pairs_required"


def test_row_dedup_delete_mask_missing_project_returns_400(tmp_path):
    store, workspace = _seed_workspace(tmp_path)
    api = DashboardApi(store, workspace_root=workspace)

    status, _headers, body = api.handle_delete(
        "/api/row-dedup/mask",
        b'{"pairs": [{"task_id": "t", "row_index": 0}]}',
    )
    assert status == 400
    result = json.loads(body.decode("utf-8"))
    assert result["error"] == "project_required"


def test_row_dedup_delete_unknown_route_returns_404(tmp_path):
    store, workspace = _seed_workspace(tmp_path)
    api = DashboardApi(store, workspace_root=workspace)

    status, _headers, body = api.handle_delete(
        "/api/nonexistent", b"{}",
    )
    assert status == 404

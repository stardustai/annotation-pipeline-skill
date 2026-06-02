"""The /api/conventions endpoint has two modes. The default (dashboard table)
is server-paginated and reads only materialized columns: it never parses
proposals_json, so rows come back with proposals=[] plus total/max_count
metadata. full=1 restores the legacy whole-project load WITH the proposals
audit trail — the TaskDrawer opts into it to look up conventions for the
current task's spans. These tests pin both shapes so the dashboard stays light
and the drawer keeps its proposals.
"""
import json

import pytest

from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def api(tmp_path):
    yield DashboardApi(SqliteStore.open(tmp_path))


def _accept(store, task_id, span, etype, project):
    """ACCEPTED task tagging ``span`` as ``etype`` — recount-only backing."""
    from annotation_pipeline_skill.core.models import ArtifactRef, Task
    from annotation_pipeline_skill.core.states import TaskStatus

    task = Task.new(task_id=task_id, pipeline_id=project,
                    source_ref={"kind": "jsonl", "payload": {
                        "rows": [{"row_index": 0, "input": span}]}})
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    rel = f"artifact_payloads/{task_id}/final.json"
    ap = store.root / rel
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text(json.dumps({"text": json.dumps({
        "rows": [{"row_index": 0, "output": {"entities": {etype: [span]}}}]})}))
    store.append_artifact(ArtifactRef.new(
        task_id=task_id, kind="annotation_result", path=rel,
        content_type="application/json"))


def _seed(api, span, n, project_id="p"):
    svc = EntityConventionService(api.store)
    for i in range(n):
        svc.record_decision(project_id=project_id, span=span, entity_type="technology",
                             source="qc_consensus", task_id=f"{span}_t{i}")
        _accept(api.store, f"{span}_t{i}", span, "technology", project_id)


def _recount(api, project_id="p"):
    EntityConventionService(api.store).recount_project(project_id=project_id)


def _get(api, qs):
    status, _headers, body = api.handle_get(f"/api/conventions?{qs}")
    return status, json.loads(body.decode("utf-8"))


def test_requires_project(api):
    status, payload = _get(api, "")
    assert status == 400 and payload["error"] == "project_required"


def test_default_mode_is_paginated_and_proposals_free(api):
    for i in range(25):
        _seed(api, f"span{i:02d}", 1)
    status, payload = _get(api, "project=p&limit=10&offset=0")
    assert status == 200
    assert payload["total"] == 25
    assert payload["limit"] == 10
    assert payload["offset"] == 0
    assert "max_count" in payload
    assert len(payload["conventions"]) == 10
    # Light loader: the table never shows the audit trail.
    assert all(c["proposals"] == [] for c in payload["conventions"])


def test_default_mode_min_count_and_search(api):
    _seed(api, "Android", 6)
    _seed(api, "Solo", 1)
    _recount(api)
    _, by_min = _get(api, "project=p&min_count=5")
    assert by_min["total"] == 1 and by_min["conventions"][0]["span"] == "Android"
    _, by_q = _get(api, "project=p&q=sol")
    assert by_q["total"] == 1 and by_q["conventions"][0]["span"] == "Solo"


def test_full_mode_returns_all_rows_with_proposals(api):
    for i in range(25):
        _seed(api, f"span{i:02d}", 1)
    _seed(api, "Android", 6)
    _recount(api)
    status, payload = _get(api, "project=p&full=1")
    assert status == 200
    # No pagination metadata, no truncation: every convention is returned.
    assert "total" not in payload
    assert len(payload["conventions"]) == 26
    # The proposals audit trail is populated (the drawer needs it).
    android = next(c for c in payload["conventions"] if c["span"] == "Android")
    assert android["proposals"]
    assert android["distinct_task_count"] == 6

import json

from annotation_pipeline_skill.core.models import ArtifactRef, Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _add_accepted_task(store, project_id, task_id, entities_by_type):
    """One ACCEPTED task whose final annotation tags {type: [spans]} in row 0."""
    task = Task.new(
        task_id=task_id, pipeline_id=project_id,
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "x"}]}},
    )
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    rel = f"artifact_payloads/{task_id}/final.json"
    abs_path = store.root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(json.dumps({"text": json.dumps({
        "rows": [{"row_index": 0, "output": {"entities": entities_by_type}}]
    })}))
    store.append_artifact(ArtifactRef.new(
        task_id=task_id, kind="annotation_result", path=rel,
        content_type="application/json",
    ))


def test_recount_fixes_frozen_dominant(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    for i in range(3):
        svc.record_decision(project_id="p", span="vue", entity_type="project",
                            source="qc_consensus", task_id=f"old-{i}")
    assert svc.list_for_project("p")[0].entity_type == "project"
    for i in range(12):
        _add_accepted_task(store, "p", f"vue-{i}", {"technology": ["vue"]})

    summary = svc.recount_project(project_id="p")

    conv = [c for c in svc.list_for_project("p") if c.span_lower == "vue"][0]
    assert conv.entity_type == "technology"
    assert conv.dominant_type == "technology"
    assert conv.distinct_task_count == 12
    assert conv.dispute_count == 0
    assert conv.dispute_pct == 0.0
    assert summary["conventions_seen"] >= 1


def test_recount_preserves_operator_lock(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p", span="edge", entity_type="product",
                        source="declared:operator-1", task_id=None)
    for i in range(3):
        _add_accepted_task(store, "p", f"edge-{i}", {"technology": ["edge"]})

    svc.recount_project(project_id="p")

    conv = [c for c in svc.list_for_project("p") if c.span_lower == "edge"][0]
    assert conv.entity_type == "product"
    assert conv.dominant_type == "technology"
    assert conv.distinct_task_count == 3
    assert conv.created_by.startswith("declared:")
    # 3 < INJECT_MIN_DISTINCT_TASKS(5): the metric gate FAILS, so a successful
    # injection here proves the created_by operator bypass is intact.
    matches = svc.find_matches_in_text("p", "we use edge in prod")
    assert any(m.span_lower == "edge" and m.entity_type == "product" for m in matches)


def test_recount_zeroes_vanished_span_without_deleting(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    for i in range(6):
        svc.record_decision(project_id="p", span="ghost", entity_type="organization",
                            source="qc_consensus", task_id=f"g-{i}")
    _add_accepted_task(store, "p", "other-0", {"technology": ["python"]})

    svc.recount_project(project_id="p")

    rows = {c.span_lower: c for c in svc.list_for_project("p")}
    assert "ghost" in rows
    assert rows["ghost"].distinct_task_count == 0
    assert rows["ghost"].dominant_type is None


def test_recount_one_vote_per_task_for_multi_type_span(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p", span="alpha", entity_type="organization",
                        source="qc_consensus", task_id="seed")
    _add_accepted_task(store, "p", "multi-0",
                       {"technology": ["alpha"], "product": ["alpha"]})

    svc.recount_project(project_id="p")

    conv = [c for c in svc.list_for_project("p") if c.span_lower == "alpha"][0]
    assert conv.distinct_task_count == 1
    assert conv.dispute_count == 0
    assert conv.dispute_pct == 0.0
    assert conv.dominant_type == max(["technology", "product"])

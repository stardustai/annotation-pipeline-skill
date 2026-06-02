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


def _add_arbiter_corrected_task(store, project_id, task_id, entities_by_type):
    """ACCEPTED task whose EFFECTIVE (latest) annotation is an ARBITER OVERRIDE
    (corrected_annotation). No human_review_answer, so it is treated as
    arbiter-ruled and must be EXCLUDED from convention votes."""
    task = Task.new(
        task_id=task_id, pipeline_id=project_id,
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "x"}]}},
    )
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    rel = f"artifact_payloads/{task_id}/{task_id}_arbiter_correction.json"
    abs_path = store.root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(json.dumps({"text": json.dumps({
        "rows": [{"row_index": 0, "output": {"entities": entities_by_type}}]
    })}))
    store.append_artifact(ArtifactRef.new(
        task_id=task_id, kind="annotation_result", path=rel,
        content_type="application/json", metadata={"source": "arbiter_correction"},
    ))


def _add_hr_final_after_arbiter_task(store, project_id, task_id, hr_entities):
    """ACCEPTED task that was arbiter-corrected (project) but then HR-corrected.
    The human_review_answer is the EFFECTIVE annotation, so it is INCLUDED
    (human had the final say — not arbiter-ruled)."""
    task = Task.new(
        task_id=task_id, pipeline_id=project_id,
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "x"}]}},
    )
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    # Earlier arbiter override.
    arb_rel = f"artifact_payloads/{task_id}/{task_id}_arbiter_correction.json"
    (store.root / arb_rel).parent.mkdir(parents=True, exist_ok=True)
    (store.root / arb_rel).write_text(json.dumps({"text": json.dumps({
        "rows": [{"row_index": 0, "output": {"entities": {"project": ["spark"]}}}]})}))
    store.append_artifact(ArtifactRef.new(
        task_id=task_id, kind="annotation_result", path=arb_rel,
        content_type="application/json", metadata={"source": "arbiter_correction"}))
    # Later HR answer (the effective final annotation).
    hr_rel = f"artifact_payloads/{task_id}/hr.json"
    (store.root / hr_rel).write_text(json.dumps({"answer": {
        "rows": [{"row_index": 0, "output": {"entities": hr_entities}}]}}))
    store.append_artifact(ArtifactRef.new(
        task_id=task_id, kind="human_review_answer", path=hr_rel,
        content_type="application/json"))


def test_recount_excludes_arbiter_corrected_tasks(tmp_path):
    """A convention vote is excluded when the task's effective annotation is an
    arbiter OVERRIDE; uncontested / annotator-wins tasks are counted."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p", span="spark", entity_type="technology",
                        source="qc_consensus", task_id="seed")
    # 3 uncontested accepted tasks tag spark=technology -> counted.
    for i in range(3):
        _add_accepted_task(store, "p", f"ok-{i}", {"technology": ["spark"]})
    # 2 arbiter-corrected tasks tag spark=project -> EXCLUDED.
    for i in range(2):
        _add_arbiter_corrected_task(store, "p", f"arb-{i}", {"project": ["spark"]})

    svc.recount_project(project_id="p")

    conv = [c for c in svc.list_for_project("p") if c.span_lower == "spark"][0]
    assert conv.distinct_task_count == 3   # arbiter-corrected votes excluded
    assert conv.dominant_type == "technology"
    assert conv.dispute_count == 0


def test_recount_includes_hr_final_even_if_arbiter_touched(tmp_path):
    """A task that was arbiter-corrected but then HR-corrected counts via its
    HR-final annotation (human had the final say)."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p", span="spark", entity_type="project",
                        source="qc_consensus", task_id="seed")
    _add_hr_final_after_arbiter_task(store, "p", "hr-0", {"technology": ["spark"]})

    svc.recount_project(project_id="p")

    conv = [c for c in svc.list_for_project("p") if c.span_lower == "spark"][0]
    assert conv.distinct_task_count == 1
    assert conv.dominant_type == "technology"   # HR-final answer counted


def test_record_decision_does_not_maintain_empirical_columns(tmp_path):
    """Recount-only: an AUTO (qc_consensus) record_decision appends a proposal
    and bumps evidence_count, but does NOT write the empirical columns or
    auto-derive entity_type. Those stay zeroed until recount_project runs."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)

    c = svc.record_decision(project_id="p", span="kafka", entity_type="technology",
                            source="qc_consensus", task_id="t1")
    assert c.evidence_count == 1
    assert c.distinct_task_count == 0
    assert c.dominant_type is None
    assert c.dispute_pct == 0.0

    c = svc.record_decision(project_id="p", span="kafka", entity_type="project",
                            source="qc_consensus", task_id="t2")
    assert c.evidence_count == 2            # proposals still tracked
    assert c.entity_type == "technology"    # unchanged (NOT re-derived to plurality)
    assert c.distinct_task_count == 0       # still not maintained here
    assert c.dominant_type is None

    c = svc.record_decision(project_id="p", span="kafka", entity_type="product",
                            source="declared:op-9", task_id=None)
    assert c.entity_type == "product"       # operator declaration takes effect
    assert c.created_by.startswith("declared:")


def test_record_decision_then_recount_populates_columns(tmp_path):
    """After auto proposals (which don't maintain columns), recount_project is
    what makes the convention injectable."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    for i in range(5):
        svc.record_decision(project_id="p", span="kafka", entity_type="technology",
                            source="qc_consensus", task_id=f"seed-{i}")
        _add_accepted_task(store, "p", f"kafka-{i}", {"technology": ["kafka"]})
    # Before recount: not injectable (columns zeroed).
    assert svc.find_matches_in_text("p", "we run kafka here") == []
    # After recount: injectable.
    svc.recount_project(project_id="p")
    matches = svc.find_matches_in_text("p", "we run kafka here")
    assert any(m.span_lower == "kafka" and m.entity_type == "technology" for m in matches)

"""The distinct-task / dispute aggregates are materialized as columns and
maintained on every write, so the injection gate is a plain SQL predicate
(no per-row JSON parsing). These tests pin that the columns stay in sync with
what _distinct_task_tally would derive, that the migration is idempotent, and
that the injection path never needs proposals_json.
"""
import json

import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
    _distinct_task_tally,
)
from annotation_pipeline_skill.store.sqlite_store import (
    SqliteStore,
    _migrate_convention_aggregate_columns,
)


@pytest.fixture
def store(tmp_path):
    yield SqliteStore.open(tmp_path)


def _columns(store, span_lower="android"):
    return store._conn.execute(
        "SELECT distinct_task_count, dispute_count, dispute_pct, dominant_type, "
        "proposals_json FROM entity_conventions WHERE span_lower=?",
        (span_lower,),
    ).fetchone()


def _accept(store, task_id, span, etype, project="p"):
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


def test_insert_seeds_zeroed_columns_then_recount_populates(store):
    """Recount-only: record_decision insert seeds zeroed aggregates; only
    recount_project (from accepted-task annotations) populates them."""
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="p", span="Android", entity_type="technology",
        source="qc_consensus", task_id="t1",
    )
    row = _columns(store)
    assert row["distinct_task_count"] == 0
    assert row["dominant_type"] is None
    # Back with an accepted task and recount -> columns populate.
    _accept(store, "t1", "Android", "technology")
    svc.recount_project(project_id="p")
    row = _columns(store)
    assert row["distinct_task_count"] == 1
    assert row["dispute_count"] == 0
    assert row["dispute_pct"] == 0.0
    assert row["dominant_type"] == "technology"


def test_recount_populates_columns_to_match_tally(store):
    """After recount the columns match what _distinct_task_tally derives from
    the (aligned) proposals — recount is the sole maintainer now."""
    svc = EntityConventionService(store)
    # 5 tasks agree technology, 1 disagrees organization.
    for i in range(5):
        svc.record_decision(project_id="p", span="Android", entity_type="technology",
                            source="qc_consensus", task_id=f"t{i}")
        _accept(store, f"t{i}", "Android", "technology")
    svc.record_decision(project_id="p", span="Android", entity_type="organization",
                        source="qc_consensus", task_id="t_dissent")
    _accept(store, "t_dissent", "Android", "organization")
    svc.recount_project(project_id="p")
    row = _columns(store)
    proposals = json.loads(row["proposals_json"])
    dom, dist, disp, pct = _distinct_task_tally(proposals)
    # Columns equal what the tally derives (proposals align with accepted tasks).
    assert row["distinct_task_count"] == dist == 6
    assert row["dispute_count"] == disp == 1
    assert abs(row["dispute_pct"] - pct) < 1e-9
    assert row["dominant_type"] == dom == "technology"


def test_clear_dispute_sets_type_without_maintaining_columns(store):
    """Recount-only: clear_dispute resolves the operator dispute (type + status
    + created_by stamp) but does NOT recompute the empirical columns — those
    stay at their pre-call value until recount_project runs."""
    svc = EntityConventionService(store)
    conv = svc.record_decision(project_id="p", span="Stripe", entity_type="technology",
                              source="qc_consensus", task_id="t1")
    svc.store._conn.execute(
        "UPDATE entity_conventions SET status='disputed' WHERE convention_id=?",
        (conv.convention_id,))
    resolved = svc.clear_dispute(convention_id=conv.convention_id,
                                 resolved_type="organization", actor="alice")
    assert resolved.status == "active"
    assert resolved.entity_type == "organization"
    assert resolved.created_by.startswith("dispute_resolved_by:")
    # Empirical columns unchanged (still the insert's zeroed values).
    assert resolved.distinct_task_count == 0
    assert resolved.dominant_type is None


def test_migration_is_idempotent(tmp_path):
    # First open creates + migrates; second open must not error on ADD COLUMN.
    store = SqliteStore.open(tmp_path)
    EntityConventionService(store).record_decision(
        project_id="p", span="Android", entity_type="technology",
        source="qc_consensus", task_id="t1",
    )
    store.close()
    store2 = SqliteStore.open(tmp_path)  # re-open: migration runs again
    # Calling the migration directly a third time is still a no-op.
    _migrate_convention_aggregate_columns(store2._conn)
    cols = {r[1] for r in store2._conn.execute("PRAGMA table_info(entity_conventions)")}
    assert {"distinct_task_count", "dispute_count", "dispute_pct", "dominant_type"} <= cols


def test_injection_path_does_not_read_proposals_json(store):
    """_iter_injection_candidates builds from columns only — returned
    conventions carry no proposals, proving proposals_json wasn't parsed."""
    svc = EntityConventionService(store)
    for i in range(6):
        svc.record_decision(project_id="p", span="Android", entity_type="technology",
                            source="qc_consensus", task_id=f"t{i}")
        _accept(store, f"t{i}", "Android", "technology")
    svc.recount_project(project_id="p")
    cands = svc._iter_injection_candidates("p")
    assert len(cands) == 1
    c = cands[0]
    assert c.proposals == []                 # not loaded
    assert c.distinct_task_count == 6         # from column
    assert c.entity_type == "technology"
    # And it injects.
    assert "android" in {m.span_lower for m in
                         svc.find_matches_in_text("p", "my Android phone")}


def test_recount_backfills_columns_not_silently_zero(store):
    """After recount a convention has non-default columns — guards against the
    columns silently staying at their DEFAULT 0."""
    svc = EntityConventionService(store)
    for i in range(7):
        svc.record_decision(project_id="p", span="Equifax", entity_type="organization",
                            source="qc_consensus", task_id=f"t{i}")
        _accept(store, f"t{i}", "Equifax", "organization")
    svc.recount_project(project_id="p")
    row = _columns(store, "equifax")
    assert row["distinct_task_count"] == 7  # not the DEFAULT 0

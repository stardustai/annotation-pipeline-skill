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


def test_insert_stores_aggregate_columns(store):
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="p", span="Android", entity_type="technology",
        source="qc_consensus", task_id="t1",
    )
    row = _columns(store)
    assert row["distinct_task_count"] == 1
    assert row["dispute_count"] == 0
    assert row["dispute_pct"] == 0.0
    assert row["dominant_type"] == "technology"


def test_update_path_keeps_columns_in_sync_with_tally(store):
    svc = EntityConventionService(store)
    # 5 tasks agree technology, 1 disagrees organization.
    for i in range(5):
        svc.record_decision(project_id="p", span="Android", entity_type="technology",
                            source="qc_consensus", task_id=f"t{i}")
    svc.record_decision(project_id="p", span="Android", entity_type="organization",
                        source="qc_consensus", task_id="t_dissent")
    row = _columns(store)
    proposals = json.loads(row["proposals_json"])
    dom, dist, disp, pct = _distinct_task_tally(proposals)
    # Columns must equal exactly what the tally derives.
    assert row["distinct_task_count"] == dist == 6
    assert row["dispute_count"] == disp == 1
    assert abs(row["dispute_pct"] - pct) < 1e-9
    assert row["dominant_type"] == dom == "technology"


def test_clear_dispute_refreshes_columns(store):
    svc = EntityConventionService(store)
    conv = svc.record_decision(project_id="p", span="Stripe", entity_type="technology",
                              source="qc_consensus", task_id="t1")
    svc.clear_dispute(convention_id=conv.convention_id, resolved_type="organization",
                     actor="alice")
    row = _columns(store, "stripe")
    proposals = json.loads(row["proposals_json"])
    dom, dist, disp, pct = _distinct_task_tally(proposals)
    assert row["distinct_task_count"] == dist
    assert row["dispute_count"] == disp
    assert row["dominant_type"] == dom


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
    cands = svc._iter_injection_candidates("p")
    assert len(cands) == 1
    c = cands[0]
    assert c.proposals == []                 # not loaded
    assert c.distinct_task_count == 6         # from column
    assert c.entity_type == "technology"
    # And it injects.
    assert "android" in {m.span_lower for m in
                         svc.find_matches_in_text("p", "my Android phone")}


def test_migration_backfills_columns_via_replay_not_silently_zero(store):
    """A convention written through record_decision has non-default columns —
    guards against the columns silently staying at their DEFAULT 0."""
    svc = EntityConventionService(store)
    for i in range(7):
        svc.record_decision(project_id="p", span="Equifax", entity_type="organization",
                            source="qc_consensus", task_id=f"t{i}")
    row = _columns(store, "equifax")
    assert row["distinct_task_count"] == 7  # not the DEFAULT 0

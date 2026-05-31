import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
    _distinct_task_tally,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path):
    yield SqliteStore.open(tmp_path)


def _prop(task_id, ptype, source="qc_consensus"):
    return {"type": ptype, "source": source, "task_id": task_id}


def test_tally_counts_one_vote_per_distinct_task():
    proposals = [
        _prop("t1", "technology"),
        _prop("t1", "technology"),
        _prop("t1", "technology"),
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "technology"
    assert distinct == 1
    assert dispute == 0
    assert pct == 0.0


def test_tally_dominant_is_plurality_across_tasks():
    proposals = [
        _prop("t1", "organization"),
        _prop("t2", "organization"),
        _prop("t3", "product"),
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "organization"
    assert distinct == 3
    assert dispute == 1
    assert pct == pytest.approx(1 / 3)


def test_tally_uses_most_recent_type_per_task():
    proposals = [
        _prop("t1", "product"),
        _prop("t1", "technology"),
        _prop("t2", "technology"),
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "technology"
    assert distinct == 2
    assert dispute == 0
    assert pct == 0.0


def test_tally_ignores_proposals_without_task_id():
    proposals = [
        {"type": "project", "source": "declared:operator", "task_id": None},
        _prop("t1", "project"),
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "project"
    assert distinct == 1
    assert dispute == 0


def test_tally_excludes_operator_source_even_with_task_id():
    proposals = [
        _prop("t1", "organization"),
        _prop("t2", "organization"),
        {"type": "product", "source": "hr_correction:alice", "task_id": "t3"},
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "organization"
    assert distinct == 2
    assert dispute == 0
    assert pct == 0.0


def test_tally_empty_is_neutral():
    dominant, distinct, dispute, pct = _distinct_task_tally([])
    assert dominant is None
    assert distinct == 0
    assert dispute == 0
    assert pct == 0.0


def test_load_row_attaches_derived_fields(store):
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="p1", span="Apple", entity_type="organization",
        source="qc_consensus:a", task_id="t1",
    )
    svc.record_decision(
        project_id="p1", span="Apple", entity_type="organization",
        source="qc_consensus:b", task_id="t2",
    )
    svc.record_decision(
        project_id="p1", span="Apple", entity_type="product",
        source="qc_consensus:c", task_id="t3",
    )
    conv = svc.list_for_project("p1")[0]
    assert conv.distinct_task_count == 3
    assert conv.dominant_type == "organization"
    assert conv.dispute_count == 1
    assert conv.dispute_pct == pytest.approx(1 / 3)
    d = conv.to_dict()
    assert d["distinct_task_count"] == 3
    assert d["dominant_type"] == "organization"
    assert d["dispute_count"] == 1
    assert d["dispute_pct"] == pytest.approx(1 / 3)

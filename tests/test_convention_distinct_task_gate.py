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


def _proposals(store):
    import json
    rows = list(store._conn.execute("SELECT proposals_json FROM entity_conventions"))
    return json.loads(rows[0][0])


def test_same_task_same_source_type_is_noop(store):
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")  # exact repeat → no-op
    assert len(_proposals(store)) == 1


def test_different_task_same_source_type_is_recorded(store):
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t2")  # new task → new vote
    proposals = _proposals(store)
    assert len(proposals) == 2
    conv = svc.list_for_project("p1")[0]
    assert conv.distinct_task_count == 2


def test_conflict_does_not_flip_to_disputed_soft_model(store):
    svc = EntityConventionService(store)
    # Two tasks say organization, one says product → dominant=organization.
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t2")
    conv = svc.record_decision(project_id="p1", span="Apple", entity_type="product",
                              source="qc_consensus", task_id="t3")
    # Soft model: stays active, entity_type tracks the plurality winner.
    assert conv.status == "active"
    assert conv.entity_type == "organization"
    assert conv.dominant_type == "organization"
    assert conv.dispute_count == 1


def test_evidence_count_tracks_total_proposals(store):
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    conv = svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                              source="qc_consensus", task_id="t2")
    assert conv.evidence_count == 2  # two recorded proposals


def test_operator_declaration_still_wins(store):
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    conv = svc.record_decision(project_id="p1", span="Apple", entity_type="product",
                              source="declared:operator")
    # Operator declaration is final authority: type locks to declared value.
    assert conv.status == "active"
    assert conv.entity_type == "product"


def _seed_votes(svc, span, etype, n, project="p1", start=0):
    for i in range(start, start + n):
        svc.record_decision(project_id=project, span=span, entity_type=etype,
                            source="qc_consensus", task_id=f"t{i}",
                            row_content=f"{span} appears here {i}")


def test_injection_requires_five_distinct_tasks(store):
    svc = EntityConventionService(store)
    _seed_votes(svc, "Salesforce", "organization", 4)  # 4 < 5 distinct tasks
    assert svc.find_matches_in_text("p1", "We use Salesforce daily") == []
    _seed_votes(svc, "Salesforce", "organization", 1, start=4)  # now 5
    matches = svc.find_matches_in_text("p1", "We use Salesforce daily")
    assert [c.span_original for c in matches] == ["Salesforce"]


def test_injection_blocked_when_dispute_pct_too_high(store):
    svc = EntityConventionService(store)
    # 6 distinct tasks, 2 dissent → dispute_pct = 2/6 = 0.333 >= 0.20 → blocked.
    _seed_votes(svc, "Mercury", "organization", 4)
    _seed_votes(svc, "Mercury", "product", 2, start=4)
    conv = svc.list_for_project("p1")[0]
    assert conv.distinct_task_count == 6
    assert conv.dispute_pct >= 0.20
    assert svc.find_matches_in_text("p1", "Mercury launched a probe") == []


def test_injection_allowed_when_dispute_pct_under_threshold(store):
    svc = EntityConventionService(store)
    # 10 distinct tasks, 1 dissent → dispute_pct = 0.10 < 0.20 → injected.
    _seed_votes(svc, "Mercury", "organization", 9)
    _seed_votes(svc, "Mercury", "product", 1, start=9)
    conv = svc.list_for_project("p1")[0]
    assert conv.distinct_task_count == 10
    assert conv.dispute_pct < 0.20
    matches = svc.find_matches_in_text("p1", "Mercury launched a probe")
    assert [c.span_original for c in matches] == ["Mercury"]


def test_operator_declared_bypasses_distinct_task_gate(store):
    svc = EntityConventionService(store)
    # One operator declaration, zero distinct task votes → still injected.
    svc.record_decision(project_id="p1", span="Gmail", entity_type="project",
                        source="declared:operator", row_content="Gmail filters")
    matches = svc.find_matches_in_text("p1", "I set up a Gmail filter")
    assert [c.span_original for c in matches] == ["Gmail"]

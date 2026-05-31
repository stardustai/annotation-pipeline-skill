"""Tests for rebuilding the entity-convention table from accepted task
annotations under the distinct-task voting model.

Rationale: the live ``proposals_json`` is lossy (the old ``(type, source)``
dedup key suppressed cross-task votes), so the convention table cannot be
re-derived from itself. The recoverable source of truth is each accepted
task's final annotation. ``rebuild_from_accepted_tasks`` clears a project's
conventions and replays every (span, type) decision as a ``qc_consensus``
vote keyed by task_id — exactly the three-party-confirmed datapoint the new
``_distinct_task_tally`` counts.
"""
import json

import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
    extract_all_span_decisions_with_row,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path):
    yield SqliteStore.open(tmp_path)


def _ann(*entities_by_row):
    """Build a minimal annotation payload.

    Each positional arg is (row_id, content, {type: [span, ...]}).
    """
    rows = []
    for i, (row_id, content, entities) in enumerate(entities_by_row):
        rows.append({
            "row_id": row_id,
            "row_index": i,
            "content": content,
            "output": {"entities": entities},
        })
    return {"rows": rows}


# --- extract_all_span_decisions_with_row ------------------------------------


def test_extract_all_returns_every_span_with_row_info():
    payload = _ann(
        ("r1", "Crashes on Android 10", {"technology": ["Android"]}),
        ("r2", "PicsArt edits missing", {"technology": ["PicsArt"]}),
    )
    out = extract_all_span_decisions_with_row(payload)
    assert ("Android", "technology", "r1", "Crashes on Android 10") in out
    assert ("PicsArt", "technology", "r2", "PicsArt edits missing") in out


def test_extract_all_dedups_same_span_type_across_rows():
    payload = _ann(
        ("r1", "Android crashes", {"technology": ["Android"]}),
        ("r2", "Android again", {"technology": ["Android"]}),
    )
    out = extract_all_span_decisions_with_row(payload)
    androids = [d for d in out if d[0] == "Android"]
    assert len(androids) == 1
    # Carries the FIRST row's content/id.
    assert androids[0][2] == "r1"


def test_extract_all_walks_json_structures_too():
    payload = {
        "rows": [{
            "row_id": "r1",
            "row_index": 0,
            "content": "lorem",
            "output": {"json_structures": {"phrase": ["battery life"]}},
        }]
    }
    out = extract_all_span_decisions_with_row(payload)
    assert out == [("battery life", "phrase", "r1", "lorem")]


def test_extract_all_handles_non_dict_payload():
    assert extract_all_span_decisions_with_row(None) == []
    assert extract_all_span_decisions_with_row("nope") == []


# --- rebuild_from_accepted_tasks --------------------------------------------


def test_rebuild_accumulates_distinct_task_votes(store):
    svc = EntityConventionService(store)
    payloads = {
        f"task_{i}": _ann((f"r{i}", "Crashes on Android", {"technology": ["Android"]}))
        for i in range(6)
    }
    summary = svc.rebuild_from_accepted_tasks(
        project_id="v4",
        task_ids=list(payloads),
        annotation_loader=payloads.get,
    )
    assert summary["tasks_seen"] == 6
    assert summary["decisions_recorded"] == 6
    conv = svc.list_for_project("v4")[0]
    assert conv.span_original == "Android"
    assert conv.entity_type == "technology"
    # Six distinct tasks each voted once → injection-eligible (>= 5, 0 dispute).
    assert conv.distinct_task_count == 6
    assert conv.dispute_pct == 0.0


def test_rebuild_clears_existing_conventions_first(store):
    svc = EntityConventionService(store)
    # Pre-seed a stale convention that should be wiped by the rebuild.
    svc.record_decision(
        project_id="v4", span="Ghost", entity_type="person",
        source="qc_consensus", task_id="old_task",
    )
    payloads = {"task_1": _ann(("r1", "Android phone", {"technology": ["Android"]}))}
    svc.rebuild_from_accepted_tasks(
        project_id="v4", task_ids=list(payloads), annotation_loader=payloads.get,
    )
    spans = {c.span_original for c in svc.list_for_project("v4")}
    assert "Ghost" not in spans
    assert "Android" in spans


def test_rebuild_is_idempotent(store):
    svc = EntityConventionService(store)
    payloads = {
        f"t{i}": _ann((f"r{i}", "Android", {"technology": ["Android"]}))
        for i in range(3)
    }
    svc.rebuild_from_accepted_tasks(
        project_id="v4", task_ids=list(payloads), annotation_loader=payloads.get,
    )
    svc.rebuild_from_accepted_tasks(
        project_id="v4", task_ids=list(payloads), annotation_loader=payloads.get,
    )
    convs = svc.list_for_project("v4")
    assert len(convs) == 1
    assert convs[0].distinct_task_count == 3


def test_rebuild_skips_tasks_with_no_loadable_annotation(store):
    svc = EntityConventionService(store)
    payloads = {"good": _ann(("r1", "Android", {"technology": ["Android"]}))}
    summary = svc.rebuild_from_accepted_tasks(
        project_id="v4",
        task_ids=["good", "missing"],
        annotation_loader=payloads.get,  # returns None for "missing"
    )
    assert summary["tasks_seen"] == 2
    assert summary["tasks_with_spans"] == 1
    assert len(svc.list_for_project("v4")) == 1


def test_rebuild_records_qc_consensus_source_with_task_id(store):
    svc = EntityConventionService(store)
    payloads = {"task_42": _ann(("r1", "Android", {"technology": ["Android"]}))}
    svc.rebuild_from_accepted_tasks(
        project_id="v4", task_ids=["task_42"], annotation_loader=payloads.get,
    )
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    assert proposals[0]["source"] == "qc_consensus"
    assert proposals[0]["task_id"] == "task_42"

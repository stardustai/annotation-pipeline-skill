import json
from pathlib import Path

import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path):
    yield SqliteStore.open(tmp_path)


def test_record_decision_persists_row_id_in_proposal(store):
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="qc_consensus",
        task_id="task_019",
        row_id="row_18452",
        row_content="Crashes on Android 10 sometimes",
    )
    rows = list(store._conn.execute("SELECT proposals_json FROM entity_conventions"))
    proposals = json.loads(rows[0][0])
    assert proposals[0]["row_id"] == "row_18452"


def test_record_decision_persists_context_snippet_when_row_content_given(store):
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="qc_consensus",
        task_id="task_019",
        row_id="row_18452",
        row_content="The app keeps crashing on my Android phone every few hours",
    )
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    snippet = proposals[0]["context_snippet"]
    assert snippet is not None
    assert "Android" in snippet


def test_record_decision_without_row_content_leaves_snippet_none(store):
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="declared:operator",
    )
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    assert proposals[0]["row_id"] is None
    assert proposals[0].get("context_snippet") is None


def test_snippet_window_truncates_long_rows(store):
    svc = EntityConventionService(store)
    long_row = "padding " * 50 + "Android" + " padding" * 50  # ~750 chars
    svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="qc_consensus",
        task_id="task_019",
        row_id="row_99",
        row_content=long_row,
    )
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    snippet = proposals[0]["context_snippet"]
    # Window is span ± 80 chars → max 200 chars including the span itself.
    assert len(snippet) <= 200
    assert "Android" in snippet
    # The snippet was truncated, so the leading/trailing "padding" tokens
    # past the window should be absent.
    assert snippet.count("padding") < long_row.count("padding")


def test_legacy_call_signature_still_works(store):
    """Existing call sites that don't pass row_id/row_content must keep working."""
    svc = EntityConventionService(store)
    conv = svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="qc_consensus",
        task_id="task_019",
    )
    assert conv.entity_type == "technology"
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    assert proposals[0]["row_id"] is None
    assert proposals[0].get("context_snippet") is None


def test_snippet_falls_back_to_head_window_when_span_not_found(store):
    """If span text doesn't appear verbatim in row_content (e.g.,
    normalization mismatch), the snippet should fall back to a head
    window of row_content with a trailing ellipsis when truncated."""
    svc = EntityConventionService(store)
    long_row = "completely unrelated content " * 20  # ~580 chars, no "Android"
    svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="qc_consensus",
        task_id="task_019",
        row_id="row_99",
        row_content=long_row,
    )
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    snippet = proposals[0]["context_snippet"]
    assert snippet is not None  # the docstring promises this
    assert snippet.startswith("completely unrelated content")
    assert snippet.endswith("…")  # trailing ellipsis since row > 160 chars

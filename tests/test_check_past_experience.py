import itertools
import json
import tempfile
from pathlib import Path

import pytest

from annotation_pipeline_skill.llm.tools.check_past_experience import check_past_experience
from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = SqliteStore.open(Path(tmpdir))
        yield s


_seed_counter = itertools.count(1)


def _seed(svc, project, span, etype, source, task_id, row_id, row_content):
    # Suffix source per-call so the service's same-source idempotency
    # guard doesn't collapse what the tests intend as N distinct events.
    # "declared:operator:seedN" still starts with "declared:" so the
    # service's operator-declaration branch fires correctly.
    unique_source = f"{source}:seed{next(_seed_counter)}"
    svc.record_decision(
        project_id=project, span=span, entity_type=etype,
        source=unique_source, task_id=task_id, row_id=row_id, row_content=row_content,
    )


def test_unknown_span_returns_none_shape(store):
    result = check_past_experience(store, project_id="p1", entry="NeverSeen")
    assert result["entry"] == "NeverSeen"
    assert result["convention"]["status"] == "none"
    assert result["convention"]["evidence_count"] == 0
    assert result["distribution"] == {}
    assert result["examples_by_type"] == {}
    assert "wordfreq_zipf" in result["meta"]


def test_active_convention_returns_examples(store):
    svc = EntityConventionService(store)
    for i in range(5):
        _seed(svc, "p1", "Android", "technology", "qc_consensus",
              f"task_{i}", f"row_{i}", f"Crashes on Android 10 ({i})")
    result = check_past_experience(store, project_id="p1", entry="Android")
    assert result["convention"]["status"] == "active"
    assert result["convention"]["type"] == "technology"
    assert result["convention"]["evidence_count"] == 5
    assert result["distribution"] == {"technology": 5}
    assert "technology" in result["examples_by_type"]
    # ≤ 3 examples per type.
    assert len(result["examples_by_type"]["technology"]) <= 3
    # Examples carry trace prefix.
    assert all(
        s.startswith("[task_") and "/row_" in s
        for s in result["examples_by_type"]["technology"]
    )


def test_conflicting_proposals_track_dispute_soft_model(store):
    svc = EntityConventionService(store)
    _seed(svc, "p1", "Apple", "organization", "qc_consensus",
          "task_a", "row_1", "Apple's customer support helped me yesterday")
    _seed(svc, "p1", "Apple", "organization", "qc_consensus",
          "task_b", "row_2", "Apple announced a new privacy policy")
    _seed(svc, "p1", "Apple", "product", "qc_consensus",
          "task_c", "row_3", "My Apple iPad keeps crashing on updates")
    result = check_past_experience(store, project_id="p1", entry="Apple")
    # Soft model: stays active, plurality (organization) wins.
    assert result["convention"]["status"] == "active"
    assert result["convention"]["type"] == "organization"
    assert result["convention"]["dominant_type"] == "organization"
    assert result["convention"]["distinct_task_count"] == 3
    assert result["convention"]["dispute_count"] == 1
    assert result["convention"]["dispute_pct"] == pytest.approx(1 / 3)
    # Distribution still counts every proposal by its declared type.
    assert result["distribution"] == {"organization": 2, "product": 1}
    assert set(result["examples_by_type"].keys()) == {"organization", "product"}


def test_skips_proposals_without_context_snippet(store):
    """Operator declarations have no row → no example available; they
    still count toward evidence and distribution, but examples_by_type
    only contains the buckets that DO have snippets."""
    svc = EntityConventionService(store)
    # Operator declared, no row_content → no snippet.
    _seed(svc, "p1", "Apple", "organization", "declared:operator",
          None, None, None)
    # QC consensus with row_content → snippet exists.
    _seed(svc, "p1", "Apple", "organization", "qc_consensus",
          "task_a", "row_1", "Apple's customer support helped me")
    result = check_past_experience(store, project_id="p1", entry="Apple")
    assert result["distribution"]["organization"] == 2
    # Only one snippet → only one example.
    assert len(result["examples_by_type"]["organization"]) == 1


def test_generic_word_flag_for_high_freq_low_evidence(store):
    """'the' has Zipf ~7+ but no evidence → generic_word should be True."""
    result = check_past_experience(store, project_id="p1", entry="the")
    assert result["meta"]["wordfreq_zipf"] > 5.0
    assert result["meta"]["generic_word"] is True


def test_generic_word_flag_false_when_evidence_count_high(store):
    svc = EntityConventionService(store)
    for i in range(6):
        _seed(svc, "p1", "the", "project", "declared:operator", None, None, None)
    result = check_past_experience(store, project_id="p1", entry="the")
    # Still high zipf, but evidence_count >= 5 → don't flag as generic.
    assert result["meta"]["generic_word"] is False


def test_empty_entry_returns_error(store):
    with pytest.raises(ValueError):
        check_past_experience(store, project_id="p1", entry="")

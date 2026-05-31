"""The injection hot path must not pull the long tail of single-task
conventions into Python on every call.

After rebuilding from task history, a project can hold tens of thousands of
single-task conventions of which <3% can ever pass the injection gate. The
prefilter narrows the candidate set in SQL (cheap, C-level predicates) so
only gate-eligible rows get JSON-parsed/tallied in Python. It must be a
strict SUPERSET of what the gate accepts — never dropping a convention the
full scan would have injected.
"""
import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path):
    yield SqliteStore.open(tmp_path)


def _seed_eligible(svc, project_id, span, entity_type, n_tasks):
    for i in range(n_tasks):
        svc.record_decision(
            project_id=project_id, span=span, entity_type=entity_type,
            source="qc_consensus", task_id=f"{span}_t{i}",
        )


def test_prefilter_drops_singletons_but_keeps_eligible(store):
    svc = EntityConventionService(store)
    # One eligible convention (6 distinct tasks) ...
    _seed_eligible(svc, "p", "Android", "technology", 6)
    # ... and a long tail of single-task conventions that can never inject.
    for i in range(20):
        svc.record_decision(
            project_id="p", span=f"longtail{i}", entity_type="technology",
            source="qc_consensus", task_id=f"solo_{i}",
        )
    candidates = svc._iter_injection_candidates("p")
    spans = {c.span_lower for c in candidates}
    assert "android" in spans
    # The 20 singletons are filtered out in SQL.
    assert not any(s.startswith("longtail") for s in spans)


def test_prefilter_keeps_operator_declared_low_evidence(store):
    """An operator declaration injects even with a single proposal — the
    prefilter must not exclude it on the evidence_count branch."""
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="p", span="Acme", entity_type="organization",
        source="declared:operator",
    )
    candidates = svc._iter_injection_candidates("p")
    assert "acme" in {c.span_lower for c in candidates}
    # And it actually injects through the full path.
    matches = svc.find_matches_in_text("p", "I filed a complaint about Acme today")
    assert "acme" in {c.span_lower for c in matches}


def test_prefilter_catches_operator_declared_on_low_evidence_auto_span(store):
    """The hard case: a span first seen via auto consensus (few votes), then
    an operator declares on it. evidence_count stays < threshold, but the
    operator stamp on created_by must keep it in the candidate set."""
    svc = EntityConventionService(store)
    # 2 auto votes — below the distinct-task threshold on its own.
    svc.record_decision(
        project_id="p", span="Klarna", entity_type="technology",
        source="qc_consensus", task_id="a1",
    )
    svc.record_decision(
        project_id="p", span="Klarna", entity_type="technology",
        source="qc_consensus", task_id="a2",
    )
    # Operator overrides — declares it an organization.
    svc.record_decision(
        project_id="p", span="Klarna", entity_type="organization",
        source="declared:operator",
    )
    candidates = svc._iter_injection_candidates("p")
    assert "klarna" in {c.span_lower for c in candidates}
    matches = svc.find_matches_in_text("p", "Klarna offers buy-now-pay-later")
    klarna = next(c for c in matches if c.span_lower == "klarna")
    assert klarna.entity_type == "organization"  # operator wins


def test_prefilter_matches_full_scan_injection_result(store):
    """find_matches_in_text (prefiltered) must equal a brute-force apply of
    the gate over every convention in the project."""
    svc = EntityConventionService(store)
    _seed_eligible(svc, "p", "Android", "technology", 6)       # eligible
    _seed_eligible(svc, "p", "Google", "technology", 4)        # below threshold
    svc.record_decision(                                       # operator bypass
        project_id="p", span="Equifax", entity_type="organization",
        source="declared:operator",
    )
    for i in range(15):
        svc.record_decision(
            project_id="p", span=f"noise{i}word", entity_type="technology",
            source="qc_consensus", task_id=f"n_{i}",
        )
    text = "Android and Google and Equifax and noise3word appear here"

    # Brute-force expected: apply the same gate over the FULL list.
    expected = set()
    for conv in svc.list_for_project("p", include_disputed=False):
        if len(conv.span_lower) < svc.MIN_INJECTION_SPAN_LEN:
            continue
        if not svc._is_operator_declared(conv):
            if conv.distinct_task_count < svc.INJECT_MIN_DISTINCT_TASKS:
                continue
            if conv.dispute_pct >= svc.INJECT_MAX_DISPUTE_PCT:
                continue
        if conv.entity_type in svc.EXCLUDED_TYPES_FOR_INJECTION:
            continue
        if conv.span_lower in text.lower():
            expected.add(conv.span_lower)

    got = {c.span_lower for c in svc.find_matches_in_text("p", text)}
    assert got == expected
    assert "android" in got and "equifax" in got
    assert "google" not in got  # below distinct-task threshold

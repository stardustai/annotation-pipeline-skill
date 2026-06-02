"""The dashboard conventions table is server-paginated: list_for_project_page
pushes limit/offset/min_count/search into SQL and reads only the materialized
columns (never proposals_json). These tests pin pagination math, each filter,
the total/max_count metadata, and that the proposals audit trail is NOT loaded
(the table never shows it, and parsing it for tens of thousands of rows was the
original ~45MB / multi-second bottleneck).
"""
import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def svc(tmp_path):
    yield EntityConventionService(SqliteStore.open(tmp_path))


def _accept(store, task_id, span, etype, project):
    """ACCEPTED task tagging ``span`` as ``etype`` — recount-only backing."""
    import json

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


def _seed(svc, span, entity_type, n_tasks, project_id="p"):
    """Record qc_consensus proposals AND back each with an ACCEPTED task; the
    caller runs svc.recount_project(project_id) to populate the columns."""
    for i in range(n_tasks):
        svc.record_decision(
            project_id=project_id, span=span, entity_type=entity_type,
            source="qc_consensus", task_id=f"{span}_t{i}",
        )
        _accept(svc.store, f"{span}_t{i}", span, entity_type, project_id)


def test_pagination_slices_and_reports_total(svc):
    # 25 distinct spans, each a singleton convention.
    for i in range(25):
        svc.record_decision(project_id="p", span=f"span{i:02d}", entity_type="technology",
                            source="qc_consensus", task_id=f"t{i}")
    rows, total, _max = svc.list_for_project_page("p", limit=10, offset=0)
    assert total == 25
    assert len(rows) == 10
    rows2, total2, _ = svc.list_for_project_page("p", limit=10, offset=20)
    assert total2 == 25
    assert len(rows2) == 5  # last partial page
    # No overlap between page 1 and page 3.
    assert {r.convention_id for r in rows}.isdisjoint({r.convention_id for r in rows2})


def test_min_count_filter_pushed_to_sql(svc):
    _seed(svc, "Android", "technology", 6)   # 6 distinct tasks
    _seed(svc, "Google", "technology", 3)    # 3 distinct tasks
    svc.record_decision(project_id="p", span="Solo", entity_type="technology",
                        source="qc_consensus", task_id="s1")  # 1
    svc.recount_project(project_id="p")
    rows, total, _ = svc.list_for_project_page("p", min_count=5)
    spans = {r.span_lower for r in rows}
    assert total == 1
    assert spans == {"android"}


def test_search_matches_span_and_type(svc):
    _seed(svc, "Equifax", "organization", 2)
    _seed(svc, "Android", "technology", 2)
    # Match by span substring.
    rows, total, _ = svc.list_for_project_page("p", search="equi")
    assert total == 1 and rows[0].span_lower == "equifax"
    # Match by entity_type substring.
    rows, total, _ = svc.list_for_project_page("p", search="organ")
    assert total == 1 and rows[0].span_lower == "equifax"


def test_search_escapes_like_wildcards(svc):
    # A literal '%' in a span must not act as a wildcard.
    svc.record_decision(project_id="p", span="50% off", entity_type="technology",
                        source="qc_consensus", task_id="t1")
    svc.record_decision(project_id="p", span="plain", entity_type="technology",
                        source="qc_consensus", task_id="t2")
    rows, total, _ = svc.list_for_project_page("p", search="%")
    assert total == 1 and rows[0].span_lower == "50% off"


def test_max_count_is_global_not_filtered(svc):
    _seed(svc, "Android", "technology", 6)
    _seed(svc, "Google", "technology", 3)
    svc.recount_project(project_id="p")
    # Even when the filter narrows results, max_count reflects the project max.
    rows, total, max_count = svc.list_for_project_page("p", min_count=5)
    assert total == 1
    assert max_count == 6


def test_rows_carry_no_proposals(svc):
    _seed(svc, "Android", "technology", 6)
    svc.recount_project(project_id="p")
    rows, _total, _max = svc.list_for_project_page("p")
    assert rows[0].proposals == []                 # proposals_json never parsed
    assert rows[0].distinct_task_count == 6         # from materialized column


def test_ordered_by_distinct_task_count_desc(svc):
    _seed(svc, "Low", "technology", 2)
    _seed(svc, "High", "technology", 9)
    _seed(svc, "Mid", "technology", 5)
    svc.recount_project(project_id="p")
    rows, _total, _max = svc.list_for_project_page("p")
    counts = [r.distinct_task_count for r in rows]
    assert counts == sorted(counts, reverse=True)
    assert rows[0].span_lower == "high"

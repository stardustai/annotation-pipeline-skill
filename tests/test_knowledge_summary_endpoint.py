"""The /api/knowledge-summary endpoint is the Entity Knowledge panel's cheap
change-signal. Instead of auto-refetching the heavy paginated conventions /
statistics tables, the panel polls this and lights up Refresh when the
fingerprint moves. The fingerprint is (count, latest_updated_at) per subtab:
conventions count = #convention rows for the project; statistics count =
#distinct spans (matching the panel's span-level pagination). These tests pin
that contract and the per-project scoping.
"""
import json

import pytest

from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.services.entity_statistics_service import (
    EntityStatisticsService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def api(tmp_path):
    yield DashboardApi(SqliteStore.open(tmp_path))


def _get(api, qs):
    status, _headers, body = api.handle_get(f"/api/knowledge-summary?{qs}")
    return status, json.loads(body.decode("utf-8"))


def test_requires_project(api):
    status, payload = _get(api, "")
    assert status == 400 and payload["error"] == "project_required"


def test_empty_project_reports_zero_and_null_timestamp(api):
    status, payload = _get(api, "project=p")
    assert status == 200
    assert payload["conventions"] == {"count": 0, "latest_updated_at": None}
    assert payload["statistics"] == {"count": 0, "latest_updated_at": None}


def test_counts_conventions_and_distinct_stat_spans_scoped_by_project(api):
    conv = EntityConventionService(api.store)
    stats = EntityStatisticsService(api.store)
    # Project p: 2 conventions; 2 distinct stat spans (one span has 2 types,
    # so distinct-span count must be 2, NOT the 3 underlying rows).
    conv.record_decision(project_id="p", span="Android", entity_type="technology",
                         source="qc_consensus", task_id="t1")
    conv.record_decision(project_id="p", span="Solo", entity_type="technology",
                         source="qc_consensus", task_id="t2")
    stats.increment(project_id="p", span="Android", entity_type="technology")
    stats.increment(project_id="p", span="Android", entity_type="product")
    stats.increment(project_id="p", span="Solo", entity_type="technology")
    # Other project must not leak into p's counts.
    conv.record_decision(project_id="other", span="Zeta", entity_type="technology",
                         source="qc_consensus", task_id="t3")
    stats.increment(project_id="other", span="Zeta", entity_type="technology")

    status, payload = _get(api, "project=p")
    assert status == 200
    assert payload["conventions"]["count"] == 2
    assert payload["statistics"]["count"] == 2
    assert payload["conventions"]["latest_updated_at"] is not None
    assert payload["statistics"]["latest_updated_at"] is not None


def test_fingerprint_moves_when_a_convention_is_added(api):
    conv = EntityConventionService(api.store)
    conv.record_decision(project_id="p", span="Android", entity_type="technology",
                         source="qc_consensus", task_id="t1")
    _, before = _get(api, "project=p")
    conv.record_decision(project_id="p", span="Kotlin", entity_type="technology",
                         source="qc_consensus", task_id="t2")
    _, after = _get(api, "project=p")
    assert after["conventions"]["count"] == before["conventions"]["count"] + 1

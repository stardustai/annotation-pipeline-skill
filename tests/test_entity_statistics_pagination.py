"""The Statistics subtab is server-paginated, mirroring /api/conventions: the
endpoint pushes limit/offset/search into SQL and paginates at the SPAN level
(one row per span, aggregating its per-type counts), so a project with tens of
thousands of spans ships one ~10KB page instead of a ~6MB whole-table blob.
These tests pin the span-level pagination math, the total, search across both
span text and entity type, LIKE-wildcard escaping, and that a type match keeps
the span's FULL distribution (not just the matching-type row).
"""
import json

import pytest

from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.services.entity_statistics_service import (
    EntityStatisticsService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def api(tmp_path):
    yield DashboardApi(SqliteStore.open(tmp_path))


def _stats(api):
    return EntityStatisticsService(api.store)


def _get(api, qs):
    status, _headers, body = api.handle_get(f"/api/entity-statistics?{qs}")
    assert status == 200
    return json.loads(body.decode("utf-8"))


def test_pagination_slices_spans_and_reports_total(api):
    svc = _stats(api)
    for i in range(25):
        svc.increment(project_id="p", span=f"span{i:02d}", entity_type="technology")
    page = _get(api, "project=p&limit=10&offset=0")
    assert page["total"] == 25
    assert page["span_count"] == 25
    assert len(page["items"]) == 10
    page3 = _get(api, "project=p&limit=10&offset=20")
    assert page3["total"] == 25
    assert len(page3["items"]) == 5  # last partial page
    spans1 = {it["span"] for it in page["items"]}
    spans3 = {it["span"] for it in page3["items"]}
    assert spans1.isdisjoint(spans3)


def test_ordered_by_total_desc(api):
    svc = _stats(api)
    svc.increment(project_id="p", span="low", entity_type="technology", weight=2)
    svc.increment(project_id="p", span="high", entity_type="technology", weight=9)
    svc.increment(project_id="p", span="mid", entity_type="technology", weight=5)
    items = _get(api, "project=p")["items"]
    assert [it["span"] for it in items] == ["high", "mid", "low"]
    assert items[0]["total"] == 9


def test_item_aggregates_full_distribution(api):
    svc = _stats(api)
    svc.increment(project_id="p", span="apple", entity_type="organization", weight=3)
    svc.increment(project_id="p", span="apple", entity_type="technology", weight=1)
    item = _get(api, "project=p")["items"][0]
    assert item["span"] == "apple"
    assert item["distribution"] == {"organization": 3, "technology": 1}
    assert item["total"] == 4


def test_search_matches_span_and_type(api):
    svc = _stats(api)
    svc.increment(project_id="p", span="Equifax", entity_type="organization")
    svc.increment(project_id="p", span="Android", entity_type="technology")
    by_span = _get(api, "project=p&q=equi")
    assert by_span["total"] == 1 and by_span["items"][0]["span"] == "equifax"
    by_type = _get(api, "project=p&q=organ")
    assert by_type["total"] == 1 and by_type["items"][0]["span"] == "equifax"


def test_type_match_keeps_full_distribution(api):
    # Searching by a type that is only ONE of the span's types must still
    # return the span's full distribution, not just the matching-type row.
    svc = _stats(api)
    svc.increment(project_id="p", span="apple", entity_type="organization", weight=3)
    svc.increment(project_id="p", span="apple", entity_type="technology", weight=1)
    item = _get(api, "project=p&q=organ")["items"][0]
    assert item["distribution"] == {"organization": 3, "technology": 1}
    assert item["total"] == 4


def test_search_escapes_like_wildcards(api):
    svc = _stats(api)
    svc.increment(project_id="p", span="50% off", entity_type="technology")
    svc.increment(project_id="p", span="plain", entity_type="technology")
    page = _get(api, "project=p&q=%25")  # %25 == literal '%'
    assert page["total"] == 1 and page["items"][0]["span"] == "50% off"

import json

from annotation_pipeline_skill.core.models import ArtifactRef, Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.services.entity_statistics_service import (
    EntityStatisticsService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def test_posterior_audit_returns_task_deviations_and_divergent_entries(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)

    # Build prior: 12 Apple → organization (dominant, eligible)
    for _ in range(12):
        svc.increment(project_id="p", span="Apple", entity_type="organization")
    # Divergent: Microsoft has 13/12/5
    for _ in range(13):
        svc.increment(project_id="p", span="Microsoft", entity_type="organization")
    for _ in range(12):
        svc.increment(project_id="p", span="Microsoft", entity_type="project")
    for _ in range(5):
        svc.increment(project_id="p", span="Microsoft", entity_type="technology")

    # Create an accepted task whose annotation tags Apple as technology (diverges from prior).
    task = Task.new(
        task_id="t-dev", pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "Apple"}],
        }},
    )
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    rel = "artifact_payloads/t-dev/final.json"
    abs_path = store.root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(json.dumps({"text": json.dumps({
        "rows": [{"row_index": 0,
                  "output": {"entities": {"technology": ["Apple"]}}}]
    })}))
    store.append_artifact(ArtifactRef.new(
        task_id="t-dev", kind="annotation_result", path=rel,
        content_type="application/json",
    ))

    from annotation_pipeline_skill.interfaces.api import build_posterior_audit
    payload = build_posterior_audit(store, project_id="p")

    assert any(d["span"] == "Apple" and d["current_type"] == "technology"
               for d in payload["task_deviations"])
    # divergent_entries come from entity_statistics (span stored as lower-case).
    assert any(c["span"].lower() == "microsoft"
               for c in payload["divergent_entries"])
    # Each divergent entry must have type_entropy >= 0.
    for entry in payload["divergent_entries"]:
        assert "type_entropy" in entry
        assert entry["type_entropy"] >= 0.0
    # low_info_entries must be present (may be empty for these spans).
    assert "low_info_entries" in payload
    assert isinstance(payload["low_info_entries"], list)


def test_build_posterior_audit_recounts_before_auditing(tmp_path):
    """build_posterior_audit rebuilds entity_statistics from accepted tasks
    first, so a span inflated by stale historical votes collapses to its
    true distinct-task distribution and drops out of divergent."""
    import json
    from annotation_pipeline_skill.core.models import Task, ArtifactRef
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.interfaces.api import build_posterior_audit
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)

    # Stale inflated distribution that WOULD be "divergent" (org 30 / product 30,
    # total 60, no dominant >= 0.80, both >= 0.20).
    svc.increment(project_id="p", span="Apple", entity_type="organization", weight=30)
    svc.increment(project_id="p", span="Apple", entity_type="product", weight=30)

    # But the current accepted reality: 10 tasks ALL tag Apple as organization.
    for i in range(10):
        tid = f"t{i}"
        task = Task.new(task_id=tid, pipeline_id="p",
                        source_ref={"kind": "jsonl", "payload": {
                            "text": "Apple", "rows": [{"row_index": 0, "input": "Apple"}]}})
        task.status = TaskStatus.ACCEPTED
        store.save_task(task)
        rel = f"artifact_payloads/{tid}/final.json"
        ap = store.root / rel
        ap.parent.mkdir(parents=True, exist_ok=True)
        ap.write_text(json.dumps({"text": json.dumps(
            {"rows": [{"row_index": 0, "output": {"entities": {"organization": ["Apple"]}}}]})}),
            encoding="utf-8")
        store.append_artifact(ArtifactRef.new(
            task_id=tid, kind="annotation_result", path=rel,
            content_type="application/json"))

    result = build_posterior_audit(store, project_id="p")

    # After the in-handler recount, Apple is org:10 -> NOT divergent.
    spans = {e["span"] for e in result["divergent_entries"]}
    assert "apple" not in spans
    assert svc.distribution(project_id="p", span="Apple") == {"organization": 10}
    # task_deviations is ALSO computed against the recounted stats (svc.check
    # reads the fresh distribution): with all 10 tasks tagging Apple as
    # organization, none deviates from the now-honest consensus.
    assert all(d["span"].lower() != "apple" for d in result["task_deviations"])

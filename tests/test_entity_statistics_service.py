from annotation_pipeline_skill.services.entity_statistics_service import (
    EntityStatisticsService,
    VerifierResult,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def test_increment_and_distribution(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)

    svc.increment(project_id="p", span="Apple", entity_type="organization", weight=1)
    svc.increment(project_id="p", span="apple", entity_type="organization", weight=2)
    svc.increment(project_id="p", span="APPLE", entity_type="project", weight=1)

    dist = svc.distribution(project_id="p", span="Apple")
    assert dist == {"organization": 3, "project": 1}
    assert svc.total(project_id="p", span="Apple") == 4


def test_check_cold_start(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    for i in range(9):  # less than MIN_PRIOR_SAMPLES (10)
        svc.increment(project_id="p", span="Apple", entity_type="organization")
    result = svc.check(project_id="p", span="Apple", proposed_type="technology")
    assert result.status == "cold_start"


def test_check_agree_when_dominance_low(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    # 6 org + 4 project = 60/40 split; no type >= 80% → agree
    for _ in range(6):
        svc.increment(project_id="p", span="Apple", entity_type="organization")
    for _ in range(4):
        svc.increment(project_id="p", span="Apple", entity_type="project")
    result = svc.check(project_id="p", span="Apple", proposed_type="technology")
    assert result.status == "agree"


def test_check_agree_when_match_dominant(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    for _ in range(9):
        svc.increment(project_id="p", span="Apple", entity_type="organization")
    svc.increment(project_id="p", span="Apple", entity_type="project")
    result = svc.check(project_id="p", span="Apple", proposed_type="organization")
    assert result.status == "agree"


def test_check_divergent(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    for _ in range(9):
        svc.increment(project_id="p", span="Apple", entity_type="organization")
    svc.increment(project_id="p", span="Apple", entity_type="project")
    result = svc.check(project_id="p", span="Apple", proposed_type="technology")
    assert result.status == "divergent"
    assert result.dominant_type == "organization"
    assert result.dominant_count == 9
    assert result.total == 10
    assert result.distribution == {"organization": 9, "project": 1}


def test_divergent_entries(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    # Contested: 13 org + 12 project + 5 tech (top=43%, runner-up=40%)
    for _ in range(13):
        svc.increment(project_id="p", span="Microsoft", entity_type="organization")
    for _ in range(12):
        svc.increment(project_id="p", span="Microsoft", entity_type="project")
    for _ in range(5):
        svc.increment(project_id="p", span="Microsoft", entity_type="technology")
    # Not contested: 9 org + 1 project (dominant > 80%)
    for _ in range(9):
        svc.increment(project_id="p", span="Apple", entity_type="organization")
    svc.increment(project_id="p", span="Apple", entity_type="project")

    entries = svc.divergent_entries(project_id="p")
    assert len(entries) == 1
    assert entries[0]["span"] == "Microsoft" or entries[0]["span"] == "microsoft"
    assert entries[0]["prior_total"] == 30
    assert entries[0]["prior_distribution"] == {"organization": 13, "project": 12, "technology": 5}


def test_iter_span_decisions_walks_entities_and_json_structures():
    from annotation_pipeline_skill.services.entity_statistics_service import (
        iter_span_decisions,
    )
    payload = {
        "rows": [
            {
                "row_index": 0,
                "output": {
                    "entities": {
                        "organization": ["Apple", "Google"],
                        "person": ["Alice"],
                    },
                    "json_structures": {
                        "goal": ["improve perf"],
                    },
                },
            }
        ]
    }
    decisions = list(iter_span_decisions(payload))
    # Both fields contribute — entities and json_structures share the same
    # underlying (span, type) decision space; the split is a training-side
    # detail, not a semantic distinction.
    assert ("Apple", "organization") in decisions
    assert ("Google", "organization") in decisions
    assert ("Alice", "person") in decisions
    assert ("improve perf", "goal") in decisions


def test_iter_span_decisions_dedupes_cross_field_duplicates():
    from annotation_pipeline_skill.services.entity_statistics_service import (
        iter_span_decisions,
    )
    # Same span tagged 'technology' in both entities AND json_structures
    # within one task — counts as ONE decision, not two.
    payload = {
        "rows": [
            {
                "row_index": 0,
                "output": {
                    "entities": {"technology": ["Kubernetes"]},
                    "json_structures": {"technology": ["Kubernetes"]},
                },
            }
        ]
    }
    decisions = list(iter_span_decisions(payload))
    assert decisions == [("Kubernetes", "technology")]


def test_iter_span_decisions_handles_missing_fields():
    from annotation_pipeline_skill.services.entity_statistics_service import (
        iter_span_decisions,
    )
    assert list(iter_span_decisions({})) == []
    assert list(iter_span_decisions({"rows": "not a list"})) == []
    assert list(iter_span_decisions({"rows": [{"output": None}]})) == []


def test_recount_project_distinct_task_counts(tmp_path):
    """recount_project rebuilds the whole project from accepted tasks using
    distinct-task semantics: each accepted task contributes +1 per distinct
    (span, type), regardless of how many rows repeat it, and pre-existing
    inflated rows are wiped."""
    import json
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.core.models import ArtifactRef
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)

    # Pre-existing INFLATED stats that recount must overwrite.
    svc.increment(project_id="p", span="Apple", entity_type="organization", weight=99)
    svc.increment(project_id="p", span="Stale", entity_type="product", weight=40)

    def _add_accepted(task_id, annotation):
        task = Task.new(
            task_id=task_id, pipeline_id="p",
            source_ref={"kind": "jsonl", "payload": {
                "text": "x", "rows": [{"row_index": 0, "input": "x"}],
            }},
        )
        task.status = TaskStatus.ACCEPTED
        store.save_task(task)
        rel = f"artifact_payloads/{task_id}/final.json"
        abs_path = store.root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(json.dumps({"text": json.dumps(annotation)}), encoding="utf-8")
        store.append_artifact(ArtifactRef.new(
            task_id=task_id, kind="annotation_result", path=rel,
            content_type="application/json",
        ))

    # Task A tags Apple as organization in TWO rows -> counts once.
    _add_accepted("a", {"rows": [
        {"row_index": 0, "output": {"entities": {"organization": ["Apple"]}}},
        {"row_index": 1, "output": {"entities": {"organization": ["Apple"]}}},
    ]})
    # Task B tags Apple as organization (+1) AND product (+1).
    _add_accepted("b", {"rows": [
        {"row_index": 0, "output": {"entities": {"organization": ["Apple"], "product": ["Apple"]}}},
    ]})

    result = svc.recount_project(project_id="p")

    assert svc.distribution(project_id="p", span="Apple") == {
        "organization": 2, "product": 1,
    }
    # Stale span had no accepted task -> wiped entirely.
    assert svc.distribution(project_id="p", span="Stale") == {}
    assert result["apple"] == {"organization": 2, "product": 1}


def test_recount_project_only_counts_accepted_tasks(tmp_path):
    import json
    from annotation_pipeline_skill.core.models import Task, ArtifactRef
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)

    def _add(task_id, status, annotation):
        task = Task.new(
            task_id=task_id, pipeline_id="p",
            source_ref={"kind": "jsonl", "payload": {
                "text": "x", "rows": [{"row_index": 0, "input": "x"}]}},
        )
        task.status = status
        store.save_task(task)
        rel = f"artifact_payloads/{task_id}/final.json"
        abs_path = store.root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(json.dumps({"text": json.dumps(annotation)}), encoding="utf-8")
        store.append_artifact(ArtifactRef.new(
            task_id=task_id, kind="annotation_result", path=rel,
            content_type="application/json"))

    _add("acc", TaskStatus.ACCEPTED,
         {"rows": [{"row_index": 0, "output": {"entities": {"organization": ["Apple"]}}}]})
    _add("pending", TaskStatus.ANNOTATING,
         {"rows": [{"row_index": 0, "output": {"entities": {"product": ["Apple"]}}}]})

    svc.recount_project(project_id="p")
    assert svc.distribution(project_id="p", span="Apple") == {"organization": 1}

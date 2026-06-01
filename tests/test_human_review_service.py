import pytest

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.core.transitions import InvalidTransition
from annotation_pipeline_skill.services.human_review_service import HumanReviewService
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _human_review_task(store: SqliteStore) -> Task:
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)
    return task


def test_human_review_accept_records_decision_in_event_metadata(tmp_path):
    store = SqliteStore.open(tmp_path)
    _human_review_task(store)

    result = HumanReviewService(store).decide(
        task_id="task-1",
        action="accept",
        actor="algorithm-engineer",
        feedback="Labels are usable for training.",
        correction_mode="manual_annotation",
    )

    event = store.list_events("task-1")[0]
    assert result.task.status is TaskStatus.ACCEPTED
    assert event.reason == "human review accepted task"
    assert event.metadata["action"] == "accept"
    assert event.metadata["feedback"] == "Labels are usable for training."
    # accept does not create a feedback record (only request_changes/reject do)
    assert store.list_feedback("task-1") == []
    # No separate decision artifact is written
    assert [a for a in store.list_artifacts("task-1") if a.kind == "human_review_decision"] == []


def test_human_review_request_changes_creates_feedback_record(tmp_path):
    store = SqliteStore.open(tmp_path)
    _human_review_task(store)

    result = HumanReviewService(store).decide(
        task_id="task-1",
        action="request_changes",
        actor="algorithm-engineer",
        feedback="Apply the new boundary rule to all rows.",
        correction_mode="batch_code_update",
    )

    event = store.list_events("task-1")[0]
    assert result.task.status is TaskStatus.ANNOTATING
    assert event.reason == "human review requested annotator changes"
    assert event.metadata["correction_mode"] == "batch_code_update"
    feedback_items = store.list_feedback("task-1")
    assert len(feedback_items) == 1
    assert feedback_items[0].source_stage.value == "human_review"
    assert feedback_items[0].message == "Apply the new boundary rule to all rows."
    assert feedback_items[0].suggested_action == "request_changes"


def test_human_review_rejects_task(tmp_path):
    store = SqliteStore.open(tmp_path)
    _human_review_task(store)

    result = HumanReviewService(store).decide(
        task_id="task-1",
        action="reject",
        actor="algorithm-engineer",
        feedback="Source data is unusable.",
        correction_mode="manual_annotation",
    )

    assert result.task.status is TaskStatus.REJECTED
    assert store.list_events("task-1")[0].reason == "human review rejected task"


def test_human_review_rejects_actions_outside_human_review(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.QC
    store.save_task(task)

    with pytest.raises(InvalidTransition):
        HumanReviewService(store).decide(
            task_id="task-1",
            action="accept",
            actor="algorithm-engineer",
            feedback="Accept",
            correction_mode="manual_annotation",
        )


def test_submit_correction_schema_valid_answer_accepts_task(tmp_path):
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.human_review_service import HumanReviewService
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-hr",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "x",
                "annotation_guidance": {
                    "output_schema": {
                        "type": "object",
                        "required": ["entities"],
                        "properties": {"entities": {"type": "array"}},
                    }
                },
            },
        },
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    svc = HumanReviewService(store)
    result = svc.submit_correction(
        task_id="t-hr",
        answer={"entities": [{"text": "Acme", "label": "ORG"}]},
        actor="reviewer-1",
        note="manual fix",
    )

    assert result.task.status is TaskStatus.ACCEPTED
    artifacts = [a for a in store.list_artifacts("t-hr") if a.kind == "human_review_answer"]
    assert len(artifacts) == 1


def test_submit_correction_schema_invalid_answer_raises_and_keeps_status(tmp_path):
    import pytest
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.schema_validation import SchemaValidationError
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.human_review_service import HumanReviewService
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-hr-bad",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "x",
                "annotation_guidance": {
                    "output_schema": {"type": "object", "required": ["entities"]}
                },
            },
        },
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    svc = HumanReviewService(store)
    with pytest.raises(SchemaValidationError):
        svc.submit_correction(
            task_id="t-hr-bad",
            answer={"wrong_key": []},
            actor="reviewer-1",
            note=None,
        )

    task_after = store.load_task("t-hr-bad")
    assert task_after.status is TaskStatus.HUMAN_REVIEW


def test_submit_correction_missing_schema_raises(tmp_path):
    import pytest
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.schema_validation import SchemaValidationError
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.human_review_service import HumanReviewService
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-hr-noschema",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "x"}},
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    svc = HumanReviewService(store)
    with pytest.raises(SchemaValidationError) as exc:
        svc.submit_correction(
            task_id="t-hr-noschema",
            answer={"anything": True},
            actor="r",
            note=None,
        )
    assert exc.value.errors[0]["kind"] == "missing_schema"


def test_submit_correction_rejects_when_task_not_in_human_review(tmp_path):
    import pytest
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.core.transitions import InvalidTransition
    from annotation_pipeline_skill.services.human_review_service import HumanReviewService
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-pending",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"annotation_guidance": {"output_schema": {"type": "object"}}}},
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    svc = HumanReviewService(store)
    with pytest.raises(InvalidTransition):
        svc.submit_correction(task_id="t-pending", answer={}, actor="r", note=None)


def test_submit_correction_rejects_non_verbatim_spans(tmp_path):
    """Operator-submitted corrections must use spans that are verbatim
    substrings of the row's input.text — same guarantee enforced on the
    annotator and arbiter paths. Otherwise an operator could paste a
    paraphrased / normalized span and ACCEPT a task with bad data.
    """
    from annotation_pipeline_skill.core.schema_validation import SchemaValidationError

    store = SqliteStore.open(tmp_path)
    row_schema = {
        "type": "object", "additionalProperties": False, "required": ["row_index", "output"],
        "properties": {
            "row_index": {"type": "integer"},
            "output": {"type": "object",
                       "properties": {"entities": {"type": "object"}}},
        },
    }
    schema = {
        "type": "object", "additionalProperties": False, "required": ["rows"],
        "properties": {"rows": {"type": "array", "items": row_schema}},
    }
    task = Task.new(
        task_id="t-hr-vb",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "text": "alpha is mentioned",
            "rows": [{"row_index": 0, "input": "alpha is mentioned"}],
            "annotation_guidance": {"output_schema": schema},
        }},
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    svc = HumanReviewService(store)
    # "beta" is not in input "alpha is mentioned"
    with pytest.raises(SchemaValidationError) as exc:
        svc.submit_correction(
            task_id="t-hr-vb",
            answer={"rows": [{"row_index": 0, "output": {"entities": {"name": ["beta"]}}}]},
            actor="r", note=None,
        )
    assert any("non_verbatim_span" in e["kind"] for e in exc.value.errors)


def test_decide_accept_blocks_when_underlying_annotation_has_violations(tmp_path):
    """``decide(action='accept')`` refuses to ACCEPT a HR task whose latest
    annotation_result contains non-verbatim spans. Operator must fix via
    submit_correction or request_changes."""
    from annotation_pipeline_skill.core.models import ArtifactRef
    from annotation_pipeline_skill.core.schema_validation import SchemaValidationError

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-hr-accept-bad",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "text": "alpha is mentioned",
            "rows": [{"row_index": 0, "input": "alpha is mentioned"}],
        }},
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)
    # Seed a bad annotation_result on disk: contains "beta" not in input.
    import json as _json
    artifact_path = "artifact_payloads/t-hr-accept-bad/annotation.json"
    (store.root / artifact_path).parent.mkdir(parents=True, exist_ok=True)
    (store.root / artifact_path).write_text(
        _json.dumps({"text": _json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"name": ["beta"]}}}]})}),
        encoding="utf-8",
    )
    store.append_artifact(ArtifactRef.new(
        task_id="t-hr-accept-bad", kind="annotation_result", path=artifact_path,
        content_type="application/json",
    ))

    svc = HumanReviewService(store)
    with pytest.raises(SchemaValidationError) as exc:
        svc.decide(task_id="t-hr-accept-bad", action="accept",
                   actor="r", feedback="lgtm", correction_mode="manual_annotation")
    assert any("non_verbatim_span" in e["kind"] for e in exc.value.errors)
    # Task status unchanged
    assert store.load_task("t-hr-accept-bad").status is TaskStatus.HUMAN_REVIEW


def test_submit_correction_rejects_when_against_prior(tmp_path):
    import pytest
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.core.schema_validation import SchemaValidationError
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.services.human_review_service import (
        HumanReviewService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    for _ in range(12):
        svc.increment(project_id="p", span="Apple", entity_type="organization")

    schema = {"type": "object", "additionalProperties": False, "required": ["rows"],
              "properties": {"rows": {"type": "array"}}}
    task = Task.new(
        task_id="hr-1", pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "Apple is mentioned"}],
            "annotation_guidance": {"output_schema": schema},
        }},
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    hr = HumanReviewService(store)
    answer = {"rows": [{"row_index": 0, "output": {"entities": {"technology": ["Apple"]}}}]}
    with pytest.raises(SchemaValidationError) as excinfo:
        hr.submit_correction(task_id="hr-1", answer=answer, actor="op", note=None)
    assert any(
        e.get("kind") == "prior_disagreement" for e in (excinfo.value.errors or [])
    )


def test_submit_correction_with_force_bypasses_verifier(tmp_path):
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.services.human_review_service import (
        HumanReviewService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    for _ in range(12):
        svc.increment(project_id="p", span="Apple", entity_type="organization")

    schema = {"type": "object", "additionalProperties": False, "required": ["rows"],
              "properties": {"rows": {"type": "array"}}}
    task = Task.new(
        task_id="hr-2", pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "Apple is mentioned"}],
            "annotation_guidance": {"output_schema": schema},
        }},
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    hr = HumanReviewService(store)
    answer = {"rows": [{"row_index": 0, "output": {"entities": {"technology": ["Apple"]}}}]}
    result = hr.submit_correction(
        task_id="hr-2", answer=answer, actor="op", note=None, force=True,
    )
    assert result.task.status is TaskStatus.ACCEPTED
    # HR no longer bumps stats; the prior-disagreement READ gate was bypassed
    # by force, but the WRITE path is recount-only now. Distribution unchanged.
    assert svc.distribution(project_id="p", span="Apple") == {"organization": 12}


def test_submit_correction_default_path_does_not_mutate_stats(tmp_path):
    """The default (non-force) correction path is the one that previously
    carried the now-removed stat_bumps dispatch. With an answer that AGREES
    with the prior (so the READ gate passes without force), the correction
    ACCEPTS the task but must NOT mutate entity_statistics — the distribution
    only changes when an operator runs Re-check (recount)."""
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.services.human_review_service import (
        HumanReviewService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    for _ in range(12):
        svc.increment(project_id="p", span="Apple", entity_type="organization")

    schema = {"type": "object", "additionalProperties": False, "required": ["rows"],
              "properties": {"rows": {"type": "array"}}}
    task = Task.new(
        task_id="hr-3", pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "Apple is mentioned"}],
            "annotation_guidance": {"output_schema": schema}}},
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    hr = HumanReviewService(store)
    # AGREEING answer (organization matches the dominant prior) -> READ gate
    # passes without force, so this exercises the default correction path.
    answer = {"rows": [{"row_index": 0, "output": {"entities": {"organization": ["Apple"]}}}]}
    result = hr.submit_correction(task_id="hr-3", answer=answer, actor="op", note=None)

    assert result.task.status is TaskStatus.ACCEPTED
    # Default path no longer bumps stats; distribution unchanged until recount.
    assert svc.distribution(project_id="p", span="Apple") == {"organization": 12}

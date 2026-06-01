from annotation_pipeline_skill.core.models import (
    Attempt,
    FeedbackDiscussionEntry,
    FeedbackRecord,
    OutboxRecord,
    Task,
)
from annotation_pipeline_skill.core.states import AttemptStatus, FeedbackSeverity, FeedbackSource, OutboxKind, TaskStatus
from annotation_pipeline_skill.services.dashboard_service import (
    build_dashboard_stats,
    build_kanban_snapshot,
    build_project_summaries,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _feedback(task_id, feedback_id):
    fb = FeedbackRecord.new(
        task_id=task_id, attempt_id="a", source_stage=FeedbackSource.QC,
        severity=FeedbackSeverity.WARNING, category="q", message="m",
        target={}, suggested_action="annotator_rerun", created_by="qc")
    fb.feedback_id = feedback_id
    return fb


def test_open_feedback_count_excludes_consensus_resolved(tmp_path):
    # build_dashboard_stats sums per-task "open" feedback (records without a
    # consensus discussion) via one grouped query. Pin that a consensus
    # discussion removes its record from the open count, across two tasks.
    store = SqliteStore.open(tmp_path)
    for tid in ("t1", "t2"):
        task = Task.new(task_id=tid, pipeline_id="pipe", source_ref={"kind": "jsonl"})
        task.status = TaskStatus.HUMAN_REVIEW
        store.save_task(task)
    store.append_feedback(_feedback("t1", "f1"))  # open
    store.append_feedback(_feedback("t1", "f2"))  # resolved by consensus below
    store.append_feedback(_feedback("t2", "f3"))  # open
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id="t1", feedback_id="f2", role="qc", stance="agree",
        message="ok", created_by="qc", consensus=True))
    # A non-consensus discussion must NOT resolve its feedback.
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id="t2", feedback_id="f3", role="qc", stance="dispute",
        message="no", created_by="qc", consensus=False))

    assert store.open_feedback_count_for_tasks(["t1", "t2"]) == 2  # f1, f3
    assert store.open_feedback_count_for_tasks([]) == 0
    assert build_dashboard_stats(store, project_id="pipe")["open_feedback_count"] == 2


def test_dashboard_snapshot_groups_tasks_into_operational_columns(tmp_path):
    store = SqliteStore.open(tmp_path)
    pending = Task.new(task_id="task-pending", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    pending.status = TaskStatus.PENDING
    review = Task.new(task_id="task-review", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    review.status = TaskStatus.HUMAN_REVIEW
    review.modality = "image"
    review.annotation_requirements = {"annotation_types": ["bounding_box"]}
    store.save_task(pending)
    store.save_task(review)
    store.append_feedback(
        FeedbackRecord.new(
            task_id="task-review",
            attempt_id="attempt-1",
            source_stage=FeedbackSource.QC,
            severity=FeedbackSeverity.WARNING,
            category="bbox",
            message="Review box boundary",
            target={"box_id": "b1"},
            suggested_action="manual_annotation",
            created_by="qc",
        )
    )

    snapshot = build_kanban_snapshot(store)

    assert [column["id"] for column in snapshot["columns"]] == [
        "pending",
        "annotating",
        "qc",
        "arbitrating",
        "human_review",
        "accepted",
        "rejected",
    ]
    assert snapshot["columns"][0]["title"] == "Pending"
    assert snapshot["columns"][0]["cards"][0]["task_id"] == "task-pending"
    assert snapshot["columns"][4]["cards"][0]["feedback_count"] == 1
    assert snapshot["columns"][4]["cards"][0]["modality"] == "image"
    assert snapshot["columns"][4]["cards"][0]["operator_stage"] == "qc"


def test_dashboard_snapshot_indexes_attempts_feedback_and_outbox_once(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl", "row_count": 42})
    task.status = TaskStatus.PENDING
    task.current_attempt = 3
    store.save_task(task)
    store.append_attempt(
        Attempt(
            attempt_id="attempt-1",
            task_id="task-1",
            index=1,
            stage="annotation",
            status=AttemptStatus.SUCCEEDED,
            provider_id="deepseek",
        )
    )
    store.append_attempt(
        Attempt(
            attempt_id="attempt-2",
            task_id="task-1",
            index=2,
            stage="qc",
            status=AttemptStatus.FAILED,
            provider_id="glm",
        )
    )
    store.append_feedback(
        FeedbackRecord.new(
            task_id="task-1",
            attempt_id="attempt-2",
            source_stage=FeedbackSource.QC,
            severity=FeedbackSeverity.WARNING,
            category="quality",
            message="Needs review",
            target={},
            suggested_action="annotator_rerun",
            created_by="qc",
        )
    )
    store.save_outbox(OutboxRecord.new(task_id="task-1", kind=OutboxKind.STATUS, payload={}))

    card = build_kanban_snapshot(store)["columns"][0]["cards"][0]

    assert card["latest_attempt_status"] == "failed"
    assert card["pipeline_chain"] == "deepseek->glm"
    assert card["feedback_count"] == 1
    assert card["external_sync_pending"] is True
    assert card["row_count"] == 42
    assert card["attempt_count"] == 3


def test_bulk_index_methods_group_by_task_and_skip_empty(tmp_path):
    # The kanban index is built from two grouped bulk queries instead of a
    # per-task N+1. Pin their contract: attempts grouped + ordered by seq with
    # only the card columns; feedback counts only for tasks that have records.
    store = SqliteStore.open(tmp_path)
    a = Task.new(task_id="a", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    b = Task.new(task_id="b", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    store.save_task(a)
    store.save_task(b)
    store.append_attempt(Attempt(attempt_id="a1", task_id="a", index=1, stage="annotation",
                                 status=AttemptStatus.SUCCEEDED, provider_id="deepseek", model="m1"))
    store.append_attempt(Attempt(attempt_id="a2", task_id="a", index=2, stage="qc",
                                 status=AttemptStatus.FAILED, provider_id="glm", model="m2"))
    store.append_feedback(FeedbackRecord.new(
        task_id="a", attempt_id="a2", source_stage=FeedbackSource.QC,
        severity=FeedbackSeverity.WARNING, category="q", message="m",
        target={}, suggested_action="annotator_rerun", created_by="qc"))

    attempts = store.attempt_cards_by_task(["a", "b"])
    assert list(attempts.keys()) == ["a"]                 # b has no attempts
    assert [x["stage"] for x in attempts["a"]] == ["annotation", "qc"]  # ordered by seq
    assert attempts["a"][1] == {"stage": "qc", "status": "failed", "model": "m2", "provider_id": "glm"}

    counts = store.feedback_counts_by_task(["a", "b"])
    assert counts == {"a": 1}                              # b omitted (no feedback)

    # Empty input must not build a degenerate "IN ()" query.
    assert store.attempt_cards_by_task([]) == {}
    assert store.feedback_counts_by_task([]) == {}


def test_dashboard_snapshot_card_row_count_is_none_when_absent(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-no-rows", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    store.save_task(task)

    card = build_kanban_snapshot(store)["columns"][0]["cards"][0]

    assert card["row_count"] is None
    assert card["attempt_count"] == 0


def test_dashboard_snapshot_can_return_operator_stage_columns(tmp_path):
    store = SqliteStore.open(tmp_path)
    annotating = Task.new(task_id="task-annotation", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    annotating.status = TaskStatus.ANNOTATING
    review = Task.new(task_id="task-qc", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    review.status = TaskStatus.HUMAN_REVIEW
    failed = Task.new(task_id="task-failed", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    failed.status = TaskStatus.REJECTED
    store.save_task(annotating)
    store.save_task(review)
    store.save_task(failed)

    snapshot = build_kanban_snapshot(store, stage_view="operator")

    assert snapshot["stage_view"] == "operator"
    assert [column["id"] for column in snapshot["columns"]] == [
        "pending",
        "annotation",
        "qc",
        "arbitration",
        "merge",
        "failed",
        "accepted",
    ]
    assert [card["task_id"] for card in snapshot["columns"][1]["cards"]] == ["task-annotation"]
    assert [card["task_id"] for card in snapshot["columns"][2]["cards"]] == ["task-qc"]
    assert [card["task_id"] for card in snapshot["columns"][5]["cards"]] == ["task-failed"]


def test_dashboard_snapshot_filters_tasks_by_project_id(tmp_path):
    store = SqliteStore.open(tmp_path)
    alpha = Task.new(task_id="alpha-1", pipeline_id="project-alpha", source_ref={"kind": "jsonl"})
    beta = Task.new(task_id="beta-1", pipeline_id="project-beta", source_ref={"kind": "jsonl"})
    alpha.status = TaskStatus.PENDING
    beta.status = TaskStatus.PENDING
    store.save_task(alpha)
    store.save_task(beta)

    snapshot = build_kanban_snapshot(store, project_id="project-alpha")

    visible_task_ids = [
        card["task_id"]
        for column in snapshot["columns"]
        for card in column["cards"]
    ]
    assert snapshot["project_id"] == "project-alpha"
    assert visible_task_ids == ["alpha-1"]


def test_dashboard_project_summaries_group_tasks_by_pipeline_id(tmp_path):
    store = SqliteStore.open(tmp_path)
    alpha_pending = Task.new(task_id="alpha-1", pipeline_id="project-alpha", source_ref={"kind": "jsonl"})
    alpha_accepted = Task.new(task_id="alpha-2", pipeline_id="project-alpha", source_ref={"kind": "jsonl"})
    alpha_pending.status = TaskStatus.PENDING
    alpha_accepted.status = TaskStatus.ACCEPTED
    beta = Task.new(task_id="beta-1", pipeline_id="project-beta", source_ref={"kind": "jsonl"})
    beta.status = TaskStatus.PENDING
    store.save_task(alpha_pending)
    store.save_task(alpha_accepted)
    store.save_task(beta)

    snapshot = build_project_summaries(store)

    assert snapshot["projects"] == [
        {
            "project_id": "project-alpha",
            "task_count": 2,
            "status_counts": {"accepted": 1, "pending": 1},
        },
        {
            "project_id": "project-beta",
            "task_count": 1,
            "status_counts": {"pending": 1},
        },
    ]

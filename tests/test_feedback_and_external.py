from annotation_pipeline_skill.core.models import FeedbackDiscussionEntry, FeedbackRecord
from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource, OutboxKind
from annotation_pipeline_skill.services.external_task_service import ExternalTaskService
from annotation_pipeline_skill.services.feedback_service import build_feedback_bundle, build_feedback_consensus_summary
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def test_feedback_bundle_orders_records_by_creation_time(tmp_path):
    store = SqliteStore.open(tmp_path)
    first = FeedbackRecord.new(
        task_id="task-1",
        attempt_id="attempt-1",
        source_stage=FeedbackSource.QC,
        severity=FeedbackSeverity.ERROR,
        category="format",
        message="Bad JSON shape",
        target={"path": "$"},
        suggested_action="batch_code_update",
        created_by="validator",
    )
    second = FeedbackRecord.new(
        task_id="task-1",
        attempt_id="attempt-2",
        source_stage=FeedbackSource.HUMAN_REVIEW,
        severity=FeedbackSeverity.WARNING,
        category="boundary",
        message="Box is too loose",
        target={"box_id": "b1"},
        suggested_action="manual_annotation",
        created_by="reviewer",
    )
    store.append_feedback(second)
    store.append_feedback(first)

    bundle = build_feedback_bundle(store, "task-1")

    assert [item["message"] for item in bundle["items"]] == ["Bad JSON shape", "Box is too loose"]


def test_feedback_bundle_includes_discussion_and_consensus(tmp_path):
    store = SqliteStore.open(tmp_path)
    feedback = FeedbackRecord.new(
        task_id="task-1",
        attempt_id="attempt-1",
        source_stage=FeedbackSource.QC,
        severity=FeedbackSeverity.WARNING,
        category="boundary",
        message="Span boundary is too wide.",
        target={"entity": "OpenAI"},
        suggested_action="manual_annotation",
        created_by="qc-agent",
    )
    annotator_reply = FeedbackDiscussionEntry.new(
        task_id="task-1",
        feedback_id=feedback.feedback_id,
        role="annotator",
        stance="partial_agree",
        message="I agree the span should exclude punctuation, but the label is correct.",
        agreed_points=["exclude trailing punctuation"],
        disputed_points=["label should remain ORG"],
        proposed_resolution="Update span only.",
        created_by="annotator-agent",
    )
    qc_reply = FeedbackDiscussionEntry.new(
        task_id="task-1",
        feedback_id=feedback.feedback_id,
        role="qc",
        stance="agree",
        message="Agreed. Span-only update is sufficient.",
        agreed_points=["exclude trailing punctuation", "label should remain ORG"],
        proposed_resolution="Update span only.",
        consensus=True,
        created_by="qc-agent",
    )
    store.append_feedback(feedback)
    store.append_feedback_discussion(annotator_reply)
    store.append_feedback_discussion(qc_reply)

    # include_resolved=True so the consensus-closed item still surfaces — the
    # prompt path defaults to filtering closed items, but the audit / UI path
    # asks for the full history.
    bundle = build_feedback_bundle(store, "task-1", include_resolved=True)
    summary = build_feedback_consensus_summary(store, "task-1")

    assert bundle["items"][0]["discussion"][0]["stance"] == "partial_agree"
    assert bundle["items"][0]["consensus"] is True
    assert summary["can_accept_by_consensus"] is True

    # Default (include_resolved=False) drops resolved items.
    filtered = build_feedback_bundle(store, "task-1")
    assert filtered["items"] == []


def test_build_feedback_bundle_max_items_caps_to_most_recent(tmp_path):
    """max_items keeps the N most recent items by created_at."""
    store = SqliteStore.open(tmp_path)
    records = []
    for i in range(5):
        r = FeedbackRecord.new(
            task_id="task-cap",
            attempt_id=f"attempt-{i}",
            source_stage=FeedbackSource.QC,
            severity=FeedbackSeverity.ERROR,
            category="format",
            message=f"error {i}",
            target={},
            suggested_action="batch_code_update",
            created_by="qc",
        )
        store.append_feedback(r)
        records.append(r)

    # No cap: all 5 returned.
    bundle = build_feedback_bundle(store, "task-cap")
    assert len(bundle["items"]) == 5

    # max_items=3: keeps the 3 most recent (records[2], records[3], records[4]).
    capped = build_feedback_bundle(store, "task-cap", max_items=3)
    assert len(capped["items"]) == 3
    messages = [item["message"] for item in capped["items"]]
    assert messages == ["error 2", "error 3", "error 4"]

    # max_items >= total: no truncation.
    full = build_feedback_bundle(store, "task-cap", max_items=10)
    assert len(full["items"]) == 5


def test_external_task_pull_is_idempotent_and_creates_status_outbox(tmp_path):
    store = SqliteStore.open(tmp_path)
    service = ExternalTaskService(store)

    first = service.upsert_pulled_task(
        pipeline_id="pipe",
        system_id="external",
        external_task_id="42",
        payload={"text": "hello"},
    )
    second = service.upsert_pulled_task(
        pipeline_id="pipe",
        system_id="external",
        external_task_id="42",
        payload={"text": "hello again"},
    )
    record = service.enqueue_status(first, status="pending")

    assert first.task_id == second.task_id
    # qc_policy is now project-level (RuntimeConfig); external pull no longer
    # injects per-task qc_policy.
    assert "qc_policy" not in first.metadata
    assert record.kind is OutboxKind.STATUS
    assert store.list_outbox() == [record]

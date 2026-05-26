from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def build_feedback_bundle(
    store: SqliteStore,
    task_id: str,
    *,
    include_resolved: bool = False,
    max_items: int | None = None,
    latest_attempt_only: bool = False,
) -> dict:
    """Build feedback bundle for prompt context.

    By default filters out *resolved* (consensus=True) feedback items — the
    annotator/QC don't need to re-address items that already reached
    consensus, and including them just bloats the prompt. Pass
    ``include_resolved=True`` to surface the full history (e.g. for audit
    UIs that want to show closed disputes).

    ``max_items`` caps the number of items returned, keeping the most recent
    ones (sorted by created_at). Use this in prompt builders to prevent the
    feedback bundle from growing unboundedly and blowing the context window.

    ``latest_attempt_only``: when True, keep only feedback records whose
    ``attempt_id`` matches the most recent attempt (by insertion order /
    created_at). This prevents cross-round accumulation for long-running
    tasks where old attempts may have generated many irrelevant records.
    """
    discussions = sorted(store.list_feedback_discussions(task_id), key=lambda entry: entry.created_at)
    consensus_ids = {entry.feedback_id for entry in discussions if entry.consensus}

    records = sorted(store.list_feedback(task_id), key=lambda record: record.created_at)
    if not include_resolved:
        records = [r for r in records if r.feedback_id not in consensus_ids]
    if latest_attempt_only and records:
        # Use the last record's attempt_id (records sorted by created_at, so
        # this is the most recent attempt). Avoid max() on attempt_id strings
        # because lexicographic order breaks at attempt-10 ("attempt-9" > "attempt-10").
        latest_attempt_id = records[-1].attempt_id
        records = [r for r in records if r.attempt_id == latest_attempt_id]
    if max_items is not None and len(records) > max_items:
        records = records[-max_items:]
    return {
        "task_id": task_id,
        "items": [
            {
                "feedback_id": record.feedback_id,
                "attempt_id": record.attempt_id,
                "source_stage": record.source_stage.value,
                "severity": record.severity.value,
                "category": record.category,
                "message": record.message,
                "target": record.target,
                "suggested_action": record.suggested_action,
                "created_at": record.created_at.isoformat(),
                "created_by": record.created_by,
                "discussion": [
                    entry.to_dict()
                    for entry in discussions
                    if entry.feedback_id == record.feedback_id
                ],
                "consensus": record.feedback_id in consensus_ids,
            }
            for record in records
        ],
    }


def build_feedback_consensus_summary(store: SqliteStore, task_id: str) -> dict:
    feedback = store.list_feedback(task_id)
    discussions = store.list_feedback_discussions(task_id)
    consensus_feedback_ids = {
        entry.feedback_id
        for entry in discussions
        if entry.consensus
    }
    return {
        "task_id": task_id,
        "total_feedback": len(feedback),
        "consensus_feedback": len(consensus_feedback_ids),
        "open_feedback": [
            record.feedback_id
            for record in feedback
            if record.feedback_id not in consensus_feedback_ids
        ],
        "can_accept_by_consensus": bool(feedback) and len(consensus_feedback_ids) == len(feedback),
    }

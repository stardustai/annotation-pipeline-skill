from datetime import datetime, timedelta, timezone

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.services.feedback_service import build_feedback_consensus_summary
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


KANBAN_COLUMNS: list[tuple[str, str, TaskStatus]] = [
    ("pending", "Pending", TaskStatus.PENDING),
    ("annotating", "Annotating", TaskStatus.ANNOTATING),
    ("qc", "QC", TaskStatus.QC),
    ("arbitrating", "Arbitration", TaskStatus.ARBITRATING),
    ("human_review", "Human Review", TaskStatus.HUMAN_REVIEW),
    ("accepted", "Accepted", TaskStatus.ACCEPTED),
    ("rejected", "Rejected", TaskStatus.REJECTED),
]

OPERATOR_COLUMNS: list[tuple[str, str]] = [
    ("pending", "Pending"),
    ("annotation", "Annotation"),
    ("qc", "QC"),
    ("arbitration", "Arbitration"),
    ("merge", "Merge"),
    ("failed", "Failed"),
    ("accepted", "Accepted"),
]


def operator_stage(task: Task) -> str:
    if task.status is TaskStatus.PENDING:
        return "pending"
    if task.status is TaskStatus.ANNOTATING:
        return "annotation"
    if task.status is TaskStatus.ARBITRATING:
        return "arbitration"
    if task.status in {TaskStatus.QC, TaskStatus.HUMAN_REVIEW}:
        return "qc"
    if task.status is TaskStatus.ACCEPTED:
        return "accepted"
    if task.status in {TaskStatus.REJECTED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}:
        return "failed"
    return "pending"


def build_kanban_snapshot(store: SqliteStore, project_id: str | None = None, stage_view: str = "internal") -> dict:
    tasks = sorted(store.list_tasks(), key=lambda task: task.created_at)
    if project_id is not None:
        tasks = [task for task in tasks if task.pipeline_id == project_id]
    index = _dashboard_index(store)
    if stage_view == "operator":
        return {
            "project_id": project_id,
            "stage_view": "operator",
            "columns": [
                {
                    "id": column_id,
                    "title": title,
                    "cards": [_task_card(index, task) for task in tasks if operator_stage(task) == column_id],
                }
                for column_id, title in OPERATOR_COLUMNS
            ],
        }
    return {
        "project_id": project_id,
        "stage_view": "internal",
        "columns": [
                {
                    "id": column_id,
                    "title": title,
                    "cards": [_task_card(index, task) for task in tasks if task.status is status],
                }
            for column_id, title, status in KANBAN_COLUMNS
        ]
    }


THROUGHPUT_STAGES = ("annotation", "qc", "arbitration")


def build_dashboard_stats(
    store: SqliteStore,
    *,
    project_id: str | None = None,
    throughput_window_minutes: int = 5,
) -> dict:
    """Counts + per-stage throughput for the always-visible stats bar.

    Throughput is the number of attempts that completed with status='succeeded'
    in the most recent ``throughput_window_minutes`` window, grouped by stage
    (annotation / qc / arbitration). When ``project_id`` is set, both the
    counts and the throughput are scoped to that pipeline.
    """
    tasks = store.list_tasks() if project_id is None else store.list_tasks_by_pipeline(project_id)

    status_counts: dict[str, int] = {}
    open_feedback_count = 0
    for task in tasks:
        status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
        open_feedback_count += len(
            build_feedback_consensus_summary(store, task.task_id)["open_feedback"]
        )

    task_id_set = {t.task_id for t in tasks}
    outbox_pending_count = sum(
        1 for record in store.list_outbox()
        if record.status.value == "pending" and record.task_id in task_id_set
    )

    since = datetime.now(timezone.utc) - timedelta(minutes=throughput_window_minutes)
    since_iso = since.isoformat()
    raw_throughput = store.count_succeeded_attempts_since(
        since_iso,
        pipeline_id=project_id,
    )
    throughput = {stage: raw_throughput.get(stage, 0) for stage in THROUGHPUT_STAGES}

    accepted_in_window = store.count_accepted_since(since_iso, pipeline_id=project_id)
    health = store.fetch_pipeline_health_metrics(pipeline_id=project_id)

    return {
        "project_id": project_id,
        "task_count": len(tasks),
        "status_counts": status_counts,
        "open_feedback_count": open_feedback_count,
        "outbox_pending_count": outbox_pending_count,
        "throughput_per_window": throughput,
        "throughput_window_minutes": throughput_window_minutes,
        "accepted_in_window": accepted_in_window,
        "accepted_count": health["accepted_count"],
        "terminal_count": health["terminal_count"],
        "first_pass_count": health["first_pass_count"],
        "arb_entered_count": health["arb_entered_count"],
        "avg_llm_calls": health["avg_llm_calls"],
    }


def build_project_summaries(store: SqliteStore) -> dict:
    summaries: dict[str, dict] = {}
    for task in store.list_tasks():
        summary = summaries.setdefault(
            task.pipeline_id,
            {"project_id": task.pipeline_id, "task_count": 0, "status_counts": {}},
        )
        summary["task_count"] += 1
        status_counts = summary["status_counts"]
        status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1

    return {
        "projects": [
            {
                "project_id": summary["project_id"],
                "task_count": summary["task_count"],
                "status_counts": dict(sorted(summary["status_counts"].items())),
            }
            for summary in sorted(summaries.values(), key=lambda item: item["project_id"])
        ]
    }


def _dashboard_index(store: SqliteStore) -> dict:
    attempts_by_task: dict[str, list[dict]] = {}
    feedback_counts: dict[str, int] = {}
    for task in store.list_tasks():
        attempts = store.list_attempts(task.task_id)
        if attempts:
            attempts_by_task[task.task_id] = [attempt.to_dict() for attempt in attempts]
        feedback = store.list_feedback(task.task_id)
        if feedback:
            feedback_counts[task.task_id] = len(feedback)
    pending_outbox_task_ids = {
        record.task_id
        for record in store.list_outbox()
        if record.status.value == "pending"
    }
    return {
        "attempts_by_task": attempts_by_task,
        "feedback_counts": feedback_counts,
        "pending_outbox_task_ids": pending_outbox_task_ids,
    }


def _task_card(index: dict, task: Task) -> dict:
    attempts = index["attempts_by_task"].get(task.task_id, [])
    latest_attempt = attempts[-1] if attempts else None
    feedback_count = index["feedback_counts"].get(task.task_id, 0)
    annotation_types = task.annotation_requirements.get("annotation_types", [])
    raw_row_count = task.source_ref.get("row_count") if isinstance(task.source_ref, dict) else None
    try:
        row_count = int(raw_row_count) if raw_row_count is not None else None
    except (TypeError, ValueError):
        row_count = None
    annotator_model, qc_model = _stage_models(task, attempts)
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "operator_stage": operator_stage(task),
        "pipeline_chain": _pipeline_chain(task, attempts),
        "modality": task.modality,
        "annotation_types": annotation_types,
        "selected_annotator_id": task.selected_annotator_id,
        "annotator_model": annotator_model,
        "qc_model": qc_model,
        "status_age_seconds": int((datetime.now(timezone.utc) - (task.updated_at if task.updated_at.tzinfo is not None else task.updated_at.replace(tzinfo=timezone.utc))).total_seconds()),
        "latest_attempt_status": latest_attempt.get("status") if latest_attempt else None,
        "feedback_count": feedback_count,
        "retry_pending": task.next_retry_at is not None,
        "blocked": task.status is TaskStatus.BLOCKED,
        "external_sync_pending": task.task_id in index["pending_outbox_task_ids"],
        "row_count": row_count,
        "attempt_count": int(task.current_attempt or 0),
    }


def _stage_models(task: Task, attempts: list) -> tuple[str | None, str | None]:
    def latest_model(stage: str) -> str | None:
        stage_attempts = [a for a in attempts if a.get("stage") == stage and a.get("model")]
        return str(stage_attempts[-1]["model"]) if stage_attempts else None

    annotator_model = latest_model("annotation") or task.selected_annotator_id
    qc_model = latest_model("qc")
    return annotator_model, qc_model


def _pipeline_chain(task: Task, attempts) -> str:
    providers = []
    for stage in ("annotation", "qc", "merge"):
        stage_attempts = [attempt for attempt in attempts if attempt.get("stage") == stage and attempt.get("provider_id")]
        if stage_attempts:
            providers.append(str(stage_attempts[-1]["provider_id"]))
    return "->".join(providers) if providers else str(task.selected_annotator_id or "")

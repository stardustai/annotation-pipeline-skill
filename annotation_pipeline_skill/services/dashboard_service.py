from datetime import datetime, timedelta, timezone

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
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


def _task_substage(task: Task) -> str | None:
    meta = task.metadata if isinstance(task.metadata, dict) else {}
    return meta.get("runtime_substage")


def _resolve_models(store: SqliteStore, targets: list[str]) -> dict[str, str]:
    """Map each annotation target to its model name for column headers
    (e.g. annotation → qwen3.6-35b-a3b). Falls back to the target name."""
    resolved: dict[str, str] = {}
    try:
        from annotation_pipeline_skill.llm.profiles import (
            load_llm_registry,
            resolve_llm_profiles_path,
        )

        path = resolve_llm_profiles_path(
            workspace_root=store.root.parent.parent,
            project_config_root=store.root,
        )
        if path is not None:
            reg = load_llm_registry(path)
            for target in targets:
                try:
                    resolved[target] = reg.resolve(target).model
                except Exception:
                    resolved[target] = target
    except Exception:
        pass
    return resolved


# Cache the column config (mode + resolved model names) by the mtimes of the two
# files it derives from, so the 5s kanban poll doesn't re-read+parse workflow.yaml
# and llm_profiles.yaml on every request. Keyed per store; invalidated on edit.
_COLUMN_CONFIG_CACHE: dict[str, tuple] = {}


def _annotation_column_config(store: SqliteStore) -> tuple[int, list[str], str, dict[str, str]]:
    """Returns (replicas, annotator_targets, arbiter_target, model_names), cached
    by the mtimes of workflow.yaml + llm_profiles.yaml. A project without a
    workflow.yaml is the default single-annotation shape; a malformed one raises."""
    workflow_path = store.root / "workflow.yaml"
    profiles_path = store.root.parent.parent / "llm_profiles.yaml"

    def _mtime(path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    signature = (_mtime(workflow_path), _mtime(profiles_path))
    key = str(store.root)
    cached = _COLUMN_CONFIG_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    from annotation_pipeline_skill.config.loader import load_annotation_config

    ann = load_annotation_config(store.root.parent)
    replicas = int(ann.replicas or 1)
    targets = list(ann.targets or ["annotation"])
    arb_target = ann.arbiter_target or "arbiter"
    models = _resolve_models(store, [*targets, arb_target]) if replicas > 1 else {}
    result = (replicas, targets, arb_target, models)
    _COLUMN_CONFIG_CACHE[key] = (signature, result)
    return result


def _internal_columns(store: SqliteStore, tasks: list[Task]):
    """Internal-board columns as ``(column_id, title, predicate(task) -> bool)``,
    adapted to the pipeline mode.

    Multi-annotation (replicas > 1) runs the N annotators in PARALLEL then the
    arbiter, all inside the single ANNOTATING status, and disables QC. We expand
    that into a per-annotator column for each annotator plus an Arbiter column,
    using the live ``runtime_substage`` the runtime stamps on the task:
    ``annotating`` → an in-flight task shows in EVERY annotator column at once
    (they run concurrently); ``arbitrating`` → it shows only in the Arbiter
    column. The QC column is dropped — EXCEPT when a task is actually in QC (a
    scheduler-resume edge case can transition a task there), so no task is ever
    invisible. ``tasks`` is the (already project-filtered) task list."""
    def by_status(status: TaskStatus):
        return lambda task, _s=status: task.status is _s

    replicas, targets, arb_target, models = _annotation_column_config(store)
    if replicas <= 1:
        return [(cid, title, by_status(status)) for cid, title, status in KANBAN_COLUMNS]

    def in_annotating(task: Task) -> bool:
        return task.status is TaskStatus.ANNOTATING and _task_substage(task) != "arbitrating"

    def in_arbitrating(task: Task) -> bool:
        return task.status is TaskStatus.ARBITRATING or (
            task.status is TaskStatus.ANNOTATING and _task_substage(task) == "arbitrating"
        )

    cols = [("pending", "Pending", by_status(TaskStatus.PENDING))]
    for i, target in enumerate(targets):
        cols.append((f"annotator_{i + 1}", f"Annotator {i + 1}: {models.get(target, target)}", in_annotating))
    cols.append(("arbiter", f"Arbiter: {models.get(arb_target, arb_target)}", in_arbitrating))
    # Only surface QC when something is actually there (resume edge case) — keeps
    # the always-empty column hidden in the normal accept_directly flow but never
    # lets a QC-status task fall through every predicate and vanish.
    if any(task.status is TaskStatus.QC for task in tasks):
        cols.append(("qc", "QC", by_status(TaskStatus.QC)))
    cols.append(("human_review", "Human Review", by_status(TaskStatus.HUMAN_REVIEW)))
    cols.append(("accepted", "Accepted", by_status(TaskStatus.ACCEPTED)))
    cols.append(("rejected", "Rejected", by_status(TaskStatus.REJECTED)))
    return cols


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
    index = _dashboard_index(store, tasks)
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
                    "cards": [_task_card(index, task) for task in tasks if predicate(task)],
                }
            for column_id, title, predicate in _internal_columns(store, tasks)
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
    for task in tasks:
        status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1

    task_id_set = {t.task_id for t in tasks}
    # One grouped query instead of a per-task consensus-summary N+1.
    open_feedback_count = store.open_feedback_count_for_tasks(list(task_id_set))
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


def _dashboard_index(store: SqliteStore, tasks: list[Task]) -> dict:
    # Bulk-load only what the visible cards need, scoped to ``tasks`` (the
    # project's tasks when a project is selected). The previous version walked
    # the entire store calling list_attempts()/list_feedback() per task — an
    # N+1 that made a 3.6k-task board take ~3s. Two grouped queries instead.
    task_ids = [task.task_id for task in tasks]
    attempts_by_task = store.attempt_cards_by_task(task_ids)
    feedback_counts = store.feedback_counts_by_task(task_ids)
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
        "retry_wait_seconds": max(0, int((task.next_retry_at - datetime.now(timezone.utc)).total_seconds())) if task.next_retry_at is not None else None,
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

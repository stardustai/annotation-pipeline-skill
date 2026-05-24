from __future__ import annotations

from datetime import datetime, timezone

from annotation_pipeline_skill.core.runtime import (
    CapacitySnapshot,
    QueueCounts,
    RuntimeConfig,
    RuntimeSnapshot,
    RuntimeStatus,
)
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.services.dashboard_service import build_project_summaries
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def build_runtime_snapshot(
    store: SqliteStore,
    config: RuntimeConfig,
    *,
    now: datetime | None = None,
) -> RuntimeSnapshot:
    generated_at = now or datetime.now(timezone.utc)
    tasks = store.list_tasks()
    active_runs = store.list_active_runs()
    leases = store.list_runtime_leases()
    heartbeat_at = store.load_runtime_heartbeat()

    errors: list[str] = []
    heartbeat_age_seconds: int | None = None
    if heartbeat_at is None:
        errors.append("heartbeat_missing")
    else:
        heartbeat_age_seconds = int((generated_at - heartbeat_at).total_seconds())
        stale_heartbeat_after = max(config.snapshot_interval_seconds * 2, 120)
        if heartbeat_age_seconds > stale_heartbeat_after:
            errors.append("heartbeat_stale")

    stale_tasks = sorted(
        run.task_id
        for run in active_runs
        if (generated_at - run.heartbeat_at).total_seconds() > config.stale_after_seconds
    )
    if stale_tasks:
        errors.append("stale_active_runs")
    stale_leases = sorted(
        lease.lease_id
        for lease in leases
        if lease.expires_at <= generated_at
    )
    if stale_leases:
        errors.append("stale_runtime_leases")

    active_count = max(len(leases), len(active_runs))
    queue_counts = _build_queue_counts(tasks)
    capacity = CapacitySnapshot(
        max_concurrent_tasks=config.max_concurrent_tasks,
        active_count=active_count,
        available_slots=max(config.max_concurrent_tasks - active_count, 0),
    )
    _claimable = {TaskStatus.PENDING, TaskStatus.QC, TaskStatus.ARBITRATING, TaskStatus.ANNOTATING}
    due_retries = sorted(
        task.task_id
        for task in tasks
        if task.next_retry_at is not None
        and task.next_retry_at <= generated_at
        and task.status in _claimable
    )

    return RuntimeSnapshot(
        generated_at=generated_at,
        runtime_status=RuntimeStatus(
            healthy=not errors,
            heartbeat_at=heartbeat_at,
            heartbeat_age_seconds=heartbeat_age_seconds,
            active=heartbeat_at is not None,
            errors=errors,
        ),
        queue_counts=queue_counts,
        active_runs=active_runs,
        leases=leases,
        capacity=capacity,
        stale_tasks=stale_tasks,
        stale_leases=stale_leases,
        due_retries=due_retries,
        project_summaries=build_project_summaries(store)["projects"],
    )


def _build_queue_counts(tasks) -> QueueCounts:
    status_counts = {
        status: sum(1 for task in tasks if task.status is status)
        for status in TaskStatus
    }
    return QueueCounts(
        draft=status_counts[TaskStatus.DRAFT],
        pending=status_counts[TaskStatus.PENDING],
        annotating=status_counts[TaskStatus.ANNOTATING],
        qc=status_counts[TaskStatus.QC],
        arbitrating=status_counts[TaskStatus.ARBITRATING],
        human_review=status_counts[TaskStatus.HUMAN_REVIEW],
        accepted=status_counts[TaskStatus.ACCEPTED],
        rejected=status_counts[TaskStatus.REJECTED],
        blocked=status_counts[TaskStatus.BLOCKED],
        cancelled=status_counts[TaskStatus.CANCELLED],
    )

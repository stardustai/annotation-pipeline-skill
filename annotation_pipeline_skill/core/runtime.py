from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from annotation_pipeline_skill.core.models import _dt_from_str, _dt_to_str, utc_now  # noqa: F401


@dataclass(frozen=True)
class RuntimeConfig:
    # Number of concurrent worker coroutines. Each worker runs its own
    # annotation→validation→QC pipeline per task. There is no batch/cycle
    # boundary — when a worker finishes it immediately claims the next
    # PENDING task.
    max_concurrent_tasks: int = 4
    stale_after_seconds: int = 600
    retry_delay_seconds: int = 3600
    # How often the runtime observer writes a fresh RuntimeSnapshot + heartbeat
    # while the worker pool is running.
    snapshot_interval_seconds: int = 30
    max_qc_rounds: int = 3
    # Hard upper bound on a single task's full pipeline run (annot+QC+arbiter+
    # any retries). If a worker's run_task_async hangs past this, the worker
    # cancels the call so the finally clause can release the lease/active_run
    # and the task gets recycled. Default 900s (15 min) covers even the slowest
    # codex-arbitration round; any LLM call that takes longer than that is
    # almost certainly a stuck HTTP/CLI subprocess, not real work.
    worker_task_timeout_seconds: int = 900
    # On scheduler restart, give worker pool this long to naturally claim
    # ANNOTATING / QC orphans (resuming them from existing artifacts) before
    # the observer sweeps any leftovers back to PENDING. 60s is enough for
    # 24 workers to cycle through ~hundreds of in-flight tasks.
    resume_settle_seconds: int = 60
    # Max times to retry the arbiter LLM call when its corrected_annotation
    # contains a non-verbatim span. Each retry tells the model exactly which
    # span failed so it can fix the specific issue. After this many retries
    # all fail, the task falls through to HUMAN_REVIEW. Default 2 → up to 3
    # arbiter calls per dispute (initial + 2 retries).
    arbiter_verbatim_retries: int = 2
    # Project-level QC sampling policy. Applies to all tasks in this project
    # unless an individual task carries a legacy ``metadata.qc_policy`` override
    # (kept only for backward-compat with tasks imported before this lift).
    qc_sample_mode: str = "sample_ratio"   # "sample_ratio" | "sample_count"
    qc_sample_ratio: float = 1.0
    qc_sample_count: int | None = None

    def to_dict(self) -> dict:
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "stale_after_seconds": self.stale_after_seconds,
            "retry_delay_seconds": self.retry_delay_seconds,
            "snapshot_interval_seconds": self.snapshot_interval_seconds,
            "max_qc_rounds": self.max_qc_rounds,
            "worker_task_timeout_seconds": self.worker_task_timeout_seconds,
            "resume_settle_seconds": self.resume_settle_seconds,
            "arbiter_verbatim_retries": self.arbiter_verbatim_retries,
            "qc_sample_mode": self.qc_sample_mode,
            "qc_sample_ratio": self.qc_sample_ratio,
            "qc_sample_count": self.qc_sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RuntimeConfig:
        return cls(
            max_concurrent_tasks=data.get("max_concurrent_tasks", 4),
            stale_after_seconds=data.get("stale_after_seconds", 600),
            retry_delay_seconds=data.get("retry_delay_seconds", 3600),
            snapshot_interval_seconds=int(data.get("snapshot_interval_seconds", 30)),
            max_qc_rounds=data.get("max_qc_rounds", 3),
            worker_task_timeout_seconds=int(data.get("worker_task_timeout_seconds", 900)),
            resume_settle_seconds=int(data.get("resume_settle_seconds", 60)),
            arbiter_verbatim_retries=int(data.get("arbiter_verbatim_retries", 2)),
            qc_sample_mode=data.get("qc_sample_mode", "sample_ratio"),
            qc_sample_ratio=float(data.get("qc_sample_ratio", 1.0)),
            qc_sample_count=data.get("qc_sample_count"),
        )


@dataclass(frozen=True)
class AnnotationConfig:
    """Annotation-stage topology, parsed from workflow.yaml `stages.annotation`.

    replicas == 1 reproduces the legacy single-annotator flow exactly.
    replicas > 1 runs N annotators (one per entry in `targets`), keeps spans
    agreed by >= keep_threshold of them, and routes the rest to `arbiter_target`.
    """
    replicas: int = 1
    targets: list[str] = field(default_factory=lambda: ["annotation"])
    keep_threshold: int = 1
    on_disagree: str = "arbiter"   # "arbiter" (resolve+fill) | "drop" (skip below-threshold)
    arbiter_target: str = "arbiter"

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationConfig":
        data = data or {}
        replicas = int(data.get("replicas", 1))
        targets = list(data.get("targets") or ["annotation"])
        # Broadcast a single target to N replicas (same model run N times).
        if len(targets) == 1 and replicas > 1:
            targets = targets * replicas
        keep_threshold = int(data.get("keep_threshold", replicas))
        return cls(
            replicas=replicas,
            targets=targets,
            keep_threshold=keep_threshold,
            on_disagree=str(data.get("on_disagree", "arbiter")),
            arbiter_target=str(data.get("arbiter_target", "arbiter")),
        )

    def validate(self) -> None:
        if self.replicas < 1:
            raise ValueError(f"annotation.replicas must be >= 1, got {self.replicas}")
        if len(self.targets) != self.replicas:
            raise ValueError(
                f"annotation.targets must list exactly replicas={self.replicas} entries, "
                f"got {len(self.targets)}: {self.targets}"
            )
        if not (1 <= self.keep_threshold <= self.replicas):
            raise ValueError(
                f"annotation.keep_threshold must be in [1, replicas={self.replicas}], "
                f"got {self.keep_threshold}"
            )
        if self.on_disagree not in {"arbiter", "drop"}:
            raise ValueError(f"annotation.on_disagree must be 'arbiter' or 'drop', got {self.on_disagree!r}")


@dataclass(frozen=True)
class ActiveRun:
    run_id: str
    task_id: str
    stage: str
    attempt_id: str
    provider_target: str
    started_at: datetime
    heartbeat_at: datetime
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "stage": self.stage,
            "attempt_id": self.attempt_id,
            "provider_target": self.provider_target,
            "started_at": _dt_to_str(self.started_at),
            "heartbeat_at": _dt_to_str(self.heartbeat_at),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ActiveRun:
        return cls(
            run_id=data["run_id"],
            task_id=data["task_id"],
            stage=data["stage"],
            attempt_id=data["attempt_id"],
            provider_target=data["provider_target"],
            started_at=_dt_from_str(data["started_at"]),
            heartbeat_at=_dt_from_str(data["heartbeat_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class RuntimeLease:
    lease_id: str
    task_id: str
    stage: str
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime
    owner: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "stage": self.stage,
            "acquired_at": _dt_to_str(self.acquired_at),
            "heartbeat_at": _dt_to_str(self.heartbeat_at),
            "expires_at": _dt_to_str(self.expires_at),
            "owner": self.owner,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RuntimeLease:
        return cls(
            lease_id=data["lease_id"],
            task_id=data["task_id"],
            stage=data["stage"],
            acquired_at=_dt_from_str(data["acquired_at"]),
            heartbeat_at=_dt_from_str(data["heartbeat_at"]),
            expires_at=_dt_from_str(data["expires_at"]),
            owner=data["owner"],
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class RuntimeStatus:
    healthy: bool
    heartbeat_at: datetime | None
    heartbeat_age_seconds: int | None
    active: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "heartbeat_at": _dt_to_str(self.heartbeat_at),
            "heartbeat_age_seconds": self.heartbeat_age_seconds,
            "active": self.active,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RuntimeStatus:
        return cls(
            healthy=data["healthy"],
            heartbeat_at=_dt_from_str(data.get("heartbeat_at")),
            heartbeat_age_seconds=data.get("heartbeat_age_seconds"),
            active=data["active"],
            errors=data.get("errors", []),
        )


@dataclass(frozen=True)
class QueueCounts:
    pending: int
    annotating: int
    qc: int
    human_review: int
    accepted: int
    rejected: int
    draft: int = 0
    arbitrating: int = 0
    blocked: int = 0
    cancelled: int = 0

    def to_dict(self) -> dict:
        return {
            "draft": self.draft,
            "pending": self.pending,
            "annotating": self.annotating,
            "qc": self.qc,
            "arbitrating": self.arbitrating,
            "human_review": self.human_review,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "blocked": self.blocked,
            "cancelled": self.cancelled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> QueueCounts:
        return cls(
            draft=data["draft"],
            pending=data["pending"],
            annotating=data["annotating"],
            qc=data["qc"],
            arbitrating=data.get("arbitrating", 0),
            human_review=data["human_review"],
            accepted=data["accepted"],
            rejected=data["rejected"],
            blocked=data["blocked"],
            cancelled=data["cancelled"],
        )


@dataclass(frozen=True)
class CapacitySnapshot:
    max_concurrent_tasks: int
    active_count: int
    available_slots: int

    def to_dict(self) -> dict:
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "active_count": self.active_count,
            "available_slots": self.available_slots,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CapacitySnapshot:
        return cls(
            max_concurrent_tasks=data["max_concurrent_tasks"],
            active_count=data["active_count"],
            available_slots=data["available_slots"],
        )


@dataclass(frozen=True)
class RuntimeSnapshot:
    generated_at: datetime
    runtime_status: RuntimeStatus
    queue_counts: QueueCounts
    active_runs: list[ActiveRun]
    capacity: CapacitySnapshot
    stale_tasks: list[str]
    due_retries: list[str]
    project_summaries: list[dict]
    leases: list[RuntimeLease] = field(default_factory=list)
    stale_leases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": _dt_to_str(self.generated_at),
            "runtime_status": self.runtime_status.to_dict(),
            "queue_counts": self.queue_counts.to_dict(),
            "active_runs": [run.to_dict() for run in self.active_runs],
            "leases": [lease.to_dict() for lease in self.leases],
            "capacity": self.capacity.to_dict(),
            "stale_tasks": self.stale_tasks,
            "stale_leases": self.stale_leases,
            "due_retries": self.due_retries,
            "project_summaries": self.project_summaries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RuntimeSnapshot:
        return cls(
            generated_at=_dt_from_str(data["generated_at"]),
            runtime_status=RuntimeStatus.from_dict(data["runtime_status"]),
            queue_counts=QueueCounts.from_dict(data["queue_counts"]),
            active_runs=[ActiveRun.from_dict(item) for item in data.get("active_runs", [])],
            leases=[RuntimeLease.from_dict(item) for item in data.get("leases", [])],
            capacity=CapacitySnapshot.from_dict(data["capacity"]),
            stale_tasks=list(data.get("stale_tasks", [])),
            stale_leases=list(data.get("stale_leases", [])),
            due_retries=list(data.get("due_retries", [])),
            project_summaries=list(data.get("project_summaries", [])),
        )

from annotation_pipeline_skill.core.models import AuditEvent, Task, utc_now
from annotation_pipeline_skill.core.states import TaskStatus


class InvalidTransition(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.DRAFT: {TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
    TaskStatus.PENDING: {TaskStatus.ANNOTATING, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
    TaskStatus.ANNOTATING: {
        TaskStatus.QC,
        TaskStatus.PENDING,
        TaskStatus.ARBITRATING,
        TaskStatus.HUMAN_REVIEW,
        TaskStatus.ACCEPTED,
        TaskStatus.REJECTED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.QC: {
        TaskStatus.PENDING,
        TaskStatus.ACCEPTED,
        TaskStatus.ARBITRATING,
        TaskStatus.HUMAN_REVIEW,
        TaskStatus.ANNOTATING,
        TaskStatus.REJECTED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.ARBITRATING: {
        TaskStatus.PENDING,
        TaskStatus.ACCEPTED,
        TaskStatus.REJECTED,
        TaskStatus.HUMAN_REVIEW,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.HUMAN_REVIEW: {
        TaskStatus.ACCEPTED,
        TaskStatus.ANNOTATING,
        TaskStatus.ARBITRATING,
        TaskStatus.REJECTED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    },
    # ACCEPTED is the normal terminal state. Allowed outgoing transitions
    # cover operator audits — scanning already-accepted tasks for a quality
    # regression. Two paths: ARBITRATING (re-run the LLM arbiter on the
    # whole task) or HUMAN_REVIEW (operator wants to hand-correct a
    # specific span without invoking the arbiter; e.g. Posterior Audit's
    # in-place "Set type & Submit" flow uses this).
    TaskStatus.ACCEPTED: {TaskStatus.ARBITRATING, TaskStatus.HUMAN_REVIEW},
    TaskStatus.REJECTED: {TaskStatus.ARBITRATING},
    TaskStatus.BLOCKED: {TaskStatus.PENDING},
    TaskStatus.CANCELLED: set(),
}


def transition_task(
    task: Task,
    next_status: TaskStatus,
    actor: str,
    reason: str,
    stage: str,
    attempt_id: str | None = None,
    metadata: dict | None = None,
) -> AuditEvent:
    previous_status = task.status
    if next_status not in ALLOWED_TRANSITIONS[previous_status]:
        raise InvalidTransition(f"cannot transition task {task.task_id} from {previous_status.value} to {next_status.value}")

    task.status = next_status
    task.updated_at = utc_now()
    return AuditEvent.new(
        task_id=task.task_id,
        previous_status=previous_status,
        next_status=next_status,
        actor=actor,
        reason=reason,
        stage=stage,
        attempt_id=attempt_id,
        metadata=metadata,
    )

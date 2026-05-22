"""
One-off: rewind HR → ARBITRATING for tasks falsely routed there during
the 2026-05-21 DeepSeek 402 "Insufficient Balance" outage.

The arbiter's claude_cli subprocess exited rc=1 on every call (the 402
error was emitted as a `result.is_error=true` stream-json event on stdout
that the parser dropped). Each call counted as a mechanical retry; 3
retries → HR. The data is fine, the LLM just had no balance.

Eligibility (and only eligibility — script does nothing else):
  - task.status == 'human_review' RIGHT NOW
  - has an audit_event with next_status='human_review' AND
    reason LIKE '%kept failing to return a usable answer%' AND
    created_at >= --since (default 2026-05-21T20:50:00Z, when the 402
    burst started)

Action per eligible task:
  - Reset task.metadata: clear arbiter_mechanical_retries,
    arbiter_verbatim_bail_count, arbiter_last_exception_class,
    arbiter_last_exception_message (otherwise the first new failure
    immediately bumps to 4 and re-routes to HR)
  - Transition HUMAN_REVIEW → ARBITRATING with audit reason citing the
    402 outage so the trail is traceable

Usage:
    python scripts/rewind_402_false_hr.py                # dry-run, shows count + sample
    python scripts/rewind_402_false_hr.py --apply        # actually rewind
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.core.transitions import (
    InvalidTransition,
    transition_task,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true",
                help="actually rewind; without this flag the script prints "
                     "the candidate count and sample but does nothing.")
ap.add_argument("--project-root", default="projects/v3_initial_deployment")
ap.add_argument("--since", default="2026-05-21T20:50:00Z",
                help="ISO-UTC cutoff for the HR-entry timestamp; only HR "
                     "entries on or after this time are considered. Default "
                     "is the 402-burst start.")
ap.add_argument("--limit", type=int, default=None,
                help="max tasks to act on (useful for cautious first batch).")
args = ap.parse_args()

root = pathlib.Path(args.project_root) / ".annotation-pipeline"
store = SqliteStore.open(root)
conn = store._conn

# Pull eligible task_ids: current status HR + matching mechanical-fail
# audit entry since cutoff.
rows = conn.execute(
    """
    SELECT DISTINCT t.task_id
    FROM tasks t
    JOIN audit_events e ON e.task_id = t.task_id
    WHERE t.status = 'human_review'
      AND e.next_status = 'human_review'
      AND e.reason LIKE '%kept failing to return a usable answer%'
      AND e.created_at >= ?
    """,
    (args.since,),
).fetchall()
task_ids = [r[0] for r in rows]
if args.limit is not None:
    task_ids = task_ids[: args.limit]

print(f"candidate tasks: {len(task_ids)}")
if not task_ids:
    sys.exit(0)
print(f"sample (first 5): {task_ids[:5]}")

if not args.apply:
    print("\n[dry-run] re-run with --apply to actually rewind.")
    sys.exit(0)

# Apply. Each task: reset counters, transition HR → ARBITRATING.
rewound = 0
failed: list[tuple[str, str]] = []
reason = (
    "rewind: 2026-05-21 DeepSeek 402 outage falsely routed this task to "
    "HR (LocalCLIExecutionError mistakenly classified as mechanical fail "
    "before the is_error event parser was added); replaying arbitration "
    "against the now-healthy provider"
)
for tid in task_ids:
    try:
        task = store.load_task(tid)
    except Exception as exc:  # noqa: BLE001
        failed.append((tid, f"load: {exc}"))
        continue
    # Reset the per-task counters so the next arbiter pickup starts from
    # zero. Without this, a single failed call would immediately bump
    # from 3 to 4 and route straight back to HR.
    for key in (
        "arbiter_mechanical_retries",
        "arbiter_verbatim_bail_count",
        "arbiter_last_exception_class",
        "arbiter_last_exception_message",
    ):
        task.metadata.pop(key, None)
    try:
        event = transition_task(
            task,
            TaskStatus.ARBITRATING,
            actor="rewind_script",
            reason=reason,
            stage="recovery",
            metadata={
                "recovery": "rewind_402_false_hr",
                "previous_status": "human_review",
            },
        )
    except InvalidTransition as exc:
        failed.append((tid, f"transition: {exc}"))
        continue
    store.save_task(task)
    store.append_event(event)
    rewound += 1

print(f"\nrewound: {rewound}")
if failed:
    print(f"failed: {len(failed)}")
    for tid, msg in failed[:10]:
        print(f"  {tid}  {msg}")

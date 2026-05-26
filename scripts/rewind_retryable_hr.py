"""
Rewind tasks falsely / transiently routed to HUMAN_REVIEW.

Four categories, all retryable without human judgment:

  total_bail_cap (5)
    Hit the 10-bail circuit breaker with transient (rate-limit / 5xx) errors.
    Fixed: transient bails are now exempt from TOTAL_BAIL_CAP (2026-05-22).
    Action: HR → PENDING, reset worker bail counters.

  arbiter_parse_fail (61)
    Arbiter hit JSON-parse / shape errors 3-4 times in a row.
    Likely a transient LLM quality glitch; replaying arbitration may succeed.
    Action: HR → ARBITRATING, reset arbiter counters.

  second_arbiter_null_verdict (65)
    Second arbiter returned null corrected_annotation.
    Null-output bug; replaying arbitration may produce a usable verdict.
    Action: HR → ARBITRATING, reset arbiter counters.

  arbiter_uncertain (~573)
    Arbiter flagged its own answer as uncertain (tentative/unsure verdict).
    Root cause: missing row text in slim prompt (P3) + stale QC feedback from
    prior rounds (P2). Both fixed 2026-05-23. With the fix, the arbiter sees
    full row text and only the latest round's feedback, so it can be confident.
    If still uncertain after replay, the new second-arbiter path kicks in.
    Action: HR → ARBITRATING, reset arbiter counters (no uncertain flag set —
    let them go through the fresh arbiter first with the improved prompt).

Eligibility is based on the MOST RECENT audit_event that transitioned the
task TO human_review — so only tasks whose last HR entry matches the pattern
are touched (tasks that went to HR for multiple reasons use the latest reason).

Usage:
    python scripts/rewind_retryable_hr.py             # dry-run
    python scripts/rewind_retryable_hr.py --apply     # actually rewind
    python scripts/rewind_retryable_hr.py --apply --limit 20  # cautious first batch
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.core.transitions import InvalidTransition, transition_task
from annotation_pipeline_skill.store.sqlite_store import SqliteStore

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true",
                help="actually rewind; without this flag prints candidates and exits")
ap.add_argument("--project-root", default="projects/v3_initial_deployment")
ap.add_argument("--limit", type=int, default=None,
                help="max tasks to rewind per category (for cautious batching)")
args = ap.parse_args()

root = pathlib.Path(args.project_root) / ".annotation-pipeline"
store = SqliteStore.open(root)
conn = store._conn

# For each current HR task, find the most recent audit event that moved it
# INTO human_review (next_status = 'human_review').
LATEST_HR_REASON_SQL = """
WITH latest_hr AS (
  SELECT
    ae.task_id,
    ae.reason,
    ae.previous_status,
    ROW_NUMBER() OVER (PARTITION BY ae.task_id ORDER BY ae.seq DESC) AS rn
  FROM audit_events ae
  JOIN tasks t ON ae.task_id = t.task_id
  WHERE t.status = 'human_review'
    AND ae.next_status = 'human_review'
)
SELECT task_id, reason, previous_status
FROM latest_hr
WHERE rn = 1
  AND ({where})
"""

ARBITER_RESET_KEYS = (
    "arbiter_mechanical_retries",
    "arbiter_verbatim_bail_count",
    "arbiter_transient_bail_count",
    "arbiter_last_exception_class",
    "arbiter_last_exception_message",
)

WORKER_RESET_KEYS = (
    "worker_bail_count",
    "worker_permanent_bail_count",
    "last_provider_error",
)


def fetch_candidates(where_clause: str) -> list[tuple[str, str, str]]:
    sql = LATEST_HR_REASON_SQL.format(where=where_clause)
    return conn.execute(sql).fetchall()


categories = [
    {
        "name": "total_bail_cap",
        "where": "reason LIKE '%consecutive times%'",
        # HR → PENDING is not in the transition graph; use ANNOTATING as the
        # intermediate hop (same path the bulk-rewind script uses).
        "target_status": TaskStatus.ANNOTATING,
        "reset_keys": WORKER_RESET_KEYS,
        "clear_next_retry_at": True,
        "rewind_reason": (
            "rewind: task hit TOTAL_BAIL_CAP with transient (rate-limit/5xx) errors "
            "that should not escalate to HR — fixed 2026-05-22; replaying annotation"
        ),
        "recovery_tag": "rewind_transient_bail_cap",
    },
    {
        "name": "arbiter_parse_fail",
        "where": "reason LIKE '%usable answer%'",
        "target_status": TaskStatus.ARBITRATING,
        "reset_keys": ARBITER_RESET_KEYS,
        "clear_next_retry_at": False,
        "rewind_reason": (
            "rewind: arbiter hit repeated JSON-parse / shape errors (transient LLM "
            "quality glitch); replaying arbitration with reset counters"
        ),
        "recovery_tag": "rewind_arbiter_parse_fail",
    },
    {
        "name": "second_arbiter_null_verdict",
        "where": "reason LIKE '%Second arbiter%'",
        "target_status": TaskStatus.ARBITRATING,
        "reset_keys": ARBITER_RESET_KEYS,
        "clear_next_retry_at": False,
        "rewind_reason": (
            "rewind: second arbiter returned null corrected_annotation (output bug); "
            "replaying arbitration with reset counters"
        ),
        "recovery_tag": "rewind_second_arbiter_null",
    },
    {
        "name": "arbiter_uncertain",
        "where": "reason LIKE '%flagged its own answer as uncertain%'",
        "target_status": TaskStatus.ARBITRATING,
        "reset_keys": ARBITER_RESET_KEYS,
        "clear_next_retry_at": False,
        "rewind_reason": (
            "rewind: arbiter was uncertain due to missing row text in slim prompt (P3) "
            "and stale QC feedback from prior rounds (P2); both fixed 2026-05-23 — "
            "replaying arbitration with improved prompt"
        ),
        "recovery_tag": "rewind_arbiter_uncertain_p2p3_fix",
    },
]

total_candidates = 0
for cat in categories:
    rows = fetch_candidates(cat["where"])
    if args.limit is not None:
        rows = rows[: args.limit]
    cat["rows"] = rows
    print(f"  {cat['name']}: {len(rows)} candidates")
    total_candidates += len(rows)

print(f"\ntotal candidates: {total_candidates}")

if not args.apply:
    for cat in categories:
        if cat["rows"]:
            sample_ids = [r[0] for r in cat["rows"][:3]]
            print(f"  {cat['name']} sample: {sample_ids}")
    print("\n[dry-run] re-run with --apply to actually rewind.")
    sys.exit(0)

print()
grand_rewound = 0
grand_failed = 0

for cat in categories:
    rewound = 0
    failed: list[tuple[str, str]] = []
    for task_id, _reason, _prev in cat["rows"]:
        try:
            task = store.load_task(task_id)
        except Exception as exc:  # noqa: BLE001
            failed.append((task_id, f"load: {exc}"))
            continue

        for key in cat["reset_keys"]:
            task.metadata.pop(key, None)
        if cat["clear_next_retry_at"]:
            task.next_retry_at = None

        try:
            event = transition_task(
                task,
                cat["target_status"],
                actor="rewind_script",
                reason=cat["rewind_reason"],
                stage="recovery",
                metadata={
                    "recovery": cat["recovery_tag"],
                    "previous_status": "human_review",
                },
            )
        except InvalidTransition as exc:
            failed.append((task_id, f"transition: {exc}"))
            continue

        store.save_task(task)
        store.append_event(event)
        rewound += 1

    print(f"  {cat['name']}: rewound {rewound}", end="")
    if failed:
        print(f", failed {len(failed)}: {[t for t, _ in failed[:5]]}", end="")
    print()
    grand_rewound += rewound
    grand_failed += len(failed)

print(f"\ntotal rewound: {grand_rewound}  failed: {grand_failed}")

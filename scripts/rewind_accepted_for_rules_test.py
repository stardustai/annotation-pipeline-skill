"""Push 10 ACCEPTED tasks back to PENDING so they re-run annotation under
the current versioned annotation rules. Diagnostic only — used to verify
that the prompt now picks up annotation_rules.yaml content and that the
schema-driven (de-hardcoded) instructions actually flow through.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.core.transitions import transition_task, InvalidTransition
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-root", required=True, help="e.g. projects/v3_initial_deployment/.annotation-pipeline")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    store = SqliteStore.open(Path(args.store_root))
    rows = store._conn.execute(
        """
        SELECT task_id FROM tasks
        WHERE status='accepted'
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (args.limit,),
    ).fetchall()
    task_ids = [r["task_id"] for r in rows]
    print(f"Rewinding {len(task_ids)} ACCEPTED tasks → PENDING")
    for tid in task_ids:
        print(f"  {tid}")

    if not args.apply:
        print("DRY RUN — pass --apply to actually transition.", file=sys.stderr)
        return 0

    transitioned = 0
    failed = 0
    for tid in task_ids:
        task = store.load_task(tid)
        # Reset state machinery so a fresh annotation pass kicks off.
        task.metadata.pop("prior_verifier_first_arbiter_divergent", None)
        task.metadata.pop("prior_verifier_payload", None)
        task.metadata.pop("prior_verifier_action", None)
        task.metadata.pop("arbiter_mechanical_retries", None)
        # Force the scheduler to treat this as a fresh annotation run
        # rather than picking up an existing annotation_result artifact.
        task.metadata["hr_request_changes"] = True
        task.metadata["rules_smoke_test"] = True
        # Legal path: ACCEPTED → HUMAN_REVIEW → ANNOTATING. The
        # hr_request_changes flag we set above tells the scheduler this
        # is a fresh annotation pass, not a resume of an existing
        # annotation_result artifact.
        try:
            event_hr = transition_task(
                task,
                TaskStatus.HUMAN_REVIEW,
                actor="rules-smoke-test",
                reason="rewind accepted → HR (intermediate hop for re-annotation)",
                stage="recovery",
                metadata={"recovery": "rules_smoke_test"},
            )
            store.append_event(event_hr)
            event_ann = transition_task(
                task,
                TaskStatus.ANNOTATING,
                actor="rules-smoke-test",
                reason="rewind HR → ANNOTATING (re-run annotation under current rules)",
                stage="annotation",
                metadata={"recovery": "rules_smoke_test"},
            )
        except InvalidTransition as exc:
            print(f"  SKIP {tid}: {exc}", file=sys.stderr)
            failed += 1
            continue
        store.save_task(task)
        store.append_event(event_ann)
        transitioned += 1
        print(f"  OK   {tid} → ANNOTATING")

    print(f"Transitioned {transitioned}; failed {failed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

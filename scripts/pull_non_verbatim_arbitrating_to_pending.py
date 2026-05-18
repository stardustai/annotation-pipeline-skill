"""Pull ARBITRATING tasks with non-verbatim spans back to PENDING.

Companion to `recheck_prior_overrides.py`: that script transitioned 241
ACCEPTED-with-suspect-prior tasks to ARBITRATING so the (fixed) second
arbiter could re-evaluate them. But some of those tasks ALSO have
non-verbatim spans in their current annotation — these are data-quality
defects, not adjudication defects, so they should be re-annotated from
scratch rather than re-arbitrated.

This script finds ARBITRATING tasks (optionally filtered by metadata
predicate) whose latest annotation contains a verbatim violation and
transitions them ARBITRATING → PENDING so the normal annotation worker
picks them up. Clears prior_verifier_first_arbiter_divergent so the
second-arbiter resolver doesn't fight the re-annotation.

Usage:
    python -m scripts.pull_non_verbatim_arbitrating_to_pending \\
        --project v3_initial_deployment \\
        --workspace projects/v3_initial_deployment/.annotation-pipeline \\
        --only-recheck-batch \\
        --apply

Without --apply, prints what would change.
--only-recheck-batch: limit to tasks tagged by recheck_prior_overrides
                       (recovery: prior_override_audit). Default scans
                       every ARBITRATING task in the project.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from annotation_pipeline_skill.core.schema_validation import find_verbatim_violations
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.core.transitions import (
    InvalidTransition,
    transition_task,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _load_latest_annotation(store: SqliteStore, task_id: str) -> dict | None:
    arts = store.list_artifacts(task_id)
    hr = [a for a in arts if a.kind == "human_review_answer"]
    if hr:
        outer = json.loads((store.root / hr[-1].path).read_text(encoding="utf-8"))
        return outer.get("answer") if isinstance(outer, dict) else None
    anns = [a for a in arts if a.kind == "annotation_result"]
    if not anns:
        return None
    outer = json.loads((store.root / anns[-1].path).read_text(encoding="utf-8"))
    text = outer.get("text")
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--only-recheck-batch", action="store_true",
                    help="Only consider ARBITRATING tasks that came from "
                         "recheck_prior_overrides (event metadata recovery="
                         "'prior_override_audit'). Default: every ARBITRATING task.")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--actor", default="pull_non_verbatim_script")
    args = ap.parse_args(argv)

    store = SqliteStore.open(Path(args.workspace))

    recheck_task_ids: set[str] | None = None
    if args.only_recheck_batch:
        recheck_task_ids = set()
        for task in store.list_tasks_by_pipeline(args.project):
            for event in store.list_events(task.task_id):
                if event.metadata.get("recheck") == "prior_override_audit":
                    recheck_task_ids.add(task.task_id)
                    break
        print(f"Recheck batch: {len(recheck_task_ids)} tasks tagged", file=sys.stderr)

    candidates = []
    for task in store.list_tasks_by_pipeline(args.project):
        if task.status is not TaskStatus.ARBITRATING:
            continue
        if recheck_task_ids is not None and task.task_id not in recheck_task_ids:
            continue
        candidates.append(task)
    print(f"ARBITRATING tasks in scope: {len(candidates)}", file=sys.stderr)

    to_pull = []
    skipped_no_annotation = 0
    skipped_verbatim_clean = 0
    for task in candidates:
        payload = _load_latest_annotation(store, task.task_id)
        if payload is None:
            skipped_no_annotation += 1
            continue
        violations = find_verbatim_violations(task, payload)
        if not violations:
            skipped_verbatim_clean += 1
            continue
        to_pull.append((task, violations))

    print(
        f"Pull candidates (non-verbatim): {len(to_pull)} "
        f"(skipped {skipped_verbatim_clean} verbatim-clean, "
        f"{skipped_no_annotation} no-annotation)",
        file=sys.stderr,
    )

    if not args.apply:
        print("DRY RUN — no changes. Pass --apply to transition.", file=sys.stderr)
        for task, viols in to_pull[:20]:
            first = viols[0]
            print(
                f"  {task.task_id}: {len(viols)} violation(s); "
                f"e.g. span={first['span']!r} at row {first['row_index']} "
                f"field {first['field']}"
            )
        if len(to_pull) > 20:
            print(f"  … and {len(to_pull) - 20} more", file=sys.stderr)
        return 0

    transitioned = 0
    failed = 0
    for task, violations in to_pull:
        # Clear divergence flag so the second-arbiter resolver doesn't
        # try to pick this up again; clear the divergent payload too.
        task.metadata.pop("prior_verifier_first_arbiter_divergent", None)
        task.metadata.pop("prior_verifier_payload", None)
        task.metadata.pop("prior_verifier_action", None)
        # Reset the mechanical-retry counter so the re-annotation pipeline
        # gets a fresh budget.
        task.metadata.pop("arbiter_mechanical_retries", None)
        try:
            event = transition_task(
                task,
                TaskStatus.PENDING,
                actor=args.actor,
                reason=(
                    f"pull-non-verbatim: latest annotation has "
                    f"{len(violations)} non-verbatim span(s) — re-annotation "
                    f"is the right fix, not re-arbitration"
                ),
                stage="recovery",
                metadata={
                    "recovery": "non_verbatim_to_pending",
                    "violation_count": len(violations),
                    "sample_violation": {
                        "row_index": violations[0]["row_index"],
                        "field": violations[0]["field"],
                        "span": violations[0]["span"],
                    },
                },
            )
        except InvalidTransition as exc:
            print(f"  SKIP {task.task_id}: {exc}", file=sys.stderr)
            failed += 1
            continue
        store.save_task(task)
        store.append_event(event)
        transitioned += 1

    print(f"Transitioned {transitioned} → PENDING ({failed} failed)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

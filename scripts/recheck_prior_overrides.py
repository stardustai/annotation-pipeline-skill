"""Re-check ACCEPTED tasks that the buggy second-arbiter path let through.

Background: before commit 9fde921, `_resolve_first_arbiter_divergence_async`
treated `corrected_annotation: null` from the second arbiter as "implicit
agreement with first arbiter" and overrode the project prior. Production
saw obvious errors like COVID-19 → technology accepted despite an 83/55
event-dominant prior.

This script finds every ACCEPTED task with
`metadata.prior_verifier_action == 'resolved_to_first'` (the tag the old
code stamped), reconstructs the prior-divergence state from the current
annotation + entity_statistics, and routes the task back to ARBITRATING
with the divergence flag set. The scheduler then reruns the (now fixed)
second-arbiter path — explicit affirmation required, or HR.

Usage:
    python -m scripts.recheck_prior_overrides --project v3_initial_deployment \\
        --workspace projects/v3_initial_deployment/.annotation-pipeline --apply

Without --apply, prints what would change without touching the store.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.core.transitions import (
    InvalidTransition,
    transition_task,
)
from annotation_pipeline_skill.services.entity_statistics_service import (
    EntityStatisticsService,
    iter_span_decisions,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _load_latest_annotation(store: SqliteStore, task_id: str) -> dict | None:
    """Mirror `build_posterior_audit._load_annotation` — prefer the most
    recent human_review_answer, fall back to the latest annotation_result.
    """
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
    ap.add_argument("--workspace", required=True,
                    help="Path to the project's .annotation-pipeline dir")
    ap.add_argument("--project", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Actually transition tasks; without this flag, dry-run only")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of tasks reset (useful for staged rollout)")
    ap.add_argument("--actor", default="recheck_prior_overrides_script")
    args = ap.parse_args(argv)

    store = SqliteStore.open(Path(args.workspace))
    stats = EntityStatisticsService(store)

    # Find ACCEPTED tasks tagged with the bug-era resolution.
    candidates = []
    for task in store.list_tasks_by_pipeline(args.project):
        if task.status is not TaskStatus.ACCEPTED:
            continue
        if task.metadata.get("prior_verifier_action") != "resolved_to_first":
            continue
        candidates.append(task)
    print(f"Found {len(candidates)} ACCEPTED tasks with prior_verifier_action='resolved_to_first'",
          file=sys.stderr)

    to_reset: list[tuple[object, dict]] = []  # (task, prior_verifier_payload)
    skipped_no_divergence = 0
    skipped_no_annotation = 0
    for task in candidates:
        payload = _load_latest_annotation(store, task.task_id)
        if payload is None:
            skipped_no_annotation += 1
            continue
        # Find the FIRST currently-divergent (span, type) — same target the
        # original prior_verifier flagged. If none is divergent now (e.g.
        # prior shifted since acceptance), skip; the task no longer needs
        # the second-arbiter re-run.
        synth_payload: dict | None = None
        for span, entity_type in iter_span_decisions(payload):
            r = stats.check(project_id=args.project, span=span, proposed_type=entity_type)
            if r.status != "divergent":
                continue
            synth_payload = {
                "span": r.span,
                "proposed_type": r.proposed_type,
                "dominant_type": r.dominant_type,
                "dominant_count": r.dominant_count,
                "total": r.total,
                "distribution": r.distribution,
            }
            break
        if synth_payload is None:
            skipped_no_divergence += 1
            continue
        to_reset.append((task, synth_payload))
        if args.limit is not None and len(to_reset) >= args.limit:
            break

    print(
        f"Reset candidates: {len(to_reset)} "
        f"(skipped {skipped_no_divergence} no-longer-divergent, "
        f"{skipped_no_annotation} no-parseable-annotation)",
        file=sys.stderr,
    )

    if not args.apply:
        print("DRY RUN — no changes. Pass --apply to transition.", file=sys.stderr)
        for task, sp in to_reset[:20]:
            print(
                f"  {task.task_id}: span={sp['span']!r} "
                f"first_type={sp['proposed_type']!r} prior={sp['dominant_type']!r} "
                f"({sp['dominant_count']}/{sp['total']})"
            )
        if len(to_reset) > 20:
            print(f"  … and {len(to_reset) - 20} more", file=sys.stderr)
        return 0

    transitioned = 0
    failed = 0
    for task, synth_payload in to_reset:
        # Tag divergence + payload so the scheduler picks it up via
        # _resolve_first_arbiter_divergence (NOT the normal rearbitrate path).
        task.metadata["prior_verifier_first_arbiter_divergent"] = True
        task.metadata["prior_verifier_payload"] = synth_payload
        # Drop the stale action tag so the new resolution writes a fresh one.
        task.metadata.pop("prior_verifier_action", None)
        try:
            event = transition_task(
                task,
                TaskStatus.ARBITRATING,
                actor=args.actor,
                reason=(
                    "recheck-prior-overrides: previously accepted via buggy "
                    "second-arbiter silent-agreement path; rerunning with "
                    "explicit-affirmation rule"
                ),
                stage="recovery",
                metadata={
                    "recheck": "prior_override_audit",
                    "previous_action": "resolved_to_first",
                },
            )
        except InvalidTransition as exc:
            print(f"  SKIP {task.task_id}: {exc}", file=sys.stderr)
            failed += 1
            continue
        store.save_task(task)
        store.append_event(event)
        transitioned += 1

    print(f"Transitioned {transitioned} → ARBITRATING ({failed} failed)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

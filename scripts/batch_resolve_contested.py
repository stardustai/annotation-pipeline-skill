"""Batch-resolve contested spans by setting a canonical type.

Reads contested spans from /api/posterior-audit (via direct service calls,
not HTTP), filters by a predicate (top type == technology OR runner-up ==
technology), and calls EntityConventionService.record_decision +
clear_dispute as needed so the operator's pick wins regardless of prior
state (active mismatching / disputed / not yet recorded).

Usage:
    python -m scripts.batch_resolve_contested --project v3_initial_deployment \
        --target-type technology --predicate tech_in_top2 --apply

Without --apply, runs as dry-run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.services.entity_statistics_service import (
    EntityStatisticsService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default="projects/v3_initial_deployment/.annotation-pipeline",
                    help="Path to the project's .annotation-pipeline dir")
    ap.add_argument("--project", required=True)
    ap.add_argument("--target-type", required=True,
                    help="Type to declare as canonical for matching spans")
    ap.add_argument("--predicate", required=True,
                    choices=["top1", "top2"],
                    help="top1 = target type must be rank #1 in observed dist; "
                         "top2 = target type in top 2 (rank #1 or #2)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write conventions; without this flag, dry-run only")
    args = ap.parse_args(argv)

    store = SqliteStore.open(Path(args.workspace))
    stats = EntityStatisticsService(store)
    convs = EntityConventionService(store)

    contested = stats.contested_spans(project_id=args.project)
    target = args.target_type

    matched = []
    for c in contested:
        dist = c["prior_distribution"]
        entries = sorted(dist.items(), key=lambda x: -x[1])
        if args.predicate == "top1":
            if entries[0][0] != target:
                continue
        else:  # top2
            top_two = [t for t, _ in entries[:2]]
            if target not in top_two:
                continue
        matched.append(c["span"])

    print(f"Matched {len(matched)} of {len(contested)} contested spans "
          f"for predicate={args.predicate} target={target}")
    if not args.apply:
        print("Dry-run only. Pass --apply to actually write conventions.")
        for s in matched[:20]:
            print(f"  would resolve: {s}")
        if len(matched) > 20:
            print(f"  …and {len(matched) - 20} more")
        return 0

    n_new = 0
    n_dispute_resolved = 0
    n_already = 0
    n_failed = 0
    for span in matched:
        try:
            conv = convs.record_decision(
                project_id=args.project,
                span=span,
                entity_type=target,
                source="batch_operator_resolve",
            )
            if conv.status == "disputed":
                convs.clear_dispute(
                    convention_id=conv.convention_id,
                    resolved_type=target,
                    actor="batch_operator_resolve",
                    notes=f"batch-resolved to {target} via predicate={args.predicate}",
                )
                n_dispute_resolved += 1
            elif conv.entity_type == target and conv.evidence_count > 1:
                n_already += 1
            else:
                n_new += 1
        except Exception as exc:
            n_failed += 1
            print(f"  failed for {span!r}: {exc}", file=sys.stderr)

    print(f"new active conventions: {n_new}")
    print(f"dispute resolved to {target}: {n_dispute_resolved}")
    print(f"already {target} (idempotent): {n_already}")
    print(f"failed: {n_failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

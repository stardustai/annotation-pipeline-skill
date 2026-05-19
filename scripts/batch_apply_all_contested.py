"""Batch "Apply to all" for contested spans that have conventions.

For each contested span with an existing convention, this script calls
the retroactive-fix endpoint to apply that convention to all ACCEPTED
tasks in the project that have the span tagged differently.

Usage:
    python -m scripts.batch_apply_all_contested --project v3_initial_deployment --apply

Without --apply, runs as dry-run to show what will be affected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.services.entity_statistics_service import (
    EntityStatisticsService,
)
from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default="projects/v3_initial_deployment/.annotation-pipeline",
                    help="Path to the project's .annotation-pipeline dir")
    ap.add_argument("--project", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Actually apply conventions; without this flag, dry-run only")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="Number of tasks to process per API call (default: 10)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N contested spans (for testing)")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip first N contested spans (for testing)")
    args = ap.parse_args(argv)

    store = SqliteStore.open(Path(args.workspace))
    stats = EntityStatisticsService(store)
    convs = EntityConventionService(store)

    contested = stats.contested_spans(project_id=args.project)
    all_conventions = convs.list_for_project(args.project, include_disputed=True)
    convention_map = {conv.span_lower: conv for conv in all_conventions}

    # Find contested spans with conventions
    matched = []
    for c in contested:
        span_lower = c["span"].lower()
        if span_lower in convention_map:
            convention = convention_map[span_lower]
            if convention.entity_type:
                matched.append({
                    "span": c["span"],
                    "entity_type": convention.entity_type,
                })

    print(f"Found {len(matched)} contested spans with conventions out of {len(contested)} total contested spans")

    if args.offset:
        matched = matched[args.offset:]
        print(f"Skipping first {args.offset} spans")
    if args.limit:
        matched = matched[:args.limit]
        print(f"Limited to first {len(matched)} spans")

    if not matched:
        return 0

    # Initialize API handler
    api = DashboardApi(
        store,
        stores={args.project: store},
        runtime_config=None,
        runtime_once=None,
    )

    total_fixed = 0
    total_errors = 0
    total_skipped = 0

    for idx, item in enumerate(matched, 1):
        span = item["span"]
        entity_type = item["entity_type"]
        print(f"\n[{idx}/{len(matched)}] Processing '{span}' → {entity_type!r}")

        # First, do a dry-run to see how many tasks will be affected
        body = json.dumps({
            "project_id": args.project,
            "span": span,
            "entity_type": entity_type,
            "actor": "batch_apply_all_contested",
            "dry_run": True,
        }).encode("utf-8")

        status, headers, response = api.handle_post("/api/posterior-audit/retroactive-fix", body)
        if status != 200:
            print(f"  ERROR: {status} - {response.decode('utf-8', errors='ignore')}")
            total_errors += 1
            continue

        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"  ERROR: invalid response - {e}")
            total_errors += 1
            continue

        remaining = result.get("remaining", 0)
        candidate_ids = result.get("candidate_task_ids", [])
        print(f"  Dry-run: {remaining} task(s) need this convention applied")

        if not args.apply:
            if remaining > 0:
                print(f"    would affect tasks: {candidate_ids[:5]}" +
                      ("..." if len(candidate_ids) > 5 else ""))
            continue

        # Actually apply the convention
        print(f"  Applying to all tasks...")
        candidates = candidate_ids or []
        batch_size = args.batch_size

        for batch_start in range(0, len(candidates), batch_size):
            batch_ids = candidates[batch_start : batch_start + batch_size]
            body = json.dumps({
                "project_id": args.project,
                "span": span,
                "entity_type": entity_type,
                "actor": "batch_apply_all_contested",
                "task_ids": batch_ids,
            }).encode("utf-8")

            status, headers, response = api.handle_post(
                "/api/posterior-audit/retroactive-fix", body
            )
            if status != 200:
                print(f"    Batch {batch_start//batch_size + 1}: ERROR {status}")
                total_errors += len(batch_ids)
                continue

            try:
                batch_result = json.loads(response)
            except json.JSONDecodeError:
                print(f"    Batch {batch_start//batch_size + 1}: ERROR - invalid response")
                total_errors += len(batch_ids)
                continue

            fixed = batch_result.get("fixed", 0)
            skipped = batch_result.get("skipped", 0)
            errors = batch_result.get("errors", [])
            batch_num = batch_start // batch_size + 1

            print(f"    Batch {batch_num}: fixed={fixed}, skipped={skipped}, errors={len(errors)}")
            total_fixed += fixed
            total_skipped += skipped
            total_errors += len(errors)

            if errors:
                for err in errors[:3]:
                    print(f"      - {err.get('task_id', '?')}: {err.get('reason', '?')}")
                if len(errors) > 3:
                    print(f"      - ...and {len(errors) - 3} more errors")

    print(f"\n{'='*60}")
    if args.apply:
        print(f"Total applied:")
        print(f"  Fixed: {total_fixed}")
        print(f"  Skipped: {total_skipped}")
        print(f"  Errors: {total_errors}")
    else:
        print(f"Dry-run complete. Pass --apply to actually apply {len(matched)} conventions.")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

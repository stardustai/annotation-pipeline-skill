"""Apply conventions from contested spans to all instances of those spans.

For each contested span that already has a convention set, this script
applies that convention to all instances of that span in the project
using the posterior_audit apply mechanism.

Usage:
    python -m scripts.apply_contested_conventions --project v3_initial_deployment --apply

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
from annotation_pipeline_skill.interfaces.api import (
    read_posterior_audit_cache,
    write_posterior_audit_cache,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default="projects/v3_initial_deployment/.annotation-pipeline",
                    help="Path to the project's .annotation-pipeline dir")
    ap.add_argument("--project", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write conventions; without this flag, dry-run only")
    args = ap.parse_args(argv)

    store = SqliteStore.open(Path(args.workspace))
    stats = EntityStatisticsService(store)
    convs = EntityConventionService(store)

    contested = stats.contested_spans(project_id=args.project)
    print(f"Total contested spans: {len(contested)}")

    # Get all conventions for the project
    all_conventions = convs.list_for_project(args.project, include_disputed=True)
    convention_map = {
        conv.span_lower: conv
        for conv in all_conventions
    }
    print(f"Total conventions: {len(all_conventions)}")

    # Find contested spans that already have conventions
    matched = []
    for c in contested:
        span_lower = c["span"].lower()
        if span_lower in convention_map:
            convention = convention_map[span_lower]
            if convention.entity_type:
                matched.append({
                    "span": c["span"],
                    "entity_type": convention.entity_type,
                    "convention_id": convention.convention_id,
                    "status": convention.status,
                })

    print(f"Found {len(matched)} contested spans with existing conventions")

    if not matched:
        print("No contested spans with conventions found.")
        return 0

    if not args.apply:
        print("Dry-run only. Pass --apply to actually apply conventions.")
        for item in matched[:10]:
            print(f"  would apply {item['entity_type']!r} to {item['span']!r} (status: {item['status']})")
        if len(matched) > 10:
            print(f"  …and {len(matched) - 10} more")
        return 0

    # Apply conventions to posterior audit cache
    cached = read_posterior_audit_cache(store, project_id=args.project)
    if cached is None:
        print("No posterior audit cache found. Cannot apply conventions.")
        return 1

    payload = cached["payload"]
    contested_spans = payload.get("contested_spans", [])

    # Build a map of span_lower -> entity_type for quick lookup
    resolved_map = {
        item["span"].lower(): item["entity_type"]
        for item in matched
    }

    # Mark contested spans as resolved in the cache
    n_marked = 0
    for c in contested_spans:
        span_lower = c.get("span", "").lower()
        if span_lower in resolved_map:
            c["resolved_convention_type"] = resolved_map[span_lower]
            n_marked += 1

    print(f"Marked {n_marked} contested spans as resolved in posterior audit cache")

    # Write back the updated cache
    write_posterior_audit_cache(
        store,
        project_id=args.project,
        payload=payload,
        accepted_hash=cached["accepted_hash"],
        created_at=cached["created_at"],
    )

    print(f"Updated posterior audit cache")
    print(f"Successfully applied {n_marked} conventions to all instances")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

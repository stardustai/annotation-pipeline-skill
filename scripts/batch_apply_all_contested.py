"""Fix divergent annotations and contested spans that have active conventions.

Two-phase approach:
  Phase 1 — Retroactive-fix divergent tasks.
    Scan every ACCEPTED task; for each (span, type) pair where an active
    convention declares a different type, call retroactive-fix to patch the
    task annotation. After fixing, call recount_span so entity_statistics
    reflects the new uniform distribution.

  Phase 2 — Recount stale contested spans.
    Spans that were fixed in a PRIOR run already have correct task
    annotations, but entity_statistics may still show the old split
    distribution. Read the current posterior-audit cache; for each contested
    span that has an active convention and was NOT processed in Phase 1,
    call recount_span to collapse the distribution. This removes them from
    the contested list on the next rebuild.

Usage:
    python -m scripts.batch_apply_all_contested --project v3_initial_deployment --apply

Without --apply, runs as dry-run (shows what would be done, no writes).
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
    iter_span_decisions,
)
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _load_annotation(store, task) -> dict | None:
    """Read a task's current annotation, preferring human_review_answer.

    Mirrors build_posterior_audit._load_annotation exactly so that our
    divergent-task scan agrees with what the UI shows.
    """
    import re as _re
    arts = store.list_artifacts(task.task_id)
    hr = [a for a in arts if a.kind == "human_review_answer"]
    if hr:
        try:
            outer = json.loads((store.root / hr[-1].path).read_text(encoding="utf-8"))
            return outer.get("answer") if isinstance(outer, dict) else None
        except (OSError, json.JSONDecodeError):
            return None
    anns = [a for a in arts if a.kind == "annotation_result"]
    if not anns:
        return None
    try:
        outer = json.loads((store.root / anns[-1].path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    text = outer.get("text") if isinstance(outer, dict) else None
    if not isinstance(text, str):
        return None
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default="projects/v3_initial_deployment/.annotation-pipeline",
                    help="Path to the project's .annotation-pipeline dir")
    ap.add_argument("--project", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Actually apply fixes; without this flag, dry-run only")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="Number of tasks to process per API call (default: 10)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit Phase 2 recount to first N stale spans (for testing)")
    args = ap.parse_args(argv)

    store = SqliteStore.open(Path(args.workspace))
    stats_svc = EntityStatisticsService(store)

    # Build convention index (active only, no disputed)
    convention_index: dict[str, str] = {}
    for conv in EntityConventionService(store).list_for_project(args.project, include_disputed=False):
        if conv.entity_type:
            convention_index[conv.span_lower] = conv.entity_type
    print(f"Loaded {len(convention_index)} active conventions")

    # ------------------------------------------------------------------ #
    # Phase 1: scan tasks → find divergent (span, task) pairs → fix them  #
    # ------------------------------------------------------------------ #
    print("\n=== Phase 1: Fix divergent task annotations ===")
    print("Scanning accepted tasks...")

    divergent_spans: dict[str, str] = {}  # span_lower → convention entity_type
    task_count = 0
    for task in store.list_tasks_by_pipeline(args.project):
        if task.status is not TaskStatus.ACCEPTED:
            continue
        task_count += 1
        payload = _load_annotation(store, task)
        if payload is None:
            continue
        for span, entity_type in iter_span_decisions(payload):
            span_lower = span.lower()
            conv_type = convention_index.get(span_lower)
            if conv_type and conv_type != entity_type:
                divergent_spans[span_lower] = conv_type

    phase1_spans = [{"span": s, "entity_type": t} for s, t in divergent_spans.items()]
    print(f"Scanned {task_count} accepted tasks")
    print(f"Found {len(phase1_spans)} span(s) with divergent tasks needing fix")

    api = DashboardApi(
        store,
        stores={args.project: store},
        runtime_config=None,
        runtime_once=None,
    )

    total_fixed = 0
    total_errors = 0
    total_skipped = 0
    phase1_done: set[str] = set()

    for idx, item in enumerate(phase1_spans, 1):
        span = item["span"]
        entity_type = item["entity_type"]
        print(f"\n[{idx}/{len(phase1_spans)}] '{span}' → {entity_type!r}")

        body = json.dumps({
            "project_id": args.project,
            "span": span,
            "entity_type": entity_type,
            "actor": "batch_apply_divergent",
            "dry_run": True,
        }).encode("utf-8")
        status, _, response = api.handle_post("/api/posterior-audit/retroactive-fix", body)
        if status != 200:
            print(f"  ERROR dry-run: {status}")
            total_errors += 1
            continue
        result = json.loads(response)
        candidate_ids = result.get("candidate_task_ids", [])
        print(f"  {result.get('remaining', 0)} task(s) to fix")

        if not args.apply:
            if candidate_ids:
                print(f"    would affect: {candidate_ids[:5]}" +
                      ("..." if len(candidate_ids) > 5 else ""))
            continue

        if candidate_ids:
            for batch_start in range(0, len(candidate_ids), args.batch_size):
                batch_ids = candidate_ids[batch_start: batch_start + args.batch_size]
                body = json.dumps({
                    "project_id": args.project,
                    "span": span,
                    "entity_type": entity_type,
                    "actor": "batch_apply_divergent",
                    "task_ids": batch_ids,
                }).encode("utf-8")
                status, _, response = api.handle_post("/api/posterior-audit/retroactive-fix", body)
                if status != 200:
                    print(f"    Batch {batch_start // args.batch_size + 1}: ERROR {status}")
                    total_errors += len(batch_ids)
                    continue
                br = json.loads(response)
                fixed = br.get("fixed", 0)
                skipped = br.get("skipped", 0)
                errs = br.get("errors", [])
                print(f"    Batch {batch_start // args.batch_size + 1}: fixed={fixed}, skipped={skipped}, errors={len(errs)}")
                total_fixed += fixed
                total_skipped += skipped
                total_errors += len(errs)
                if errs:
                    for err in errs[:3]:
                        print(f"      - {err.get('task_id', '?')}: {err.get('reason', '?')}")

        # Recount entity_statistics so next rebuild sees the collapsed distribution
        try:
            new_dist = stats_svc.recount_span(project_id=args.project, span=span)
            print(f"  Recounted: {new_dist}")
        except Exception as exc:
            print(f"  Recount failed: {exc}")

        phase1_done.add(span)

    # ------------------------------------------------------------------ #
    # Phase 2: recount contested spans with conventions (stale statistics) #
    # ------------------------------------------------------------------ #
    print(f"\n=== Phase 2: Recount stale contested spans with conventions ===")

    # Read the posterior audit cache to find contested spans with conventions
    # that were NOT handled in Phase 1 (tasks already correct, stats stale).
    try:
        row = store._conn.execute(
            "SELECT payload_json FROM posterior_audit_cache WHERE project_id=?",
            (args.project,),
        ).fetchone()
        cache_payload = json.loads(row["payload_json"]) if row else {}
    except Exception:
        cache_payload = {}

    contested = cache_payload.get("divergent_entries", [])
    stale_spans = []
    for c in contested:
        span_lower = (c.get("span") or "").lower()
        if span_lower in convention_index and span_lower not in phase1_done:
            stale_spans.append(span_lower)

    if args.limit:
        stale_spans = stale_spans[:args.limit]
    print(f"Found {len(stale_spans)} contested span(s) with stale statistics to recount")

    for idx, span in enumerate(stale_spans, 1):
        conv_type = convention_index[span]
        print(f"  [{idx}/{len(stale_spans)}] recount '{span}' (convention → {conv_type!r})")
        if not args.apply:
            continue
        try:
            new_dist = stats_svc.recount_span(project_id=args.project, span=span)
            print(f"    New dist: {new_dist}")
        except Exception as exc:
            print(f"    Recount failed: {exc}")

    print(f"\n{'=' * 60}")
    if args.apply:
        print(f"Phase 1 — Fixed: {total_fixed}  Skipped: {total_skipped}  Errors: {total_errors}")
        print(f"Phase 2 — Recounted {len(stale_spans)} stale span(s)")
        print(f"\nReady for rebuild. The contested spans with conventions should now collapse.")
    else:
        print(f"Dry-run complete.")
        print(f"  Phase 1: would fix {len(phase1_spans)} span(s)")
        print(f"  Phase 2: would recount {len(stale_spans)} stale span(s)")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Retroactively reject the substation-equipment-report template batch.

Identifies tasks whose input rows match the synthetic-template regex and
transitions them from ACCEPTED to REJECTED with an operator audit event.

Why move-to-reject instead of hard-delete:
  - Reversible (manual_drag REJECTED → ARBITRATING/ACCEPTED).
  - Preserves artifacts + audit history.
  - One status transition per task; the standard entity_statistics
    re-bootstrap (only counts ACCEPTED) automatically picks up the change.

Dry-run by default; --apply does the writes. After --apply, the caller
should run ``scripts/rebootstrap_stats_merged.py --apply`` to flush stats
+ posterior_audit_cache.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from annotation_pipeline_skill.core.models import TaskStatus
from annotation_pipeline_skill.core.transitions import (
    InvalidTransition,
    transition_task,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


# Distinctive template signature: any one of these substrings in a row's
# input.text is enough to flag the task. Designed to NOT match natural
# project content (status reports, code reviews, etc.).
TEMPLATE_RE = re.compile(
    r"(?:substation [A-Z]{3}-SS-\d+"
    r"|reported equipment [A-Z]{3}-SS-\d+-[A-Z]{3}-\d+"
    r"|health score \d+\.?\d*, 30-day failure probability"
    r"|recommended action (?:monitor|schedule_service|replace_component))",
    re.IGNORECASE,
)

REJECT_REASON = (
    "合成的变电站设备巡检模板数据，499 个 task 高度雷同（只是数字和站点 ID 变），"
    "对项目 entity_statistics 是噪声，整批 reject"
)
REJECT_STAGE = "template_dataset_cleanup"
REJECT_ACTOR = "operator"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--project-root", default="projects/v3_initial_deployment",
        help="Path to the project directory (contains .annotation-pipeline/)",
    )
    ap.add_argument("--apply", action="store_true",
                    help="Actually transition tasks; without this flag, dry-run only")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.project_root) / ".annotation-pipeline"
    store = SqliteStore.open(root)

    # Locate the project pipeline id from any existing task
    row = store._conn.execute("SELECT pipeline_id FROM tasks LIMIT 1").fetchone()
    if row is None:
        print("no tasks in this store", file=sys.stderr); return 1
    project_id = row["pipeline_id"]

    # Identify template tasks among ACCEPTED only
    candidates: list[tuple[str, int, int]] = []  # (task_id, n_template_rows, n_rows)
    for r in store._conn.execute(
        "SELECT task_id, source_ref_json FROM tasks "
        "WHERE pipeline_id=? AND status='accepted' "
        "ORDER BY task_id",
        (project_id,),
    ).fetchall():
        src = json.loads(r["source_ref_json"])
        rows_data = src.get("payload", {}).get("rows", [])
        if not rows_data:
            continue
        n_match = sum(
            1 for row in rows_data
            if isinstance(row.get("input"), str) and TEMPLATE_RE.search(row["input"])
        )
        if n_match > 0:
            candidates.append((r["task_id"], n_match, len(rows_data)))

    print(f"template-matching ACCEPTED tasks: {len(candidates)}")
    if not candidates:
        print("nothing to reject")
        return 0

    pure = sum(1 for _, m, n in candidates if m == n)
    partial = sum(1 for _, m, n in candidates if m < n)
    print(f"  pure-template (all rows match):  {pure}")
    print(f"  partial-template (some rows):    {partial}")
    if partial:
        print(f"  ⚠ partial-template tasks (review before applying):")
        for tid, m, n in candidates:
            if m < n:
                print(f"      {tid}  ({m}/{n} rows match)")
    print()
    print(f"task_id range: {candidates[0][0]} → {candidates[-1][0]}")

    if not args.apply:
        print("\n[dry-run] no writes. Pass --apply to commit transitions.")
        return 0

    print(f"\n[apply] transitioning {len(candidates)} tasks ACCEPTED → REJECTED")
    moved = skipped = 0
    for tid, m, n in candidates:
        try:
            t = store.load_task(tid)
        except (FileNotFoundError, KeyError):
            skipped += 1; continue
        if t.status is not TaskStatus.ACCEPTED:
            skipped += 1; continue
        try:
            ev = transition_task(
                t, TaskStatus.REJECTED,
                actor=REJECT_ACTOR,
                reason=REJECT_REASON,
                stage=REJECT_STAGE,
                attempt_id=None,
                metadata={
                    "rejection_kind": "template_dataset_cleanup",
                    "template_name": "substation_equipment_report",
                    "previous_status": "accepted",
                    "template_match_rows": m,
                    "total_rows": n,
                    "reversible_via": "manual_drag to ARBITRATING or ACCEPTED",
                },
            )
            store.save_task(t)
            store.append_event(ev)
            moved += 1
        except InvalidTransition as exc:
            print(f"  skip {tid}: {exc}")
            skipped += 1

    print(f"[apply] moved={moved}  skipped={skipped}")
    print("\nNext step: scripts/rebootstrap_stats_merged.py --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Recompute entity_statistics from ALL ACCEPTED tasks under the merged
(span, type) decision space (entities + json_structures unified).

Run AFTER landing the iter_span_decisions merge — historical stats only
counted the entities field, so the table is now under-counted relative to
what posterior audit / conventions / contested-span audit should see.

Steps:
  1. Truncate entity_statistics for the project.
  2. Walk every ACCEPTED task's latest annotation_result, run the merged
     iter_span_decisions, bump counts.
  3. Clear posterior_audit_cache so the next GET triggers a fresh scan.

Idempotent. Dry-run by default; --apply does the writes.
"""
import argparse, json, pathlib, sys
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.core.models import TaskStatus
from annotation_pipeline_skill.services.entity_statistics_service import (
    EntityStatisticsService, iter_span_decisions,
)
from annotation_pipeline_skill.runtime.subagent_cycle import _parse_llm_json

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--project-root", default="projects/v3_initial_deployment")
args = ap.parse_args()

root = pathlib.Path(args.project_root) / ".annotation-pipeline"
store = SqliteStore(root=root, db_path=root / "db.sqlite")
project_id = None
# Find the pipeline_id by reading one task
row = store._conn.execute(
    "SELECT pipeline_id FROM tasks LIMIT 1"
).fetchone()
if row is None:
    print("no tasks found"); sys.exit(1)
project_id = row["pipeline_id"]

def load_latest_annotation(task_id):
    arts = [a for a in store.list_artifacts(task_id) if a.kind == "annotation_result"]
    if not arts:
        return None
    art = arts[-1]
    p = root / art.path
    if not p.exists():
        return None
    try:
        outer = json.loads(p.read_text())
        text = outer.get("text") if isinstance(outer, dict) else None
        return _parse_llm_json(text) if isinstance(text, str) else None
    except Exception:
        return None

# Tally first
from collections import defaultdict
tallies = defaultdict(int)  # (span_lower, entity_type) -> count
n_tasks = n_decisions = 0
for t in store.list_tasks_by_pipeline(project_id):
    if t.status is not TaskStatus.ACCEPTED:
        continue
    payload = load_latest_annotation(t.task_id)
    if payload is None:
        continue
    n_tasks += 1
    seen_task = set()
    for span, etype in iter_span_decisions(payload):
        if not span or not etype:
            continue
        key = (span.strip().lower(), etype)
        if not key[0] or not key[1]:
            continue
        if key in seen_task:
            continue
        seen_task.add(key)
        tallies[key] += 1
        n_decisions += 1

# Compare to current stats
cur = store._conn.execute(
    "SELECT span_lower, entity_type, count FROM entity_statistics WHERE project_id=?",
    (project_id,),
).fetchall()
cur_total = sum(r["count"] for r in cur)
new_total = sum(tallies.values())
print(f"Tasks processed: {n_tasks}")
print(f"Decisions tallied (merged): {n_decisions}")
print(f"Unique (span,type) pairs: {len(tallies)}")
print(f"Current entity_statistics total count: {cur_total}")
print(f"Recomputed total count: {new_total}")
print(f"Delta: {new_total - cur_total}")
print()

# Quick preview: top-10 new (span, type) keys that aren't yet in stats
existing = {(r["span_lower"], r["entity_type"]): r["count"] for r in cur}
new_pairs = sorted(
    ((k, v) for k, v in tallies.items() if k not in existing),
    key=lambda x: -x[1],
)[:10]
print("Top 10 brand-new (span, type) pairs entering stats:")
for (s, t), c in new_pairs:
    print(f"  {s!r:30s} → {t:15s}  ×{c}")

if not args.apply:
    print("\n[dry-run] not writing. re-run with --apply to commit.")
    sys.exit(0)

# Apply: truncate + reseed
print("\n[apply] truncating entity_statistics for this project...")
store._conn.execute("DELETE FROM entity_statistics WHERE project_id=?", (project_id,))
svc = EntityStatisticsService(store)
for (span_lower, etype), count in tallies.items():
    svc.increment(project_id=project_id, span=span_lower, entity_type=etype, weight=count)
print("[apply] clearing posterior_audit_cache...")
store._conn.execute("DELETE FROM posterior_audit_cache WHERE project_id=?", (project_id,))
store._conn.commit()
print(f"[apply] done — {len(tallies)} (span,type) rows, posterior cache invalidated")

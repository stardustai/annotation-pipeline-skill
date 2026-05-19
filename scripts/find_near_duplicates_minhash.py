"""Find near-duplicate ACCEPTED tasks via MinHash + LSH.

Pipeline: load every accepted task, extract canonical text, compute
MinHash signatures, build the LSH index at the requested Jaccard
threshold, and emit a ClusterReport JSON. With --apply, transition
every non-representative task in every cluster from ACCEPTED to
REJECTED (audit stage="similarity_dedup_minhash").

Dry-run is the default. Output JSON is written even in dry-run so the
operator can review clusters before deciding.

Usage:
    .venv/bin/python scripts/find_near_duplicates_minhash.py \\
        --project-root projects/v3_initial_deployment \\
        --shingle-size 5 --jaccard-threshold 0.7 \\
        --report-path /tmp/minhash-clusters.json

    # apply (after reviewing the report):
    .venv/bin/python scripts/find_near_duplicates_minhash.py \\
        --project-root projects/v3_initial_deployment --apply
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime, timezone

from annotation_pipeline_skill.core.models import TaskStatus
from annotation_pipeline_skill.core.transitions import (
    InvalidTransition,
    transition_task,
)
from annotation_pipeline_skill.similarity.clusters import (
    ClusterReport,
    pick_representative,
)
from annotation_pipeline_skill.similarity.extractors import canonical_task_text
from annotation_pipeline_skill.similarity.minhash import MinHashLSHFinder
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


REJECT_STAGE = "similarity_dedup_minhash"
REJECT_ACTOR = "operator"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--shingle-size", type=int, default=5)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--jaccard-threshold", type=float, default=0.7)
    ap.add_argument(
        "--report-path",
        default="/tmp/minhash-clusters.json",
        help="Where to write the ClusterReport JSON (always written, even in dry-run)",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="Transition all non-representative cluster members ACCEPTED → REJECTED",
    )
    args = ap.parse_args(argv)

    root = pathlib.Path(args.project_root) / ".annotation-pipeline"
    store = SqliteStore.open(root)
    row = store._conn.execute("SELECT pipeline_id FROM tasks LIMIT 1").fetchone()
    if row is None:
        print("no tasks in store", file=sys.stderr); return 1
    project_id = row["pipeline_id"]

    print(f"loading ACCEPTED tasks for project {project_id!r}…")
    finder = MinHashLSHFinder(
        shingle_size=args.shingle_size,
        num_perm=args.num_perm,
        jaccard_threshold=args.jaccard_threshold,
    )
    n_added = 0
    for t in store.list_tasks_by_pipeline(project_id):
        if t.status is not TaskStatus.ACCEPTED:
            continue
        text = canonical_task_text(t)
        if not text.strip():
            continue
        finder.add(t.task_id, text)
        n_added += 1
    print(f"indexed {n_added} ACCEPTED tasks")

    clusters = finder.clusters(include_singletons=False)
    n_dup = sum(len(c.task_ids) for c in clusters)
    print(f"found {len(clusters)} clusters covering {n_dup} tasks "
          f"(at jaccard ≥ {args.jaccard_threshold})")

    report = ClusterReport(
        project_id=project_id,
        method="minhash",
        params={
            "shingle_size": args.shingle_size,
            "num_perm": args.num_perm,
            "jaccard_threshold": args.jaccard_threshold,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        clusters=clusters,
    )
    report_path = pathlib.Path(args.report_path)
    report.to_json_file(report_path)
    print(f"report written → {report_path}")

    # Preview top 5 clusters
    print("\ntop clusters by size:")
    for c in clusters[:5]:
        rep = pick_representative(c)
        print(f"  cluster {c.cluster_id}  size={len(c.task_ids)}  sim={c.similarity:.3f}"
              f"  rep={rep}")
        sample = [t for t in c.task_ids if t != rep][:3]
        for tid in sample:
            print(f"    would-reject {tid}")

    if not args.apply:
        print("\n[dry-run] no transitions. Re-run with --apply to commit "
              "(after reviewing the report).")
        return 0

    to_reject = report.tasks_to_reject()
    print(f"\n[apply] transitioning {len(to_reject)} tasks ACCEPTED → REJECTED")
    moved = skipped = 0
    cluster_by_task = {tid: c for c in clusters for tid in c.task_ids}
    for tid in to_reject:
        try:
            t = store.load_task(tid)
        except (FileNotFoundError, KeyError):
            skipped += 1; continue
        if t.status is not TaskStatus.ACCEPTED:
            skipped += 1; continue
        c = cluster_by_task[tid]
        rep = pick_representative(c)
        try:
            ev = transition_task(
                t, TaskStatus.REJECTED,
                actor=REJECT_ACTOR,
                reason=(
                    f"MinHash 近重复检测 (jaccard ≥ {args.jaccard_threshold})；"
                    f"簇 {c.cluster_id}（{len(c.task_ids)} 个 task，相似度 "
                    f"{c.similarity:.2f}）保留代表 {rep}"
                ),
                stage=REJECT_STAGE,
                attempt_id=None,
                metadata={
                    "rejection_kind": "similarity_dedup_minhash",
                    "cluster_id": c.cluster_id,
                    "cluster_size": len(c.task_ids),
                    "cluster_similarity": c.similarity,
                    "representative_task_id": rep,
                    "jaccard_threshold": args.jaccard_threshold,
                    "shingle_size": args.shingle_size,
                    "previous_status": "accepted",
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

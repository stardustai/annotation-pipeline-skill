"""Find near-duplicate ROWS across ACCEPTED tasks; optionally mask them.

Sibling of scripts/find_semantic_clusters.py but at row granularity:
each (task_id, row_index) is the unit of comparison. Row pairs above
``--jaccard-threshold`` (for MinHash) or ``--cosine-threshold`` (for
embedding profiles — same flag, threshold semantics differ per metric)
land in the same cluster; non-representative members of each cluster
can be masked in place via row_masks.

Masked rows become invisible at every downstream read boundary:
canonical_task_text, iter_span_decisions, export, scatter coords. The
task stays ACCEPTED.

Usage:
    .venv/bin/python scripts/row_dedup_scan.py \\
        --project-root projects/v3_initial_deployment \\
        --profile MinHash --jaccard-threshold 0.5 \\
        --report-path /tmp/row-dedup-clusters.json

    # apply after reviewing the report:
    .venv/bin/python scripts/row_dedup_scan.py \\
        --project-root projects/v3_initial_deployment --profile MinHash \\
        --jaccard-threshold 0.5 --apply
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

from annotation_pipeline_skill.services.row_dedup_service import RowDedupService
from annotation_pipeline_skill.similarity.profiles import load_similarity_profiles
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _resolve_profiles_path(workspace_root: pathlib.Path) -> pathlib.Path:
    ws = workspace_root / "similarity_profiles.yaml"
    if ws.exists():
        return ws
    raise FileNotFoundError(
        f"no similarity_profiles.yaml found at {ws} — create one with at "
        f"least one profile (see annotation_pipeline_skill/similarity/profiles.py)"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--profile", required=True,
                    help="Profile name from similarity_profiles.yaml (e.g. MinHash, jina_small)")
    ap.add_argument("--profiles-yaml", default=None,
                    help="Path to similarity_profiles.yaml; default: <workspace>/similarity_profiles.yaml")
    ap.add_argument(
        "--jaccard-threshold", type=float, default=0.5,
        help="Jaccard for minhash provider, cosine for embedding providers (same flag, "
             "different metric — read the cluster report's params.metric field for clarity)",
    )
    ap.add_argument("--statuses", default=None,
                    help='Comma-separated TaskStatus values to include; default: all stages')
    ap.add_argument("--report-path", default="/tmp/row-dedup-clusters.json",
                    help="Where to write the cluster JSON (always written, even in dry-run)")
    ap.add_argument("--apply", action="store_true",
                    help="Mask all non-representative cluster members in row_masks")
    args = ap.parse_args(argv)

    project_root = pathlib.Path(args.project_root)
    root = project_root / ".annotation-pipeline"
    store = SqliteStore.open(root)
    row = store._conn.execute("SELECT pipeline_id FROM tasks LIMIT 1").fetchone()
    if row is None:
        print("no tasks in store", file=sys.stderr); return 1
    project_id = row["pipeline_id"]

    profiles_path = (
        pathlib.Path(args.profiles_yaml) if args.profiles_yaml
        else _resolve_profiles_path(project_root.parent)
    )
    profiles = load_similarity_profiles(profiles_path)
    if args.profile not in profiles:
        print(
            f"profile {args.profile!r} not in {profiles_path}; "
            f"available: {sorted(profiles)}",
            file=sys.stderr,
        )
        return 1

    statuses: list[str] | None = (
        [s.strip() for s in args.statuses.split(",") if s.strip()]
        if args.statuses else None
    )

    svc = RowDedupService(store, profiles)
    print(f"scanning rows for project {project_id!r} with profile {args.profile!r}…")
    payload = svc.scan_rows(
        project_id=project_id,
        profile_name=args.profile,
        statuses=statuses,
        jaccard_threshold=args.jaccard_threshold,
    )

    metric = payload["params"]["metric"]
    print(f"  row_count: {payload['row_count']}  task_count: {payload['task_count']}")
    print(f"  clusters: {len(payload['clusters'])}  metric: {metric}")
    cache_stats = payload["params"].get("embedding_cache") or {}
    print(f"  embedding cache hits/misses: {cache_stats.get('hits')}/{cache_stats.get('misses')}")

    report_path = pathlib.Path(args.report_path)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report → {report_path}")

    if payload["clusters"]:
        print(f"\ntop clusters by size:")
        for c in payload["clusters"][:5]:
            members = c["members"]
            rep = sorted(members, key=lambda m: (str(m["task_id"]), int(m["row_index"])))[0]
            print(
                f"  {c['cluster_id']:8s} size={len(members):3d} "
                f"{metric}={c['similarity']:.3f}  rep={rep['task_id']}:{rep['row_index']}"
            )

    if not args.apply:
        print("\n[dry-run] no masks applied. Re-run with --apply after reviewing report.")
        return 0

    if not payload["clusters"]:
        print("\nno clusters → nothing to mask")
        return 0

    print(f"\n[apply] masking non-representative rows in {len(payload['clusters'])} clusters…")
    moved = skipped = 0
    profile = profiles[args.profile]
    for c in payload["clusters"]:
        result = svc.mask_duplicates(
            project_id=project_id,
            members=c["members"],
            cluster_id=c["cluster_id"],
            similarity=c["similarity"],
            profile_name=args.profile,
            model=profile.model,
        )
        moved += result.get("masked", 0)
        skipped += result.get("skipped", 0)
    print(f"[apply] masked={moved}  skipped={skipped}")
    print(
        f"\nNext step: re-run scripts/rebootstrap_stats_merged.py --apply "
        f"if you want entity_statistics to reflect the new masks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

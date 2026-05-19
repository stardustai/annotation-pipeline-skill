"""Find semantic-similar ACCEPTED tasks via embedding + UMAP + HDBSCAN.

Pipeline:
  1. Load every ACCEPTED task, extract canonical text.
  2. Embed via the configured SimilarityProfile (default: jina_small).
  3. UMAP-project to 2D.
  4. HDBSCAN cluster.
  5. Emit ClusterReport JSON + per-task (x, y, cluster_id) coords.
  6. (Optional) PNG scatter plot.

With --apply, transition non-representative cluster members ACCEPTED →
REJECTED (audit stage="similarity_dedup_embedding"). Dry-run is the
default — always inspect the cluster report first; embedding clusters
are looser than MinHash clusters so false positives matter more.

Usage:
    .venv/bin/python scripts/find_semantic_clusters.py \\
        --project-root projects/v3_initial_deployment \\
        --profile jina_small \\
        --min-cluster-size 5 \\
        --report-path /tmp/embedding-clusters.json \\
        --coords-path /tmp/embedding-coords.json \\
        --plot-path /tmp/embedding-scatter.png
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

from annotation_pipeline_skill.core.models import TaskStatus
from annotation_pipeline_skill.core.transitions import (
    InvalidTransition,
    transition_task,
)
from annotation_pipeline_skill.similarity.clusters import (
    Cluster,
    ClusterReport,
    pick_representative,
)
from annotation_pipeline_skill.similarity.embeddings import build_embedding_client
from annotation_pipeline_skill.similarity.extractors import canonical_task_text
from annotation_pipeline_skill.similarity.profiles import load_similarity_profiles
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


REJECT_STAGE = "similarity_dedup_embedding"
REJECT_ACTOR = "operator"


def _resolve_profiles_path(workspace_root: pathlib.Path) -> pathlib.Path:
    # workspace-global > project-local
    ws = workspace_root / "similarity_profiles.yaml"
    if ws.exists():
        return ws
    raise FileNotFoundError(
        f"no similarity_profiles.yaml found at {ws} — create one with at "
        f"least one profile (see docs in similarity/profiles.py)"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--profile", required=True,
                    help="Profile name from similarity_profiles.yaml")
    ap.add_argument("--profiles-yaml", default=None,
                    help="Path to similarity_profiles.yaml; default: <workspace>/similarity_profiles.yaml")
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--min-cluster-size", type=int, default=5,
                    help="HDBSCAN min_cluster_size; smaller = more, looser clusters")
    ap.add_argument("--report-path", default="/tmp/embedding-clusters.json")
    ap.add_argument("--coords-path", default="/tmp/embedding-coords.json")
    ap.add_argument("--plot-path", default=None,
                    help="Optional path to write a PNG scatter plot")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    project_root = pathlib.Path(args.project_root)
    root = project_root / ".annotation-pipeline"
    store = SqliteStore.open(root)
    row = store._conn.execute("SELECT pipeline_id FROM tasks LIMIT 1").fetchone()
    if row is None:
        print("no tasks in store", file=sys.stderr); return 1
    project_id = row["pipeline_id"]

    profiles_path = pathlib.Path(args.profiles_yaml) if args.profiles_yaml \
        else _resolve_profiles_path(project_root.parent)
    profiles = load_similarity_profiles(profiles_path)
    if args.profile not in profiles:
        print(f"profile {args.profile!r} not in {profiles_path}; "
              f"available: {sorted(profiles)}", file=sys.stderr); return 1
    profile = profiles[args.profile]

    print(f"loading ACCEPTED tasks for project {project_id!r}…")
    task_ids: list[str] = []
    texts: list[str] = []
    for t in store.list_tasks_by_pipeline(project_id):
        if t.status is not TaskStatus.ACCEPTED:
            continue
        text = canonical_task_text(t)
        if not text.strip():
            continue
        task_ids.append(t.task_id)
        texts.append(text)
    print(f"  {len(task_ids)} tasks to embed")

    print(f"embedding via {profile.provider}:{profile.model} (batch={profile.batch_size})…")
    client = build_embedding_client(profile)
    emb = client.embed(texts)
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass
    print(f"  vectors: {emb.vectors.shape}")

    import umap, hdbscan
    print("UMAP-projecting to 2D…")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=42,
    )
    coords = reducer.fit_transform(emb.vectors)
    print(f"HDBSCAN clustering (min_cluster_size={args.min_cluster_size})…")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  found {n_clusters} clusters; {(labels == -1).sum()} tasks marked noise")

    # Build clusters
    cluster_members: dict[int, list[str]] = {}
    for tid, lbl in zip(task_ids, labels):
        if lbl == -1:
            continue
        cluster_members.setdefault(int(lbl), []).append(tid)
    clusters: list[Cluster] = []
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    for lbl, members in cluster_members.items():
        # Average pairwise cosine similarity inside the cluster
        idxs = [id_to_idx[m] for m in members]
        sims = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = emb.vectors[idxs[i]], emb.vectors[idxs[j]]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                sims.append(float(np.dot(a, b) / denom) if denom else 0.0)
        avg_sim = float(np.mean(sims)) if sims else 1.0
        clusters.append(
            Cluster(
                cluster_id=f"emb-{lbl}",
                task_ids=sorted(members),
                method="embedding",
                similarity=avg_sim,
            )
        )
    clusters.sort(key=lambda c: len(c.task_ids), reverse=True)

    report = ClusterReport(
        project_id=project_id,
        method="embedding",
        params={
            "profile": args.profile,
            "model": profile.model,
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "min_cluster_size": args.min_cluster_size,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        clusters=clusters,
    )
    report.to_json_file(args.report_path)
    print(f"report → {args.report_path}")

    # Coords output: one record per task with 2D position + cluster_id
    coords_out = [
        {"task_id": tid, "x": float(coords[i, 0]), "y": float(coords[i, 1]),
         "cluster_id": (f"emb-{int(labels[i])}" if labels[i] != -1 else None)}
        for i, tid in enumerate(task_ids)
    ]
    pathlib.Path(args.coords_path).write_text(
        json.dumps(coords_out, indent=2), encoding="utf-8",
    )
    print(f"coords → {args.coords_path}")

    if args.plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        # Noise = grey, clusters = colored
        noise_mask = labels == -1
        ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
                   c="lightgrey", s=4, alpha=0.5, label="noise")
        non_noise = ~noise_mask
        if non_noise.any():
            ax.scatter(coords[non_noise, 0], coords[non_noise, 1],
                       c=labels[non_noise], s=8, cmap="tab20", alpha=0.8)
        ax.set_title(f"{project_id} — {len(task_ids)} tasks, {n_clusters} clusters")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(args.plot_path, dpi=120)
        print(f"plot → {args.plot_path}")

    print("\ntop clusters by size:")
    for c in clusters[:5]:
        rep = pick_representative(c)
        print(f"  {c.cluster_id}  size={len(c.task_ids)}  cos={c.similarity:.3f}  rep={rep}")

    if not args.apply:
        print("\n[dry-run] no transitions. Review the report + plot first.")
        return 0

    to_reject = report.tasks_to_reject()
    print(f"\n[apply] {len(to_reject)} tasks → REJECTED")
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
                    f"embedding 语义聚类 ({profile.model}, cos≈{c.similarity:.2f})；"
                    f"簇 {c.cluster_id}（{len(c.task_ids)} 个 task）保留代表 {rep}"
                ),
                stage=REJECT_STAGE,
                metadata={
                    "rejection_kind": "similarity_dedup_embedding",
                    "cluster_id": c.cluster_id,
                    "cluster_size": len(c.task_ids),
                    "cluster_similarity": c.similarity,
                    "representative_task_id": rep,
                    "embedding_profile": args.profile,
                    "embedding_model": profile.model,
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
    print("Next step: scripts/rebootstrap_stats_merged.py --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

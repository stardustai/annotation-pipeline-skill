"""DistributionService — embed → UMAP → HDBSCAN scatter for the Distribution tab.

Wraps the same pipeline that ``scripts/find_semantic_clusters.py`` runs
offline, but:

  - status filter is parametric (default all-stages for colour-by-status);
  - writes the result to the ``distribution_cache`` table keyed by
    (project_id, profile_name);
  - ``get_cache_state`` surfaces a ``stale`` flag via content_hash comparison;
  - ``reject_duplicates`` drives ACCEPTED→REJECTED transitions with rich audit
    metadata (cluster context).
"""
from __future__ import annotations

import numpy as np
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.core.transitions import InvalidTransition, transition_task
from annotation_pipeline_skill.interfaces.api import (
    compute_distribution_content_hash,
    read_distribution_cache,
    write_distribution_cache,
)
from annotation_pipeline_skill.similarity.clusters import Cluster
from annotation_pipeline_skill.similarity.embeddings import build_embedding_client
from annotation_pipeline_skill.similarity.extractors import canonical_task_text
from annotation_pipeline_skill.similarity.profiles import SimilarityProfile
from annotation_pipeline_skill.store.sqlite_store import SqliteStore

_REJECT_STAGE = "similarity_dedup_embedding"


class DistributionService:
    """Service that builds a 2-D scatter of all project tasks (colour-by-stage)
    and can reject duplicate tasks found by the embedding cluster pipeline.
    """

    def __init__(
        self,
        store: SqliteStore,
        profiles: dict[str, SimilarityProfile],
    ) -> None:
        self._store = store
        self._profiles = profiles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        *,
        project_id: str,
        profile_name: str,
        statuses: list[str] | None = None,
        min_cluster_size: int = 5,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
    ) -> dict:
        """Run the end-to-end pipeline and return (and cache) the scatter payload.

        Steps: load tasks → canonical_task_text → embed → UMAP → HDBSCAN.
        The result is written to the ``distribution_cache`` table before being
        returned.

        Parameters
        ----------
        project_id:
            Pipeline/project identifier used to scope task queries.
        profile_name:
            Key into the ``profiles`` dict passed to ``__init__``.
        statuses:
            List of ``TaskStatus`` string values to include.  ``None`` means
            all stages (default), which is the right setting for the scatter
            whose whole point is colouring by status.
        min_cluster_size:
            HDBSCAN ``min_cluster_size`` parameter.
        umap_neighbors:
            UMAP ``n_neighbors`` parameter.
        umap_min_dist:
            UMAP ``min_dist`` parameter.

        Returns
        -------
        dict with keys:
            ``params``, ``clusters``, ``coords``, ``task_count``.
        """
        if profile_name not in self._profiles:
            raise KeyError(
                f"profile {profile_name!r} not found; available: {sorted(self._profiles)}"
            )
        profile = self._profiles[profile_name]

        # Convert status strings to enum set for filtering (validates them).
        status_enums: set[TaskStatus] | None
        if statuses is not None:
            status_enums = {TaskStatus(s) for s in statuses}
        else:
            status_enums = None

        # --- Load tasks -------------------------------------------------
        task_ids: list[str] = []
        task_statuses: list[str] = []
        texts: list[str] = []
        for task in self._store.list_tasks_by_pipeline(project_id):
            if status_enums is not None and task.status not in status_enums:
                continue
            text = canonical_task_text(task)
            task_ids.append(task.task_id)
            task_statuses.append(task.status.value)
            texts.append(text)

        # --- Embed -------------------------------------------------------
        client = build_embedding_client(profile)
        try:
            emb = client.embed(texts) if texts else None
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

        # --- UMAP + HDBSCAN (skip when no tasks) -------------------------
        coords_xy: list[tuple[float, float]] = []
        labels: list[int] = []
        clusters: list[Cluster] = []

        if emb is not None and len(task_ids) > 0:
            import umap
            import hdbscan as _hdbscan

            # Clamp n_neighbors so UMAP doesn't crash on tiny inputs.
            n_neighbors = min(umap_neighbors, max(1, len(task_ids) - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=umap_min_dist,
                random_state=42,
            )
            coords_arr = reducer.fit_transform(emb.vectors)
            clusterer = _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels_arr = clusterer.fit_predict(coords_arr)

            coords_xy = [
                (float(coords_arr[i, 0]), float(coords_arr[i, 1]))
                for i in range(len(task_ids))
            ]
            labels = list(labels_arr.tolist())

            # Build Cluster objects with average pairwise cosine similarity.
            cluster_members: dict[int, list[str]] = {}
            for tid, lbl in zip(task_ids, labels):
                if lbl == -1:
                    continue
                cluster_members.setdefault(int(lbl), []).append(tid)

            id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
            for lbl, members in cluster_members.items():
                idxs = [id_to_idx[m] for m in members]
                sims: list[float] = []
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        a = emb.vectors[idxs[i]]
                        b = emb.vectors[idxs[j]]
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

        # --- Build payload -----------------------------------------------
        label_by_id: dict[str, int] = dict(zip(task_ids, labels))
        generated_at = datetime.now(timezone.utc).isoformat()

        coords_out: list[dict[str, Any]] = []
        for i, tid in enumerate(task_ids):
            lbl = label_by_id.get(tid, -1)
            coords_out.append({
                "task_id": tid,
                "x": coords_xy[i][0] if coords_xy else 0.0,
                "y": coords_xy[i][1] if coords_xy else 0.0,
                "status": task_statuses[i],
                "cluster_id": f"emb-{lbl}" if lbl != -1 else None,
                "text_preview": texts[i][:120],
            })

        payload: dict[str, Any] = {
            "params": {
                "profile": profile_name,
                "model": profile.model,
                "min_cluster_size": min_cluster_size,
                "umap_neighbors": umap_neighbors,
                "umap_min_dist": umap_min_dist,
                "statuses": statuses,
                "generated_at": generated_at,
            },
            "clusters": [asdict(c) for c in clusters],
            "coords": coords_out,
            "task_count": len(task_ids),
        }

        # --- Cache write --------------------------------------------------
        content_hash = compute_distribution_content_hash(
            self._store,
            project_id=project_id,
            statuses=statuses,
        )
        write_distribution_cache(
            self._store,
            project_id=project_id,
            profile_name=profile_name,
            payload=payload,
            content_hash=content_hash,
            created_at=generated_at,
        )

        return payload

    def get_cache_state(self, *, project_id: str, profile_name: str) -> dict:
        """Return cache presence and staleness information.

        Returns
        -------
        dict with keys:
            ``cached`` (bool), ``payload`` (dict | None),
            ``generated_at`` (str | None), ``cached_content_hash`` (str | None),
            ``current_content_hash`` (str), ``stale`` (bool).

        ``current_content_hash`` is computed for the same status filter that
        the cache was built with (recovered from ``payload.params.statuses``).
        When there is no cache, the all-status hash is used.
        """
        row = read_distribution_cache(
            self._store, project_id=project_id, profile_name=profile_name,
        )

        if row is None:
            current_hash = compute_distribution_content_hash(
                self._store, project_id=project_id, statuses=None,
            )
            return {
                "cached": False,
                "payload": None,
                "generated_at": None,
                "cached_content_hash": None,
                "current_content_hash": current_hash,
                "stale": False,
            }

        cached_payload = row["payload"]
        cached_statuses: list[str] | None = (
            cached_payload.get("params", {}).get("statuses")
        )
        current_hash = compute_distribution_content_hash(
            self._store,
            project_id=project_id,
            statuses=cached_statuses,
        )
        cached_hash = row["content_hash"]
        return {
            "cached": True,
            "payload": cached_payload,
            "generated_at": row["created_at"],
            "cached_content_hash": cached_hash,
            "current_content_hash": current_hash,
            "stale": current_hash != cached_hash,
        }

    def reject_duplicates(
        self,
        *,
        project_id: str,
        task_ids: list[str],
        cluster_id: str | None = None,
        representative_task_id: str | None = None,
        cluster_similarity: float | None = None,
        embedding_profile: str = "",
        embedding_model: str = "",
        actor: str = "operator",
    ) -> dict:
        """Transition each task in ``task_ids`` from ACCEPTED → REJECTED.

        Non-ACCEPTED tasks are skipped without error (idempotent — calling
        again with the same IDs returns ``moved=0, skipped=N``).

        An audit event is written with ``stage='similarity_dedup_embedding'``
        and metadata that records the cluster context for traceability.

        Returns
        -------
        dict with keys ``moved`` (int), ``skipped`` (int),
        ``skipped_task_ids`` (list[str]).
        """
        moved = 0
        skipped_task_ids: list[str] = []
        n = len(task_ids)
        sim_str = f"{cluster_similarity:.2f}" if cluster_similarity is not None else "n/a"
        reason = (
            f"embedding 语义聚类 ({embedding_model}, cos≈{sim_str})；"
            f"簇 {cluster_id}（{n} 个 task）保留代表 {representative_task_id}"
        )

        for tid in task_ids:
            try:
                task = self._store.load_task(tid)
            except (FileNotFoundError, KeyError):
                skipped_task_ids.append(tid)
                continue

            if task.status is not TaskStatus.ACCEPTED:
                skipped_task_ids.append(tid)
                continue

            try:
                event = transition_task(
                    task,
                    TaskStatus.REJECTED,
                    actor=actor,
                    reason=reason,
                    stage=_REJECT_STAGE,
                    metadata={
                        "rejection_kind": "similarity_dedup_embedding",
                        "cluster_id": cluster_id,
                        "cluster_size": n,
                        "cluster_similarity": cluster_similarity,
                        "representative_task_id": representative_task_id,
                        "embedding_profile": embedding_profile,
                        "embedding_model": embedding_model,
                        "previous_status": "accepted",
                        "reversible_via": "manual_drag to ARBITRATING or ACCEPTED",
                    },
                )
                self._store.save_task(task)
                self._store.append_event(event)
                moved += 1
            except InvalidTransition:
                skipped_task_ids.append(tid)

        return {
            "moved": moved,
            "skipped": len(skipped_task_ids),
            "skipped_task_ids": skipped_task_ids,
        }

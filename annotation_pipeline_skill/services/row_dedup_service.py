"""RowDedupService — find and mask near-duplicate rows across tasks.

Scans all rows in a project's tasks (post-mask), embeds them via the
configured SimilarityProfile, clusters them by similarity, and exposes
a ``mask_duplicates`` method to suppress all but the representative row
in each cluster.

Two clustering paths:
  - ``minhash`` provider: uses MinHashLSHFinder (Jaccard threshold on
    word-level shingles). ``jaccard_threshold`` is interpreted as Jaccard.
  - Any other provider (e.g. ``jina_http``, ``random``): uses brute-force
    cosine similarity + union-find connected components.
    ``jaccard_threshold`` is interpreted as cosine similarity threshold
    in this case. The ``metric`` field in the returned payload indicates
    which interpretation was used.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np

from annotation_pipeline_skill.services.row_mask_service import RowMaskService
from annotation_pipeline_skill.similarity.embedding_cache import text_content_hash
from annotation_pipeline_skill.similarity.embeddings import build_embedding_client
from annotation_pipeline_skill.similarity.profiles import SimilarityProfile
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.util.text import truncate_to_words


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _brute_force_neighbors(
    vectors: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Return connected-components clusters from pairwise cosine similarity.

    Parameters
    ----------
    vectors:
        (N, dim) float32 array, one row per item.
    threshold:
        Minimum cosine similarity for an edge.

    Returns
    -------
    List of clusters; each cluster is a sorted list of row indices.
    Only clusters with >= 2 members are returned.
    """
    n = vectors.shape[0]
    if n == 0:
        return []

    # Normalize rows to unit length.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / norms  # (N, dim)

    # Build adjacency via pairwise dot products.
    sim_matrix = normed @ normed.T  # (N, N)

    # Union-Find
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if float(sim_matrix[i, j]) >= threshold:
                _union(i, j)

    # Collect components.
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[_find(i)].append(i)

    return [sorted(members) for members in groups.values() if len(members) >= 2]


def _compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _avg_pairwise_cosine(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 1.0
    total, n = 0.0, 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            total += _compute_cosine_similarity(vectors[i], vectors[j])
            n += 1
    return total / n if n else 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RowDedupService:
    """Scan project rows for near-duplicates and mask cluster members.

    Mirrors DistributionService's structure but operates at the
    (task_id, row_index) granularity rather than the task granularity,
    and uses MinHashLSHFinder or brute-force cosine clustering instead of
    UMAP+HDBSCAN.
    """

    def __init__(
        self,
        store: SqliteStore,
        profiles: dict[str, SimilarityProfile],
    ) -> None:
        self._store = store
        self._profiles = profiles

    def scan_rows(
        self,
        *,
        project_id: str,
        profile_name: str,
        statuses: list[str] | None = None,
        jaccard_threshold: float = 0.5,
    ) -> dict:
        """Scan all rows in a project for near-duplicates.

        Parameters
        ----------
        project_id:
            Pipeline/project identifier.
        profile_name:
            Key into the ``profiles`` dict passed to ``__init__``.
        statuses:
            List of status strings to include (``None`` = all statuses).
        jaccard_threshold:
            Similarity threshold for clustering. Interpreted as Jaccard
            for ``minhash`` provider, as cosine similarity for others.

        Returns
        -------
        dict with keys ``params``, ``clusters``, ``row_count``, ``task_count``.
        """
        from annotation_pipeline_skill.interfaces.api import (
            compute_row_dedup_content_hash,
            write_row_dedup_cache,
        )
        from annotation_pipeline_skill.similarity.row_embedding_cache import RowEmbeddingCache

        if profile_name not in self._profiles:
            raise KeyError(
                f"profile {profile_name!r} not found; available: {sorted(self._profiles)}"
            )
        profile = self._profiles[profile_name]
        is_minhash = profile.provider == "minhash"

        # Determine metric string for payload
        metric = "jaccard" if is_minhash else "cosine"

        # Compute salt for content hashing (same scheme as DistributionService)
        if is_minhash:
            hash_salt = f"minhash-w{profile.shingle_size}-p{profile.num_perm}"
        else:
            hash_salt = f"{profile.provider}:{profile.model}"

        # Load tasks, filtering by status if requested
        from annotation_pipeline_skill.core.states import TaskStatus

        all_tasks = self._store.list_tasks_by_pipeline(project_id)
        if statuses is not None:
            status_enums = {TaskStatus(s) for s in statuses}
            all_tasks = [t for t in all_tasks if t.status in status_enums]

        task_ids = [t.task_id for t in all_tasks]

        # Bulk-fetch masked row indices upfront
        mask_svc = RowMaskService(self._store)
        masked_by_task = mask_svc.masked_indices_by_task(task_ids)

        # Collect (task_id, row_index, row_text) triplets — INCLUDING
        # masked rows. They still participate in clustering so the
        # operator can see them in their original cluster (marked with
        # ``masked: true``), but they are barred from being the rep via
        # ``rep_exclude`` and are tagged in the output so the UI can
        # render them with a "masked" badge instead of a checkbox.
        triplets: list[tuple[str, int, str]] = []  # (task_id, row_index, row_text)
        contributing_task_ids: set[str] = set()
        # Set of "task_id:row_index" keys (matching the MinHashLSHFinder
        # key format) for masked rows — passed as ``rep_exclude`` so a
        # masked row never gets chosen as the cluster representative.
        masked_member_keys: set[str] = set()

        for task in all_tasks:
            rows = (
                task.source_ref.get("payload", {}).get("rows", [])
                if isinstance(task.source_ref, dict)
                else []
            )
            if not isinstance(rows, list):
                rows = []

            masked = masked_by_task.get(task.task_id, set())
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_index = row.get("row_index")
                if not isinstance(row_index, int):
                    continue
                input_val = row.get("input")
                if not isinstance(input_val, str):
                    continue
                triplets.append((task.task_id, row_index, input_val))
                contributing_task_ids.add(task.task_id)
                if row_index in masked:
                    masked_member_keys.add(f"{task.task_id}:{row_index}")

        # Compute content hashes for each row
        content_hashes = [
            text_content_hash(f"{hash_salt}|{text}") for _, _, text in triplets
        ]

        # Cache lookup
        row_cache = RowEmbeddingCache(self._store)
        specs = [
            (tid, idx, h)
            for (tid, idx, _), h in zip(triplets, content_hashes)
        ]
        cached = row_cache.get_many(profile_name=profile_name, specs=specs)

        # Separate hits from misses
        miss_indices = [
            i for i, (tid, idx, _) in enumerate(triplets)
            if (tid, idx) not in cached
        ]
        miss_texts = [triplets[i][2] for i in miss_indices]

        cache_stats = {
            "hits": len(triplets) - len(miss_indices),
            "misses": len(miss_indices),
        }

        # Embed misses
        fresh_vectors: dict[int, np.ndarray] = {}
        if miss_texts:
            client = build_embedding_client(profile)
            try:
                fresh_result = client.embed(miss_texts)
            finally:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass

            miss_dim = int(fresh_result.vectors.shape[1]) if fresh_result.vectors.size else 0

            # Persist to cache
            put_entries = []
            for k, i in enumerate(miss_indices):
                tid, idx, _ = triplets[i]
                vec = fresh_result.vectors[k]
                fresh_vectors[i] = vec
                put_entries.append((tid, idx, content_hashes[i], vec))

            row_cache.put_many(
                profile_name=profile_name,
                model=profile.model,
                dim=miss_dim,
                entries=put_entries,
            )

        # Build full vector array: stitch cached + fresh
        # For minhash clustering, we use MinHashLSHFinder directly on text.
        # For other providers, we need the vectors for cosine clustering.

        clusters_out: list[dict[str, Any]] = []

        if triplets:
            if is_minhash:
                from annotation_pipeline_skill.similarity.minhash import MinHashLSHFinder

                finder = MinHashLSHFinder(
                    shingle_size=profile.shingle_size,
                    num_perm=profile.num_perm,
                    jaccard_threshold=jaccard_threshold,
                )
                for tid, idx, text in triplets:
                    finder.add(f"{tid}:{idx}", text)

                # Pass masked_member_keys as rep_exclude so the finder
                # picks an UN-masked row as each cluster's representative
                # AND uses that rep for the step-2 verification. Masked
                # rows still survive verification (J(masked, rep) ≥ t)
                # and remain as cluster members — they're just tagged
                # ``masked: true`` below so the UI renders a badge.
                raw_clusters = finder.clusters(
                    include_singletons=False,
                    rep_exclude=masked_member_keys,
                )

                # Build text preview and word count lookups once
                preview_by_key: dict[str, str] = {}
                word_count_by_key: dict[str, int] = {}
                for t2, i2, txt2 in triplets:
                    preview_by_key[f"{t2}:{i2}"] = truncate_to_words(txt2, 100)
                    word_count_by_key[f"{t2}:{i2}"] = len(txt2.split())

                for ci, cluster in enumerate(sorted(raw_clusters, key=lambda c: len(c.task_ids), reverse=True)):
                    # Re-derive the rep (lex-smallest UN-masked) so we can
                    # compute per-member similarity-to-rep. This must match
                    # the rep the finder used in step-2 verification: same
                    # rule (lex-smallest, excluding masked) so members stay
                    # consistent. We record ``sim_to_rep`` on every member
                    # so the UI slider can filter members in real time
                    # without re-scanning — drag the slider up, members
                    # with sim_to_rep < new_threshold disappear.
                    keys_sorted = sorted(cluster.task_ids)
                    eligible = [k for k in keys_sorted if k not in masked_member_keys]
                    rep_key = eligible[0] if eligible else keys_sorted[0]
                    rep_mh = finder._minhashes.get(rep_key)

                    members = []
                    for member_key in cluster.task_ids:
                        last_colon = member_key.rfind(":")
                        if last_colon < 0:
                            continue
                        m_tid = member_key[:last_colon]
                        m_idx = int(member_key[last_colon + 1:])
                        # Member's direct MinHash-estimated Jaccard with
                        # the rep. Rep itself gets 1.0. Used for real-time
                        # slider filtering on the frontend.
                        if member_key == rep_key:
                            sim_to_rep = 1.0
                        elif rep_mh is None:
                            sim_to_rep = 0.0
                        else:
                            mh = finder._minhashes.get(member_key)
                            sim_to_rep = float(mh.jaccard(rep_mh)) if mh is not None else 0.0
                        members.append({
                            "task_id": m_tid,
                            "row_index": m_idx,
                            "text_preview": preview_by_key.get(member_key, ""),
                            "masked": member_key in masked_member_keys,
                            "sim_to_rep": sim_to_rep,
                            "word_count": word_count_by_key.get(member_key, 0),
                        })
                    clusters_out.append({
                        "cluster_id": f"row-{ci}",
                        "members": members,
                        "similarity": float(cluster.similarity),
                        "method": "minhash",
                    })

            else:
                # Build full vector array
                dim_sample: np.ndarray | None = None
                for i, (tid, idx, _) in enumerate(triplets):
                    if (tid, idx) in cached:
                        dim_sample = cached[(tid, idx)].vector
                        break
                    if i in fresh_vectors:
                        dim_sample = fresh_vectors[i]
                        break

                if dim_sample is not None:
                    dim = int(dim_sample.shape[0])
                    all_vecs = np.zeros((len(triplets), dim), dtype=np.float32)
                    for i, (tid, idx, _) in enumerate(triplets):
                        if (tid, idx) in cached:
                            all_vecs[i] = cached[(tid, idx)].vector
                        elif i in fresh_vectors:
                            all_vecs[i] = fresh_vectors[i]
                        # else: zero vector (shouldn't happen)

                    raw_components = _brute_force_neighbors(all_vecs, jaccard_threshold)

                    def _is_masked_pos(pos: int) -> bool:
                        tid, idx, _ = triplets[pos]
                        return f"{tid}:{idx}" in masked_member_keys

                    for ci, component in enumerate(
                        sorted(raw_components, key=len, reverse=True)
                    ):
                        # Step 2: rep-anchored verification. The rep is
                        # the lex-smallest UN-masked member (so masked
                        # rows can stay in the cluster as members but
                        # never serve as rep). Members are kept iff
                        # their direct cosine with the rep ≥ threshold.
                        rep_pos: int | None = None
                        if len(component) >= 2:
                            sorted_by_key = sorted(
                                component,
                                key=lambda p: (str(triplets[p][0]), int(triplets[p][1])),
                            )
                            eligible_reps = [p for p in sorted_by_key if not _is_masked_pos(p)]
                            rep_pos = eligible_reps[0] if eligible_reps else sorted_by_key[0]
                            rep_vec = all_vecs[rep_pos]
                            kept = [rep_pos]
                            for pos in sorted_by_key:
                                if pos == rep_pos:
                                    continue
                                if _compute_cosine_similarity(all_vecs[pos], rep_vec) >= jaccard_threshold:
                                    kept.append(pos)
                            component = kept

                        if len(component) < 2:
                            continue

                        # Per-member similarity to rep (= cosine in this
                        # path) so the UI slider can filter members in
                        # real-time without re-scanning.
                        rep_vec = all_vecs[rep_pos] if rep_pos is not None else None
                        members = []
                        component_vecs = []
                        for pos in component:
                            tid, idx, text = triplets[pos]
                            if pos == rep_pos:
                                sim_to_rep = 1.0
                            elif rep_vec is None:
                                sim_to_rep = 0.0
                            else:
                                sim_to_rep = _compute_cosine_similarity(all_vecs[pos], rep_vec)
                            members.append({
                                "task_id": tid,
                                "row_index": idx,
                                "text_preview": truncate_to_words(text, 100),
                                "masked": _is_masked_pos(pos),
                                "sim_to_rep": float(sim_to_rep),
                                "word_count": len(text.split()),
                            })
                            component_vecs.append(all_vecs[pos])

                        avg_sim = _avg_pairwise_cosine(component_vecs)
                        clusters_out.append({
                            "cluster_id": f"row-{ci}",
                            "members": members,
                            "similarity": float(avg_sim),
                            "method": "embedding",
                        })

        generated_at = datetime.now(timezone.utc).isoformat()

        payload: dict[str, Any] = {
            "params": {
                "profile": profile_name,
                "provider": profile.provider,
                "model": profile.model,
                "jaccard_threshold": jaccard_threshold,
                "metric": metric,
                "statuses": statuses,
                "generated_at": generated_at,
                "embedding_cache": cache_stats,
            },
            "clusters": clusters_out,
            "row_count": len(triplets),
            "task_count": len(contributing_task_ids),
        }

        # Write to cache
        content_hash = compute_row_dedup_content_hash(
            self._store,
            project_id=project_id,
            statuses=statuses,
        )
        write_row_dedup_cache(
            self._store,
            project_id=project_id,
            profile_name=profile_name,
            payload=payload,
            content_hash=content_hash,
            created_at=generated_at,
        )

        return payload

    def get_cache_state(self, *, project_id: str, profile_name: str) -> dict:
        """Return cache presence and staleness for a (project, profile) pair.

        Mirrors DistributionService.get_cache_state.
        """
        from annotation_pipeline_skill.interfaces.api import (
            compute_row_dedup_content_hash,
            read_row_dedup_cache,
        )

        row = read_row_dedup_cache(
            self._store, project_id=project_id, profile_name=profile_name,
        )
        if row is None:
            current_hash = compute_row_dedup_content_hash(
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
        current_hash = compute_row_dedup_content_hash(
            self._store,
            project_id=project_id,
            statuses=cached_statuses,
        )
        cached_hash = row["content_hash"]

        # Overlay the current row_masks state on the cached cluster
        # members. The scan that produced this cache filtered out
        # already-masked rows, but masks applied AFTER the scan
        # (typical workflow: scan once, then apply masks) won't show
        # up in the cached payload. We surface them here so the UI
        # can visibly tag those members without paying for a re-scan.
        cached_keys: set[tuple[str, int]] = set()
        for cluster in cached_payload.get("clusters", []) or []:
            for m in cluster.get("members", []) or []:
                if not isinstance(m, dict):
                    continue
                tid = m.get("task_id")
                ridx = m.get("row_index")
                if isinstance(tid, str) and isinstance(ridx, int):
                    cached_keys.add((tid, ridx))

        currently_masked: list[list] = []
        if cached_keys:
            mask_rows = self._store._conn.execute(
                "SELECT task_id, row_index FROM row_masks"
            ).fetchall()
            for mr in mask_rows:
                key = (mr["task_id"], int(mr["row_index"]))
                if key in cached_keys:
                    currently_masked.append([mr["task_id"], int(mr["row_index"])])

        return {
            "cached": True,
            "payload": cached_payload,
            "generated_at": row["created_at"],
            "cached_content_hash": cached_hash,
            "current_content_hash": current_hash,
            "stale": current_hash != cached_hash,
            "currently_masked": currently_masked,
        }

    def mask_duplicates(
        self,
        *,
        project_id: str,
        members: list[dict],  # [{task_id, row_index}, ...]
        cluster_id: str,
        similarity: float,
        profile_name: str,
        model: str,
    ) -> dict:
        """Mask all cluster members except the representative.

        The representative is the member with the smallest
        ``(task_id, row_index)`` pair (lexicographic on task_id, then
        numeric on row_index). The remaining members get a row_mask
        applied via RowMaskService.

        Idempotent — calling again with the same members returns
        ``{masked: 0, skipped: N}`` (all already masked).

        Returns
        -------
        dict with keys ``masked`` (int), ``skipped`` (int).
        """
        if not members:
            return {"masked": 0, "skipped": 0}

        mask_svc = RowMaskService(self._store)

        # Find representative: smallest (task_id, row_index) lexicographically
        def _sort_key(m: dict) -> tuple[str, int]:
            return (str(m["task_id"]), int(m["row_index"]))

        sorted_members = sorted(members, key=_sort_key)
        representative = sorted_members[0]
        rep_key = _sort_key(representative)

        # Determine metric from profile if available
        profile = self._profiles.get(profile_name)
        if profile is not None:
            masked_by = (
                "row_dedup_jaccard" if profile.provider == "minhash"
                else "row_dedup_cosine"
            )
            metric = "jaccard" if profile.provider == "minhash" else "cosine"
        else:
            # Fallback: infer from model name hint or use generic
            masked_by = "row_dedup"
            metric = "unknown"

        # Collect peer rows (all except representative)
        peer_rows = [
            {"task_id": m["task_id"], "row_index": int(m["row_index"])}
            for m in sorted_members[1:]
        ]

        # Check which rows are already masked (to count skipped)
        to_mask = sorted_members[1:]
        if not to_mask:
            return {"masked": 0, "skipped": 0}

        # Bulk-fetch current masks for the affected tasks
        affected_task_ids = list({m["task_id"] for m in to_mask})
        existing_by_task = mask_svc.masked_indices_by_task(affected_task_ids)

        masks_to_apply = []
        skipped = 0
        for m in to_mask:
            tid = m["task_id"]
            idx = int(m["row_index"])
            if idx in existing_by_task.get(tid, set()):
                skipped += 1
                continue
            masks_to_apply.append({
                "task_id": tid,
                "row_index": idx,
                "reason": f"near-duplicate of {rep_key[0]}:{rep_key[1]} in cluster {cluster_id}",
                "masked_by": masked_by,
                "metadata": {
                    "cluster_id": cluster_id,
                    "cluster_similarity": similarity,
                    "peer_rows": peer_rows,
                    "representative": {
                        "task_id": representative["task_id"],
                        "row_index": int(representative["row_index"]),
                    },
                    "metric": metric,
                    "profile_name": profile_name,
                    "model": model,
                },
            })

        applied = mask_svc.apply_many(masks_to_apply)
        return {"masked": applied, "skipped": skipped}

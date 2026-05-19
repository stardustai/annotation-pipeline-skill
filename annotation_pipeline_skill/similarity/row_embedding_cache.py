"""Per-row embedding cache (SQLite-backed).

Parallel to TaskEmbeddingCache but keyed by (task_id, row_index, profile_name)
instead of (task_id, profile_name). Used by RowDedupService to avoid
re-embedding rows that haven't changed between dedup scans.

Vectors are stored as little-endian float32 BLOBs — same layout as
task_embeddings. Content hash is sha256 of the salted row text (same
salt scheme as TaskEmbeddingCache) so text changes auto-invalidate.

Chunk lookups at 400 specs per query (3-tuple keys are denser in the
expression tree than the 2-tuple task_embeddings lookups — 800/2 = 400).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from annotation_pipeline_skill.similarity.embedding_cache import text_content_hash  # re-export
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@dataclass(frozen=True)
class CachedRowEmbedding:
    task_id: str
    row_index: int
    content_hash: str
    vector: np.ndarray


class RowEmbeddingCache:
    """Cache for per-row embeddings keyed by (task_id, row_index, profile_name).

    ``get_many`` returns only entries whose stored content_hash exactly matches
    the requested hash — stale rows (text changed) are silently dropped, forcing
    the caller to re-embed.
    """

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def get_many(
        self,
        *,
        profile_name: str,
        specs: list[tuple[str, int, str]],  # (task_id, row_index, content_hash)
    ) -> dict[tuple[str, int], CachedRowEmbedding]:
        """Bulk lookup.

        Returns a dict keyed by (task_id, row_index). Only entries whose
        stored content_hash matches the requested hash are returned.
        """
        if not specs:
            return {}

        # Build a lookup: (task_id, row_index) → wanted_hash
        wanted: dict[tuple[str, int], str] = {}
        for tid, idx, h in specs:
            wanted[(tid, int(idx))] = h

        out: dict[tuple[str, int], CachedRowEmbedding] = {}

        # Chunk at 400 specs: each spec is a 3-tuple, generating a 3-column
        # WHERE expression, so the effective expression-tree cost is ~3×
        # higher than the 1-column task_embeddings case. 400 keeps us well
        # below SQLite's 1000-node default limit.
        spec_list = list(wanted.keys())
        for i in range(0, len(spec_list), 400):
            batch = spec_list[i : i + 400]
            # Build: WHERE profile_name=? AND (task_id=? AND row_index=?) OR ...
            # Use a row-constructor for clarity and safety.
            pair_placeholders = ",".join("(?,?)" for _ in batch)
            params: list[object] = [profile_name]
            for tid, idx in batch:
                params.extend([tid, idx])
            rows = self._store._conn.execute(
                f"SELECT task_id, row_index, content_hash, dim, vector "
                f"FROM row_embeddings "
                f"WHERE profile_name=? AND (task_id, row_index) IN ({pair_placeholders})",
                params,
            ).fetchall()
            for r in rows:
                key = (r["task_id"], int(r["row_index"]))
                if wanted.get(key) != r["content_hash"]:
                    continue  # stale — text changed since last embed
                vec = np.frombuffer(r["vector"], dtype=np.float32)
                if int(r["dim"]) and vec.shape[0] != int(r["dim"]):
                    continue  # stored dim disagrees — skip
                out[key] = CachedRowEmbedding(
                    task_id=r["task_id"],
                    row_index=int(r["row_index"]),
                    content_hash=r["content_hash"],
                    vector=vec.copy(),
                )
        return out

    def put_many(
        self,
        *,
        profile_name: str,
        model: str,
        dim: int,
        entries: list[tuple[str, int, str, np.ndarray]],
        # (task_id, row_index, content_hash, vector)
    ) -> None:
        """Bulk UPSERT. Coerces vectors to float32 before persisting."""
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for tid, idx, h, vec in entries:
            v = np.asarray(vec, dtype=np.float32)
            rows.append((tid, int(idx), profile_name, model, dim, h, v.tobytes(), now))
        self._store._conn.executemany(
            "INSERT INTO row_embeddings "
            "(task_id, row_index, profile_name, model, dim, content_hash, vector, created_at) "
            "VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(task_id, row_index, profile_name) DO UPDATE SET "
            "model=excluded.model, dim=excluded.dim, "
            "content_hash=excluded.content_hash, "
            "vector=excluded.vector, "
            "created_at=excluded.created_at",
            rows,
        )
        self._store._conn.commit()

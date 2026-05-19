"""Per-task embedding cache (SQLite-backed).

Avoids re-running the embedding server on every distribution scan. Keyed
by ``(task_id, profile_name)``; the row carries a ``content_hash`` of
the canonical text so we can detect input changes and trigger a
re-embed without explicit invalidation.

Vectors are stored as little-endian float32 BLOBs — ~4 KB per task for
the local Jina v5 small (dim=1024), comfortably under SQLite's 1 GB
default row limit for thousands of tasks.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def text_content_hash(text: str) -> str:
    """Stable fingerprint of a task's canonical text. Used as the cache
    invalidation key: the same text always produces the same hash, and a
    one-character edit produces a different hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class CachedEmbedding:
    task_id: str
    content_hash: str
    vector: np.ndarray


class TaskEmbeddingCache:
    def __init__(self, store: SqliteStore):
        self.store = store

    def get_many(
        self,
        *,
        profile_name: str,
        task_specs: list[tuple[str, str]],
    ) -> dict[str, CachedEmbedding]:
        """Bulk lookup. ``task_specs`` is ``[(task_id, content_hash), ...]``.

        Returns only the entries whose stored content_hash exactly matches
        the requested hash — stale rows (text changed) are NOT returned,
        forcing the caller to re-embed. Missing rows are NOT returned.
        """
        if not task_specs:
            return {}
        hashes_by_id = {tid: h for tid, h in task_specs}
        out: dict[str, CachedEmbedding] = {}
        # Chunked IN-clause query — SQLite default expression-tree depth
        # is 1000, so chunk to 800 to stay well clear.
        ids = list(hashes_by_id)
        for i in range(0, len(ids), 800):
            batch = ids[i : i + 800]
            placeholders = ",".join("?" * len(batch))
            rows = self.store._conn.execute(
                f"SELECT task_id, content_hash, dim, vector "
                f"FROM task_embeddings "
                f"WHERE profile_name=? AND task_id IN ({placeholders})",
                [profile_name, *batch],
            ).fetchall()
            for r in rows:
                tid = r["task_id"]
                if hashes_by_id.get(tid) != r["content_hash"]:
                    continue  # text changed since last embed; cache miss
                vec = np.frombuffer(r["vector"], dtype=np.float32)
                if int(r["dim"]) and vec.shape[0] != int(r["dim"]):
                    continue  # stored dim disagrees with payload — skip
                out[tid] = CachedEmbedding(
                    task_id=tid,
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
        entries: list[tuple[str, str, np.ndarray]],
    ) -> None:
        """Bulk UPSERT. ``entries`` is ``[(task_id, content_hash, vector), ...]``.

        Vector dtype is coerced to float32 before persisting so all rows
        share a known byte layout regardless of caller arithmetic.
        """
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for tid, h, vec in entries:
            v = np.asarray(vec, dtype=np.float32)
            rows.append(
                (tid, profile_name, model, dim, h, v.tobytes(), now)
            )
        self.store._conn.executemany(
            "INSERT INTO task_embeddings "
            "(task_id, profile_name, model, dim, content_hash, vector, created_at) "
            "VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(task_id, profile_name) DO UPDATE SET "
            "model=excluded.model, dim=excluded.dim, "
            "content_hash=excluded.content_hash, "
            "vector=excluded.vector, "
            "created_at=excluded.created_at",
            rows,
        )
        self.store._conn.commit()

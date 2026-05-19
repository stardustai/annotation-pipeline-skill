"""Row-level mask service.

A row in ``row_masks`` is treated as nonexistent by every downstream
consumer that reads task data — export, entity statistics, posterior
audit, scatter. The task itself stays in whatever status it had; only
the specific (task_id, row_index) entries disappear at the read boundary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@dataclass(frozen=True)
class RowMask:
    task_id: str
    row_index: int
    reason: str
    masked_by: str
    masked_at: str
    metadata: dict | None = None


class RowMaskService:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def list_for_task(self, task_id: str) -> list[RowMask]:
        rows = self._store._conn.execute(
            "SELECT task_id, row_index, reason, masked_by, masked_at, metadata_json "
            "FROM row_masks WHERE task_id=? ORDER BY row_index",
            (task_id,),
        ).fetchall()
        return [self._row_to_mask(r) for r in rows]

    def masked_indices_for_task(self, task_id: str) -> set[int]:
        """Cheaper than list_for_task when callers only need the set of
        masked row_index values — used at every read boundary."""
        rows = self._store._conn.execute(
            "SELECT row_index FROM row_masks WHERE task_id=?",
            (task_id,),
        ).fetchall()
        return {int(r["row_index"]) for r in rows}

    def masked_indices_by_task(self, task_ids: list[str]) -> dict[str, set[int]]:
        """Bulk variant — one SELECT for all tasks (chunked at 800 ids to
        stay under SQLite's expression-tree depth limit). Used by the
        scatter and stats code paths that walk every task."""
        out: dict[str, set[int]] = {tid: set() for tid in task_ids}
        if not task_ids:
            return out
        ids = list(task_ids)
        for i in range(0, len(ids), 800):
            batch = ids[i:i + 800]
            placeholders = ",".join("?" * len(batch))
            rows = self._store._conn.execute(
                f"SELECT task_id, row_index FROM row_masks "
                f"WHERE task_id IN ({placeholders})",
                batch,
            ).fetchall()
            for r in rows:
                out[r["task_id"]].add(int(r["row_index"]))
        return out

    def list_for_project(self, project_id: str) -> list[RowMask]:
        """All masks for tasks belonging to a project. Implemented via JOIN
        on the tasks table."""
        rows = self._store._conn.execute(
            "SELECT m.task_id, m.row_index, m.reason, m.masked_by, "
            "       m.masked_at, m.metadata_json "
            "FROM row_masks m "
            "JOIN tasks t ON t.task_id = m.task_id "
            "WHERE t.pipeline_id=? "
            "ORDER BY m.task_id, m.row_index",
            (project_id,),
        ).fetchall()
        return [self._row_to_mask(r) for r in rows]

    def apply(
        self,
        *,
        task_id: str,
        row_index: int,
        reason: str,
        masked_by: str,
        metadata: dict | None = None,
    ) -> RowMask:
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata, ensure_ascii=False) if metadata is not None else None
        self._store._conn.execute(
            "INSERT INTO row_masks "
            "(task_id, row_index, reason, masked_by, masked_at, metadata_json) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(task_id, row_index) DO UPDATE SET "
            "reason=excluded.reason, masked_by=excluded.masked_by, "
            "masked_at=excluded.masked_at, metadata_json=excluded.metadata_json",
            (task_id, row_index, reason, masked_by, now, meta_str),
        )
        self._store._conn.commit()
        return RowMask(
            task_id=task_id, row_index=row_index, reason=reason,
            masked_by=masked_by, masked_at=now, metadata=metadata,
        )

    def apply_many(self, masks: list[dict]) -> int:
        """Bulk UPSERT. Each dict needs ``task_id``, ``row_index``, ``reason``,
        ``masked_by``, optional ``metadata``. Returns the count written."""
        if not masks:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for m in masks:
            md = m.get("metadata")
            md_str = json.dumps(md, ensure_ascii=False) if md is not None else None
            rows.append((m["task_id"], int(m["row_index"]), m["reason"],
                         m["masked_by"], now, md_str))
        self._store._conn.executemany(
            "INSERT INTO row_masks "
            "(task_id, row_index, reason, masked_by, masked_at, metadata_json) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(task_id, row_index) DO UPDATE SET "
            "reason=excluded.reason, masked_by=excluded.masked_by, "
            "masked_at=excluded.masked_at, metadata_json=excluded.metadata_json",
            rows,
        )
        self._store._conn.commit()
        return len(rows)

    def remove(self, *, task_id: str, row_index: int) -> bool:
        cur = self._store._conn.execute(
            "DELETE FROM row_masks WHERE task_id=? AND row_index=?",
            (task_id, row_index),
        )
        self._store._conn.commit()
        return cur.rowcount > 0

    def remove_many(self, pairs: list[tuple[str, int]]) -> int:
        if not pairs:
            return 0
        cur = self._store._conn.executemany(
            "DELETE FROM row_masks WHERE task_id=? AND row_index=?",
            pairs,
        )
        self._store._conn.commit()
        return cur.rowcount

    def _row_to_mask(self, r) -> RowMask:
        return RowMask(
            task_id=r["task_id"],
            row_index=int(r["row_index"]),
            reason=r["reason"],
            masked_by=r["masked_by"],
            masked_at=r["masked_at"],
            metadata=(
                json.loads(r["metadata_json"]) if r["metadata_json"] else None
            ),
        )


def filter_masked_rows(payload: dict | None, masked_indices: set[int]) -> dict | None:
    """Return a SHALLOW copy of ``payload`` with rows whose ``row_index``
    is in ``masked_indices`` removed.

    Works on both:
      - source_ref.payload shape: ``{"rows": [{"row_index": int, "input": str, ...}, ...]}``
      - annotation_result shape: ``{"rows": [{"row_index": int, "output": {...}}, ...]}``

    No-op when ``masked_indices`` is empty or payload doesn't have a list
    of dict rows. Returns the input reference unchanged in those cases so
    callers can keep the variable name without conditionals.
    """
    if not masked_indices or not isinstance(payload, dict):
        return payload
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return payload
    kept = []
    for r in rows:
        if not isinstance(r, dict):
            kept.append(r)
            continue
        idx = r.get("row_index")
        if isinstance(idx, int) and idx in masked_indices:
            continue
        kept.append(r)
    if len(kept) == len(rows):
        return payload  # nothing actually masked in this payload
    return {**payload, "rows": kept}

"""Extract a single canonical text per task for similarity comparison.

Concatenates row inputs ordered by row_index. Deterministic so the same
task always produces the same text — both MinHash signatures and
embeddings are sensitive to byte-level changes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def canonical_task_text(task: Any, *, store: "SqliteStore | None" = None) -> str:
    """Concatenate ``task.source_ref.payload.rows[*].input`` ordered by
    ``row_index``, joined with newlines. Returns ``""`` when the task
    has no parseable rows.

    Rows missing ``row_index`` (or with a non-int value) fall back to
    their position in the payload list, which keeps the output stable
    for partially-typed data but means the ordering of a mixed batch
    isn't intuitive — feed clean per-task payloads if order matters.

    Parameters
    ----------
    store:
        Optional ``SqliteStore``. When provided, rows whose ``row_index``
        is listed in ``row_masks`` for this task are silently excluded
        before building the canonical text. Callers that don't pass a
        store get the original behaviour unchanged.
    """
    source_ref = getattr(task, "source_ref", None)
    if not isinstance(source_ref, dict):
        return ""
    payload = source_ref.get("payload")
    if not isinstance(payload, dict):
        return ""

    # Apply row-level masks when a store is available.
    if store is not None:
        from annotation_pipeline_skill.services.row_mask_service import (
            RowMaskService,
            filter_masked_rows,
        )
        task_id = getattr(task, "task_id", None)
        if task_id is not None:
            masked = RowMaskService(store).masked_indices_for_task(task_id)
            payload = filter_masked_rows(payload, masked)

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return ""
    pairs: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        idx = row.get("row_index") if isinstance(row.get("row_index"), int) else i
        text = row.get("input")
        if isinstance(text, str):
            pairs.append((idx, text))
    pairs.sort(key=lambda p: p[0])
    return "\n".join(text for _, text in pairs)

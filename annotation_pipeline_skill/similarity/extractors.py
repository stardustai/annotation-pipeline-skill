"""Extract a single canonical text per task for similarity comparison.

Concatenates row inputs ordered by row_index. Deterministic so the same
task always produces the same text — both MinHash signatures and
embeddings are sensitive to byte-level changes.
"""
from __future__ import annotations

from typing import Any


def canonical_task_text(task: Any) -> str:
    """Concatenate ``task.source_ref.payload.rows[*].input`` ordered by
    ``row_index``, joined with newlines. Returns ``""`` when the task
    has no parseable rows.

    Rows missing ``row_index`` (or with a non-int value) fall back to
    their position in the payload list, which keeps the output stable
    for partially-typed data but means the ordering of a mixed batch
    isn't intuitive — feed clean per-task payloads if order matters.
    """
    source_ref = getattr(task, "source_ref", None)
    if not isinstance(source_ref, dict):
        return ""
    payload = source_ref.get("payload")
    if not isinstance(payload, dict):
        return ""
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

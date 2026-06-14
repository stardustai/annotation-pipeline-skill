"""Pure consensus logic for N-way (duplicate) annotation.

Given N independent annotation drafts of the same task, compute:
  - a consensus payload: spans agreed by >= keep_threshold drafts
  - a disagreement list: spans present in some-but-fewer drafts (for the arbiter)

No I/O, no LLM calls — fully unit-testable. The runtime layer wires this to
the actual annotators + arbiter.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterator

SpanItem = tuple[int, str, str, str]  # (row_index, field, type, span)
_FIELDS = ("entities", "json_structures")


def iter_span_items(payload: dict) -> Iterator[SpanItem]:
    """Yield (row_index, field, type, span) for every span in a parsed
    annotation payload {"rows": [{"row_index", "output": {...}}]}."""
    if not isinstance(payload, dict):
        return
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        ri = row.get("row_index", 0)
        out = row.get("output") or {}
        if not isinstance(out, dict):
            continue
        for field in _FIELDS:
            buckets = out.get(field) or {}
            if not isinstance(buckets, dict):
                continue
            for typ, spans in buckets.items():
                if not isinstance(spans, list):
                    continue
                for span in spans:
                    if isinstance(span, str) and span:
                        yield (ri, field, typ, span)

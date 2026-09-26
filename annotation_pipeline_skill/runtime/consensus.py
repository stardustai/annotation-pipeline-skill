"""Pure consensus logic for N-way (duplicate) annotation.

Given N independent annotation drafts of the same task, compute:
  - a consensus payload: spans agreed by >= keep_threshold drafts
  - a disagreement list: spans present in some-but-fewer drafts (for the arbiter)

No I/O, no LLM calls — fully unit-testable. The runtime layer wires this to
the actual annotators + arbiter.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Iterator

SpanItem = tuple[int, str, str, str]  # (row_index, field, type, span)
_FIELDS = ("entities", "json_structures")


def source_row_ids(source_rows: list) -> dict[int, str]:
    """Map ``row_index -> row_id`` for the task's own source rows, in source order.

    The task's source rows are the only authority on which rows exist and what
    they are called; model output is aligned onto this map, never the other
    way round. A source row without ``row_index`` is identified by its
    position; one without ``row_id`` (a dataset that doesn't name its rows) by
    ``row-<row_index>``.
    """
    ids: dict[int, str] = {}
    for i, row in enumerate(source_rows or []):
        if not isinstance(row, dict):
            continue
        ri = row.get("row_index") if isinstance(row.get("row_index"), int) else i
        rid = row.get("row_id")
        ids[ri] = rid if isinstance(rid, str) and rid else f"row-{ri}"
    return ids


def iter_span_items(payload: dict) -> Iterator[SpanItem]:
    """Yield (row_index, field, type, span) for every span in a parsed
    annotation payload {"rows": [{"row_index", "output": {...}}]}.

    Rows without an integer ``row_index`` are skipped: they cannot be placed
    on a source row (defaulting them to 0 once invented a phantom row 0)."""
    if not isinstance(payload, dict):
        return
    for row in payload.get("rows") or []:
        if not isinstance(row, dict) or not isinstance(row.get("row_index"), int):
            continue
        ri = row["row_index"]
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


def build_consensus(
    drafts: list[dict], keep_threshold: int, *, source_rows: list,
) -> tuple[dict, list[dict]]:
    """Return (consensus_payload, disagreements).

    consensus_payload: {"rows":[...]} with exactly one row per task source row,
      in source order, carrying the source ``row_index`` + ``row_id`` and only
      the spans whose (row, field, type, span) support is >= keep_threshold.
    disagreements: list of {"row_index","field","type","span","support"} for
      items with 0 < support < keep_threshold.

    Draft rows whose ``row_index`` is not a source row (a draft that invented a
    row, or emitted one without keys) contribute nothing.
    """
    ids = source_row_ids(source_rows)
    counts: Counter[SpanItem] = Counter()
    for draft in drafts:
        # de-dup within a single draft so a draft can't vote twice
        counts.update(item for item in set(iter_span_items(draft)) if item[0] in ids)

    by_row: dict[int, dict] = {ri: {f: {} for f in _FIELDS} for ri in ids}
    disagreements: list[dict] = []
    for item, support in counts.items():
        ri, field, typ, span = item
        if support >= keep_threshold:
            by_row[ri][field].setdefault(typ, []).append(span)
        else:
            disagreements.append(
                {"row_index": ri, "field": field, "type": typ, "span": span, "support": support}
            )
    return {"rows": [_row_out(ri, ids[ri], by_row[ri]) for ri in ids]}, disagreements


def align_rows_to_source(payload: dict, source_rows: list) -> dict:
    """Place a model-emitted payload onto the task's own source rows.

    Rows are matched by ``row_index`` (the key the model was shown next to each
    input); ``row_id`` is always taken from the source row, never from the
    model. Rows whose ``row_index`` is not a source row are dropped, and rows
    sharing a ``row_index`` (the arbiter has been observed to emit a filled copy
    plus an empty scaffold) are merged by unioning spans per (field, type).
    Output follows source order; a source row the model omitted stays absent so
    row-coverage validation still reports it.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        return payload
    ids = source_row_ids(source_rows)
    merged: dict[int, dict] = {}
    for row in payload["rows"]:
        if not isinstance(row, dict) or row.get("row_index") not in ids:
            continue
        dst = merged.setdefault(row["row_index"], {f: {} for f in _FIELDS})
        out = row.get("output") or {}
        if not isinstance(out, dict):
            continue
        for field in _FIELDS:
            buckets = out.get(field) or {}
            if not isinstance(buckets, dict):
                continue
            for typ, spans in buckets.items():
                if isinstance(spans, list):
                    dst[field].setdefault(typ, set()).update(
                        s for s in spans if isinstance(s, str) and s
                    )
    return {"rows": [_row_out(ri, ids[ri], merged[ri]) for ri in ids if ri in merged]}


def _row_out(row_index: int, row_id: str, spans: dict) -> dict:
    # Always emit BOTH fields (as {} when empty): the output schema marks
    # entities + json_structures as required on every row. Span lists are
    # sorted so artifact bytes are reproducible regardless of draft order.
    out = {f: {t: sorted(s) for t, s in spans[f].items() if s} for f in _FIELDS}
    return {"row_index": row_index, "row_id": row_id, "output": out}


def build_arbiter_merge_prompt(
    *, source_rows: list,
    consensus: dict, disagreements: list[dict],
) -> str:
    """Build the user prompt for the arbiter merge call. The arbiter receives,
    per source row, its ``row_index`` + ``row_id`` + input text, the
    already-agreed consensus, and the disputed spans, and must return the FINAL
    annotation: keep consensus, resolve each disagreement per the rules
    (选择题), and add only clearly-required missed spans (补漏).

    ``row_id`` is shown next to ``row_index`` because the output schema requires
    both: an arbiter shown only ``row_index`` invents ``row_id`` = str(row_index).
    """
    ids = source_row_ids(source_rows)
    inputs = {
        (r.get("row_index") if isinstance(r.get("row_index"), int) else i):
            str(r.get("input") or r.get("text") or "")
        for i, r in enumerate(source_rows or []) if isinstance(r, dict)
    }
    agreed = {row["row_index"]: row.get("output", {}) for row in consensus.get("rows") or []}
    rows = [
        {
            "row_index": ri,
            "row_id": rid,
            "input": inputs.get(ri, ""),
            "agreed": agreed.get(ri, {}),
            "disputed": [d for d in disagreements if d["row_index"] == ri],
        }
        for ri, rid in ids.items()
    ]
    return (
        "你是标注仲裁器(arbiter)。每行给出 row_index、row_id、input、已一致的 agreed 标注、以及 disputed(只有部分草稿标了的 span)。\n"
        "产出每行正确的最终标注:\n"
        "- 保留 agreed。\n"
        "- 对 disputed 做选择题:按规则选对的 type;不该标的删。\n"
        "- 补漏:规则明确要求但所有草稿都漏的 span 才补(verbatim)。\n"
        "- 每个 span 必须是该行 input 的 verbatim 子串。\n"
        "- 输出的 rows 与下面给出的行一一对应、顺序相同:每行原样复制它的 row_index 和 row_id,不增行、不删行、不改编号。\n\n"
        "严格输出 JSON:{\"rows\":[{\"row_index\":int,\"row_id\":str,\"output\":{\"entities\":{...},\"json_structures\":{...}}}]}\n\n"
        + json.dumps(rows, ensure_ascii=False)
    )

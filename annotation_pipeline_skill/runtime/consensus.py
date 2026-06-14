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


def build_consensus(drafts: list[dict], keep_threshold: int) -> tuple[dict, list[dict]]:
    """Return (consensus_payload, disagreements).

    consensus_payload: same {"rows":[...]} shape, containing only spans whose
      (row, field, type, span) support count is >= keep_threshold.
    disagreements: list of {"row_index","field","type","span","support"} for
      items with 0 < support < keep_threshold.
    """
    counts: Counter[SpanItem] = Counter()
    for draft in drafts:
        # de-dup within a single draft so a draft can't vote twice
        counts.update(set(iter_span_items(draft)))

    kept: list[SpanItem] = []
    disagreements: list[dict] = []
    for item, support in counts.items():
        if support >= keep_threshold:
            kept.append(item)
        else:
            ri, field, typ, span = item
            disagreements.append(
                {"row_index": ri, "field": field, "type": typ, "span": span, "support": support}
            )

    # Rebuild a payload from kept items, preserving row order seen across drafts.
    row_order: list[int] = []
    seen_rows: set[int] = set()
    for draft in drafts:
        for row in draft.get("rows") or []:
            ri = row.get("row_index", 0) if isinstance(row, dict) else 0
            if ri not in seen_rows:
                seen_rows.add(ri); row_order.append(ri)
    by_row: dict[int, dict] = {ri: {"entities": {}, "json_structures": {}} for ri in row_order}
    for ri, field, typ, span in kept:
        by_row.setdefault(ri, {"entities": {}, "json_structures": {}})[field].setdefault(typ, []).append(span)
    rows_out = []
    for ri in row_order:
        out = {f: {t: s for t, s in by_row[ri][f].items() if s} for f in _FIELDS}
        out = {f: v for f, v in out.items() if v}
        rows_out.append({"row_index": ri, "output": out})
    return {"rows": rows_out}, disagreements


import json as _json


def build_arbiter_merge_prompt(
    *, row_inputs: dict[int, str], drafts: list[dict],
    consensus: dict, disagreements: list[dict],
) -> str:
    """Build the user prompt for the arbiter merge call. The arbiter receives the
    per-row input text, the already-agreed consensus, and the disputed spans, and
    must return the FINAL annotation: keep consensus, resolve each disagreement per
    the rules (选择题), and add only clearly-required missed spans (补漏)."""
    rows = []
    for row in consensus.get("rows") or []:
        ri = row.get("row_index", 0)
        rows.append({
            "row_index": ri,
            "input": row_inputs.get(ri, ""),
            "agreed": row.get("output", {}),
            "disputed": [d for d in disagreements if d["row_index"] == ri],
        })
    return (
        "你是标注仲裁器(arbiter)。每行给出 input、已一致的 agreed 标注、以及 disputed(只有部分草稿标了的 span)。\n"
        "产出每行正确的最终标注:\n"
        "- 保留 agreed。\n"
        "- 对 disputed 做选择题:按规则选对的 type;不该标的删。\n"
        "- 补漏:规则明确要求但所有草稿都漏的 span 才补(verbatim)。\n"
        "- 每个 span 必须是该行 input 的 verbatim 子串。\n\n"
        "严格输出 JSON:{\"rows\":[{\"row_index\":int,\"output\":{\"entities\":{...},\"json_structures\":{...}}}]}\n\n"
        + _json.dumps(rows, ensure_ascii=False)
    )

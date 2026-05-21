"""Pure-function implementation of the check_past_experience MCP tool.

The MCP server wrapper (kb_server.py, Task 7) is thin — it forwards the
JSON-RPC arguments here and serializes the returned dict back over
stdio. Keeping the logic separate makes it unit-testable without
launching a subprocess.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.similarity.diverse import select_diverse_examples
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.text.wordfreq_utils import wordfreq_score


_MAX_EXAMPLES_PER_TYPE = 3
_GENERIC_WORD_ZIPF = 5.0
_GENERIC_WORD_MIN_EVIDENCE = 5


def check_past_experience(
    store: SqliteStore,
    *,
    project_id: str,
    entry: str,
) -> dict[str, Any]:
    """Return past-annotation evidence for a candidate span.

    Output shape — see
    docs/superpowers/specs/2026-05-19-annotation-knowledge-base-design.md
    section "Tool Contract" for field semantics.
    """
    if not entry or not entry.strip():
        raise ValueError("entry is required")

    span_lower = entry.strip().lower()
    row = store._conn.execute(
        "SELECT convention_id, entity_type, status, evidence_count, proposals_json "
        "FROM entity_conventions WHERE project_id=? AND span_lower=?",
        (project_id, span_lower),
    ).fetchone()

    zipf = wordfreq_score(entry)

    if row is None:
        return {
            "entry": entry,
            "convention": {"status": "none", "type": None, "evidence_count": 0},
            "distribution": {},
            "examples_by_type": {},
            "meta": {
                "wordfreq_zipf": round(zipf, 3),
                "generic_word": zipf >= _GENERIC_WORD_ZIPF,
            },
        }

    proposals = json.loads(row["proposals_json"] or "[]")

    # Distribution counts every proposal by its declared type.
    distribution = Counter(
        p["type"] for p in proposals
        if isinstance(p, dict) and isinstance(p.get("type"), str)
    )

    # Group context snippets by type, formatted with trace prefix.
    snippets_by_type: dict[str, list[str]] = {}
    for p in proposals:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type")
        snippet = p.get("context_snippet")
        if not (isinstance(ptype, str) and isinstance(snippet, str) and snippet.strip()):
            continue
        task_id = p.get("task_id") or "?"
        row_id = p.get("row_id") or "?"
        formatted = f"[{task_id}/{row_id}] {snippet}"
        snippets_by_type.setdefault(ptype, []).append(formatted)

    examples_by_type = {
        ptype: select_diverse_examples(snippets, k=_MAX_EXAMPLES_PER_TYPE)
        for ptype, snippets in snippets_by_type.items()
    }

    evidence_count = row["evidence_count"]
    return {
        "entry": entry,
        "convention": {
            "status": row["status"],
            "type": row["entity_type"],
            "evidence_count": evidence_count,
        },
        "distribution": dict(distribution),
        "examples_by_type": examples_by_type,
        "meta": {
            "wordfreq_zipf": round(zipf, 3),
            "generic_word": (
                zipf >= _GENERIC_WORD_ZIPF and evidence_count < _GENERIC_WORD_MIN_EVIDENCE
            ),
        },
    }

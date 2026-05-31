"""Per-project entity convention store.

Accumulates "lesson learned" entity-type decisions from QC consensus,
arbiter rulings, HR feedback, and operator declarations. Each subsequent
task gets the matching conventions injected into its annotator/QC/arbiter
prompts so ambiguous spans (Gmail = project, Apple = organization, etc.)
get consistent classification.

Case-insensitive matching, original-case storage. Soft dispute model:
automated (qc_consensus) conflicts on the same span do NOT mark the
convention 'disputed'; instead the plurality winner across distinct tasks
becomes the current type and disagreement is tracked numerically as
dispute_pct (enforced softly at injection time). Only an operator can set
'disputed' status, and only an operator declaration / clear_dispute can
change an operator-locked type.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from annotation_pipeline_skill.store.sqlite_store import SqliteStore

_SNIPPET_WINDOW = 80  # chars before and after the span hit


def _build_context_snippet(span: str, row_content: str | None) -> str | None:
    """Build a ~200-char window around the first case-insensitive
    occurrence of ``span`` in ``row_content``. Returns ``None`` only when
    ``row_content`` is falsy. When the span is not found (e.g. normalization
    mismatch), returns the first ~160 chars of ``row_content`` as a
    fallback so the row is still surfaced as evidence.
    """
    if not row_content:
        return None
    hit = row_content.lower().find(span.lower())
    if hit < 0:
        # Span not present in row_content (e.g., normalization mismatch);
        # still surface the row as evidence by returning a head window.
        head = row_content[: _SNIPPET_WINDOW * 2].strip()
        suffix = "…" if len(row_content) > _SNIPPET_WINDOW * 2 else ""
        return f"{head}{suffix}"
    start = max(0, hit - _SNIPPET_WINDOW)
    end = min(len(row_content), hit + len(span) + _SNIPPET_WINDOW)
    snippet = row_content[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(row_content) else ""
    return f"{prefix}{snippet}{suffix}"


# Source prefixes that indicate a human override (operator / HR), NOT a
# three-party LLM consensus. These are EXCLUDED from the distinct-task tally
# (they don't count as "三方一致" consensus votes) and instead take effect via
# the operator-declared injection bypass.
_OPERATOR_DECLARATION_SOURCE_PREFIXES: tuple[str, ...] = (
    "declared:", "hr_correction:", "posterior_audit_operator",
    "batch_operator_resolve", "batch_operator_correction",
    "dispute_resolved_by:",
)


def _is_operator_source(source: Any) -> bool:
    return isinstance(source, str) and any(
        source.startswith(p) for p in _OPERATOR_DECLARATION_SOURCE_PREFIXES
    )


def _distinct_task_tally(
    proposals: list[dict[str, Any]],
) -> tuple[str | None, int, int, float]:
    """Aggregate three-party-consensus proposals into a one-vote-per-task tally.

    A "distinct task" is a unique ``task_id`` whose proposal came from the
    three-party consensus path (``source="qc_consensus"``: annotator + QC +
    prior verifier agree). Each such task contributes a SINGLE vote; that
    task's vote is the type of its MOST RECENT consensus proposal (later
    proposals overwrite earlier ones, so a task that changed its mind votes
    for its final answer).

    EXCLUDED from the tally:
      - proposals with no ``task_id`` (operator declarations, dispute
        resolutions), and
      - proposals from operator/HR sources (see
        ``_OPERATOR_DECLARATION_SOURCE_PREFIXES``) — those are human
        overrides, not three-party consensus, and take effect via the
        operator-declared injection bypass instead.

    Returns ``(dominant_type, distinct_task_count, dispute_count,
    dispute_pct)`` where:
      - ``dominant_type`` is the plurality winner across task votes (ties
        broken deterministically by the larger type string), or ``None``
        when there are no consensus task votes.
      - ``dispute_count`` is the number of consensus tasks whose vote !=
        dominant.
      - ``dispute_pct`` is ``dispute_count / distinct_task_count`` (0.0 when
        there are no consensus task votes).
    """
    votes: dict[str, str] = {}  # task_id -> most-recent consensus type
    for p in proposals:
        if not isinstance(p, dict):
            continue
        task_id = p.get("task_id")
        ptype = p.get("type")
        if not task_id or not isinstance(ptype, str):
            continue
        if _is_operator_source(p.get("source")):
            continue  # human override, not a three-party consensus vote
        votes[task_id] = ptype  # later proposals overwrite → most-recent wins
    distinct = len(votes)
    if distinct == 0:
        return (None, 0, 0, 0.0)
    tally = Counter(votes.values())
    # Plurality; deterministic tiebreak on (count, type) so equal counts
    # resolve consistently regardless of insertion order.
    dominant_type = max(tally.items(), key=lambda kv: (kv[1], kv[0]))[0]
    dispute_count = distinct - tally[dominant_type]
    return (dominant_type, distinct, dispute_count, dispute_count / distinct)


@dataclass(frozen=True)
class EntityConvention:
    convention_id: str
    project_id: str
    span_lower: str
    span_original: str
    entity_type: str | None
    status: str   # 'active' or 'disputed'
    evidence_count: int
    proposals: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    created_by: str
    notes: str | None = None
    # Derived from ``proposals`` (one vote per distinct task). Not stored as
    # columns — computed in ``_load_row`` so ``proposals_json`` stays the
    # single source of truth.
    distinct_task_count: int = 0
    dispute_count: int = 0
    dispute_pct: float = 0.0
    dominant_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "convention_id": self.convention_id,
            "project_id": self.project_id,
            "span": self.span_original,
            "entity_type": self.entity_type,
            "status": self.status,
            "evidence_count": self.evidence_count,
            "proposals": self.proposals,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "notes": self.notes,
            "distinct_task_count": self.distinct_task_count,
            "dispute_count": self.dispute_count,
            "dispute_pct": self.dispute_pct,
            "dominant_type": self.dominant_type,
        }


class EntityConventionService:
    def __init__(self, store: SqliteStore):
        self.store = store

    def record_decision(
        self,
        *,
        project_id: str,
        span: str,
        entity_type: str,
        source: str,
        task_id: str | None = None,
        row_id: str | None = None,
        row_content: str | None = None,
        notes: str | None = None,
    ) -> EntityConvention:
        """Upsert a convention. Rules:
        - first time → insert as 'active'
        - automated proposals (qc_consensus etc.) → append a proposal and set
          entity_type to the plurality winner across distinct tasks. Conflicts
          do NOT flip the convention to 'disputed' (soft model); disagreement
          is tracked numerically (dispute_pct) and enforced at injection time.
        - explicit operator declaration ("declared:...") → wins, locks the
          chosen type, clears any prior dispute.
        - already operator-'disputed' → append the proposal but leave the
          status alone until an operator clears it.
        """
        if not span or not entity_type:
            raise ValueError("span and entity_type are required")
        span_lower = span.strip().lower()
        now = datetime.now(timezone.utc)
        proposal = {
            "type": entity_type,
            "source": source,
            "task_id": task_id,
            "row_id": row_id,
            "context_snippet": _build_context_snippet(span, row_content),
            "notes": notes,
            "at": now.isoformat(),
        }
        conn = self.store._conn
        row = conn.execute(
            "SELECT * FROM entity_conventions WHERE project_id=? AND span_lower=?",
            (project_id, span_lower),
        ).fetchone()
        if row is None:
            conv_id = f"conv-{uuid4().hex[:16]}"
            conn.execute(
                """
                INSERT INTO entity_conventions
                (convention_id, project_id, span_lower, span_original, entity_type,
                 status, evidence_count, proposals_json, created_at, updated_at,
                 created_by, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conv_id, project_id, span_lower, span.strip(), entity_type,
                    "active", 1, json.dumps([proposal]),
                    now.isoformat(), now.isoformat(), source, notes,
                ),
            )
            return self._load_row(conn.execute(
                "SELECT * FROM entity_conventions WHERE convention_id=?", (conv_id,)
            ).fetchone())

        proposals = json.loads(row["proposals_json"] or "[]")
        # Idempotent re-clicks: if the most recent proposal came from the same
        # source with the same type, treat the new call as a no-op. Without
        # this, an operator double-clicking the same button (or refreshing
        # the page and re-clicking) inflates evidence_count and the
        # button-tally shown in the UI. Genuine independent corroborations
        # (annotator + qc + arbiter or a fresh operator session) still
        # increment because their source strings differ or there's an
        # intervening proposal.
        last = proposals[-1] if proposals else None
        if (
            isinstance(last, dict)
            and last.get("type") == entity_type
            and last.get("source") == source
            and last.get("task_id") == task_id
        ):
            return self._load_row(row)
        proposals.append(proposal)
        dominant_type, _distinct, _dispute, _pct = _distinct_task_tally(proposals)
        # evidence_count is now a plain display counter: the total number of
        # recorded proposals. The injection gate uses distinct_task_count /
        # dispute_pct (derived in _load_row), NOT this field.
        new_count = len(proposals)
        if source.startswith("declared:"):
            # Explicit operator declaration is the final authority: it always
            # wins, clears any prior dispute, and locks in the chosen type.
            new_status = "active"
            new_type = entity_type
        elif row["status"] == "disputed":
            # A convention disputed by an operator stays disputed until the
            # operator clears it; automated proposals only append evidence.
            new_status = "disputed"
            new_type = row["entity_type"]
        elif any(
            isinstance(p, dict) and _is_operator_source(p.get("source"))
            for p in proposals
        ):
            # An operator has declared a policy for this span at some point in
            # the chain. The operator's call is the final authority and is NOT
            # silently overridden by later auto consensus — keep the locked
            # type (consistent with the operator-declared injection bypass).
            # Only another operator action (declared:/clear_dispute) can change
            # it. The proposal is still appended as evidence.
            new_status = "active"
            new_type = row["entity_type"]
        else:
            # Soft model: automated conflicts NEVER hard-flip to 'disputed'.
            # The plurality winner (one vote per distinct task) becomes the
            # convention's current type; disagreement is tracked numerically
            # via dispute_pct and enforced softly at injection time.
            new_status = "active"
            new_type = dominant_type if dominant_type is not None else row["entity_type"]
        conn.execute(
            """
            UPDATE entity_conventions
            SET entity_type=?, status=?, evidence_count=?, proposals_json=?, updated_at=?
            WHERE convention_id=?
            """,
            (new_type, new_status, new_count, json.dumps(proposals),
             now.isoformat(), row["convention_id"]),
        )
        return self._load_row(conn.execute(
            "SELECT * FROM entity_conventions WHERE convention_id=?",
            (row["convention_id"],),
        ).fetchone())

    def delete_for_span(self, *, project_id: str, span: str) -> bool:
        """Hard-delete a convention by (project_id, span). Used by the Manual
        Review UI when the operator clicks the already-selected button to
        cancel their choice. Returns True if a row was removed.
        """
        span_lower = span.strip().lower()
        cur = self.store._conn.execute(
            "DELETE FROM entity_conventions WHERE project_id=? AND span_lower=?",
            (project_id, span_lower),
        )
        return cur.rowcount > 0

    def clear_dispute(
        self,
        *,
        convention_id: str,
        resolved_type: str,
        actor: str,
        notes: str | None = None,
    ) -> EntityConvention:
        """Operator resolves a disputed convention by picking a winning type."""
        now = datetime.now(timezone.utc)
        conn = self.store._conn
        row = conn.execute(
            "SELECT * FROM entity_conventions WHERE convention_id=?",
            (convention_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"convention {convention_id} not found")
        proposals = json.loads(row["proposals_json"] or "[]")
        proposals.append({
            "type": resolved_type,
            "source": f"dispute_resolved_by:{actor}",
            "notes": notes,
            "at": now.isoformat(),
        })
        conn.execute(
            """
            UPDATE entity_conventions
            SET entity_type=?, status='active', proposals_json=?, updated_at=?
            WHERE convention_id=?
            """,
            (resolved_type, json.dumps(proposals), now.isoformat(), convention_id),
        )
        return self._load_row(conn.execute(
            "SELECT * FROM entity_conventions WHERE convention_id=?",
            (convention_id,),
        ).fetchone())

    def list_for_project(
        self, project_id: str, *, include_disputed: bool = True
    ) -> list[EntityConvention]:
        conn = self.store._conn
        q = "SELECT * FROM entity_conventions WHERE project_id=?"
        params: tuple[Any, ...] = (project_id,)
        if not include_disputed:
            q += " AND status='active'"
        q += " ORDER BY evidence_count DESC, updated_at DESC"
        rows = conn.execute(q, params).fetchall()
        return [self._load_row(r) for r in rows]

    def rebuild_from_accepted_tasks(
        self,
        *,
        project_id: str,
        task_ids: Iterable[str],
        annotation_loader: Callable[[str], "dict | None"],
    ) -> dict[str, int]:
        """Rebuild this project's conventions from accepted task annotations.

        The live ``proposals_json`` is lossy — the old ``(type, source)`` dedup
        key suppressed cross-task votes, so the convention table cannot be
        re-derived from itself. The recoverable source of truth is each
        accepted task's final annotation: under the three-party (prelabel +
        QC + arbiter) consensus model, an accepted task is a confirmed
        datapoint for EVERY span it labels.

        This method:
          1. DELETEs all existing conventions for ``project_id``.
          2. Replays every (span, type) decision from each task's annotation
             as a ``qc_consensus`` proposal keyed by that task's id.

        Because the dedup key is ``(source, type, task_id)``, each task
        contributes at most one vote per (span, type), so
        ``distinct_task_count`` accumulates exactly one vote per task.

        ``annotation_loader`` maps a task_id to its parsed annotation payload
        (pass ``HumanReviewService._latest_annotation_payload``); it returns
        ``None`` for tasks with no loadable annotation, which are skipped.
        The whole rebuild runs in a single transaction so a crash leaves the
        prior table intact.

        Returns a summary dict with ``tasks_seen``, ``tasks_with_spans`` and
        ``decisions_recorded`` counts.
        """
        conn = self.store._conn
        tasks_seen = 0
        tasks_with_spans = 0
        decisions_recorded = 0
        with conn:
            conn.execute(
                "DELETE FROM entity_conventions WHERE project_id=?", (project_id,)
            )
            for tid in task_ids:
                tasks_seen += 1
                payload = annotation_loader(tid)
                if not isinstance(payload, dict):
                    continue
                pairs = extract_all_span_decisions_with_row(payload)
                if pairs:
                    tasks_with_spans += 1
                for span, entity_type, row_id, row_content in pairs:
                    self.record_decision(
                        project_id=project_id,
                        span=span,
                        entity_type=entity_type,
                        source="qc_consensus",
                        task_id=tid,
                        row_id=row_id,
                        row_content=row_content,
                    )
                    decisions_recorded += 1
        return {
            "tasks_seen": tasks_seen,
            "tasks_with_spans": tasks_with_spans,
            "decisions_recorded": decisions_recorded,
        }

    # Skip conventions whose span is shorter than this (digits, single
    # letters, "CA" etc). They substring-match almost any input and pollute
    # the prompt with noise like "'1' → entities.number".
    MIN_INJECTION_SPAN_LEN = 4
    # Injection gate (replaces the old evidence_count threshold). An
    # auto-accumulated convention is injected only once enough DISTINCT tasks
    # have voted for it AND the cross-task disagreement is low.
    # One task = one vote (see _distinct_task_tally). Operator-declared
    # conventions bypass both thresholds.
    INJECT_MIN_DISTINCT_TASKS = 5
    INJECT_MAX_DISPUTE_PCT = 0.20
    # Prefixes of ``created_by`` (or proposal source) that indicate an
    # explicit operator/HR declaration — these bypass the injection gate.
    OPERATOR_DECLARATION_SOURCE_PREFIXES: tuple[str, ...] = (
        _OPERATOR_DECLARATION_SOURCE_PREFIXES
    )
    # Entity types whose conventions we never inject — the catch-all type
    # is by design generic and shouldn't override the LLM's judgment.
    EXCLUDED_TYPES_FOR_INJECTION: tuple[str, ...] = ("entity",)

    @classmethod
    def _is_operator_declared(cls, conv: "EntityConvention") -> bool:
        """True if any proposal in the convention's history came from an
        operator/HR declaration. A single explicit operator declaration
        anywhere in the chain is enough to bypass the evidence threshold —
        the operator's call wins over later auto-accumulated qc_consensus
        votes for the same span."""
        if conv.created_by and any(
            conv.created_by.startswith(p) for p in cls.OPERATOR_DECLARATION_SOURCE_PREFIXES
        ):
            return True
        for prop in conv.proposals or ():
            if not isinstance(prop, dict):
                continue
            src = prop.get("source") or ""
            if isinstance(src, str) and any(
                src.startswith(p) for p in cls.OPERATOR_DECLARATION_SOURCE_PREFIXES
            ):
                return True
        return False

    def find_matches_in_text(
        self, project_id: str, text: str
    ) -> list[EntityConvention]:
        """Return active conventions whose span occurs as a word-boundary
        match in ``text`` (case-insensitive). Disputed conventions are
        excluded — the runtime should not inject contradictory guidance.

        Match rules:
          - Span must be at least ``MIN_INJECTION_SPAN_LEN`` characters
            long; shorter spans substring-match too liberally.
          - Span must appear at a word boundary (or as a complete token).
            For pure-ASCII spans we use ``\\b``; for spans containing CJK
            or other non-word characters we fall back to plain substring
            since ``\\b`` doesn't apply there.
          - The convention must have at least
            ``INJECT_MIN_DISTINCT_TASKS`` distinct tasks voting for it AND a
            cross-task ``dispute_pct < INJECT_MAX_DISPUTE_PCT`` — UNLESS it was
            operator-declared, in which case one declaration counts as policy.
        """
        if not text:
            return []
        text_lower = text.lower()
        out: list[EntityConvention] = []
        for conv in self.list_for_project(project_id, include_disputed=False):
            if not conv.span_lower:
                continue
            if len(conv.span_lower) < self.MIN_INJECTION_SPAN_LEN:
                continue
            if not self._is_operator_declared(conv):
                if conv.distinct_task_count < self.INJECT_MIN_DISTINCT_TASKS:
                    continue
                if conv.dispute_pct >= self.INJECT_MAX_DISPUTE_PCT:
                    continue
            if conv.entity_type in self.EXCLUDED_TYPES_FOR_INJECTION:
                continue
            if not _span_in_text_at_word_boundary(conv.span_lower, text_lower):
                continue
            out.append(conv)
        return out

    def _load_row(self, row: sqlite3.Row) -> EntityConvention:
        proposals = json.loads(row["proposals_json"] or "[]")
        dominant_type, distinct_tasks, dispute_count, dispute_pct = (
            _distinct_task_tally(proposals)
        )
        return EntityConvention(
            convention_id=row["convention_id"],
            project_id=row["project_id"],
            span_lower=row["span_lower"],
            span_original=row["span_original"],
            entity_type=row["entity_type"],
            status=row["status"],
            evidence_count=row["evidence_count"],
            proposals=proposals,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            created_by=row["created_by"],
            notes=row["notes"],
            distinct_task_count=distinct_tasks,
            dispute_count=dispute_count,
            dispute_pct=dispute_pct,
            dominant_type=dominant_type,
        )


def extract_all_span_decisions_with_row(
    payload: Any,
) -> list[tuple[str, str, str | None, str | None]]:
    """Return every (span, type, row_id, row_content) decision in an
    annotation payload, deduped per ``(span_lower, type)``.

    Unlike ``extract_entity_type_decisions_with_row`` (which diffs against a
    prior annotation and emits only spans whose type CHANGED), this emits ALL
    span/type decisions present in the annotation. It is the "full
    derivation" extractor used by ``rebuild_from_accepted_tasks``: under the
    three-party consensus model, an accepted task is a confirmed datapoint for
    every span it labels, not just the ones that differed from the prelabel.

    Walks BOTH ``entities`` and ``json_structures`` (the same union
    ``iter_span_decisions`` uses). For each ``(span_lower, type)`` pair the
    carried ``row_id``/``row_content`` is from the FIRST row where it appears,
    used to build a context snippet. ``row_content`` falls back from the
    row's ``"content"`` to its ``"text"`` field; ``None`` if neither is a
    string.
    """
    out: list[tuple[str, str, str | None, str | None]] = []
    if not isinstance(payload, dict):
        return out
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return out
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("row_id") if isinstance(row.get("row_id"), str) else None
        content = row.get("content") or row.get("text")
        if not isinstance(content, str):
            content = None
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        for field_key in ("entities", "json_structures"):
            field_val = output.get(field_key)
            if not isinstance(field_val, dict):
                continue
            for typ, items in field_val.items():
                if not isinstance(items, list):
                    continue
                for span in items:
                    if not (isinstance(span, str) and span.strip()):
                        continue
                    key = (span.strip().lower(), typ)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((span.strip(), typ, row_id, content))
    return out


def extract_entity_type_decisions(
    prior_annotation: Any,
    new_annotation: Any,
) -> list[tuple[str, str]]:
    """Walk both annotations and return (span, new_type) for every entity
    whose type differs between prior and new. Used to auto-collect
    conventions when HR submits a correction or arbiter applies a fix.

    Returns spans where:
      - new annotation has the span under type X
      - prior annotation either didn't have the span, or had it under type Y != X
    Json_structures collisions are NOT considered (phrases play multiple
    legitimate roles; type "fixes" there usually aren't meaningful).
    """
    def _index_entities(annotation: Any) -> dict[tuple[int, str], str]:
        # (row_index, span_lower) -> type
        index: dict[tuple[int, str], str] = {}
        if not isinstance(annotation, dict):
            return index
        rows = annotation.get("rows")
        if not isinstance(rows, list):
            return index
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_idx = row.get("row_index") if isinstance(row.get("row_index"), int) else 0
            output = row.get("output")
            if not isinstance(output, dict):
                continue
            entities = output.get("entities")
            if not isinstance(entities, dict):
                continue
            for typ, items in entities.items():
                if not isinstance(items, list):
                    continue
                for s in items:
                    if isinstance(s, str) and s.strip():
                        # First-seen wins per row+span (consistent with within-row dedupe)
                        index.setdefault((row_idx, s.strip().lower()), typ)
        return index

    prior_index = _index_entities(prior_annotation)
    new_index = _index_entities(new_annotation)
    decisions: list[tuple[str, str]] = []
    seen_spans: set[str] = set()
    # Walk new — for any (span, type) that wasn't in prior, or had a different
    # type in prior, record one decision (use the original case from the new
    # annotation by looking it up again).
    if isinstance(new_annotation, dict):
        rows = new_annotation.get("rows", [])
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            row_idx = row.get("row_index") if isinstance(row.get("row_index"), int) else 0
            entities = row.get("output", {}).get("entities") if isinstance(row.get("output"), dict) else None
            if not isinstance(entities, dict):
                continue
            for typ, items in entities.items():
                if not isinstance(items, list):
                    continue
                for s in items:
                    if not isinstance(s, str) or not s.strip():
                        continue
                    key = (row_idx, s.strip().lower())
                    prior_type = prior_index.get(key)
                    if prior_type == typ:
                        continue
                    span_key = s.strip().lower()
                    if span_key in seen_spans:
                        continue
                    seen_spans.add(span_key)
                    decisions.append((s.strip(), typ))
    return decisions


def extract_entity_type_decisions_with_row(
    prior_annotation: Any,
    new_annotation: Any,
    source_rows: list[dict] | None = None,
) -> list[tuple[str, str, str | None, str | None]]:
    """Like ``extract_entity_type_decisions`` but also returns ``row_id`` and
    ``row_content`` for each decision.

    Returns list of ``(span, new_type, row_id, row_content)`` tuples.
    ``row_id`` is taken from the annotation row that triggered the decision.
    ``row_content`` is looked up from ``source_rows`` by ``row_id`` (the
    ``"content"`` field falls back to ``"text"``); ``None`` if the row
    isn't in ``source_rows`` or ``source_rows`` is not provided.

    Diff semantic matches ``extract_entity_type_decisions``: a decision is
    emitted only when ``new`` has the span under type X and ``prior`` either
    didn't have the span at the same row_index, or had it under type Y != X.
    Within a single batch, the first occurrence of a span wins (matching
    the existing dedupe behavior).
    """
    # Reuse the same prior-index logic as extract_entity_type_decisions
    # (the diff semantics are identical; we just also carry row data).
    def _index_entities(annotation: Any) -> dict[tuple[int, str], str]:
        index: dict[tuple[int, str], str] = {}
        if not isinstance(annotation, dict):
            return index
        rows = annotation.get("rows")
        if not isinstance(rows, list):
            return index
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_idx = row.get("row_index") if isinstance(row.get("row_index"), int) else 0
            output = row.get("output")
            if not isinstance(output, dict):
                continue
            entities = output.get("entities")
            if not isinstance(entities, dict):
                continue
            for typ, items in entities.items():
                if not isinstance(items, list):
                    continue
                for s in items:
                    if isinstance(s, str) and s.strip():
                        index.setdefault((row_idx, s.strip().lower()), typ)
        return index

    prior_index = _index_entities(prior_annotation)

    # Build a row_id → content lookup from source_rows.
    content_by_row_id: dict[str, str] = {}
    if isinstance(source_rows, list):
        for sr in source_rows:
            if not isinstance(sr, dict):
                continue
            rid = sr.get("row_id")
            if not isinstance(rid, str):
                continue
            content = sr.get("content") or sr.get("text")
            if isinstance(content, str):
                content_by_row_id[rid] = content

    decisions: list[tuple[str, str, str | None, str | None]] = []
    seen_spans: set[str] = set()

    if not isinstance(new_annotation, dict):
        return decisions
    rows = new_annotation.get("rows", [])
    if not isinstance(rows, list):
        return decisions

    for row in rows:
        if not isinstance(row, dict):
            continue
        row_idx = row.get("row_index") if isinstance(row.get("row_index"), int) else 0
        row_id = row.get("row_id") if isinstance(row.get("row_id"), str) else None
        row_content = content_by_row_id.get(row_id) if row_id else None
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        entities = output.get("entities")
        if not isinstance(entities, dict):
            continue
        for typ, items in entities.items():
            if not isinstance(items, list):
                continue
            for s in items:
                if not isinstance(s, str) or not s.strip():
                    continue
                key = (row_idx, s.strip().lower())
                prior_type = prior_index.get(key)
                if prior_type == typ:
                    continue
                span_key = s.strip().lower()
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)
                decisions.append((s.strip(), typ, row_id, row_content))
    return decisions


def _span_in_text_at_word_boundary(span: str, text: str) -> bool:
    """Case-insensitive word-boundary match. Both args expected lowercase.

    For ASCII spans (e.g. "Gmail", "Mitul Mallik") we require ``\\b`` on both
    ends so "CA" doesn't match the "ca" inside "callable" or "decade". For
    spans containing CJK or other non-``\\w`` characters, ``\\b`` doesn't
    apply meaningfully — fall back to plain substring.
    """
    import re

    if not span or not text:
        return False
    # If the span has any non-ASCII letter / digit / underscore chars, just
    # use substring matching.
    if not all(ord(c) < 128 and (c.isalnum() or c in " .-_'") for c in span):
        return span in text
    pattern = r"(?<!\w)" + re.escape(span) + r"(?!\w)"
    return bool(re.search(pattern, text))

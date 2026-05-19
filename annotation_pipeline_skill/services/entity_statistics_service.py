"""Per-project span/type frequency table used as external verifier.

Distinct from ``entity_conventions`` (which holds the high-trust subset
of decisions injected into prompts). ``entity_statistics`` accumulates
ALL ACCEPTED decisions — annotator+QC, arbiter, HR — without filtering.
HR decisions count with extra weight because they are the only
ground-truth source.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from annotation_pipeline_skill.store.sqlite_store import SqliteStore


# Verifier tuning constants. Kept module-level so call sites can introspect
# them in tests and operators can override via project workflow.yaml later.
MIN_PRIOR_SAMPLES = 10
DOMINANCE_THRESHOLD = 0.80
HR_WEIGHT = 5
MIN_CONTESTED_SAMPLES = 10
MIN_RUNNER_UP_SHARE = 0.20


@dataclass(frozen=True)
class VerifierResult:
    """Outcome of one PriorVerifier.check() call.

    status:
      - 'agree'      — proposed_type matches the dominant prior, or no clear
                       dominant exists (prior insufficiently opinionated).
      - 'cold_start' — fewer than MIN_PRIOR_SAMPLES total observations.
      - 'divergent'  — clear dominant prior disagrees with proposed_type.
    """
    status: str
    span: str
    proposed_type: str
    dominant_type: str | None = None
    dominant_count: int = 0
    total: int = 0
    distribution: dict[str, int] | None = None


class EntityStatisticsService:
    def __init__(self, store: SqliteStore):
        self.store = store

    def increment(
        self,
        *,
        project_id: str,
        span: str,
        entity_type: str,
        weight: int = 1,
    ) -> None:
        """UPSERT count += weight on (project_id, span_lower, entity_type)."""
        if not span or not entity_type or weight <= 0:
            return
        span_lower = span.strip().lower()
        if not span_lower:
            return
        now = datetime.now(timezone.utc).isoformat()
        self.store._conn.execute(
            """
            INSERT INTO entity_statistics (project_id, span_lower, entity_type, count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(project_id, span_lower, entity_type) DO UPDATE SET
                count = count + excluded.count,
                updated_at = excluded.updated_at
            """,
            (project_id, span_lower, entity_type, weight, now),
        )

    def recount_span(self, *, project_id: str, span: str) -> dict[str, int]:
        """Rebuild entity_statistics for one span from the ground truth:
        the current annotation of every ACCEPTED task in the project
        that mentions the span.

        Why this exists: entity_statistics is a vote-accumulator (every
        ACCEPTED decision +1, every HR commit +5). Over a project's
        lifetime it accumulates inflated historical counts that don't
        match current task state — especially after bulk Apply-to-all
        sweeps. Recounting after a sweep gives an honest "how many
        ACCEPTED tasks currently tag this span as what" distribution,
        so contested-span classification reflects reality.

        Returns the new distribution (entity_type → count) so callers
        can preview the effect.
        """
        import json as _json
        import re as _re

        span_strip = span.strip()
        if not span_strip:
            return {}
        span_lower = span_strip.lower()
        # Same prefilter the retroactive-fix endpoint uses — narrows from
        # ~all tasks to ~few hundred candidates.
        span_lower_json = _json.dumps(span_lower, ensure_ascii=True)[1:-1]
        rows = self.store._conn.execute(
            "SELECT task_id FROM tasks "
            "WHERE pipeline_id=? AND status='accepted' "
            "AND (lower(source_ref_json) LIKE ? OR lower(source_ref_json) LIKE ?)",
            (project_id, f"%{span_lower}%", f"%{span_lower_json}%"),
        ).fetchall()

        # Reuse the same "prefer human_review_answer, fall back to
        # annotation_result" load semantics that build_posterior_audit
        # and find_typical_text_for_span use.
        def _load_latest(task_id: str) -> dict | None:
            arts = self.store.list_artifacts(task_id)
            hr = [a for a in arts if a.kind == "human_review_answer"]
            if hr:
                try:
                    outer = _json.loads((self.store.root / hr[-1].path).read_text(encoding="utf-8"))
                    return outer.get("answer") if isinstance(outer, dict) else None
                except (OSError, _json.JSONDecodeError):
                    return None
            anns = [a for a in arts if a.kind == "annotation_result"]
            if not anns:
                return None
            try:
                outer = _json.loads((self.store.root / anns[-1].path).read_text(encoding="utf-8"))
            except (OSError, _json.JSONDecodeError):
                return None
            text = outer.get("text") if isinstance(outer, dict) else None
            if not isinstance(text, str):
                return None
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
            try:
                return _json.loads(text)
            except (ValueError, _json.JSONDecodeError):
                return None

        new_counts: dict[str, int] = {}
        for row in rows:
            payload = _load_latest(row["task_id"])
            if not isinstance(payload, dict):
                continue
            # Find every (Instagram, type) occurrence in this task's
            # rows. iter_span_decisions already does the walk for us.
            seen_types: set[str] = set()
            for s, t in iter_span_decisions(payload):
                if s.lower() == span_lower:
                    seen_types.add(t)
            # Count this task once per distinct type it tags the span
            # as. Most tasks have only one (correct) type — multi-type
            # tasks are unusual and indicate per-row context variance.
            for t in seen_types:
                new_counts[t] = new_counts.get(t, 0) + 1

        # Replace entity_statistics rows for this (project, span)
        # atomically: DELETE then INSERT. Other spans untouched.
        now = datetime.now(timezone.utc).isoformat()
        with self.store._conn:
            self.store._conn.execute(
                "DELETE FROM entity_statistics WHERE project_id=? AND span_lower=?",
                (project_id, span_lower),
            )
            for entity_type, count in new_counts.items():
                self.store._conn.execute(
                    "INSERT INTO entity_statistics "
                    "(project_id, span_lower, entity_type, count, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (project_id, span_lower, entity_type, count, now),
                )
        return new_counts

    def distribution(self, *, project_id: str, span: str) -> dict[str, int]:
        """Return {entity_type: count} for the given span. Empty if unseen."""
        span_lower = span.strip().lower()
        if not span_lower:
            return {}
        rows = self.store._conn.execute(
            "SELECT entity_type, count FROM entity_statistics "
            "WHERE project_id = ? AND span_lower = ?",
            (project_id, span_lower),
        ).fetchall()
        return {r["entity_type"]: r["count"] for r in rows}

    def total(self, *, project_id: str, span: str) -> int:
        return sum(self.distribution(project_id=project_id, span=span).values())

    def check(
        self,
        *,
        project_id: str,
        span: str,
        proposed_type: str,
    ) -> VerifierResult:
        dist = self.distribution(project_id=project_id, span=span)
        total = sum(dist.values())
        if total < MIN_PRIOR_SAMPLES:
            return VerifierResult(
                status="cold_start",
                span=span,
                proposed_type=proposed_type,
                total=total,
                distribution=dist or None,
            )
        dominant_type = max(dist, key=dist.get)
        dominant_count = dist[dominant_type]
        if dominant_count / total < DOMINANCE_THRESHOLD:
            return VerifierResult(
                status="agree",
                span=span,
                proposed_type=proposed_type,
                dominant_type=dominant_type,
                dominant_count=dominant_count,
                total=total,
                distribution=dist,
            )
        if dominant_type == proposed_type:
            return VerifierResult(
                status="agree",
                span=span,
                proposed_type=proposed_type,
                dominant_type=dominant_type,
                dominant_count=dominant_count,
                total=total,
                distribution=dist,
            )
        return VerifierResult(
            status="divergent",
            span=span,
            proposed_type=proposed_type,
            dominant_type=dominant_type,
            dominant_count=dominant_count,
            total=total,
            distribution=dist,
        )

    def contested_spans(self, *, project_id: str) -> list[dict[str, Any]]:
        """Return spans where the prior distribution has no clear winner.

        Criteria (all required):
          - total >= MIN_CONTESTED_SAMPLES
          - no type >= DOMINANCE_THRESHOLD (would be "settled")
          - at least two types each >= MIN_RUNNER_UP_SHARE (genuine split)
        """
        rows = self.store._conn.execute(
            "SELECT span_lower, entity_type, count FROM entity_statistics "
            "WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        per_span: dict[str, dict[str, int]] = {}
        for r in rows:
            per_span.setdefault(r["span_lower"], {})[r["entity_type"]] = r["count"]
        out: list[dict[str, Any]] = []
        for span, dist in per_span.items():
            total = sum(dist.values())
            if total < MIN_CONTESTED_SAMPLES:
                continue
            shares = sorted(
                ((t, c / total) for t, c in dist.items()), key=lambda kv: kv[1], reverse=True
            )
            top_share = shares[0][1]
            if top_share >= DOMINANCE_THRESHOLD:
                continue
            second_share = shares[1][1] if len(shares) > 1 else 0.0
            if second_share < MIN_RUNNER_UP_SHARE:
                continue
            out.append({
                "span": span,
                "prior_total": total,
                "prior_distribution": dist,
                "top_share": round(top_share, 3),
                "runner_up_share": round(second_share, 3),
            })
        out.sort(key=lambda r: r["prior_total"], reverse=True)
        return out


def iter_span_decisions(payload: Any) -> "list[tuple[str, str]]":
    """Yield (span, type) pairs from an annotation payload.

    Walks BOTH ``rows[*].output.entities[type] = [span, ...]`` AND
    ``rows[*].output.json_structures[type] = [phrase, ...]``. The two
    fields exist for training-pipeline reasons (model can't carry all 20+
    labels at once, so they're split by surface length), but semantically
    they share the same goal — labeling spans with a type. Statistics,
    conventions, and divergent-annotation audit therefore operate on the
    UNION of (span, type) decisions regardless of source field.

    Per-task per-(span, type) deduplication: a phrase that appears in both
    ``entities.technology`` and ``json_structures.technology`` in the same
    task is counted ONCE (not twice). This matters because we don't want
    the same task to register two votes for the same decision via a
    cross-field violation.

    Skips non-string spans, empty spans, and non-conforming structures.
    """
    out: list[tuple[str, str]] = []
    if not isinstance(payload, dict):
        return out
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return out
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        for field_key in ("entities", "json_structures"):
            field = output.get(field_key)
            if not isinstance(field, dict):
                continue
            for typ, items in field.items():
                if not isinstance(items, list):
                    continue
                for span in items:
                    if isinstance(span, str) and span.strip():
                        key = (span, typ)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(key)
    return out

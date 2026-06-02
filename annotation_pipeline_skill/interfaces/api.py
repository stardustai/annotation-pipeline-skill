from __future__ import annotations

import argparse
import io
import json
import threading
import traceback
import uuid
import zipfile
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import yaml

from annotation_pipeline_skill.core.models import AuditEvent, FeedbackDiscussionEntry, Task, utc_now
from annotation_pipeline_skill.core.qc_policy import build_qc_policy, validate_qc_sample_options
from annotation_pipeline_skill.core.runtime import RuntimeConfig, RuntimeSnapshot
from annotation_pipeline_skill.core.schema_validation import SchemaValidationError, load_project_output_schema
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.core.transitions import InvalidTransition, transition_task
from annotation_pipeline_skill.services.coordinator_service import CoordinatorService
from annotation_pipeline_skill.services.feedback_service import build_feedback_consensus_summary
from annotation_pipeline_skill.services.dashboard_service import (
    build_dashboard_stats,
    build_kanban_snapshot,
    build_project_summaries,
)
from annotation_pipeline_skill.services.human_review_service import HumanReviewService
from annotation_pipeline_skill.services.outbox_dispatch_service import build_outbox_summary
from annotation_pipeline_skill.runtime.monitor import validate_runtime_snapshot
from annotation_pipeline_skill.runtime.snapshot import build_runtime_snapshot
from annotation_pipeline_skill.services.provider_config_service import build_provider_config_snapshot, save_provider_config
from annotation_pipeline_skill.services.readiness_service import build_readiness_report
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.llm.profiles import ProfileValidationError


_BACKGROUND_JOBS: dict[str, dict[str, Any]] = {}
_BACKGROUND_JOBS_LOCK = threading.Lock()
# Tracks "is there an in-flight long-running job of `kind` for this project"
# so duplicate POSTs don't pile up parallel scans. Key = (project_id, kind).
_BACKGROUND_INFLIGHT: dict[tuple[str, str], str] = {}


def _pid_alive(pid: int) -> bool:
    """Return True if the given PID is still running (Linux/macOS)."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _start_background_job(
    *,
    kind: str,
    project_id: str,
    target: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
) -> dict[str, Any]:
    """Spawn `target(*args, **kwargs)` in a daemon thread and return the
    job descriptor immediately. Caller hands the job_id back to the
    frontend, which polls GET /api/jobs/<id> for status.

    Idempotent on (project_id, kind): if an in-flight job of the same
    kind exists, returns it instead of spawning a duplicate. Lets the
    UI's "scan started" button safely no-op on double-click.

    Background jobs run on detached threads — they don't block the
    request handler. SQLite is WAL so reads remain available while
    writes happen.
    """
    inflight_key = (project_id, kind)
    with _BACKGROUND_JOBS_LOCK:
        existing_id = _BACKGROUND_INFLIGHT.get(inflight_key)
        if existing_id is not None:
            job = _BACKGROUND_JOBS.get(existing_id)
            if job is not None and job.get("status") == "running":
                return dict(job)
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "kind": kind,
            "project_id": project_id,
            "status": "running",
            "started_at": utc_now().isoformat(),
            "finished_at": None,
            "error": None,
            "result": None,
        }
        _BACKGROUND_JOBS[job_id] = job
        _BACKGROUND_INFLIGHT[inflight_key] = job_id

    def _run() -> None:
        try:
            result = target(*args, **(kwargs or {}))
            with _BACKGROUND_JOBS_LOCK:
                _BACKGROUND_JOBS[job_id]["status"] = "done"
                _BACKGROUND_JOBS[job_id]["finished_at"] = utc_now().isoformat()
                _BACKGROUND_JOBS[job_id]["result"] = result if isinstance(result, dict) else None
        except Exception as exc:  # noqa: BLE001
            with _BACKGROUND_JOBS_LOCK:
                _BACKGROUND_JOBS[job_id]["status"] = "error"
                _BACKGROUND_JOBS[job_id]["finished_at"] = utc_now().isoformat()
                _BACKGROUND_JOBS[job_id]["error"] = f"{type(exc).__name__}: {exc}"
                _BACKGROUND_JOBS[job_id]["traceback"] = traceback.format_exc()
        finally:
            with _BACKGROUND_JOBS_LOCK:
                # Clear the inflight slot whether success or failure;
                # next POST can spawn a fresh attempt.
                if _BACKGROUND_INFLIGHT.get(inflight_key) == job_id:
                    _BACKGROUND_INFLIGHT.pop(inflight_key, None)

    threading.Thread(target=_run, name=f"bg-{kind}-{job_id}", daemon=True).start()
    return dict(job)


def _read_background_job(job_id: str) -> dict[str, Any] | None:
    with _BACKGROUND_JOBS_LOCK:
        job = _BACKGROUND_JOBS.get(job_id)
        return dict(job) if job is not None else None


def build_posterior_audit(store, *, project_id: str) -> dict:
    """Recount entity_statistics for the project from the current annotation
    of every ACCEPTED task (distinct-task semantics), then compare each
    task's (span, type) decisions to the freshly-recounted stats. Return
    task-level deviations and project-level contested spans.

    SIDE EFFECT: this persists a full rebuild of entity_statistics for the
    project (DELETE + INSERT via recount_project) — it is NOT a pure read.
    Safe because the sole caller is the manual/background "Re-check" job.
    """
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
        iter_span_decisions,
    )
    from annotation_pipeline_skill.runtime.subagent_cycle import _parse_llm_json
    import json as _json
    import re

    def _load_annotation(task):
        arts = store.list_artifacts(task.task_id)
        hr = [a for a in arts if a.kind == "human_review_answer"]
        if hr:
            outer = _json.loads((store.root / hr[-1].path).read_text(encoding="utf-8"))
            return outer.get("answer") if isinstance(outer, dict) else None
        anns = [a for a in arts if a.kind == "annotation_result"]
        if not anns:
            return None
        outer = _json.loads((store.root / anns[-1].path).read_text(encoding="utf-8"))
        text = outer.get("text")
        if not isinstance(text, str):
            return None
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        try:
            return _parse_llm_json(text)
        except (ValueError, _json.JSONDecodeError):
            return None

    svc = EntityStatisticsService(store)
    # Re-check == re-count: rebuild entity_statistics from the current
    # annotation of every accepted task (distinct-task semantics) BEFORE
    # auditing. This replaces the old lifetime vote-accumulator reading,
    # so divergent/low-info/deviation flags reflect current reality rather
    # than inflated historical counts.
    # NOTE: the deviations loop below re-reads accepted-task artifacts that
    # recount_project just scanned; the double-scan is an accepted YAGNI
    # tradeoff for a manual/background op (not worth folding into one pass).
    svc.recount_project(project_id=project_id)

    # Conventions are recount-only too: rebuild each convention's empirical
    # fields from the same current accepted annotations BEFORE we read the
    # convention_index below, so a frozen-wrong convention (e.g. vue->project
    # while every accepted task now tags it technology) is corrected and the
    # audit reflects current policy. Operator/HR locks are preserved inside
    # recount_project. SIDE EFFECT: persists column updates to entity_conventions.
    from annotation_pipeline_skill.services.entity_convention_service import (
        EntityConventionService,
    )
    EntityConventionService(store).recount_project(project_id=project_id)

    # Build the operator-declared convention index. Active conventions
    # (i.e., the operator has made an explicit policy call for this span)
    # override the empirical prior — they should suppress deviation /
    # contested flagging so the operator doesn't keep seeing the same
    # span show up after they've already adjudicated it. Disputed
    # conventions don't suppress (the policy itself is in conflict).
    from annotation_pipeline_skill.services.entity_convention_service import (
        EntityConventionService,
    )
    convention_index: dict[str, str] = {}
    for c in EntityConventionService(store).list_for_project(project_id, include_disputed=False):
        if c.entity_type:  # disputed conventions have entity_type=None — skip
            convention_index[c.span_lower] = c.entity_type

    deviations = []
    for task in store.list_tasks_by_pipeline(project_id):
        if task.status is not TaskStatus.ACCEPTED:
            continue
        payload = _load_annotation(task)
        if payload is None:
            continue
        for span, entity_type in iter_span_decisions(payload):
            # Operator-declared convention takes precedence over prior.
            conv_type = convention_index.get(span.lower())
            if conv_type is not None:
                # If convention matches the task's type → not divergent.
                # If convention is "not_an_entity", the task SHOULD have
                # dropped this span; flagging is still useful but framed
                # differently — keep the deviation visible so operator can
                # apply the fix.
                if conv_type == entity_type:
                    continue
                if conv_type == "not_an_entity":
                    # Operator declared this span shouldn't be tagged at
                    # all but the task still has it — keep flagging until
                    # the operator submits the delete fix.
                    pass
                # else: convention says some other type but task disagrees;
                # this is a real divergence vs the operator's policy. Fall
                # through and emit a deviation, but using conv_type as the
                # target instead of the empirical dominant.
            r = svc.check(project_id=project_id, span=span, proposed_type=entity_type)
            if r.status != "divergent":
                continue
            deviations.append({
                "task_id": task.task_id,
                "row_index": 0,  # iter_span_decisions doesn't currently yield row_index;
                                 # UI can still show "task-level" without it.
                "span": r.span,
                "current_type": r.proposed_type,
                "prior_dominant_type": conv_type or r.dominant_type,
                "prior_distribution": r.distribution,
                "prior_total": r.total,
            })

    # Annotate (don't filter) contested spans with an explicit
    # operator-declared convention. The contested classification is
    # driven by entity_statistics — after Apply-to-all + recount_span,
    # the distribution naturally collapses to a single type and the row
    # falls out of contested on its own (no filter needed). Keeping the
    # `resolved_convention_type` field for the UI's "✓ set" badge when
    # the row IS still present for some other reason (stats not yet
    # recounted, or partial Apply).
    divergent_all = svc.divergent_entries(project_id=project_id)
    divergent_entries = []
    for c in divergent_all:
        conv_type = convention_index.get(c.get("span", "").lower())
        entry = {**c, "type_entropy": _type_entropy(c.get("prior_distribution") or {})}
        if conv_type is not None:
            entry = {**entry, "resolved_convention_type": conv_type}
        divergent_entries.append(entry)

    # low_info_entries: ALL spans in entity_statistics with high wordfreq,
    # regardless of divergent/settled status or convention.
    LOW_INFO_THRESHOLD = 4.0
    all_stat_rows = store._conn.execute(
        "SELECT span_lower, entity_type, count FROM entity_statistics WHERE project_id = ?",
        (project_id,),
    ).fetchall()
    all_per_span: dict[str, dict[str, int]] = {}
    for r in all_stat_rows:
        all_per_span.setdefault(r["span_lower"], {})[r["entity_type"]] = r["count"]
    low_info_entries = []
    for span, dist in all_per_span.items():
        wf = _wordfreq_score(span)
        if wf >= LOW_INFO_THRESHOLD:
            low_info_entries.append({
                "span": span,
                "prior_total": sum(dist.values()),
                "prior_distribution": dist,
                "wordfreq": round(wf, 3),
            })
    low_info_entries.sort(key=lambda r: r["wordfreq"], reverse=True)

    return {
        "task_deviations": deviations,
        "divergent_entries": divergent_entries,
        "low_info_entries": low_info_entries,
    }


def _type_entropy(dist: dict[str, int]) -> float:
    import math as _math
    total = sum(dist.values())
    if not total:
        return 0.0
    return -sum((c / total) * _math.log2(c / total) for c in dist.values() if c > 0)


def _wordfreq_score(span: str) -> float:
    from annotation_pipeline_skill.text.wordfreq_utils import wordfreq_score
    return wordfreq_score(span)


def find_typical_text_for_span(
    store,
    *,
    project_id: str,
    span: str,
    exclude_task_ids: list[str] | None = None,
    exclude_keys: list[str] | None = None,
    task_id_filter: str | None = None,
    source_only: bool = False,
) -> dict | None:
    """Return one random ACCEPTED task whose *annotation* tags this exact span
    as an entity or json_structures phrase. Returns the row's input.text so
    the UI can render context with the span highlighted.

    The earlier substring-in-input search was wrong for short spans — `"app"`
    would surface text containing "applications" / "applies" / "happen",
    none of which were annotated as the entity. By matching on the
    annotation output instead, we only show samples where the span was
    actually tagged.

    Strategy: cheap source_ref LIKE prefilter (the span must appear
    somewhere in the input text for it to ever be tagged) → load that
    task's annotation_result → look for the EXACT span string in any
    row's entities/json_structures lists → return the corresponding
    row's input.text.

    When ``source_only=True`` the annotation membership check is skipped:
    we return any row whose input text contains the span as a whole word
    (word-boundary match). Useful for Low-Info Entries where the span may
    have been stripped from annotations but still exists in source text.

    Result shape: ``{"task_id": str, "row_index": int, "text": str}`` or None.
    """
    import random
    import re as _re
    span_lower = span.lower()
    # source_ref_json is stored with ensure_ascii=True, so non-ASCII chars
    # appear as Unicode escapes (e.g. "毫秒" → "毫秒"). A LIKE
    # pattern with the raw char would miss them — match the JSON-escaped
    # form too. For ASCII-only spans both forms collapse to the same string.
    span_lower_json = json.dumps(span_lower, ensure_ascii=True)[1:-1]
    if task_id_filter:
        rows = store._conn.execute(
            "SELECT task_id, source_ref_json FROM tasks "
            "WHERE pipeline_id=? AND task_id=? "
            "AND (lower(source_ref_json) LIKE ? OR lower(source_ref_json) LIKE ?)",
            (project_id, task_id_filter,
             f"%{span_lower}%", f"%{span_lower_json}%"),
        ).fetchall()
    else:
        rows = store._conn.execute(
            "SELECT task_id, source_ref_json FROM tasks "
            "WHERE pipeline_id=? AND status='accepted' "
            "AND (lower(source_ref_json) LIKE ? OR lower(source_ref_json) LIKE ?)",
            (project_id, f"%{span_lower}%", f"%{span_lower_json}%"),
        ).fetchall()
    excluded = set(exclude_task_ids or [])
    excluded_keys = set(exclude_keys or [])
    candidates: list[dict] = []

    # Local import to keep module import cycle small.
    from annotation_pipeline_skill.runtime.subagent_cycle import _parse_llm_json

    def _load_annotation(task_id: str) -> dict | None:
        arts = store.list_artifacts(task_id)
        hr = [a for a in arts if a.kind == "human_review_answer"]
        if hr:
            try:
                outer = json.loads((store.root / hr[-1].path).read_text(encoding="utf-8"))
                return outer.get("answer") if isinstance(outer, dict) else None
            except (json.JSONDecodeError, OSError):
                return None
        anns = [a for a in arts if a.kind == "annotation_result"]
        if not anns:
            return None
        try:
            outer = json.loads((store.root / anns[-1].path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        text = outer.get("text") if isinstance(outer, dict) else None
        if not isinstance(text, str):
            return None
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
        try:
            return _parse_llm_json(text)
        except (ValueError, json.JSONDecodeError):
            return None

    def _row_has_span_in_annotation(ann_row: dict, span_lc: str) -> bool:
        """True if ann_row.output.entities[*] OR json_structures[*] contains
        the span (case-insensitive exact match)."""
        output = ann_row.get("output") if isinstance(ann_row, dict) else None
        if not isinstance(output, dict):
            return False
        for key in ("entities", "json_structures"):
            container = output.get(key)
            if not isinstance(container, dict):
                continue
            for _type, items in container.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, str) and item.lower() == span_lc:
                        return True
        return False

    # Compile word-boundary pattern for source_only mode once, outside loop.
    import re as _re_inner
    _span_word_re = _re_inner.compile(
        r"(?<![^\W\d_])" + _re_inner.escape(span) + r"(?![^\W\d_])",
        _re_inner.IGNORECASE,
    ) if source_only else None

    for r in rows:
        if r["task_id"] in excluded:
            continue
        try:
            sr = json.loads(r["source_ref_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        payload = sr.get("payload") if isinstance(sr, dict) else None
        if not isinstance(payload, dict):
            continue
        input_rows = payload.get("rows")
        if not isinstance(input_rows, list):
            continue

        if source_only:
            # Skip annotation loading entirely; just find rows where the span
            # appears as a whole word in the input text.
            for ir in input_rows:
                if not isinstance(ir, dict):
                    continue
                text = ir.get("input")
                if isinstance(text, dict):
                    text = text.get("text")
                if not isinstance(text, str):
                    continue
                if not _span_word_re.search(text):  # type: ignore[union-attr]
                    continue
                row_index = ir.get("row_index", 0) if isinstance(ir.get("row_index"), int) else 0
                key = f"{r['task_id']}:{row_index}"
                if key in excluded_keys:
                    continue
                candidates.append({
                    "task_id": r["task_id"],
                    "row_index": row_index,
                    "text": text,
                })
                if len(candidates) >= 200:
                    break
        else:
            ann = _load_annotation(r["task_id"])
            if not isinstance(ann, dict):
                continue
            ann_rows = ann.get("rows")
            if not isinstance(ann_rows, list):
                continue
            # Index annotation rows by row_index for join with input rows.
            ann_by_idx: dict[int, dict] = {}
            for i, ar in enumerate(ann_rows):
                if not isinstance(ar, dict):
                    continue
                idx = ar.get("row_index") if isinstance(ar.get("row_index"), int) else i
                ann_by_idx[idx] = ar
            for ir in input_rows:
                if not isinstance(ir, dict):
                    continue
                text = ir.get("input")
                if isinstance(text, dict):
                    text = text.get("text")
                if not isinstance(text, str):
                    continue
                row_index = ir.get("row_index", 0) if isinstance(ir.get("row_index"), int) else 0
                ann_row = ann_by_idx.get(row_index)
                if not ann_row or not _row_has_span_in_annotation(ann_row, span_lower):
                    continue
                key = f"{r['task_id']}:{row_index}"
                if key in excluded_keys:
                    continue
                candidates.append({
                    "task_id": r["task_id"],
                    "row_index": row_index,
                    "text": text,
                })
                if len(candidates) >= 200:
                    break
        if len(candidates) >= 200:
            break
    if not candidates:
        return None
    return random.choice(candidates)


def compute_accepted_hash(store, *, project_id: str) -> str:
    """SHA-256 over `(task_id, updated_at)` of every ACCEPTED task in the
    project, sorted by task_id. Cheap (no artifact reads) and changes
    whenever a task transitions in/out of ACCEPTED or its annotation gets
    updated (every transition bumps updated_at).
    """
    import hashlib
    rows = store._conn.execute(
        "SELECT task_id, updated_at FROM tasks "
        "WHERE pipeline_id=? AND status='accepted' "
        "ORDER BY task_id",
        (project_id,),
    ).fetchall()
    h = hashlib.sha256()
    for r in rows:
        h.update(r["task_id"].encode("utf-8"))
        h.update(b"\x1f")  # unit separator
        h.update(r["updated_at"].encode("utf-8"))
        h.update(b"\x1e")  # record separator
    return f"sha256:{h.hexdigest()[:16]}:n={len(rows)}"


def read_posterior_audit_cache(store, *, project_id: str) -> dict | None:
    row = store._conn.execute(
        "SELECT payload_json, accepted_hash, created_at "
        "FROM posterior_audit_cache WHERE project_id=?",
        (project_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "payload": json.loads(row["payload_json"]),
        "accepted_hash": row["accepted_hash"],
        "created_at": row["created_at"],
    }


def write_posterior_audit_cache(
    store,
    *,
    project_id: str,
    payload: dict,
    accepted_hash: str,
    created_at: str,
) -> None:
    store._conn.execute(
        "INSERT INTO posterior_audit_cache "
        "(project_id, payload_json, accepted_hash, created_at) "
        "VALUES (?,?,?,?) "
        "ON CONFLICT(project_id) DO UPDATE SET "
        "payload_json=excluded.payload_json, "
        "accepted_hash=excluded.accepted_hash, "
        "created_at=excluded.created_at",
        (project_id, json.dumps(payload, ensure_ascii=False), accepted_hash, created_at),
    )
    store._conn.commit()


def read_distribution_cache(
    store, *, project_id: str, profile_name: str,
) -> dict | None:
    row = store._conn.execute(
        "SELECT payload_json, content_hash, created_at "
        "FROM distribution_cache WHERE project_id=? AND profile_name=?",
        (project_id, profile_name),
    ).fetchone()
    if row is None:
        return None
    return {
        "payload": json.loads(row["payload_json"]),
        "content_hash": row["content_hash"],
        "created_at": row["created_at"],
    }


def write_distribution_cache(
    store, *, project_id: str, profile_name: str,
    payload: dict, content_hash: str, created_at: str,
) -> None:
    store._conn.execute(
        "INSERT INTO distribution_cache "
        "(project_id, profile_name, payload_json, content_hash, created_at) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT(project_id, profile_name) DO UPDATE SET "
        "payload_json=excluded.payload_json, "
        "content_hash=excluded.content_hash, "
        "created_at=excluded.created_at",
        (project_id, profile_name,
         json.dumps(payload, ensure_ascii=False),
         content_hash, created_at),
    )
    store._conn.commit()


def compute_distribution_content_hash(
    store, *, project_id: str, statuses: list[str] | None = None,
) -> str:
    """Fingerprint the input set the distribution scan operates on.

    Includes (task_id, status, source_ref_json sha256) for every task in
    the configured status filter, ordered by task_id. Status is in the
    hash because moving a task ACCEPTED→REJECTED changes the scatter
    even though the underlying text didn't.
    """
    import hashlib
    h = hashlib.sha256()
    if statuses:
        placeholders = ",".join("?" * len(statuses))
        sql = (
            "SELECT task_id, status, source_ref_json FROM tasks "
            f"WHERE pipeline_id=? AND status IN ({placeholders}) "
            "ORDER BY task_id"
        )
        params = [project_id, *statuses]
    else:
        sql = (
            "SELECT task_id, status, source_ref_json FROM tasks "
            "WHERE pipeline_id=? ORDER BY task_id"
        )
        params = [project_id]
    rows = store._conn.execute(sql, params).fetchall()
    for r in rows:
        text = r["source_ref_json"] or ""
        ref_h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        h.update(f"{r['task_id']}|{r['status']}|{ref_h}\n".encode("utf-8"))
    return f"sha256:{h.hexdigest()[:16]}:n={len(rows)}"


def read_row_dedup_cache(
    store, *, project_id: str, profile_name: str,
) -> dict | None:
    row = store._conn.execute(
        "SELECT payload_json, content_hash, created_at "
        "FROM row_dedup_cache WHERE project_id=? AND profile_name=?",
        (project_id, profile_name),
    ).fetchone()
    if row is None:
        return None
    return {
        "payload": json.loads(row["payload_json"]),
        "content_hash": row["content_hash"],
        "created_at": row["created_at"],
    }


def write_row_dedup_cache(
    store, *, project_id: str, profile_name: str,
    payload: dict, content_hash: str, created_at: str,
) -> None:
    store._conn.execute(
        "INSERT INTO row_dedup_cache "
        "(project_id, profile_name, payload_json, content_hash, created_at) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT(project_id, profile_name) DO UPDATE SET "
        "payload_json=excluded.payload_json, "
        "content_hash=excluded.content_hash, "
        "created_at=excluded.created_at",
        (project_id, profile_name,
         json.dumps(payload, ensure_ascii=False),
         content_hash, created_at),
    )
    store._conn.commit()


def compute_row_dedup_content_hash(
    store, *, project_id: str, statuses: list[str] | None = None,
) -> str:
    """Fingerprint the row-level input that RowDedupService operates on.

    Incorporates (task_id, status, source_ref_json sha256) for every task
    in the status filter, PLUS the set of active row_masks for those tasks.
    Row masks are included so that masking a row (or removing a mask) marks
    the cache stale — the cluster list changes even though no task text
    changed.
    """
    import hashlib as _hashlib
    h = _hashlib.sha256()
    if statuses:
        placeholders = ",".join("?" * len(statuses))
        sql = (
            "SELECT task_id, status, source_ref_json FROM tasks "
            f"WHERE pipeline_id=? AND status IN ({placeholders}) "
            "ORDER BY task_id"
        )
        params: list[object] = [project_id, *statuses]
    else:
        sql = (
            "SELECT task_id, status, source_ref_json FROM tasks "
            "WHERE pipeline_id=? ORDER BY task_id"
        )
        params = [project_id]
    rows = store._conn.execute(sql, params).fetchall()
    for r in rows:
        text = r["source_ref_json"] or ""
        ref_h = _hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        h.update(f"{r['task_id']}|{r['status']}|{ref_h}\n".encode("utf-8"))
    # Also include row masks so masking a row marks the cache stale.
    task_ids = [r["task_id"] for r in rows]
    if task_ids:
        id_placeholders = ",".join("?" * len(task_ids))
        mask_rows = store._conn.execute(
            f"SELECT task_id, row_index FROM row_masks "
            f"WHERE task_id IN ({id_placeholders}) "
            "ORDER BY task_id, row_index",
            task_ids,
        ).fetchall()
        for mr in mask_rows:
            h.update(f"mask:{mr['task_id']}:{mr['row_index']}\n".encode("utf-8"))
    return f"sha256:{h.hexdigest()[:16]}:n={len(rows)}"


def build_type_statistics(store, *, project_id: str) -> dict:
    """Walk every ACCEPTED task in the project and aggregate counts by
    entity type AND by json_structure phrase type. The "Statistics"
    dashboard view renders these distributions.

    Returns:
      {
        "entities": {type: {tasks: int, occurrences: int}, ...},
        "json_structures": {type: {tasks: int, phrases: int}, ...},
        "scanned_tasks": int,
        "skipped_tasks": int,
      }

    `occurrences` counts every (span, type) pair (a span tagged in N
    rows of one task counts N). `tasks` counts distinct tasks where
    that type appears at all. Similar for json_structures.phrases.
    """
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.runtime.subagent_cycle import _parse_llm_json
    import re as _re

    def _load_annotation(task):
        arts = store.list_artifacts(task.task_id)
        hr = [a for a in arts if a.kind == "human_review_answer"]
        if hr:
            try:
                outer = json.loads((store.root / hr[-1].path).read_text(encoding="utf-8"))
                return outer.get("answer") if isinstance(outer, dict) else None
            except (json.JSONDecodeError, OSError):
                return None
        anns = [a for a in arts if a.kind == "annotation_result"]
        # Walk in reverse: skip empty / unparseable artifacts (some failure
        # modes write text="" — see the AnnotationView fallback).
        for art in reversed(anns):
            try:
                outer = json.loads((store.root / art.path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            text = outer.get("text") if isinstance(outer, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            cleaned = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
            try:
                return _parse_llm_json(cleaned)
            except (ValueError, json.JSONDecodeError):
                continue
        return None

    entity_occurrences: dict[str, int] = {}
    entity_tasks: dict[str, int] = {}
    js_phrases: dict[str, int] = {}
    js_tasks: dict[str, int] = {}
    scanned = 0
    skipped = 0
    for task in store.list_tasks_by_pipeline(project_id):
        if task.status is not TaskStatus.ACCEPTED:
            continue
        payload = _load_annotation(task)
        if not isinstance(payload, dict):
            skipped += 1
            continue
        rows = payload.get("rows")
        if not isinstance(rows, list):
            skipped += 1
            continue
        task_entity_types: set[str] = set()
        task_js_types: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            output = row.get("output")
            if not isinstance(output, dict):
                continue
            entities = output.get("entities")
            if isinstance(entities, dict):
                for typ, items in entities.items():
                    if not isinstance(items, list):
                        continue
                    real_items = [s for s in items if isinstance(s, str) and s.strip()]
                    if not real_items:
                        continue
                    entity_occurrences[typ] = entity_occurrences.get(typ, 0) + len(real_items)
                    task_entity_types.add(typ)
            js = output.get("json_structures")
            if isinstance(js, dict):
                for typ, phrases in js.items():
                    if not isinstance(phrases, list):
                        continue
                    real_phrases = [p for p in phrases if isinstance(p, str) and p.strip()]
                    if not real_phrases:
                        continue
                    js_phrases[typ] = js_phrases.get(typ, 0) + len(real_phrases)
                    task_js_types.add(typ)
        for typ in task_entity_types:
            entity_tasks[typ] = entity_tasks.get(typ, 0) + 1
        for typ in task_js_types:
            js_tasks[typ] = js_tasks.get(typ, 0) + 1
        scanned += 1

    return {
        "entities": {
            typ: {"tasks": entity_tasks.get(typ, 0), "occurrences": entity_occurrences[typ]}
            for typ in sorted(entity_occurrences, key=lambda k: -entity_occurrences[k])
        },
        "json_structures": {
            typ: {"tasks": js_tasks.get(typ, 0), "phrases": js_phrases[typ]}
            for typ in sorted(js_phrases, key=lambda k: -js_phrases[k])
        },
        "scanned_tasks": scanned,
        "skipped_tasks": skipped,
    }


def read_type_statistics_cache(store, *, project_id: str) -> dict | None:
    row = store._conn.execute(
        "SELECT payload_json, created_at FROM type_statistics_cache WHERE project_id=?",
        (project_id,),
    ).fetchone()
    if row is None:
        return None
    return {"payload": json.loads(row["payload_json"]), "created_at": row["created_at"]}


def write_type_statistics_cache(
    store, *, project_id: str, payload: dict, created_at: str,
) -> None:
    store._conn.execute(
        "INSERT INTO type_statistics_cache (project_id, payload_json, created_at) "
        "VALUES (?,?,?) "
        "ON CONFLICT(project_id) DO UPDATE SET "
        "payload_json=excluded.payload_json, created_at=excluded.created_at",
        (project_id, json.dumps(payload, ensure_ascii=False), created_at),
    )
    store._conn.commit()


CONFIG_FILE_DEFINITIONS: dict[str, str] = {
    "annotators.yaml": "Annotation Agents",
    "workflow.yaml": "Workflow",
    "external_tasks.yaml": "External Task API",
    "callbacks.yaml": "Callbacks",
    # llm_profiles.yaml is workspace-global; edit via the Providers panel,
    # not the per-project Config panel.
    # annotation_rules are versioned documents in the DB — not a config file.
}


def _load_active_rules(store) -> tuple[str, str]:
    """Return (version_label, content) of the latest annotation_rules document version."""
    try:
        doc_rows = store._conn.execute(
            "SELECT document_id, metadata_json FROM documents"
        ).fetchall()
    except Exception:
        return "unknown", ""
    target_doc_id: str | None = None
    for r in doc_rows:
        try:
            meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
        except Exception:
            meta = {}
        if isinstance(meta, dict) and meta.get("role") == "annotation_rules":
            target_doc_id = r["document_id"]
            break
    if not target_doc_id:
        return "unknown", ""
    try:
        ver_row = store._conn.execute(
            "SELECT version, content_path FROM document_versions "
            "WHERE document_id = ? ORDER BY created_at DESC LIMIT 1",
            (target_doc_id,),
        ).fetchone()
    except Exception:
        return "unknown", ""
    if ver_row is None or not ver_row["content_path"]:
        return "unknown", ""
    content_file = store.root / ver_row["content_path"]
    if not content_file.exists():
        return ver_row["version"], ""
    return ver_row["version"], content_file.read_text(encoding="utf-8")


class DashboardApi:
    def __init__(
        self,
        store: SqliteStore,
        *,
        stores: dict[str, SqliteStore] | None = None,
        default_store_key: str | None = None,
        runtime_once: Callable[[], RuntimeSnapshot] | None = None,
        runtime_config: RuntimeConfig | None = None,
        workspace_root: Path | None = None,
    ):
        self.store = store
        self._stores = stores or {}
        self._default_store_key = default_store_key
        self.runtime_once = runtime_once
        self.runtime_config = runtime_config or RuntimeConfig()
        # workspace_root holds the dir that owns the shared llm_profiles.yaml.
        # When provided, it's used for both reads (with project-local fallback)
        # and writes (always to the workspace path). Defaults to the parent of
        # the store's project root (i.e. <store.root>/../.. → workspace).
        if workspace_root is not None:
            self.workspace_root: Path = Path(workspace_root)
        else:
            self.workspace_root = store.root.parent.parent

    def _resolve_store(self, query: dict[str, list[str]]) -> SqliteStore:
        key = query.get("store", [None])[0]
        if key and key in self._stores:
            return self._stores[key]
        if self._default_store_key and self._default_store_key in self._stores:
            return self._stores[self._default_store_key]
        return self.store

    def handle_get(self, path: str) -> tuple[int, dict[str, str], bytes]:
        parsed_path = urlparse(path)
        route = parsed_path.path
        query = parse_qs(parsed_path.query)
        store = self._resolve_store(query)
        project_id = query.get("project", [None])[0]
        stage_view = query.get("stage_view", ["internal"])[0]
        if route == "/api/health":
            return self._json_response(200, {"ok": True})
        if route == "/api/stores":
            return self._json_response(200, {
                "workspace_path": str(self.workspace_root),
                "stores": self._stores_list(),
            })
        if route == "/api/projects":
            return self._json_response(200, build_project_summaries(store))
        if route == "/api/kanban":
            return self._json_response(200, build_kanban_snapshot(store, project_id=project_id, stage_view=stage_view))
        if route == "/api/dashboard-stats":
            return self._json_response(200, build_dashboard_stats(store, project_id=project_id))
        if route == "/api/schema":
            schema = load_project_output_schema(store.root) if store else None
            return self._json_response(200, {"schema": schema})
        if route == "/api/guidelines":
            return self._json_response(200, self._guidelines_response(store))
        if route == "/api/config":
            return self._json_response(200, {"files": self._config_files(store)})
        if route == "/api/providers":
            return self._provider_config_response(store)
        if route == "/api/annotators":
            return self._annotators_response(store)
        if route == "/api/coordinator":
            return self._json_response(
                200,
                CoordinatorService(store, workspace_root=self.workspace_root).build_report(project_id=project_id),
            )
        if route == "/api/events":
            try:
                limit = max(1, min(500, int(query.get("limit", ["100"])[0])))
            except ValueError:
                limit = 100
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
            except ValueError:
                offset = 0
            events, total = store.list_events_paginated(
                pipeline_id=project_id, limit=limit, offset=offset,
            )
            return self._json_response(200, {
                "events": [event.to_dict() for event in events],
                "total": total,
                "limit": limit,
                "offset": offset,
            })
        if route == "/api/alerts":
            # Read tail of <store_root>/alerts.jsonl. Lines are JSON
            # objects written by SubagentRuntime._emit_provider_alert,
            # _emit_enum_coerce_alert, and LocalRuntimeScheduler.
            # _write_health_alert. Cheap O(N) read since the file caps
            # at a few hundred KB in practice (alerts are deduped /
            # rare); if it grows unbounded a future patch can rotate.
            try:
                limit = max(1, min(500, int(query.get("limit", ["100"])[0])))
            except ValueError:
                limit = 100
            alerts_path = store.root / "alerts.jsonl"
            entries: list[dict] = []
            if alerts_path.exists():
                try:
                    raw_lines = alerts_path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    raw_lines = []
                # Take the last `limit` lines, parse each.
                for line in raw_lines[-limit:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue
            # Reverse-chronological for the UI.
            entries.reverse()
            return self._json_response(200, {
                "alerts": entries,
                "total_lines": len(entries),
                "alerts_path": str(alerts_path.relative_to(store.root)),
            })
        if route == "/api/readiness":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            return self._json_response(200, build_readiness_report(store, project_id))
        if route == "/api/export-file":
            rel_path = query.get("path", [None])[0]
            if not rel_path:
                return self._json_response(400, {"error": "path_required"})
            candidate = (store.root / rel_path).resolve()
            if not str(candidate).startswith(str(store.root.resolve())):
                return self._json_response(403, {"error": "forbidden"})
            if not candidate.exists():
                return self._json_response(404, {"error": "not_found"})
            filename = candidate.name
            return (200, {"content-type": "application/octet-stream", "content-disposition": f'attachment; filename="{filename}"'}, candidate.read_bytes())
        if route == "/api/export-zip":
            export_id = query.get("export_id", [None])[0]
            if not export_id:
                return self._json_response(400, {"error": "export_id_required"})
            export_dir = (store.root / "exports" / export_id).resolve()
            if not str(export_dir).startswith(str(store.root.resolve())):
                return self._json_response(403, {"error": "forbidden"})
            if not export_dir.exists():
                return self._json_response(404, {"error": "not_found"})
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(export_dir.iterdir()):
                    if f.is_file():
                        zf.write(f, arcname=f"{export_id}/{f.name}")
            zip_bytes = buf.getvalue()
            return (200, {
                "content-type": "application/zip",
                "content-disposition": f'attachment; filename="{export_id}.zip"',
                "content-length": str(len(zip_bytes)),
            }, zip_bytes)
        if route == "/api/outbox":
            return self._json_response(200, build_outbox_summary(store, project_id=project_id))
        if route == "/api/conventions":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            from annotation_pipeline_skill.services.entity_convention_service import (
                EntityConventionService,
            )
            # full=1 restores the legacy whole-project load (parses
            # proposals_json, ships every row). The TaskDrawer needs the
            # proposals audit trail and looks up conventions for the current
            # task's spans, so it opts into this heavier path. The dashboard
            # table (default) must NOT use it.
            if query.get("full", [""])[0] in ("1", "true"):
                convs = EntityConventionService(store).list_for_project(project_id)
                return self._json_response(200, {
                    "conventions": [c.to_dict() for c in convs],
                })
            # Server-side pagination + filtering: a rebuilt project holds tens
            # of thousands of conventions, so returning them all (with the full
            # proposals audit trail) is a multi-second, ~45MB response. The
            # table only ever shows one page and never the proposals, so push
            # limit/offset/min_count/search into SQL and read materialized
            # columns only (no proposals_json parse).
            try:
                limit = max(1, min(500, int(query.get("limit", ["100"])[0])))
            except ValueError:
                limit = 100
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
            except ValueError:
                offset = 0
            try:
                min_count = max(0, int(query.get("min_count", ["0"])[0]))
            except ValueError:
                min_count = 0
            search = query.get("q", [None])[0]
            convs, total, max_count = EntityConventionService(store).list_for_project_page(
                project_id,
                limit=limit,
                offset=offset,
                min_count=min_count,
                search=search,
            )
            return self._json_response(200, {
                "conventions": [c.to_dict() for c in convs],
                "total": total,
                "limit": limit,
                "offset": offset,
                "max_count": max_count,
            })
        if route == "/api/knowledge-summary":
            # Cheap change-signal for the Entity Knowledge panel. The panel
            # loads a paginated snapshot once and does NOT auto-poll (re-running
            # the heavy paginated queries every few seconds would undo the
            # pagination win and yank the table out from under an operator who's
            # auditing it). Instead the panel polls this endpoint and, when the
            # fingerprint moves, shows a "click Refresh" badge without touching
            # the table. The fingerprint is (count, MAX(updated_at)) per subtab:
            # any insert moves count, any in-place update moves the timestamp.
            # Both queries are covered by (project_id, span_lower) indexes.
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            conn = store._conn
            conv = conn.execute(
                "SELECT COUNT(*) AS n, MAX(updated_at) AS latest "
                "FROM entity_conventions WHERE project_id = ?",
                [project_id],
            ).fetchone()
            stats = conn.execute(
                "SELECT COUNT(DISTINCT span_lower) AS n, MAX(updated_at) AS latest "
                "FROM entity_statistics WHERE project_id = ?",
                [project_id],
            ).fetchone()
            return self._json_response(200, {
                "conventions": {"count": conv["n"] or 0, "latest_updated_at": conv["latest"]},
                "statistics": {"count": stats["n"] or 0, "latest_updated_at": stats["latest"]},
            })
        if route == "/api/posterior-audit":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            # GET returns the cached scan result + a staleness flag derived
            # from comparing the accepted_hash captured at cache time to the
            # current hash. Running a fresh scan requires POST (cheap GET so
            # auto-load on page mount doesn't pay the scan cost).
            cached = read_posterior_audit_cache(store, project_id=project_id)
            current_hash = compute_accepted_hash(store, project_id=project_id)
            if cached is None:
                return self._json_response(200, {
                    "cached": False,
                    "payload": None,
                    "generated_at": None,
                    "cached_accepted_hash": None,
                    "current_accepted_hash": current_hash,
                    "stale": False,
                })
            return self._json_response(200, {
                "cached": True,
                "payload": cached["payload"],
                "generated_at": cached["created_at"],
                "cached_accepted_hash": cached["accepted_hash"],
                "current_accepted_hash": current_hash,
                "stale": cached["accepted_hash"] != current_hash,
            })
        if route == "/api/distribution":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            profile_name = query.get("profile", ["jina_small"])[0]
            profiles_path = self.workspace_root / "similarity_profiles.yaml"
            try:
                from annotation_pipeline_skill.similarity.profiles import load_similarity_profiles
                profiles = load_similarity_profiles(profiles_path)
            except FileNotFoundError:
                profiles = {}
            from annotation_pipeline_skill.services.distribution_service import DistributionService
            svc = DistributionService(store, profiles)
            state = svc.get_cache_state(project_id=project_id, profile_name=profile_name)
            state["available_profiles"] = sorted(profiles.keys())
            return self._json_response(200, state)
        if route == "/api/row-dedup":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            profile_name = query.get("profile", ["MinHash"])[0]
            profiles_path = self.workspace_root / "similarity_profiles.yaml"
            try:
                from annotation_pipeline_skill.similarity.profiles import load_similarity_profiles
                profiles = load_similarity_profiles(profiles_path)
            except FileNotFoundError:
                profiles = {}
            from annotation_pipeline_skill.services.row_dedup_service import RowDedupService
            svc = RowDedupService(store, profiles)
            state = svc.get_cache_state(project_id=project_id, profile_name=profile_name)
            state["available_profiles"] = sorted(profiles.keys())
            return self._json_response(200, state)
        if route.startswith("/api/jobs/"):
            job_id = route.removeprefix("/api/jobs/").strip("/")
            if not job_id:
                return self._json_response(400, {"error": "job_id_required"})
            job = _read_background_job(job_id)
            if job is None:
                return self._json_response(404, {"error": "job_not_found"})
            return self._json_response(200, job)
        if route == "/api/type-statistics":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            cached = read_type_statistics_cache(store, project_id=project_id)
            if cached is None:
                return self._json_response(200, {
                    "cached": False, "payload": None, "generated_at": None,
                })
            return self._json_response(200, {
                "cached": True,
                "payload": cached["payload"],
                "generated_at": cached["created_at"],
            })
        if route == "/api/typical-text":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            span = query.get("span", [None])[0]
            if not span:
                return self._json_response(400, {"error": "span_required"})
            exclude = query.get("exclude", [])
            exclude_keys = query.get("exclude_key", [])
            task_id_filter = query.get("task", [None])[0]
            source_only = query.get("source_only", ["0"])[0] in ("1", "true", "yes")
            result = find_typical_text_for_span(
                store,
                project_id=project_id,
                span=span,
                exclude_task_ids=exclude,
                exclude_keys=exclude_keys,
                task_id_filter=task_id_filter,
                source_only=source_only,
            )
            if result is None:
                return self._json_response(200, {"found": False})
            return self._json_response(200, {"found": True, **result})
        if route == "/api/entity-statistics":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            # Server-side pagination, mirroring /api/conventions. A rebuilt
            # project holds tens of thousands of distinct spans (~6MB if shipped
            # whole), but the table shows one page. The natural unit is the
            # span (one row per span, aggregating its per-type counts), so we
            # paginate at the span level: pick the page of spans first, then
            # load only those spans' distribution rows.
            conn = store._conn
            try:
                limit = max(1, min(500, int(query.get("limit", ["100"])[0])))
            except ValueError:
                limit = 100
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
            except ValueError:
                offset = 0
            term = (query.get("q", [""])[0] or "").strip().lower()
            # Optional search: restrict to spans whose text OR any of their
            # entity types match. Resolve the matching span set first so a
            # type match (e.g. "organization") keeps the span's FULL
            # distribution rather than dropping its other-type rows.
            filter_sql = ""
            filter_params: list = []
            if term:
                esc = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                like = f"%{esc}%"
                filter_sql = (
                    " AND span_lower IN (SELECT span_lower FROM entity_statistics "
                    "WHERE project_id = ? AND (span_lower LIKE ? ESCAPE '\\' "
                    "OR lower(entity_type) LIKE ? ESCAPE '\\'))"
                )
                filter_params = [project_id, like, like]
            total = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT span_lower FROM entity_statistics "
                f"WHERE project_id = ?{filter_sql} GROUP BY span_lower)",
                [project_id, *filter_params],
            ).fetchone()[0]
            page_spans = conn.execute(
                f"SELECT span_lower, SUM(count) AS total FROM entity_statistics "
                f"WHERE project_id = ?{filter_sql} GROUP BY span_lower "
                f"ORDER BY total DESC, span_lower ASC LIMIT ? OFFSET ?",
                [project_id, *filter_params, limit, offset],
            ).fetchall()
            items: list[dict] = []
            if page_spans:
                names = [r["span_lower"] for r in page_spans]
                placeholders = ",".join("?" for _ in names)
                dist_rows = conn.execute(
                    f"SELECT span_lower, entity_type, count FROM entity_statistics "
                    f"WHERE project_id = ? AND span_lower IN ({placeholders})",
                    [project_id, *names],
                ).fetchall()
                by_span: dict[str, dict] = {}
                for r in dist_rows:
                    entry = by_span.setdefault(
                        r["span_lower"], {"span": r["span_lower"], "distribution": {}, "total": 0}
                    )
                    entry["distribution"][r["entity_type"]] = r["count"]
                    entry["total"] += r["count"]
                # Preserve the page's (total DESC, span ASC) ordering.
                items = [by_span[name] for name in names if name in by_span]
            return self._json_response(200, {
                "items": items,
                "total": total,
                "span_count": total,
                "limit": limit,
                "offset": offset,
            })
        if route == "/api/runtime":
            return self._json_response(200, self._runtime_snapshot(store).to_dict())
        if route == "/api/runtime/monitor":
            return self._json_response(200, validate_runtime_snapshot(self._runtime_snapshot(store)))
        if route == "/api/documents":
            return self._json_response(200, {"documents": [doc.to_dict() for doc in store.list_documents()]})
        if route == "/api/annotation-rules-document":
            return self._annotation_rules_document_response(store)
        if route.startswith("/api/documents/"):
            remainder = route.removeprefix("/api/documents/")
            parts = remainder.split("/")
            if len(parts) == 1 and parts[0]:
                return self._document_detail_response(store, parts[0])
            if len(parts) == 2 and parts[1] == "versions":
                return self._json_response(200, {"versions": [v.to_dict() for v in store.list_document_versions(parts[0])]})
            if len(parts) == 3 and parts[1] == "versions" and parts[2]:
                try:
                    ver = store.load_document_version(parts[2])
                except FileNotFoundError:
                    return self._json_response(404, {"error": "version_not_found"})
                return self._json_response(200, ver.to_dict())
        if route.startswith("/api/tasks/") and route.endswith("/deviations"):
            task_id = route.removeprefix("/api/tasks/").removesuffix("/deviations").strip("/")
            return self._task_deviations_response(store, task_id)
        if route.startswith("/api/tasks/"):
            task_id = route.removeprefix("/api/tasks/")
            if not task_id:
                return self._json_response(404, {"error": "not_found"})
            return self._task_detail_response(store, task_id)
        return self._json_response(404, {"error": "not_found"})

    def handle_put(self, path: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        parsed_path = urlparse(path)
        route = parsed_path.path
        query = parse_qs(parsed_path.query)
        store = self._resolve_store(query)
        if route.startswith("/api/tasks/") and route.endswith("/qc-policy"):
            task_id = route.removeprefix("/api/tasks/").removesuffix("/qc-policy").strip("/")
            return self._update_task_qc_policy_response(store, task_id, body)
        if route.startswith("/api/config/"):
            config_id = route.removeprefix("/api/config/")
            return self._update_config_response(store, config_id, body)
        if route == "/api/providers":
            return self._update_provider_config_response(store, body)
        if route == "/api/annotators":
            return self._update_annotators_response(store, body)
        if route == "/api/schema":
            return self._update_project_schema_response(store, body)
        return self._json_response(404, {"error": "not_found"})

    def handle_post(self, path: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        parsed_path = urlparse(path)
        route = parsed_path.path
        query = parse_qs(parsed_path.query)
        store = self._resolve_store(query)
        project_id = query.get("project", [None])[0]
        if route == "/api/runtime/run-once":
            return self._runtime_run_once_response()
        if route == "/api/runtime/start":
            return self._runtime_start_response(store)
        if route == "/api/runtime/stop":
            return self._runtime_stop_response(store)
        if route == "/api/posterior-audit/retroactive-fix":
            return self._post_posterior_audit_retroactive_fix(store, body)
        if route == "/api/entity-statistics/recount":
            return self._post_entity_statistics_recount(store, body)
        if route == "/api/posterior-audit":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})

            def _rebuild_posterior_audit() -> dict:
                payload = build_posterior_audit(store, project_id=project_id)
                accepted_hash = compute_accepted_hash(store, project_id=project_id)
                created_at = utc_now().isoformat()
                write_posterior_audit_cache(
                    store, project_id=project_id, payload=payload,
                    accepted_hash=accepted_hash, created_at=created_at,
                )
                return {"generated_at": created_at}

            job = _start_background_job(
                kind="posterior_audit_rebuild",
                project_id=project_id,
                target=_rebuild_posterior_audit,
            )
            return self._json_response(202, {"started": True, "job": job})
        if route == "/api/type-statistics":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})

            def _rebuild_type_statistics() -> dict:
                payload = build_type_statistics(store, project_id=project_id)
                created_at = utc_now().isoformat()
                write_type_statistics_cache(
                    store, project_id=project_id, payload=payload, created_at=created_at,
                )
                return {"generated_at": created_at}

            job = _start_background_job(
                kind="type_statistics_rebuild",
                project_id=project_id,
                target=_rebuild_type_statistics,
            )
            return self._json_response(202, {"started": True, "job": job})
        if route == "/api/distribution/scan":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
            if not isinstance(data, dict):
                return self._json_response(400, {"error": "invalid_payload"})
            profile_name = data.get("profile", "jina_small")
            statuses = data.get("statuses")  # None = all stages
            min_cluster_size = int(data.get("min_cluster_size", 5))
            umap_neighbors = int(data.get("umap_neighbors", 15))
            umap_min_dist = float(data.get("umap_min_dist", 0.1))
            profiles_path = self.workspace_root / "similarity_profiles.yaml"
            try:
                from annotation_pipeline_skill.similarity.profiles import load_similarity_profiles
                profiles = load_similarity_profiles(profiles_path)
            except FileNotFoundError:
                return self._json_response(400, {"error": "profiles_file_missing"})
            from annotation_pipeline_skill.services.distribution_service import DistributionService
            svc = DistributionService(store, profiles)

            def _run_distribution_scan() -> dict:
                payload = svc.scan(
                    project_id=project_id,
                    profile_name=profile_name,
                    statuses=statuses,
                    min_cluster_size=min_cluster_size,
                    umap_neighbors=umap_neighbors,
                    umap_min_dist=umap_min_dist,
                )
                generated_at = payload.get("params", {}).get(
                    "generated_at", utc_now().isoformat(),
                )
                return {"generated_at": generated_at}

            job = _start_background_job(
                kind=f"distribution_scan:{profile_name}",
                project_id=project_id,
                target=_run_distribution_scan,
            )
            return self._json_response(202, {
                "started": True,
                "job": job,
                "available_profiles": sorted(profiles.keys()),
            })
        if route == "/api/distribution/reject":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
            if not isinstance(data, dict):
                return self._json_response(400, {"error": "invalid_payload"})
            task_ids = data.get("task_ids")
            if not isinstance(task_ids, list) or not task_ids:
                return self._json_response(400, {"error": "task_ids_required"})
            profiles_path = self.workspace_root / "similarity_profiles.yaml"
            try:
                from annotation_pipeline_skill.similarity.profiles import load_similarity_profiles
                profiles = load_similarity_profiles(profiles_path)
            except FileNotFoundError:
                profiles = {}
            from annotation_pipeline_skill.services.distribution_service import DistributionService
            svc = DistributionService(store, profiles)
            result = svc.reject_duplicates(
                project_id=project_id,
                task_ids=task_ids,
                cluster_id=data.get("cluster_id"),
                representative_task_id=data.get("representative_task_id"),
                cluster_similarity=data.get("cluster_similarity"),
                embedding_profile=data.get("embedding_profile", ""),
                embedding_model=data.get("embedding_model", ""),
                actor=data.get("actor", "operator"),
            )
            return self._json_response(200, result)
        if route == "/api/row-dedup/scan":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
            if not isinstance(data, dict):
                return self._json_response(400, {"error": "invalid_payload"})
            profile_name = data.get("profile", "MinHash")
            statuses = data.get("statuses")  # None = all stages
            jaccard_threshold = float(data.get("jaccard_threshold", 0.5))
            profiles_path = self.workspace_root / "similarity_profiles.yaml"
            try:
                from annotation_pipeline_skill.similarity.profiles import load_similarity_profiles
                profiles = load_similarity_profiles(profiles_path)
            except FileNotFoundError:
                return self._json_response(400, {"error": "profiles_file_missing"})
            from annotation_pipeline_skill.services.row_dedup_service import RowDedupService
            svc = RowDedupService(store, profiles)
            # Run synchronously (scan_rows is typically fast; background job
            # can be wired up later if needed)
            try:
                svc.scan_rows(
                    project_id=project_id,
                    profile_name=profile_name,
                    statuses=statuses,
                    jaccard_threshold=jaccard_threshold,
                )
            except KeyError as exc:
                return self._json_response(400, {"error": "unknown_profile", "detail": str(exc)})
            state = svc.get_cache_state(project_id=project_id, profile_name=profile_name)
            state["available_profiles"] = sorted(profiles.keys())
            return self._json_response(200, state)
        if route == "/api/row-dedup/mask":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
            if not isinstance(data, dict):
                return self._json_response(400, {"error": "invalid_payload"})
            members = data.get("members")
            if not isinstance(members, list) or not members:
                return self._json_response(400, {"error": "members_required"})
            cluster_id = data.get("cluster_id", "")
            cluster_similarity = float(data.get("cluster_similarity", 0.0))
            embedding_profile = data.get("embedding_profile", "")
            embedding_model = data.get("embedding_model", "")
            profiles_path = self.workspace_root / "similarity_profiles.yaml"
            try:
                from annotation_pipeline_skill.similarity.profiles import load_similarity_profiles
                profiles = load_similarity_profiles(profiles_path)
            except FileNotFoundError:
                profiles = {}
            from annotation_pipeline_skill.services.row_dedup_service import RowDedupService
            svc = RowDedupService(store, profiles)
            result = svc.mask_duplicates(
                project_id=project_id,
                members=members,
                cluster_id=cluster_id,
                similarity=cluster_similarity,
                profile_name=embedding_profile,
                model=embedding_model,
            )
            return self._json_response(200, result)
        if route == "/api/documents":
            return self._post_document_response(store, body)
        if route == "/api/annotation-rules-document/versions":
            return self._post_annotation_rules_document_version(store, body)
        if route.startswith("/api/documents/") and route.endswith("/versions"):
            doc_id = route.removeprefix("/api/documents/").removesuffix("/versions").strip("/")
            return self._post_document_version_response(store, doc_id, body)
        if route.startswith("/api/tasks/") and route.endswith("/human-review"):
            task_id = route.removeprefix("/api/tasks/").removesuffix("/human-review").strip("/")
            return self._post_human_review_response(store, task_id, body)
        if route.startswith("/api/tasks/") and route.endswith("/human_review_correction"):
            task_id = route.removeprefix("/api/tasks/").removesuffix("/human_review_correction").strip("/")
            return self._post_human_review_correction(store, task_id, body)
        if route.startswith("/api/tasks/") and route.endswith("/feedback-discussions"):
            task_id = route.removeprefix("/api/tasks/").removesuffix("/feedback-discussions").strip("/")
            return self._post_feedback_discussion_response(store, task_id, body)
        if route.startswith("/api/tasks/") and route.endswith("/move"):
            task_id = route.removeprefix("/api/tasks/").removesuffix("/move").strip("/")
            return self._post_task_move_response(store, task_id, body)
        if route.startswith("/api/tasks/") and route.endswith("/posterior-fix"):
            task_id = route.removeprefix("/api/tasks/").removesuffix("/posterior-fix").strip("/")
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
            if not isinstance(payload, dict):
                return self._json_response(400, {"error": "invalid_payload"})
            span = payload.get("span")
            current_type = payload.get("current_type")
            new_type = payload.get("new_type")  # may be None or "not_an_entity" for delete
            actor = payload.get("actor") or "posterior_audit_operator"
            # Default True (back-compat); the UI checkbox sends an explicit
            # bool. When False, the fix patches this task only and is NOT
            # promoted to a project-wide convention.
            save_flag = payload.get("save_as_convention", True)
            save_as_convention = bool(save_flag)
            if not span or not current_type:
                return self._json_response(400, {"error": "span_and_current_type_required"})
            try:
                result = HumanReviewService(store).apply_posterior_fix(
                    task_id=task_id, span=span, current_type=current_type,
                    new_type=new_type, actor=actor,
                    save_as_convention=save_as_convention,
                )
                # Surgically update the Posterior Audit cache: remove the
                # (task_id, span, current_type) deviations that this fix
                # resolved and refresh the accepted_hash so the next GET
                # returns an up-to-date, non-stale view. Without this,
                # operators see "no change" on refresh after Submit and
                # have to click Re-check to trigger a full rescan.
                task = store.load_task(task_id)
                project_for_cache = task.pipeline_id
                cached = read_posterior_audit_cache(store, project_id=project_for_cache)
                if cached is not None:
                    payload_in_cache = cached["payload"]
                    devs = payload_in_cache.get("task_deviations", [])
                    kept = [
                        d for d in devs
                        if not (
                            d.get("task_id") == task_id
                            and d.get("span") == span
                            and d.get("current_type") == current_type
                        )
                    ]
                    payload_in_cache["task_deviations"] = kept
                    # If this fix also wrote a project-wide convention,
                    # stamp the matching divergent_entries rows so the
                    # Contested tab's badge reflects the new policy on
                    # refresh (mirrors /api/conventions cache surgery).
                    if save_as_convention:
                        decided_type = new_type or "not_an_entity"
                        span_lower = span.strip().lower()
                        for c in payload_in_cache.get("divergent_entries", []):
                            if c.get("span", "").lower() == span_lower:
                                c["resolved_convention_type"] = decided_type
                    new_hash = compute_accepted_hash(store, project_id=project_for_cache)
                    from annotation_pipeline_skill.core.models import utc_now as _utc_now
                    write_posterior_audit_cache(
                        store,
                        project_id=project_for_cache,
                        payload=payload_in_cache,
                        accepted_hash=new_hash,
                        created_at=_utc_now().isoformat(),
                    )
                return self._json_response(200, result)
            except InvalidTransition as exc:
                return self._json_response(409, {"error": "invalid_state", "detail": str(exc)})
            except SchemaValidationError as exc:
                return self._json_response(400, {"error": "schema_invalid", "detail": str(exc), "errors": exc.errors})
            except Exception as exc:  # noqa: BLE001
                return self._json_response(500, {"error": "internal", "detail": str(exc)})
        if route == "/api/conventions":
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
            if not isinstance(payload, dict):
                return self._json_response(400, {"error": "invalid_payload"})
            return self._post_convention_response(store, project_id, payload)
        if route.startswith("/api/conventions/") and route.endswith("/resolve"):
            conv_id = route.removeprefix("/api/conventions/").removesuffix("/resolve").strip("/")
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
            if not isinstance(payload, dict):
                return self._json_response(400, {"error": "invalid_payload"})
            return self._post_convention_resolve_response(store, conv_id, payload)
        if route == "/api/conventions/clear":
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
            if not isinstance(payload, dict):
                return self._json_response(400, {"error": "invalid_payload"})
            from annotation_pipeline_skill.services.entity_convention_service import (
                EntityConventionService,
            )
            pid = payload.get("project_id") or project_id
            span = payload.get("span")
            if not pid:
                return self._json_response(400, {"error": "project_required"})
            if not isinstance(span, str) or not span.strip():
                return self._json_response(400, {"error": "span_required"})
            removed = EntityConventionService(store).delete_for_span(
                project_id=pid, span=span,
            )
            # Surgery: strip `resolved_convention_type` from any cached
            # posterior-audit contested row matching this span. Without
            # this the dashboard keeps showing the green "✓ set: type"
            # badge after Unset because the cached payload still has
            # the field — the GET reads cache, doesn't recompute.
            if removed:
                try:
                    cached = read_posterior_audit_cache(store, project_id=pid)
                    if cached is not None:
                        cache_payload = cached["payload"]
                        span_lower = span.strip().lower()
                        divergent = cache_payload.get("divergent_entries", [])
                        changed = False
                        for c in divergent:
                            if (c.get("span") or "").lower() == span_lower and \
                               "resolved_convention_type" in c:
                                c.pop("resolved_convention_type", None)
                                changed = True
                        if changed:
                            cache_payload["divergent_entries"] = divergent
                            write_posterior_audit_cache(
                                store, project_id=pid, payload=cache_payload,
                                accepted_hash=cached["accepted_hash"],
                                created_at=cached["created_at"],
                            )
                except Exception:  # noqa: BLE001 — cache update is best-effort
                    pass
            return self._json_response(200, {"removed": removed})
        return self._json_response(404, {"error": "not_found"})

    def _post_convention_response(self, store: SqliteStore, project_id: str | None, body: dict) -> tuple:
        from annotation_pipeline_skill.services.entity_convention_service import (
            EntityConventionService,
        )
        pid = body.get("project_id") or project_id
        if not pid:
            return self._json_response(400, {"error": "project_required"})
        span = body.get("span")
        entity_type = body.get("entity_type")
        if not isinstance(span, str) or not span.strip():
            return self._json_response(400, {"error": "span_required"})
        if not isinstance(entity_type, str) or not entity_type.strip():
            return self._json_response(400, {"error": "entity_type_required"})
        actor = body.get("actor") or "operator"
        try:
            conv = EntityConventionService(store).record_decision(
                project_id=pid,
                span=span,
                entity_type=entity_type,
                source=f"declared:{actor}",
                task_id=body.get("task_id"),
                notes=body.get("notes"),
            )
        except (ValueError, TypeError) as exc:
            return self._json_response(400, {"error": str(exc)})
        # Stamp this span's `resolved_convention_type` on the cached
        # Posterior Audit `divergent_entries` and `low_info_entries` so a
        # refresh after Set Convention shows the badge inline (UI annotates
        # the row instead of hiding it — operator wants to observe the change).
        cached = read_posterior_audit_cache(store, project_id=pid)
        if cached is not None:
            payload_in_cache = cached["payload"]
            span_lower = span.strip().lower()
            divergent = payload_in_cache.get("divergent_entries", [])
            mutated = False
            for c in divergent:
                if c.get("span", "").lower() == span_lower:
                    c["resolved_convention_type"] = entity_type
                    mutated = True
            if mutated:
                write_posterior_audit_cache(
                    store,
                    project_id=pid,
                    payload=payload_in_cache,
                    accepted_hash=cached["accepted_hash"],
                    created_at=cached["created_at"],
                )
        return self._json_response(200, conv.to_dict())

    def _post_convention_resolve_response(self, store: SqliteStore, convention_id: str, body: dict) -> tuple:
        from annotation_pipeline_skill.services.entity_convention_service import (
            EntityConventionService,
        )
        resolved = body.get("entity_type")
        if not isinstance(resolved, str) or not resolved.strip():
            return self._json_response(400, {"error": "entity_type_required"})
        actor = body.get("actor") or "operator"
        try:
            conv = EntityConventionService(store).clear_dispute(
                convention_id=convention_id,
                resolved_type=resolved,
                actor=actor,
                notes=body.get("notes"),
            )
        except KeyError:
            return self._json_response(404, {"error": "convention_not_found"})
        return self._json_response(200, conv.to_dict())

    def _stores_list(self) -> list[dict]:
        result = []
        for key, s in self._stores.items():
            tasks = s.list_tasks()
            result.append({
                "key": key,
                "name": s.root.parent.name,
                "path": str(s.root.parent),
                "pipeline_count": len({task.pipeline_id for task in tasks}),
                "task_count": len(tasks),
            })
        return result

    def _runtime_snapshot(self, store: SqliteStore) -> RuntimeSnapshot:
        from datetime import datetime, timezone

        snapshot = store.load_runtime_snapshot()
        if snapshot is None:
            rebuilt = build_runtime_snapshot(store, self.runtime_config)
            status = replace(
                rebuilt.runtime_status,
                healthy=False,
                active=False,
                errors=sorted(set([*rebuilt.runtime_status.errors, "runtime_snapshot_missing"])),
            )
            return replace(rebuilt, runtime_status=status)

        # The stored snapshot's heartbeat_at/age are frozen at write time.
        # Re-read the live heartbeat file so a stopped scheduler (which clears
        # the file on exit) is immediately reflected as unhealthy.
        now = datetime.now(timezone.utc)
        old = snapshot.runtime_status
        errors: list[str] = []
        heartbeat_age_seconds: int | None = None
        live_heartbeat_at = store.load_runtime_heartbeat()  # None if file deleted
        if live_heartbeat_at is None:
            errors.append("heartbeat_missing")
        else:
            heartbeat_age_seconds = int((now - live_heartbeat_at).total_seconds())
            stale_after = max(self.runtime_config.snapshot_interval_seconds * 2, 120)
            if heartbeat_age_seconds > stale_after:
                errors.append("heartbeat_stale")

        fresh_status = replace(
            old,
            healthy=not errors,
            heartbeat_at=live_heartbeat_at,
            heartbeat_age_seconds=heartbeat_age_seconds,
            errors=errors,
        )
        return replace(snapshot, runtime_status=fresh_status)

    def _runtime_run_once_response(self) -> tuple[int, dict[str, str], bytes]:
        if self.runtime_once is None:
            return self._json_response(409, {"error": "runtime_runner_unavailable"})
        snapshot = self.runtime_once()
        return self._json_response(200, {"ok": True, "snapshot": snapshot.to_dict()})

    def _runtime_start_response(self, store: SqliteStore) -> tuple[int, dict[str, str], bytes]:
        import shutil
        import subprocess

        owner_path = store.root / "runtime" / "scheduler_owner.json"
        if owner_path.exists():
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
                pid = owner.get("pid")
                if pid and _pid_alive(pid):
                    return self._json_response(409, {
                        "error": "already_running",
                        "pid": pid,
                        "project_root": str(store.root.parent),
                    })
            except (json.JSONDecodeError, OSError):
                pass

        project_root = store.root.parent
        binary = shutil.which("annotation-pipeline")
        if not binary:
            return self._json_response(500, {"error": "annotation-pipeline binary not found on PATH"})

        log_path = self.workspace_root / "runtime.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log_f:
            proc = subprocess.Popen(
                [binary, "runtime", "run", "--project-root", str(project_root)],
                stdout=log_f,
                stderr=log_f,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        return self._json_response(200, {
            "ok": True,
            "pid": proc.pid,
            "project_root": str(project_root),
        })

    def _runtime_stop_response(self, store: SqliteStore) -> tuple[int, dict[str, str], bytes]:
        import os
        import signal

        owner_path = store.root / "runtime" / "scheduler_owner.json"
        if not owner_path.exists():
            return self._json_response(404, {"error": "no_scheduler_running"})
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return self._json_response(500, {"error": "cannot_read_owner", "detail": str(exc)})

        pid = owner.get("pid")
        if not pid:
            return self._json_response(404, {"error": "no_pid_in_owner"})
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            owner_path.unlink(missing_ok=True)
            return self._json_response(404, {"error": "process_not_found"})
        except PermissionError:
            return self._json_response(403, {"error": "permission_denied"})
        # Clear owner file and heartbeat immediately so the next health check
        # reflects the stopped state without waiting for the staleness window.
        owner_path.unlink(missing_ok=True)
        store.clear_runtime_heartbeat()
        return self._json_response(200, {"ok": True, "pid": pid})

    def _provider_config_response(self, store: SqliteStore) -> tuple[int, dict[str, str], bytes]:
        try:
            return self._json_response(
                200,
                build_provider_config_snapshot(store.root, workspace_root=self.workspace_root),
            )
        except (FileNotFoundError, OSError, ProfileValidationError) as exc:
            return self._json_response(
                400,
                {
                    "config_valid": False,
                    "error": "invalid_provider_config",
                    "detail": str(exc),
                },
            )

    def _update_provider_config_response(self, store: SqliteStore, body: bytes) -> tuple[int, dict[str, str], bytes]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})
        try:
            snapshot = save_provider_config(store.root, payload, workspace_root=self.workspace_root)
        except (FileNotFoundError, OSError, ProfileValidationError) as exc:
            return self._json_response(400, {"error": "invalid_provider_config", "detail": str(exc)})
        return self._json_response(200, snapshot)

    def _update_project_schema_response(self, store: SqliteStore, body: bytes) -> tuple[int, dict[str, str], bytes]:
        """PUT /api/schema — persist edited output_schema.json after JSON-Schema metaschema validation."""
        from annotation_pipeline_skill.core.schema_validation import PROJECT_SCHEMA_FILENAME
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        # Accept either {"schema": {...}} or the raw schema object.
        schema = payload.get("schema") if isinstance(payload, dict) and "schema" in payload else payload
        if not isinstance(schema, dict):
            return self._json_response(400, {"error": "invalid_payload", "detail": "schema must be an object"})
        # Validate that the supplied document is itself a valid JSON Schema.
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            return self._json_response(400, {"error": "invalid_schema", "detail": str(exc)})
        path = store.root / PROJECT_SCHEMA_FILENAME
        try:
            path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            return self._json_response(500, {"error": "write_failed", "detail": str(exc)})
        return self._json_response(200, {"schema": schema, "path": str(path.relative_to(store.root))})

    def handle_delete(self, path: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        parsed_path = urlparse(path)
        route = parsed_path.path
        query = parse_qs(parsed_path.query)
        store = self._resolve_store(query)
        project_id = query.get("project", [None])[0]
        if route == "/api/row-dedup/mask":
            if not project_id:
                return self._json_response(400, {"error": "project_required"})
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
            if not isinstance(data, dict):
                return self._json_response(400, {"error": "invalid_payload"})
            pairs_raw = data.get("pairs")
            if not isinstance(pairs_raw, list) or not pairs_raw:
                return self._json_response(400, {"error": "pairs_required"})
            from annotation_pipeline_skill.services.row_mask_service import RowMaskService
            mask_svc = RowMaskService(store)
            pairs = [
                (p["task_id"], int(p["row_index"]))
                for p in pairs_raw
                if isinstance(p, dict) and "task_id" in p and "row_index" in p
            ]
            removed = mask_svc.remove_many(pairs)
            return self._json_response(200, {"removed": removed})
        return self._json_response(404, {"error": "not_found"})

    def _json_response(self, status: int, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        return status, {"content-type": "application/json"}, body

    def _document_detail_response(self, store: SqliteStore, document_id: str) -> tuple[int, dict[str, str], bytes]:
        try:
            doc = store.load_document(document_id)
        except FileNotFoundError:
            return self._json_response(404, {"error": "document_not_found"})
        versions = store.list_document_versions(document_id)
        return self._json_response(200, {"document": doc.to_dict(), "versions": [v.to_dict() for v in versions]})

    def _post_document_response(self, store: SqliteStore, body: bytes) -> tuple[int, dict[str, str], bytes]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})
        from annotation_pipeline_skill.core.models import AnnotationDocument
        doc = AnnotationDocument.new(
            title=str(payload.get("title") or ""),
            description=str(payload.get("description") or ""),
            created_by=str(payload.get("created_by") or "operator"),
            metadata=dict(payload.get("metadata") or {}),
        )
        store.save_document(doc)
        return self._json_response(200, doc.to_dict())

    def _post_document_version_response(self, store: SqliteStore, doc_id: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        try:
            store.load_document(doc_id)
        except FileNotFoundError:
            return self._json_response(404, {"error": "document_not_found"})
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})
        from annotation_pipeline_skill.core.models import AnnotationDocumentVersion
        ver = AnnotationDocumentVersion.new(
            document_id=doc_id,
            version=str(payload.get("version") or "v1"),
            content=str(payload.get("content") or ""),
            changelog=str(payload.get("changelog") or ""),
            created_by=str(payload.get("created_by") or "operator"),
            metadata=dict(payload.get("metadata") or {}),
        )
        store.save_document_version(ver)
        return self._json_response(200, ver.to_dict())

    # ---- annotation rules singleton document ---------------------------

    _ANNOTATION_RULES_ROLE = "annotation_rules"

    def _find_or_create_annotation_rules_doc(self, store: SqliteStore):
        """Return the singleton AnnotationDocument that holds annotation rules."""
        from annotation_pipeline_skill.core.models import AnnotationDocument

        for d in store.list_documents():
            if d.metadata.get("role") == self._ANNOTATION_RULES_ROLE:
                return d
        doc = AnnotationDocument.new(
            title="Annotation Rules",
            description="Project-level annotation rules injected into annotator / QC / arbiter prompts.",
            created_by="system",
            metadata={"role": self._ANNOTATION_RULES_ROLE},
        )
        store.save_document(doc)
        return doc

    def _annotation_rules_document_response(
        self, store: SqliteStore
    ) -> tuple[int, dict[str, str], bytes]:
        doc = self._find_or_create_annotation_rules_doc(store)
        versions = store.list_document_versions(doc.document_id)
        versions_sorted = sorted(versions, key=lambda v: v.created_at, reverse=True)
        latest = versions_sorted[0] if versions_sorted else None
        return self._json_response(
            200,
            {
                "document": doc.to_dict(),
                "versions": [v.to_dict() for v in versions_sorted],
                "latest_version_id": latest.version_id if latest else None,
            },
        )

    def _post_annotation_rules_document_version(
        self, store: SqliteStore, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})
        from annotation_pipeline_skill.core.models import AnnotationDocumentVersion

        doc = self._find_or_create_annotation_rules_doc(store)
        version_label = str(payload.get("version") or "").strip()
        if not version_label:
            existing = store.list_document_versions(doc.document_id)
            version_label = f"v{len(existing) + 1}"
        ver = AnnotationDocumentVersion.new(
            document_id=doc.document_id,
            version=version_label,
            content=str(payload.get("content") or ""),
            changelog=str(payload.get("changelog") or ""),
            created_by=str(payload.get("created_by") or "operator"),
            metadata=dict(payload.get("metadata") or {}),
        )
        store.save_document_version(ver)
        return self._json_response(200, ver.to_dict())

    def _task_deviations_response(self, store: SqliteStore, task_id: str) -> tuple[int, dict[str, str], bytes]:
        """Return per-(span, type) deviations for a single task against
        the project's entity_statistics. Used by Manual Review to surface
        prior-disagreeing spans inline so the operator can decide whether
        to apply a posterior fix while reviewing.

        Honors active operator conventions the same way build_posterior_audit
        does: if a convention pins this span to a type, skip when matched.
        """
        try:
            task = store.load_task(task_id)
        except (FileNotFoundError, KeyError):
            return self._json_response(404, {"error": "task_not_found"})
        from annotation_pipeline_skill.services.entity_statistics_service import (
            EntityStatisticsService,
            iter_span_decisions,
        )
        from annotation_pipeline_skill.services.entity_convention_service import (
            EntityConventionService,
        )
        from annotation_pipeline_skill.runtime.subagent_cycle import _parse_llm_json
        import json as _json
        import re

        # Load latest annotation/HR answer payload.
        arts = store.list_artifacts(task_id)
        payload: dict | None = None
        hr = [a for a in arts if a.kind == "human_review_answer"]
        if hr:
            outer = _json.loads((store.root / hr[-1].path).read_text(encoding="utf-8"))
            payload = outer.get("answer") if isinstance(outer, dict) else None
        else:
            anns = [a for a in arts if a.kind == "annotation_result"]
            if anns:
                outer = _json.loads((store.root / anns[-1].path).read_text(encoding="utf-8"))
                text = outer.get("text") if isinstance(outer, dict) else None
                if isinstance(text, str):
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
                    try:
                        payload = _parse_llm_json(text)
                    except (ValueError, _json.JSONDecodeError):
                        payload = None
        if payload is None:
            return self._json_response(200, {"task_id": task_id, "deviations": []})

        convention_index: dict[str, str] = {}
        for c in EntityConventionService(store).list_for_project(task.pipeline_id, include_disputed=False):
            if c.entity_type:
                convention_index[c.span_lower] = c.entity_type

        svc = EntityStatisticsService(store)
        deviations: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for span, entity_type in iter_span_decisions(payload):
            key = (span, entity_type)
            if key in seen:
                continue
            seen.add(key)
            conv_type = convention_index.get(span.lower())
            if conv_type is not None and conv_type == entity_type:
                continue  # operator-declared policy matches; not divergent
            r = svc.check(project_id=task.pipeline_id, span=span, proposed_type=entity_type)
            if r.status != "divergent":
                continue
            deviations.append({
                "span": r.span,
                "current_type": r.proposed_type,
                "prior_dominant_type": conv_type or r.dominant_type,
                "prior_total": r.total,
                "prior_distribution": r.distribution,
                "has_convention": conv_type is not None,
            })
        return self._json_response(200, {"task_id": task_id, "deviations": deviations})

    def _task_detail_response(self, store: SqliteStore, task_id: str) -> tuple[int, dict[str, str], bytes]:
        try:
            task = store.load_task(task_id)
        except FileNotFoundError:
            return self._json_response(404, {"error": "task_not_found"})

        # Apply row-level mask filtering at the read boundary so the
        # annotator-facing payload never includes rows that have been
        # masked (whether by row-dedup auto-mask, manual mask, etc.).
        # Mask state is sourced from the row_masks table.
        from annotation_pipeline_skill.services.row_mask_service import (
            RowMaskService,
            apply_masks_to_task,
        )
        masked_task = apply_masks_to_task(store, task)
        masked_indices = sorted(
            RowMaskService(store).masked_indices_for_task(task_id)
        )

        artifacts = [
            {**artifact.to_dict(), "payload": self._read_artifact_payload(store, artifact.path)}
            for artifact in store.list_artifacts(task_id)
        ]
        return self._json_response(
            200,
            {
                "task": masked_task.to_dict(),
                # Surface the indices of rows that were filtered out — useful
                # for the UI to show a "N rows masked" hint and for debugging.
                "masked_row_indices": masked_indices,
                "attempts": [attempt.to_dict() for attempt in store.list_attempts(task_id)],
                "artifacts": artifacts,
                "events": [event.to_dict() for event in store.list_events(task_id)],
                "feedback": [feedback.to_dict() for feedback in store.list_feedback(task_id)],
                "feedback_discussions": [
                    entry.to_dict()
                    for entry in store.list_feedback_discussions(task_id)
                ],
                "feedback_consensus": build_feedback_consensus_summary(store, task_id),
            },
        )

    def _post_feedback_discussion_response(self, store: SqliteStore, task_id: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        try:
            task = store.load_task(task_id)
        except FileNotFoundError:
            return self._json_response(404, {"error": "task_not_found"})
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})

        feedback_id = str(payload.get("feedback_id") or "")
        if feedback_id not in {feedback.feedback_id for feedback in store.list_feedback(task_id)}:
            return self._json_response(400, {"error": "unknown_feedback_id"})
        entry = FeedbackDiscussionEntry.new(
            task_id=task_id,
            feedback_id=feedback_id,
            role=str(payload.get("role") or "annotator"),
            stance=str(payload.get("stance") or "comment"),
            message=str(payload.get("message") or ""),
            agreed_points=list(payload.get("agreed_points") or []),
            disputed_points=list(payload.get("disputed_points") or []),
            proposed_resolution=payload.get("proposed_resolution"),
            consensus=bool(payload.get("consensus", False)),
            created_by=str(payload.get("created_by") or payload.get("role") or "unknown"),
            metadata=dict(payload.get("metadata") or {}),
        )
        store.append_feedback_discussion(entry)

        consensus = build_feedback_consensus_summary(store, task_id)
        if consensus["can_accept_by_consensus"] and task.status in {TaskStatus.QC, TaskStatus.HUMAN_REVIEW}:
            event = transition_task(
                task,
                TaskStatus.ACCEPTED,
                actor=entry.created_by,
                reason="feedback consensus accepted by annotator and qc",
                stage="qc",
                metadata={"feedback_id": feedback_id, "discussion_entry_id": entry.entry_id},
            )
            store.append_event(event)
            store.save_task(task)

        return self._json_response(
            200,
            {
                "entry": entry.to_dict(),
                "feedback_consensus": consensus,
                "task": store.load_task(task_id).to_dict(),
            },
        )

    # Whitelist of manual moves a human can request through the UI's drag-drop.
    # Drag actions that overlap with HR Decision (HR → Accepted / Rejected) are
    # intentionally absent — those go through the HR Decision form instead.
    _MANUAL_MOVE_WHITELIST: dict[TaskStatus, set[TaskStatus]] = {
        TaskStatus.REJECTED: {TaskStatus.ARBITRATING},
        TaskStatus.HUMAN_REVIEW: {TaskStatus.ARBITRATING, TaskStatus.PENDING},
        TaskStatus.ACCEPTED: {TaskStatus.HUMAN_REVIEW},
    }

    def _post_task_move_response(self, store: SqliteStore, task_id: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})
        target_value = str(payload.get("target_status") or "").lower()
        reason = str(payload.get("reason") or "").strip()
        actor = str(payload.get("actor") or "human")
        try:
            target_status = TaskStatus(target_value)
        except ValueError:
            return self._json_response(400, {"error": "invalid_target_status", "detail": target_value})
        if not reason:
            return self._json_response(400, {"error": "reason_required"})
        try:
            task = store.load_task(task_id)
        except (FileNotFoundError, KeyError):
            return self._json_response(404, {"error": "task_not_found"})
        # Block manual move when the task is actively leased by the runtime —
        # avoids racing with an in-flight worker on the same row.
        active_leases = [
            lease for lease in store.list_runtime_leases() if lease.task_id == task_id
        ]
        if active_leases:
            return self._json_response(409, {"error": "task_in_flight", "detail": "task is currently being processed by the runtime"})
        allowed = self._MANUAL_MOVE_WHITELIST.get(task.status, set())
        if target_status not in allowed:
            return self._json_response(
                400,
                {
                    "error": "manual_move_not_allowed",
                    "detail": f"cannot manually move {task.status.value} → {target_status.value}",
                    "allowed_targets": sorted(s.value for s in allowed),
                },
            )
        try:
            event = transition_task(
                task,
                target_status,
                actor=actor,
                reason=f"manual_drag: {reason}",
                stage="manual_move",
                metadata={"via": "manual_drag", "manual_target": target_status.value},
            )
        except InvalidTransition as exc:
            return self._json_response(400, {"error": "invalid_transition", "detail": str(exc)})
        # If the target is PENDING (HR → Annotating), reset the retry counter
        # so the annotator gets a fresh budget on the next worker pickup.
        if target_status is TaskStatus.PENDING:
            task.current_attempt = 0
        store.save_task(task)
        store.append_event(event)
        return self._json_response(200, {"ok": True, "task": task.to_dict()})

    def _post_entity_statistics_recount(
        self, store: SqliteStore, body: bytes,
    ) -> tuple[int, dict[str, str], bytes]:
        """Rebuild entity_statistics for one span from the current state
        of ACCEPTED tasks (see EntityStatisticsService.recount_span).

        Body: {"project_id", "span"}
        Returns: {"distribution": {type: count}, "total": int}
        """
        from annotation_pipeline_skill.services.entity_statistics_service import (
            EntityStatisticsService,
        )
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})
        project_id = payload.get("project_id")
        span = payload.get("span")
        if not project_id or not span:
            return self._json_response(400, {"error": "project_and_span_required"})
        try:
            dist = EntityStatisticsService(store).recount_span(
                project_id=project_id, span=span,
            )
        except Exception as exc:  # noqa: BLE001
            return self._json_response(500, {"error": "recount_failed", "detail": str(exc)})
        return self._json_response(200, {
            "distribution": dist,
            "total": sum(dist.values()),
        })

    def _post_posterior_audit_retroactive_fix(
        self, store: SqliteStore, body: bytes,
    ) -> tuple[int, dict[str, str], bytes]:
        """Declare a project convention for `span -> entity_type` AND
        retroactively patch every ACCEPTED task in the project whose
        current annotation has the span tagged differently.

        Body: {
          "project_id", "span", "entity_type", "actor",
          "batch_size": int | null,
          "task_ids": list[str] | null  # optional. When set, skip the
                                        # candidate scan and process ONLY
                                        # these specific task_ids (each
                                        # still checked for whether the
                                        # span is currently tagged as a
                                        # different type — idempotent).
                                        # When null, scan the whole project.
        }

        First call (no task_ids): server scans the project, returns the
        full candidate list in `candidate_task_ids`. Caller stores that
        and on subsequent polls passes a slice of N (matching
        `batch_size`) so the server skips the O(N_accepted_tasks) scan.

        `entity_type` may be the sentinel "not_an_entity" (or null), in
        which case matching tasks have the span removed entirely.

        Returns: {
          convention,
          fixed: int,
          skipped: int,
          errors: [{task_id, reason}],
          remaining: int,                  # only meaningful in scan mode
          candidate_task_ids: list[str]    # full candidate list, scan-mode only
              | null,
          done: bool,
        }
        record_decision is idempotent on identical (source, type), so
        multiple polls don't inflate the convention's evidence_count.
        """
        from annotation_pipeline_skill.core.states import TaskStatus
        from annotation_pipeline_skill.services.entity_convention_service import (
            EntityConventionService,
        )
        from annotation_pipeline_skill.services.human_review_service import (
            HumanReviewService,
        )
        from annotation_pipeline_skill.services.entity_statistics_service import (
            iter_span_decisions,
        )
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})
        project_id = payload.get("project_id")
        span = payload.get("span")
        entity_type_raw = payload.get("entity_type")
        actor = payload.get("actor") or "posterior_audit_retroactive"
        batch_size_raw = payload.get("batch_size")
        try:
            batch_size = int(batch_size_raw) if batch_size_raw is not None else None
        except (TypeError, ValueError):
            batch_size = None
        if batch_size is not None and batch_size < 1:
            batch_size = None
        task_ids_raw = payload.get("task_ids")
        explicit_task_ids: list[str] | None
        if isinstance(task_ids_raw, list) and task_ids_raw:
            explicit_task_ids = [str(t) for t in task_ids_raw if isinstance(t, str)]
        else:
            explicit_task_ids = None
        # Dry-run mode: scan + return candidate_task_ids without applying
        # any fix and without writing the convention. Used by the
        # confirmation modal to surface the real (task-count) impact
        # before the operator commits.
        dry_run = bool(payload.get("dry_run"))
        # Optional: skip the project convention write while still
        # patching tasks. Used when the operator wants a one-off bulk
        # task fix without polluting the convention table (e.g. for
        # context-specific spans that shouldn't influence future tasks).
        # Defaults to True for back-compat.
        set_convention = bool(payload.get("set_convention", True))
        if not project_id or not span:
            return self._json_response(400, {"error": "project_and_span_required"})
        if entity_type_raw is None:
            entity_type_str = "not_an_entity"
        else:
            entity_type_str = str(entity_type_raw)
        # "not_an_entity" means delete the span; apply_posterior_fix takes
        # None for that, and EntityConventionService stores the sentinel
        # string so callers can distinguish "no entity here" from "no
        # convention declared".
        fix_new_type = None if entity_type_str == "not_an_entity" else entity_type_str

        conv_svc = EntityConventionService(store)
        conv = None
        if not dry_run and set_convention:
            try:
                conv = conv_svc.record_decision(
                    project_id=project_id,
                    span=span,
                    entity_type=entity_type_str,
                    source=f"declared:{actor}",
                    task_id=None,
                )
                if conv.status == "disputed":
                    conv = conv_svc.clear_dispute(
                        convention_id=conv.convention_id,
                        resolved_type=entity_type_str,
                        actor=actor,
                        notes="resolved via posterior audit retroactive fix",
                    )
            except (ValueError, TypeError) as exc:
                return self._json_response(
                    400, {"error": "convention_failed", "detail": str(exc)},
                )

        hr = HumanReviewService(store)
        span_lower = span.lower()
        # Two modes:
        # 1. Scan mode (no task_ids in body): walk the project, return
        #    candidate_task_ids so the caller can iterate.
        # 2. Process mode (task_ids in body): trust the caller's list,
        #    skip the O(N_accepted) scan. Each task is still validated
        #    individually (idempotent — already-fixed tasks are skipped).
        candidate_task_ids: list[str] | None = None
        candidates: list[tuple[str, str, str]] = []  # (task_id, span_form, current_type)
        skipped = 0
        if explicit_task_ids is not None:
            for tid in explicit_task_ids:
                try:
                    task = store.load_task(tid)
                except (FileNotFoundError, KeyError):
                    skipped += 1
                    continue
                if task.status is not TaskStatus.ACCEPTED:
                    skipped += 1
                    continue
                ann = hr._latest_annotation_payload(tid)
                if not isinstance(ann, dict):
                    skipped += 1
                    continue
                # Collect ALL (span_form, type) pairs for this span in this
                # task — a span can appear in multiple entity-type buckets
                # (e.g. cross-type collision, or different row capitalisation).
                # Without collecting all of them, only the first encountered
                # type gets fixed and subsequent calls are needed.
                task_pairs: list[tuple[str, str]] = []
                for s, t in iter_span_decisions(ann):
                    if s.lower() == span_lower and t != entity_type_str:
                        task_pairs.append((s, t))
                if not task_pairs:
                    skipped += 1
                    continue
                for span_form, current_type in task_pairs:
                    candidates.append((tid, span_form, current_type))
        else:
            # Full scan. Pre-filter via SQL LIKE on source_ref_json so we
            # only read annotations for tasks whose input could possibly
            # contain the span. Match both raw + JSON-Unicode-escaped
            # form (Chinese / Japanese / Korean spans are stored escaped).
            span_lower_json = json.dumps(span_lower, ensure_ascii=True)[1:-1]
            prefilter_rows = store._conn.execute(
                "SELECT task_id FROM tasks "
                "WHERE pipeline_id=? AND status='accepted' "
                "AND (lower(source_ref_json) LIKE ? OR lower(source_ref_json) LIKE ?)",
                (project_id, f"%{span_lower}%", f"%{span_lower_json}%"),
            ).fetchall()
            prefiltered_ids = [r["task_id"] for r in prefilter_rows]
            for tid in prefiltered_ids:
                ann = hr._latest_annotation_payload(tid)
                if not isinstance(ann, dict):
                    continue
                task_pairs = []
                for s, t in iter_span_decisions(ann):
                    if s.lower() == span_lower and t != entity_type_str:
                        task_pairs.append((s, t))
                for span_form, current_type in task_pairs:
                    candidates.append((tid, span_form, current_type))
            # candidate_task_ids is the unique set of affected tasks (for the
            # UI's task-count display and batch iteration). The candidates list
            # may have multiple entries per task when the span appears under
            # more than one entity type.
            seen: set[str] = set()
            candidate_task_ids = []
            for tid, _, _ in candidates:
                if tid not in seen:
                    seen.add(tid)
                    candidate_task_ids.append(tid)

        # Process the requested batch.
        if dry_run:
            # Preview only — return the candidate list without applying.
            return self._json_response(200, {
                "convention": None,
                "fixed": 0,
                "skipped": skipped,
                "errors": [],
                "remaining": len(candidates),
                "done": False,
                "candidate_task_ids": candidate_task_ids,
                "dry_run": True,
            })
        if batch_size is not None:
            to_process = candidates[:batch_size]
        else:
            to_process = candidates
        fixed = 0
        errors: list[dict[str, str]] = []
        for task_id, span_form, current_type in to_process:
            try:
                hr.apply_posterior_fix(
                    task_id=task_id,
                    span=span_form,
                    current_type=current_type,
                    new_type=fix_new_type,
                    actor=actor,
                    save_as_convention=False,  # convention already declared above
                )
                fixed += 1
            except Exception as exc:  # noqa: BLE001
                errors.append({"task_id": task_id, "reason": str(exc)})
        # In scan mode, `remaining` reflects the global remainder. In
        # explicit-task_ids mode the caller knows the global picture
        # already, so `remaining` is just "what we couldn't process in
        # this call" — typically 0 since batch_size = len(task_ids).
        remaining = max(0, len(candidates) - len(to_process))

        # Cache surgery: drop deviations / contested entries this fix
        # resolves so the dashboard reflects the change on next GET
        # without forcing a full rescan.
        try:
            cached = read_posterior_audit_cache(store, project_id=project_id)
            if cached is not None:
                cache_payload = cached["payload"]
                devs = cache_payload.get("task_deviations", [])
                kept_devs = [
                    d for d in devs
                    if not (
                        (d.get("span") or "").lower() == span_lower
                    )
                ]
                divergent = cache_payload.get("divergent_entries", [])
                kept_divergent = [
                    c for c in divergent
                    if (c.get("span") or "").lower() != span_lower
                ]
                low_info = cache_payload.get("low_info_entries", [])
                kept_low_info = [
                    c for c in low_info
                    if (c.get("span") or "").lower() != span_lower
                ]
                cache_payload["task_deviations"] = kept_devs
                cache_payload["divergent_entries"] = kept_divergent
                cache_payload["low_info_entries"] = kept_low_info
                # PRESERVE the old accepted_hash so the dashboard's stale-
                # cache indicator fires: we've patched N task annotations
                # (changing their updated_at), so the cache is no longer
                # in sync with reality — even though we cleaned out the
                # specific deviations we just resolved, NEW divergences
                # may have been introduced (e.g. retro-patched span
                # interacts with other spans in the same task). Operator
                # should re-check to confirm.
                write_posterior_audit_cache(
                    store, project_id=project_id, payload=cache_payload,
                    accepted_hash=cached["accepted_hash"],
                    created_at=cached["created_at"],
                )
        except Exception:  # noqa: BLE001 — cache update is best-effort
            pass

        return self._json_response(200, {
            "convention": (
                None if conv is None else {
                    "convention_id": conv.convention_id,
                    "span": conv.span_original or span,
                    "entity_type": conv.entity_type,
                    "status": conv.status,
                }
            ),
            "fixed": fixed,
            "skipped": skipped,
            "errors": errors,
            "remaining": remaining,
            "done": remaining == 0,
            "candidate_task_ids": candidate_task_ids,
        })

    def _post_human_review_response(self, store: SqliteStore, task_id: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})
        try:
            picks = payload.get("picks")
            if not isinstance(picks, list):
                picks = None
            result = HumanReviewService(store).decide(
                task_id=task_id,
                action=str(payload.get("action") or ""),
                actor=str(payload.get("actor") or "human-reviewer"),
                feedback=str(payload.get("feedback") or ""),
                correction_mode=str(payload.get("correction_mode") or "manual_annotation"),
                picks=picks,
            )
        except FileNotFoundError:
            return self._json_response(404, {"error": "task_not_found"})
        except SchemaValidationError as exc:
            return self._json_response(
                422,
                {
                    "error": "validation_blocked",
                    "detail": str(exc),
                    "errors": exc.errors,
                },
            )
        except (InvalidTransition, ValueError) as exc:
            return self._json_response(400, {"error": "invalid_human_review_decision", "detail": str(exc)})
        return self._json_response(200, result.to_dict())

    def _post_human_review_correction(
        self, store: SqliteStore, task_id: str, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return self._json_response(400, {"error": "invalid_json"})
        if not isinstance(data, dict):
            return self._json_response(400, {"error": "invalid_json"})
        actor = data.get("actor")
        answer = data.get("answer")
        note = data.get("note")
        if not isinstance(actor, str) or not actor.strip():
            return self._json_response(400, {"error": "actor_required"})
        if not isinstance(answer, dict):
            return self._json_response(400, {"error": "answer_must_be_object"})

        svc = HumanReviewService(store)
        try:
            result = svc.submit_correction(task_id=task_id, answer=answer, actor=actor, note=note)
        except FileNotFoundError:
            return self._json_response(404, {"error": "task_not_found"})
        except InvalidTransition as exc:
            return self._json_response(409, {"error": "invalid_transition", "detail": str(exc)})
        except SchemaValidationError as exc:
            return self._json_response(400, {"error": "schema_validation_failed", "details": exc.errors})
        return self._json_response(200, result.to_dict())

    def _update_task_qc_policy_response(self, store: SqliteStore, task_id: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        try:
            task = store.load_task(task_id)
        except FileNotFoundError:
            return self._json_response(404, {"error": "task_not_found"})
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})

        try:
            policy = self._build_task_qc_policy(task, payload)
        except ValueError as exc:
            return self._json_response(400, {"error": "invalid_qc_policy", "detail": str(exc)})

        previous_policy = dict(task.metadata.get("qc_policy") or {})
        task.metadata["row_count"] = self._task_row_count(task)
        task.metadata["qc_policy"] = policy
        task.updated_at = utc_now()
        store.save_task(task)
        store.append_event(
            AuditEvent.new(
                task_id=task.task_id,
                previous_status=task.status,
                next_status=task.status,
                actor=str(payload.get("actor") or "algorithm-engineer"),
                reason="qc policy updated",
                stage="qc",
                metadata={"previous_qc_policy": previous_policy, "qc_policy": policy},
            )
        )
        return self._task_detail_response(store, task_id)

    def _build_task_qc_policy(self, task: Task, payload: dict) -> dict:
        row_count = self._task_row_count(task)
        mode = str(payload.get("mode") or "")
        if mode == "all_rows":
            return build_qc_policy(row_count=row_count)
        if mode == "sample_count":
            sample_count = payload.get("sample_count")
            if isinstance(sample_count, bool) or not isinstance(sample_count, int):
                raise ValueError("sample_count must be an integer")
            validate_qc_sample_options(sample_count, None)
            return build_qc_policy(row_count=row_count, qc_sample_count=sample_count)
        if mode == "sample_ratio":
            sample_ratio = payload.get("sample_ratio")
            if isinstance(sample_ratio, bool) or not isinstance(sample_ratio, (int, float)):
                raise ValueError("sample_ratio must be a number")
            validate_qc_sample_options(None, float(sample_ratio))
            return build_qc_policy(row_count=row_count, qc_sample_ratio=float(sample_ratio))
        raise ValueError("mode must be all_rows, sample_count, or sample_ratio")

    def _task_row_count(self, task: Task) -> int:
        metadata_row_count = task.metadata.get("row_count")
        if isinstance(metadata_row_count, int) and metadata_row_count >= 0:
            return metadata_row_count
        payload = task.source_ref.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return len(payload["rows"])
        return 1

    def _read_artifact_payload(self, store: SqliteStore, relative_path: str) -> Any:
        path = store.root / relative_path
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _annotators_response(self, store: SqliteStore) -> tuple[int, dict[str, str], bytes]:
        path = store.root / "annotators.yaml"
        data: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if isinstance(loaded, dict):
                    data = loaded
            except yaml.YAMLError:
                data = {}
        annotators_dict = data.get("annotators", {}) if isinstance(data.get("annotators"), dict) else {}
        annotators: list[dict[str, Any]] = []
        for annotator_id, body in annotators_dict.items():
            if not isinstance(body, dict):
                continue
            annotators.append({
                "id": str(annotator_id),
                "display_name": body.get("display_name", ""),
                "provider_target": body.get("provider_target", ""),
                "llm_profile": body.get("llm_profile", ""),
                "enabled": bool(body.get("enabled", True)),
                "modalities": list(body.get("modalities", []) or []),
                "annotation_types": list(body.get("annotation_types", []) or []),
                "input_artifact_kinds": list(body.get("input_artifact_kinds", []) or []),
                "output_artifact_kinds": list(body.get("output_artifact_kinds", []) or []),
                "preview_renderer_id": body.get("preview_renderer_id"),
            })
        sampling = data.get("sampling", {}) if isinstance(data.get("sampling"), dict) else {}
        available_profiles: list[str] = []
        targets: dict[str, str] = {}
        try:
            snap = build_provider_config_snapshot(
                store.root,
                workspace_root=self.workspace_root,
            )
            available_profiles = [str(p.get("name")) for p in snap.get("profiles", []) if p.get("name")]
            raw_targets = snap.get("targets") or {}
            if isinstance(raw_targets, dict):
                targets = {str(k): str(v) for k, v in raw_targets.items()}
        except (FileNotFoundError, ProfileValidationError):
            available_profiles = []
        return self._json_response(200, {
            "annotators": annotators,
            "sampling": sampling,
            "available_profiles": available_profiles,
            "stage_targets": targets,
        })

    def _update_annotators_response(self, store: SqliteStore, body: bytes) -> tuple[int, dict[str, str], bytes]:
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            return self._json_response(400, {"error": "invalid_json", "detail": str(exc)})
        if not isinstance(payload, dict):
            return self._json_response(400, {"error": "invalid_payload"})

        # Load existing annotators.yaml so we can preserve the annotators block
        # when the form only sends sampling / stage_targets edits. The selector
        # still needs the annotators dict on disk to route tasks.
        path = store.root / "annotators.yaml"
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if isinstance(loaded, dict):
                    existing = loaded
            except yaml.YAMLError:
                existing = {}

        # Annotators are optional in the payload. When omitted, keep whatever
        # is already on disk (the form no longer surfaces them for editing).
        if "annotators" in payload:
            annotators_input = payload["annotators"]
            if not isinstance(annotators_input, list):
                return self._json_response(400, {"error": "invalid_annotators"})
            annotators_dict: dict[str, Any] = {}
            for item in annotators_input:
                if not isinstance(item, dict):
                    continue
                annotator_id = str(item.get("id", "")).strip()
                if not annotator_id:
                    continue
                entry: dict[str, Any] = {
                    "display_name": item.get("display_name", ""),
                    "modalities": list(item.get("modalities", []) or []),
                    "annotation_types": list(item.get("annotation_types", []) or []),
                    "input_artifact_kinds": list(item.get("input_artifact_kinds", []) or []),
                    "output_artifact_kinds": list(item.get("output_artifact_kinds", []) or []),
                    "provider_target": item.get("provider_target", ""),
                    "enabled": bool(item.get("enabled", True)),
                }
                llm_profile = item.get("llm_profile")
                if llm_profile:
                    entry["llm_profile"] = llm_profile
                preview_renderer_id = item.get("preview_renderer_id")
                if preview_renderer_id:
                    entry["preview_renderer_id"] = preview_renderer_id
                annotators_dict[annotator_id] = entry
        else:
            existing_annotators = existing.get("annotators")
            annotators_dict = existing_annotators if isinstance(existing_annotators, dict) else {}

        sampling = payload.get("sampling", {})
        if not isinstance(sampling, dict):
            sampling = {}
        out: dict[str, Any] = {"annotators": annotators_dict}
        if sampling:
            out["sampling"] = sampling
        text = yaml.safe_dump(out, sort_keys=False, default_flow_style=False, allow_unicode=True)
        path.write_text(text, encoding="utf-8")

        # Optional: update workspace llm_profiles.yaml `targets` mapping when
        # the form sends stage_targets. This is the actual stage → profile
        # binding the runtime resolves at dispatch time (annotation/qc/arbiter/
        # coordinator). The per-annotator `llm_profile` field above is metadata
        # only; runtime ignores it.
        stage_targets = payload.get("stage_targets")
        if isinstance(stage_targets, dict) and stage_targets:
            try:
                self._update_stage_targets(stage_targets)
            except (OSError, yaml.YAMLError, ProfileValidationError) as exc:
                return self._json_response(
                    400,
                    {"error": "stage_targets_save_failed", "detail": str(exc)},
                )
        return self._json_response(200, {"ok": True})

    def _update_stage_targets(self, stage_targets: dict[str, Any]) -> None:
        """Merge stage→profile updates into the workspace llm_profiles.yaml.

        Loads the current file, validates each profile name exists, replaces
        the targets block, writes back. Other top-level keys (profiles, limits)
        are preserved verbatim.
        """
        from annotation_pipeline_skill.llm.profiles import LLM_PROFILES_FILENAME, resolve_llm_profiles_path

        existing_path = resolve_llm_profiles_path(workspace_root=self.workspace_root)
        if existing_path is None:
            existing_path = self.workspace_root / LLM_PROFILES_FILENAME
        raw: dict[str, Any] = {}
        if existing_path.exists():
            loaded = yaml.safe_load(existing_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                raw = loaded
        profiles = raw.get("profiles") or {}
        valid_names = set(profiles.keys()) if isinstance(profiles, dict) else set()
        clean_targets: dict[str, str] = {}
        for stage, profile_name in stage_targets.items():
            if not isinstance(stage, str) or not isinstance(profile_name, str):
                continue
            stage = stage.strip()
            profile_name = profile_name.strip()
            if not stage or not profile_name:
                continue
            if profile_name not in valid_names:
                raise ProfileValidationError(
                    f"stage target {stage} → {profile_name} references missing profile"
                )
            clean_targets[stage] = profile_name
        raw["targets"] = clean_targets
        target_path = self.workspace_root / LLM_PROFILES_FILENAME
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    def _guidelines_response(self, store: SqliteStore | None) -> dict[str, Any]:
        if store is None:
            return {"guidelines": []}
        # config.json lives in the project dir, one level above the store root
        config_path = store.root.parent / "config.json"
        if not config_path.exists():
            return {"guidelines": []}
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"guidelines": []}
        annotation_rules = config.get("annotation_rules", {})
        guidelines = []
        for label, value in annotation_rules.items():
            if not isinstance(value, str) or not value.endswith(".md"):
                continue
            path = Path(value)
            guidelines.append({
                "label": label,
                "path": value,
                "filename": path.name,
                "exists": path.exists(),
                "content": path.read_text(encoding="utf-8") if path.exists() else None,
            })
        return {"guidelines": guidelines}

    def _config_files(self, store: SqliteStore) -> list[dict[str, Any]]:
        files = []
        for config_id, title in CONFIG_FILE_DEFINITIONS.items():
            path = store.root / config_id
            files.append(
                {
                    "id": config_id,
                    "title": title,
                    "path": str(path),
                    "exists": path.exists(),
                    "content": path.read_text(encoding="utf-8") if path.exists() else "",
                }
            )
        return files

    def _update_config_response(self, store: SqliteStore, config_id: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        if config_id not in CONFIG_FILE_DEFINITIONS:
            return self._json_response(404, {"error": "config_not_found"})
        content = body.decode("utf-8")
        try:
            yaml.safe_load(content) if content.strip() else None
        except yaml.YAMLError as exc:
            return self._json_response(400, {"error": "invalid_yaml", "detail": str(exc)})
        path = store.root / config_id
        path.write_text(content, encoding="utf-8")
        return self._json_response(200, {"ok": True, "id": config_id})

MIME_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
}

_STATIC_ROOT: Path | None = None


def _find_static_root() -> Path | None:
    candidates = [
        Path(__file__).parent.parent.parent / "web" / "dist",
    ]
    for path in candidates:
        if (path / "index.html").exists():
            return path
    return None


def make_handler(api: DashboardApi, static_root: Path | None = None) -> type[BaseHTTPRequestHandler]:
    resolved_static = static_root or _find_static_root()

    class DashboardRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = self.path.split("?", 1)[0]
            if route.startswith("/api/"):
                status, headers, body = api.handle_get(self.path)
                self._send(status, headers, body)
                return
            if resolved_static is not None:
                self._serve_static(route)
                return
            status, headers, body = api.handle_get(self.path)
            self._send(status, headers, body)

        def _serve_static(self, route: str) -> None:
            assert resolved_static is not None
            rel = route.lstrip("/") or "index.html"
            candidate = (resolved_static / rel).resolve()
            if not str(candidate).startswith(str(resolved_static.resolve())):
                self._send(403, {}, b"Forbidden")
                return
            if not candidate.exists() or candidate.is_dir():
                candidate = resolved_static / "index.html"
            suffix = candidate.suffix.lower()
            content_type = MIME_TYPES.get(suffix, "application/octet-stream")
            body = candidate.read_bytes()
            # Cache strategy: index.html must always revalidate (entry
            # point — points to the current hashed asset URLs). Hashed
            # assets under /assets/ are safe to cache forever because
            # Vite changes the filename on rebuild. Without this, browsers
            # heuristic-cache the old JS for hours and users see stale
            # behavior even after the server has the new bundle.
            headers = {"content-type": content_type}
            name = candidate.name.lower()
            try:
                rel_resolved = candidate.relative_to(resolved_static.resolve()).as_posix()
            except ValueError:
                rel_resolved = name
            if name == "index.html":
                headers["cache-control"] = "no-cache, must-revalidate"
            elif rel_resolved.startswith("assets/"):
                headers["cache-control"] = "public, max-age=31536000, immutable"
            else:
                headers["cache-control"] = "no-cache"
            self._send(200, headers, body)

        def do_PUT(self) -> None:
            content_length = int(self.headers.get("content-length", "0"))
            request_body = self.rfile.read(content_length)
            status, headers, body = api.handle_put(self.path, request_body)
            self._send(status, headers, body)

        def do_POST(self) -> None:
            content_length = int(self.headers.get("content-length", "0"))
            request_body = self.rfile.read(content_length)
            status, headers, body = api.handle_post(self.path, request_body)
            self._send(status, headers, body)

        def do_DELETE(self) -> None:
            content_length = int(self.headers.get("content-length", "0"))
            request_body = self.rfile.read(content_length)
            status, headers, body = api.handle_delete(self.path, request_body)
            self._send(status, headers, body)

        def _send(self, status: int, headers: dict[str, str], body: bytes) -> None:
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardRequestHandler


def serve_dashboard_api(
    store: SqliteStore,
    host: str,
    port: int,
    *,
    stores: dict[str, SqliteStore] | None = None,
    default_store_key: str | None = None,
    runtime_once: Callable[[], RuntimeSnapshot] | None = None,
    runtime_config: RuntimeConfig | None = None,
    workspace_root: Path | None = None,
) -> None:
    server = ThreadingHTTPServer(
        (host, port),
        make_handler(DashboardApi(
            store,
            stores=stores,
            default_store_key=default_store_key,
            runtime_once=runtime_once,
            runtime_config=runtime_config,
            workspace_root=workspace_root,
        )),
    )
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local annotation dashboard API.")
    parser.add_argument("store_root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8509)
    args = parser.parse_args()
    serve_dashboard_api(SqliteStore.open(args.store_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

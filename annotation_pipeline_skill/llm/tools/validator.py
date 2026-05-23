"""Self-check tools for annotator + arbiter subagents.

Two pure functions:

- ``check_annotation_draft(store, args)`` — run the project's deterministic
  validators (schema / verbatim / cross-type / trailing-punct / row coverage)
  against a proposed annotation payload. Returns a structured violations
  dict so the agent can fix → re-check → submit in-session, avoiding the
  external mech_fail retry loop.
- ``lookup_row_text(store, args)`` — fetch one row's exact ``input.text`` so
  the agent can re-extract a verbatim span when ``check_annotation_draft``
  reports a verbatim_violation.

Tool schemas + dispatch wiring live in
``annotation_pipeline_skill.llm.tool_registry``.
"""
from __future__ import annotations

import logging
from typing import Any

from annotation_pipeline_skill.core.schema_validation import (
    SchemaValidationError,
    find_cross_type_collisions,
    find_trailing_punctuation_spans,
    find_verbatim_violations,
    validate_payload_against_task_schema,
)
from annotation_pipeline_skill.services.row_mask_service import apply_masks_to_task
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


logger = logging.getLogger("annotation_pipeline_skill.llm.tools.validator")


# Cap on per-violation-category list size returned to the LLM. Without a
# cap a schema-level error on a 50-row task can blow the tool result past
# the context window; the agent only needs a representative sample to act.
MAX_VIOLATIONS_PER_CATEGORY = 25


def check_annotation_draft(store: SqliteStore, arguments: dict) -> dict[str, Any]:
    """Run all mechanical validators on the draft payload."""
    task_id = arguments["task_id"]
    payload = arguments["payload"]
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        return {
            "ok": False,
            "error": (
                "payload must be an object with a 'rows' list — pass your full "
                'draft like {"rows": [{"row_index": 0, "row_id": "...", "output": '
                '{"entities": {}, "json_structures": {}}}, ...]}'
            ),
        }
    try:
        task = store.load_task(task_id)
    except (KeyError, FileNotFoundError):
        return {"ok": False, "error": f"task not found: {task_id}"}

    # Apply row masks: the annotator/arbiter prompts already have masked rows
    # filtered out, so they shouldn't be expected to produce them. Coverage
    # check must exclude masked rows for the same reason.
    masked_task = apply_masks_to_task(store, task)

    violations: dict[str, list] = {}

    # Schema validation FIRST. If types/keys are invalid the row-level
    # checks below may misclassify their failures, so resolve schema first
    # and let the agent fix shape issues before re-validating spans.
    try:
        validate_payload_against_task_schema(task, payload, store=store)
    except SchemaValidationError as exc:
        errs = getattr(exc, "errors", None) or [{"message": str(exc)}]
        violations["schema_errors"] = errs[:MAX_VIOLATIONS_PER_CATEGORY]

    vb = find_verbatim_violations(masked_task, payload)
    if vb:
        violations["verbatim_violations"] = vb[:MAX_VIOLATIONS_PER_CATEGORY]

    ct = find_cross_type_collisions(payload)
    if ct:
        violations["cross_type_collisions"] = ct[:MAX_VIOLATIONS_PER_CATEGORY]

    tp = find_trailing_punctuation_spans(masked_task, payload)
    if tp:
        violations["trailing_punctuation"] = tp[:MAX_VIOLATIONS_PER_CATEGORY]

    # Row coverage against the MASKED task. Masked rows are absent from the
    # prompt input, so the agent doesn't (and shouldn't) include them. Using
    # the unmasked task here would falsely flag every masked task.
    try:
        source_rows = masked_task.source_ref["payload"]["rows"]
        source_ids = {
            r["row_id"] for r in source_rows
            if isinstance(r, dict) and isinstance(r.get("row_id"), str)
        }
        corr_rows = payload.get("rows", [])
        corr_ids = {
            r["row_id"] for r in corr_rows
            if isinstance(r, dict) and isinstance(r.get("row_id"), str)
        }
        missing = sorted(source_ids - corr_ids)
        if missing:
            violations["row_coverage_missing"] = missing[:MAX_VIOLATIONS_PER_CATEGORY]
    except (KeyError, TypeError):
        # Tasks without a clean rows[] structure get a no-op coverage check
        # rather than a tool error — schema_errors above will already have
        # flagged the structural issue.
        pass

    ok = not violations
    return {
        "ok": ok,
        "violations": violations,
        "next_action": (
            "All checks passed — emit your final JSON now."
            if ok
            else "Fix the listed violations and call check_annotation_draft again."
        ),
    }


def lookup_row_text(store: SqliteStore, arguments: dict) -> dict[str, Any]:
    """Return one row's input + metadata."""
    task_id = arguments["task_id"]
    try:
        task = store.load_task(task_id)
    except (KeyError, FileNotFoundError):
        return {"error": f"task not found: {task_id}"}
    rows = (
        task.source_ref.get("payload", {}).get("rows", [])
        if isinstance(task.source_ref, dict)
        else []
    )
    if not isinstance(rows, list):
        return {"error": "task has no rows[]"}

    if "row_index" in arguments and arguments["row_index"] is not None:
        idx = arguments["row_index"]
        for r in rows:
            if isinstance(r, dict) and r.get("row_index") == idx:
                return _row_response(r)
        return {"error": f"row_index={idx} not found"}
    if "row_id" in arguments and arguments["row_id"]:
        rid = arguments["row_id"]
        for r in rows:
            if isinstance(r, dict) and r.get("row_id") == rid:
                return _row_response(r)
        return {"error": f"row_id={rid!r} not found"}
    return {"error": "specify row_index or row_id"}


def _row_response(row: dict) -> dict[str, Any]:
    return {
        "row_index": row.get("row_index"),
        "row_id": row.get("row_id"),
        "input": row.get("input"),
    }

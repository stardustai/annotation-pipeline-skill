"""Stdio MCP server exposing draft self-check tools for the annotation pipeline.

Launched by Claude CLI via ``--mcp-config``. Holds a read-only SqliteStore
connection and exposes two tools the annotator / arbiter can call BEFORE
emitting their final JSON:

- ``check_annotation_draft`` — run the project's deterministic checks
  (schema, verbatim, cross-type collisions, trailing-punct boundary, row
  coverage) against a proposed annotation payload. Returns the structured
  violations list. The agent iterates: fix → re-check → submit only when
  ``ok=true``. Catches the "non-verbatim span in an unwatched row"
  failure mode that today causes a wasteful mech_fail retry loop.
- ``lookup_row_text`` — fetch one row's exact ``input.text`` plus metadata.
  Use when ``check_annotation_draft`` reports a verbatim violation: the
  agent reads the original text and re-extracts the span byte-for-byte
  instead of guessing.

Invocation:
    python -m annotation_pipeline_skill.mcp.validator_server \\
        --project-root <annotation-pipeline workspace>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from annotation_pipeline_skill.core.schema_validation import (
    SchemaValidationError,
    find_cross_type_collisions,
    find_trailing_punctuation_spans,
    find_verbatim_violations,
    validate_payload_against_task_schema,
)
from annotation_pipeline_skill.services.row_mask_service import apply_masks_to_task
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


logger = logging.getLogger("annotation_validator_mcp")


# Cap on per-violation-category list size returned to the LLM. Without a
# cap a schema-level error on a 50-row task can blow the tool result past
# the context window; the agent only needs a representative sample to act.
MAX_VIOLATIONS_PER_CATEGORY = 25


def build_server(*, project_root: Path) -> Server:
    server: Server = Server("annotation-validator")
    store = SqliteStore.open(project_root)

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="check_annotation_draft",
                description=(
                    "Validate a draft annotation against the project's deterministic "
                    "checks BEFORE you submit your final JSON. Returns the list of "
                    "violations the pipeline will reject (so you can fix them in this "
                    "same session instead of getting bounced back). Call this on your "
                    "FULL proposed {rows: [...]} payload — all source rows, not just "
                    "the ones you changed. On any non-empty violations list, fix your "
                    "draft and call again. Submit your final JSON only when ok=true. "
                    "Checks: schema validation (entity/structure type names match the "
                    "enum, required fields present), verbatim (every span is a "
                    "byte-for-byte substring of the matching row's input.text), "
                    "cross-type collisions (same span tagged as two entity types in "
                    "one row), trailing-punctuation boundary spans, row coverage "
                    "(every non-masked source row is present)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task ID this draft is for (matches "
                                           "the task_id in the prompt input).",
                        },
                        "payload": {
                            "type": "object",
                            "description": (
                                'Your draft annotation, shape {"rows": [{"row_index": '
                                'int, "row_id": str, "output": {"entities": {...}, '
                                '"json_structures": {...}}}, ...]}. Include every row, '
                                "not a slim subset."
                            ),
                        },
                    },
                    "required": ["task_id", "payload"],
                },
            ),
            Tool(
                name="lookup_row_text",
                description=(
                    "Fetch the exact input.text and metadata for one row of a task. "
                    "Use this when check_annotation_draft reports a verbatim "
                    "violation: read the original text and re-extract the span "
                    "byte-for-byte (the pipeline matches by substring with no "
                    "normalization). Specify either row_index (0-based int) OR "
                    "row_id (string); row_index is faster."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "row_index": {
                            "type": "integer",
                            "description": "0-based row index.",
                        },
                        "row_id": {"type": "string"},
                    },
                    "required": ["task_id"],
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "check_annotation_draft":
                payload = check_annotation_draft(store, arguments)
            elif name == "lookup_row_text":
                payload = lookup_row_text(store, arguments)
            else:
                payload = {"error": f"unknown tool: {name}"}
        except KeyError as exc:
            payload = {"error": f"missing argument: {exc.args[0]}"}
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            logger.exception("validator tool %s failed", name, exc_info=exc)
            payload = {"error": f"tool failure: {type(exc).__name__}: {exc!s}"}
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    return server


def check_annotation_draft(store: SqliteStore, arguments: dict) -> dict[str, Any]:
    """Run all mechanical validators on the draft payload. Pure function so
    tests can hit it without spinning up the MCP server."""
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
    """Return one row's input + metadata. The agent uses this to re-extract
    verbatim spans without dumping the entire task back into the prompt."""
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="annotation-validator-mcp-server")
    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Path to the annotation-pipeline workspace root (contains db.sqlite).",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    server = build_server(project_root=args.project_root)

    async def _run() -> None:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()

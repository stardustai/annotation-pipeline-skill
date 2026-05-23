"""Tool schemas + dispatch wiring for the Anthropic SDK client.

Source of truth for every tool the annotator / QC / arbiter subagents can
call. Schemas use the Anthropic-API shape ``{name, description, input_schema}``.
Tool names are bare identifiers (``check_annotation_draft``) — historically
they carried an ``mcp__<server>__<tool>`` prefix because claude CLI rewrote
MCP tool names that way before exposing them to the model. The SDK runtime
calls the tools directly so the prefix served no purpose; dropped.

A profile's ``tools`` field is a list of group names (``"annotation-validator"``,
``"annotation-kb"``). ``build_tool_registry`` looks up which schemas + dispatch
callables to register for that group.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, NamedTuple

from annotation_pipeline_skill.llm.tools.check_past_experience import (
    check_past_experience,
)
from annotation_pipeline_skill.llm.tools.validator import (
    check_annotation_draft,
    lookup_row_text,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


# ---- Tool schemas (Anthropic-API shape: {name, description, input_schema}) ----

CHECK_ANNOTATION_DRAFT_SCHEMA: dict[str, Any] = {
    "name": "check_annotation_draft",
    "description": (
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
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": (
                    "The task ID this draft is for (matches the task_id "
                    "in the prompt input)."
                ),
            },
            "payload": {
                "type": "object",
                "description": (
                    'Your draft annotation, shape {"rows": [{"row_index": '
                    'int, "row_id": str, "output": {"entities": {...}, '
                    '"json_structures": {...}}}, ...]}. Include every '
                    "row, not a slim subset."
                ),
            },
        },
        "required": ["task_id", "payload"],
    },
}


LOOKUP_ROW_TEXT_SCHEMA: dict[str, Any] = {
    "name": "lookup_row_text",
    "description": (
        "Fetch the exact input.text and metadata for one row of a task. "
        "Use this when check_annotation_draft reports a verbatim "
        "violation: read the original text and re-extract the span "
        "byte-for-byte (the pipeline matches by substring with no "
        "normalization). Specify either row_index (0-based int) OR "
        "row_id (string); row_index is faster."
    ),
    "input_schema": {
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
}


CHECK_PAST_EXPERIENCE_SCHEMA: dict[str, Any] = {
    "name": "check_past_experience",
    "description": (
        "Query the project's annotation history for a candidate "
        "entity/span. Returns the current convention (if any), the "
        "distribution of past type proposals, up to 3 diverse sentence-"
        "level examples per type, and a wordfreq Zipf score. Use this "
        "BEFORE deciding the type of an ambiguous or unfamiliar span — "
        "past decisions and concrete row examples beat statistical "
        "summaries for in-context generalization."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entry": {
                "type": "string",
                "description": (
                    "The candidate span text (case-insensitive lookup)."
                ),
            },
        },
        "required": ["entry"],
    },
}


# ---- Registry --------------------------------------------------------------

ToolDispatch = Callable[[dict[str, Any]], Awaitable[Any]]


class ToolEntry(NamedTuple):
    schema: dict[str, Any]
    dispatch: ToolDispatch


def build_tool_registry(
    *,
    store: SqliteStore | None,
    project_id: str | None,
    tool_group_names: set[str],
) -> dict[str, ToolEntry]:
    """Build ``{tool_name: ToolEntry}`` based on the tool groups a profile
    enabled. A profile that declares no groups gets an empty registry; the
    SDK client sends ``tools=[]`` and the model can't call any tool.
    """
    registry: dict[str, ToolEntry] = {}

    if "annotation-validator" in tool_group_names:
        if store is None:
            raise ValueError(
                "annotation-validator tools require a store; "
                "pass store=... when constructing AnthropicSDKClient"
            )

        async def _check_draft(args: dict[str, Any]) -> Any:
            return check_annotation_draft(store, args)

        async def _lookup_row(args: dict[str, Any]) -> Any:
            return lookup_row_text(store, args)

        registry[CHECK_ANNOTATION_DRAFT_SCHEMA["name"]] = ToolEntry(
            schema=CHECK_ANNOTATION_DRAFT_SCHEMA,
            dispatch=_check_draft,
        )
        registry[LOOKUP_ROW_TEXT_SCHEMA["name"]] = ToolEntry(
            schema=LOOKUP_ROW_TEXT_SCHEMA,
            dispatch=_lookup_row,
        )

    if "annotation-kb" in tool_group_names:
        if store is None or not project_id:
            raise ValueError(
                "annotation-kb tools require a store and project_id; "
                "pass them when constructing AnthropicSDKClient"
            )

        async def _check_kb(args: dict[str, Any]) -> Any:
            return check_past_experience(
                store, project_id=project_id, entry=args["entry"]
            )

        registry[CHECK_PAST_EXPERIENCE_SCHEMA["name"]] = ToolEntry(
            schema=CHECK_PAST_EXPERIENCE_SCHEMA,
            dispatch=_check_kb,
        )

    return registry

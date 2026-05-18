from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from annotation_pipeline_skill.core.models import Task

if TYPE_CHECKING:
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore


PROJECT_SCHEMA_FILENAME = "output_schema.json"


class SchemaValidationError(ValueError):
    def __init__(self, message: str, errors: list[dict]):
        super().__init__(message)
        self.errors = errors


def load_output_schema(task: Task) -> dict | None:
    """Return inline per-task output_schema if present.

    Backward compat path for tasks imported before the schema moved to the
    project level. Callers that want full resolution (with project fallback)
    should use :func:`resolve_output_schema`.
    """
    payload = task.source_ref.get("payload") if isinstance(task.source_ref, dict) else None
    if not isinstance(payload, dict):
        return None
    guidance = payload.get("annotation_guidance")
    if not isinstance(guidance, dict):
        return None
    schema = guidance.get("output_schema")
    return schema if isinstance(schema, dict) else None


def load_project_output_schema(project_config_root: Path) -> dict | None:
    """Read ``<project_config_root>/output_schema.json`` if it exists."""
    path = Path(project_config_root) / PROJECT_SCHEMA_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def resolve_output_schema(task: Task, store: "SqliteStore | None") -> dict | None:
    """Resolve a task's output schema, preferring inline then project-level."""
    inline = load_output_schema(task)
    if inline is not None:
        return inline
    if store is None:
        return None
    return load_project_output_schema(store.root)


def find_verbatim_violations(
    task: "Task",
    payload: Any,
) -> "list[dict[str, Any]]":
    """Walk every entity and json_structures phrase in ``payload`` and return
    violations: spans that aren't a verbatim substring of the corresponding
    row's input.text. Returns an empty list when everything is verbatim.

    Mirrors the runtime's pipeline-side verbatim check; lifted here so HR /
    audit / arbiter paths can use the same logic without touching the
    SubagentRuntime instance.
    """
    if not isinstance(payload, dict):
        return []
    rows_out = payload.get("rows")
    if not isinstance(rows_out, list):
        return []
    source_payload = task.source_ref.get("payload") if isinstance(task.source_ref, dict) else None
    if not isinstance(source_payload, dict):
        return []
    source_rows = source_payload.get("rows")
    if not isinstance(source_rows, list):
        return []
    input_by_index: dict[int, str] = {}
    for i, r in enumerate(source_rows):
        if not isinstance(r, dict):
            continue
        idx = r.get("row_index") if isinstance(r.get("row_index"), int) else i
        text = r.get("input")
        if isinstance(text, str):
            input_by_index[idx] = text
    violations: list[dict[str, Any]] = []
    for r in rows_out:
        if not isinstance(r, dict):
            continue
        row_index = r.get("row_index") if isinstance(r.get("row_index"), int) else 0
        input_text = input_by_index.get(row_index)
        if not input_text:
            continue
        output = r.get("output")
        if not isinstance(output, dict):
            continue
        for typ_dict_key in ("entities", "json_structures"):
            type_dict = output.get(typ_dict_key)
            if not isinstance(type_dict, dict):
                continue
            for type_name, items in type_dict.items():
                if not isinstance(items, list):
                    continue
                for span in items:
                    if isinstance(span, str) and span and span not in input_text:
                        violations.append({
                            "row_index": row_index,
                            "field": f"{typ_dict_key}.{type_name}",
                            "span": span,
                        })
    return violations


_VERBATIM_ALIGN_TRIM_CHARS = " \t\n\r　.,;:!?。，；：！？\"'`“”‘’«»【】「」"


def try_align_to_verbatim(span: str, input_text: str) -> str | None:
    """Return a verbatim form of ``span`` if ``span`` is almost-verbatim in
    ``input_text`` but differs only by surrounding whitespace, punctuation,
    or quote characters. Returns ``None`` if the span cannot be aligned
    safely.

    Safety contract — alignment may ONLY remove characters from either end
    of ``span`` (whitespace / sentence punctuation / quote chars). It may
    NOT change letters, digits, or any internal character. This guarantees
    we never silently rewrite the semantic content of an entity. The
    caller can show the alignment to an auditor and trust it didn't change
    what the span means — only where the span starts/ends.

    Aligned result must be a non-empty substring of ``input_text``.
    """
    if not isinstance(span, str) or not span or not isinstance(input_text, str) or not input_text:
        return None
    if span in input_text:
        return None  # already verbatim, caller shouldn't have asked
    stripped = span.strip(_VERBATIM_ALIGN_TRIM_CHARS)
    if stripped and stripped != span and stripped in input_text:
        return stripped
    return None


def find_duplicate_spans(payload: Any) -> "list[dict[str, Any]]":
    """Return any spans that appear more than once under the same
    entities/json_structures type within a single row. Always a model
    quality issue (the type set should be deduped); safe to auto-fix
    at write time, but worth surfacing as a WARNING so the annotator
    sees it in the next round's feedback bundle.
    """
    if not isinstance(payload, dict):
        return []
    rows_out = payload.get("rows")
    if not isinstance(rows_out, list):
        return []
    dups: list[dict[str, Any]] = []
    for r in rows_out:
        if not isinstance(r, dict):
            continue
        row_index = r.get("row_index") if isinstance(r.get("row_index"), int) else 0
        output = r.get("output")
        if not isinstance(output, dict):
            continue
        for typ_dict_key in ("entities", "json_structures"):
            type_dict = output.get(typ_dict_key)
            if not isinstance(type_dict, dict):
                continue
            for type_name, items in type_dict.items():
                if not isinstance(items, list):
                    continue
                seen: set[str] = set()
                for span in items:
                    if not isinstance(span, str):
                        continue
                    if span in seen:
                        dups.append({
                            "row_index": row_index,
                            "field": f"{typ_dict_key}.{type_name}",
                            "span": span,
                        })
                    else:
                        seen.add(span)
    return dups


_TRAILING_SENTENCE_PUNCT = ".,;:!?。，；：！？"


def find_trailing_punctuation_spans(task: "Task", payload: Any) -> "list[dict[str, Any]]":
    """Return entity / json_structures spans that end with sentence-ending
    punctuation when the same string without the trailing punctuation is
    also a verbatim substring of input.text.

    Rule: the entity is the name itself, not the sentence boundary. If the
    annotator emits "Mitul Mallik." (with period) and the row's input also
    contains "Mitul Mallik" (without), the trailing period is sentence
    punctuation rather than part of the entity → BLOCK so the annotator
    re-emits the trimmed form.

    No-op when the trimmed form isn't in input (the period may genuinely
    be part of an abbreviation like "Inc." with no "Inc" elsewhere).
    """
    if not isinstance(payload, dict):
        return []
    rows_out = payload.get("rows")
    if not isinstance(rows_out, list):
        return []
    source_payload = task.source_ref.get("payload") if isinstance(task.source_ref, dict) else None
    if not isinstance(source_payload, dict):
        return []
    source_rows = source_payload.get("rows")
    if not isinstance(source_rows, list):
        return []
    input_by_index: dict[int, str] = {}
    for i, r in enumerate(source_rows):
        if not isinstance(r, dict):
            continue
        idx = r.get("row_index") if isinstance(r.get("row_index"), int) else i
        text = r.get("input")
        if isinstance(text, str):
            input_by_index[idx] = text
    findings: list[dict[str, Any]] = []
    for r in rows_out:
        if not isinstance(r, dict):
            continue
        row_index = r.get("row_index") if isinstance(r.get("row_index"), int) else 0
        input_text = input_by_index.get(row_index)
        if not input_text:
            continue
        output = r.get("output")
        if not isinstance(output, dict):
            continue
        for typ_dict_key in ("entities", "json_structures"):
            type_dict = output.get(typ_dict_key)
            if not isinstance(type_dict, dict):
                continue
            for type_name, items in type_dict.items():
                if not isinstance(items, list):
                    continue
                for span in items:
                    if not isinstance(span, str) or not span:
                        continue
                    if span[-1] not in _TRAILING_SENTENCE_PUNCT:
                        continue
                    trimmed = span.rstrip(_TRAILING_SENTENCE_PUNCT)
                    if not trimmed or trimmed == span:
                        continue
                    if trimmed not in input_text:
                        # Trailing punct is genuinely part of the entity
                        # (abbreviation like "Inc.") — skip.
                        continue
                    findings.append({
                        "row_index": row_index,
                        "field": f"{typ_dict_key}.{type_name}",
                        "span": span,
                        "trimmed": trimmed,
                    })
    return findings


def find_cross_type_collisions(payload: Any) -> "list[dict[str, Any]]":
    """Return entity spans tagged under two or more entity types in the
    same row. The schema permits this (different types are separate keys),
    but in practice a span should resolve to one entity type per
    occurrence; collisions usually mean the annotator hedged. Treated as
    BLOCKING at validation time so the annotator picks one. Only checks
    ``entities`` (json_structures is allowed to overlap by design — a
    phrase can be both a "goal" and a "constraint").
    """
    if not isinstance(payload, dict):
        return []
    rows_out = payload.get("rows")
    if not isinstance(rows_out, list):
        return []
    collisions: list[dict[str, Any]] = []
    for r in rows_out:
        if not isinstance(r, dict):
            continue
        row_index = r.get("row_index") if isinstance(r.get("row_index"), int) else 0
        output = r.get("output")
        if not isinstance(output, dict):
            continue
        entities = output.get("entities")
        if not isinstance(entities, dict):
            continue
        seen_at: dict[str, str] = {}
        for type_name, items in entities.items():
            if not isinstance(items, list):
                continue
            for span in items:
                if not isinstance(span, str):
                    continue
                if span in seen_at and seen_at[span] != type_name:
                    collisions.append({
                        "row_index": row_index,
                        "span": span,
                        "types": [seen_at[span], type_name],
                    })
                else:
                    seen_at[span] = type_name
    return collisions


def validate_payload_against_task_schema(
    task: Task,
    payload: Any,
    *,
    store: "SqliteStore | None" = None,
) -> None:
    schema = resolve_output_schema(task, store)
    if schema is None:
        raise SchemaValidationError(
            "task has no output_schema",
            [{"kind": "missing_schema", "message": "task has no output_schema"}],
        )
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    if errors:
        raise SchemaValidationError(
            f"schema validation failed with {len(errors)} error(s)",
            [
                {
                    "kind": "schema_error",
                    "path": "/".join(str(p) for p in err.absolute_path),
                    "message": err.message,
                }
                for err in errors
            ],
        )

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
    of ``span`` (whitespace / sentence punctuation / quote chars) OR
    re-insert whitespace that's literally present in ``input_text`` at the
    matching position. It may NOT change letters, digits, or any internal
    non-whitespace character. This guarantees we never silently rewrite the
    semantic content of an entity. The caller can show the alignment to an
    auditor and trust it didn't change what the span means — only where the
    span starts/ends or how internal whitespace breaks up the characters.

    Aligned result must be a non-empty substring of ``input_text``.
    """
    if not isinstance(span, str) or not span or not isinstance(input_text, str) or not input_text:
        return None
    if span in input_text:
        return None  # already verbatim, caller shouldn't have asked
    stripped = span.strip(_VERBATIM_ALIGN_TRIM_CHARS)
    if stripped and stripped != span and stripped in input_text:
        return stripped
    # Internal-whitespace recovery: the LLM collapsed whitespace inside the
    # span (common when the input is space-separated for visual layout, e.g.
    # CJK OCR output "1 7 6 4"). Re-insert exactly the whitespace that's in
    # the input. SAFE because:
    #   (a) we never add a non-whitespace char that's not already in input
    #   (b) the result is by construction a substring of input_text
    #   (c) the non-whitespace character sequence is preserved end-to-end
    aligned = _align_collapsing_whitespace(stripped or span, input_text)
    if aligned is not None:
        return aligned
    return None


def _align_collapsing_whitespace(span: str, input_text: str) -> str | None:
    """If ``span`` is the whitespace-collapsed form of some substring of
    ``input_text``, return that substring (with input's whitespace preserved).
    Otherwise return ``None``.

    Example: span="1764", input_text="...年（1 7 6 4 年）..." → "1 7 6 4"

    Determinism: returns the FIRST match in input_text.
    Safety: by construction the returned string is a slice of input_text,
    and its non-whitespace characters equal those of ``span``.
    """
    # Strip whitespace from span to get the character sequence we need to
    # match (ignoring all whitespace).
    span_no_ws = "".join(ch for ch in span if not ch.isspace())
    if not span_no_ws:
        return None
    n = len(input_text)
    target = span_no_ws
    tlen = len(target)
    for start in range(n):
        if input_text[start].isspace():
            continue
        # Try to match target starting here, skipping whitespace in input.
        i = start
        j = 0
        last_nonws = start - 1
        while i < n and j < tlen:
            ch = input_text[i]
            if ch.isspace():
                i += 1
                continue
            if ch != target[j]:
                break
            last_nonws = i
            i += 1
            j += 1
        if j == tlen:
            # Matched all of target. The span is input_text[start : last_nonws+1].
            candidate = input_text[start:last_nonws + 1]
            # Sanity: don't return the original if it happens to equal span
            # (then it'd be verbatim already and we wouldn't have been called).
            if candidate and candidate != span:
                return candidate
    return None


def auto_fix_shared_type_field_in_place(payload: Any) -> int:
    """Move spans of a SHARED type to the correct field based on the
    word-count rule. Returns the number of (row, span) relocations.

    - multi-word span in ``entities.<shared>``  → move to ``json_structures.<shared>``
    - single-word span in ``json_structures.<shared>`` → move to ``entities.<shared>``
    - same span in both fields → keep only the correct one

    Letter content untouched; the span string is identical, only the
    containing field changes. Safe to run before other validators.
    """
    if not isinstance(payload, dict):
        return 0
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return 0
    moves = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        output = r.get("output")
        if not isinstance(output, dict):
            continue
        entities = output.setdefault("entities", {}) if isinstance(output.get("entities"), dict) or output.get("entities") is None else None
        phrases = output.setdefault("json_structures", {}) if isinstance(output.get("json_structures"), dict) or output.get("json_structures") is None else None
        if not isinstance(entities, dict) or not isinstance(phrases, dict):
            continue
        for typ in _SHARED_TYPES:
            e_list = entities.get(typ)
            p_list = phrases.get(typ)
            if not isinstance(e_list, list):
                e_list = None
            if not isinstance(p_list, list):
                p_list = None
            if e_list is None and p_list is None:
                continue
            # Collect what we have, then re-partition by word count.
            e_set = list(e_list or [])
            p_set = list(p_list or [])
            union = []
            seen_union: set[str] = set()
            for s in e_set + p_set:
                if isinstance(s, str) and s.strip() and s not in seen_union:
                    seen_union.add(s)
                    union.append(s)
            new_entities: list[str] = []
            new_phrases: list[str] = []
            for s in union:
                if _word_count(s) == 1:
                    new_entities.append(s)
                else:
                    new_phrases.append(s)
            # Compute whether anything actually moved (compared to original
            # exact lists).
            before = (tuple(e_list or ()), tuple(p_list or ()))
            after = (tuple(new_entities), tuple(new_phrases))
            if before != after:
                # Count items that switched fields (set-symmetric-diff over
                # field membership).
                e_before, p_before = set(e_set), set(p_set)
                e_after, p_after = set(new_entities), set(new_phrases)
                moves += len((e_after - e_before) | (p_after - p_before))
                if new_entities:
                    entities[typ] = new_entities
                elif typ in entities:
                    del entities[typ]
                if new_phrases:
                    phrases[typ] = new_phrases
                elif typ in phrases:
                    del phrases[typ]
    return moves


def auto_fix_safe_spans_in_place(task: "Task", payload: Any) -> int:
    """Mutate ``payload`` in place: rewrite spans whose only defect is
    surrounding whitespace / sentence punctuation / quote characters into
    the form that's both verbatim in input.text AND not trailing-punct
    flagged, AND relocate shared-type spans (``technology``) to the
    correct field by word count. Returns the total count of rewrites +
    relocations.

    Boundary rules (preserve letters, only move start/end):

    1. If the span is non-verbatim but ``try_align_to_verbatim`` yields a
       verbatim form that differs only by trim-safe chars, use that. Catches
       things like ``"Mitul Mallik."`` when input has ``"Mitul Mallik "`` or
       wrapping quotes ``"foo"`` when input has ``foo``.
    2. If the span IS already verbatim but its punct-trimmed form is ALSO
       verbatim, prefer the trimmed form. Mirrors
       ``find_trailing_punctuation_spans`` semantics: the entity is the name,
       not the sentence boundary.

    Field-routing rule (preserve content + type, only move which field):

    3. Spans of a SHARED type (only ``technology`` today) get routed to the
       field implied by word count: single-word → ``entities.<type>``;
       multi-word → ``json_structures.<type>``. Same span appearing in both
       fields gets collapsed to the correct one.

    Letter-level edits are NEVER performed; type changes are never inferred.
    Spans that can't be safely fixed are left untouched and will surface to
    the normal validation failure path.
    """
    if not isinstance(payload, dict):
        return 0
    source_payload = task.source_ref.get("payload") if isinstance(task.source_ref, dict) else None
    if not isinstance(source_payload, dict):
        return 0
    source_rows = source_payload.get("rows")
    if not isinstance(source_rows, list):
        return 0
    input_by_index: dict[int, str] = {}
    for i, r in enumerate(source_rows):
        if not isinstance(r, dict):
            continue
        idx = r.get("row_index") if isinstance(r.get("row_index"), int) else i
        text = r.get("input")
        if isinstance(text, str):
            input_by_index[idx] = text
    rows_out = payload.get("rows")
    if not isinstance(rows_out, list):
        return 0
    rewrites = 0
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
                for i, span in enumerate(items):
                    if not isinstance(span, str) or not span:
                        continue
                    if span in input_text:
                        # Already verbatim — preempt trailing-punct flag if
                        # the trimmed form is also verbatim.
                        trimmed = span.rstrip(_TRAILING_SENTENCE_PUNCT)
                        if trimmed and trimmed != span and trimmed in input_text:
                            items[i] = trimmed
                            rewrites += 1
                        continue
                    aligned = try_align_to_verbatim(span, input_text)
                    if aligned is not None:
                        items[i] = aligned
                        rewrites += 1
    # Shared-type field routing runs AFTER span-boundary alignment so
    # trim-safe rewrites are settled before we count words.
    rewrites += auto_fix_shared_type_field_in_place(payload)
    return rewrites


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


# Types that exist in BOTH `entityType` and `jsonStructureType` enums.
# These are the only types where a span could legitimately be placed in
# either field — and therefore the only types that need the word-count
# routing rule. Adding more shared types here should be safe (the validator
# just considers more types) but in practice the schema designer makes
# this list very short on purpose.
_SHARED_TYPES = ("technology",)


def _word_count(span: str) -> int:
    """Cheap whitespace-delimited word count. Works for English-style spans;
    CJK runs are typically one 'word' under this measure, which is the
    behavior we want for the entity-vs-phrase routing rule (a single
    Chinese name = entity, a multi-character sentence = phrase)."""
    return len([w for w in span.strip().split() if w])


def _schema_type_enums(schema: Any) -> tuple[set[str], set[str]]:
    """Extract (entityType_enum, jsonStructureType_enum) from a resolved
    output_schema. Returns empty sets if the schema is missing or not
    structured as expected. Used by apply-path routing to decide which
    field a (span, type) decision should land in.

    Handles two shapes:
      - ``{"enum": ["person", "organization", ...]}``
      - ``{"oneOf": [{"const": "person", "description": "..."}, ...]}``
    The schema authoring tool emits the oneOf form so model-facing
    descriptions ride along; earlier versions used the bare enum form.
    """
    if not isinstance(schema, dict):
        return set(), set()
    defs = schema.get("$defs") or schema.get("definitions") or {}
    entity_def = defs.get("entityType") if isinstance(defs, dict) else None
    phrase_def = defs.get("jsonStructureType") if isinstance(defs, dict) else None

    def _values(defn: Any) -> set[str]:
        if not isinstance(defn, dict):
            return set()
        bare = defn.get("enum")
        if isinstance(bare, list):
            return {v for v in bare if isinstance(v, str)}
        one_of = defn.get("oneOf")
        if isinstance(one_of, list):
            return {
                item["const"]
                for item in one_of
                if isinstance(item, dict) and isinstance(item.get("const"), str)
            }
        return set()

    return _values(entity_def), _values(phrase_def)


def resolve_apply_field(
    span: str, target_type: str, schema: Any,
) -> "tuple[str | None, str | None]":
    """Decide which annotation field (entities or json_structures) a
    (span, target_type) operator decision should land in.

    Returns ``(field_key, error)``:
      - ``field_key`` is "entities" or "json_structures" on success, None on error
      - ``error`` is a human-readable reason string on rejection, None on success

    Rules:
      - target_type in entityType only → "entities"
      - target_type in jsonStructureType only → "json_structures"
      - target_type in BOTH (shared, currently just "technology") → word count
        decides: single-word → "entities", multi-word → "json_structures"
      - target_type in NEITHER → reject (unknown type for this project)
    """
    if not isinstance(target_type, str) or not target_type.strip():
        return None, "target_type is required"
    entity_enum, phrase_enum = _schema_type_enums(schema)
    if not entity_enum and not phrase_enum:
        # No schema available — fall back to word-count heuristic so we
        # don't block the apply just because the schema couldn't be
        # resolved. The downstream verbatim/schema check still catches
        # truly invalid types.
        return ("entities" if _word_count(span) == 1 else "json_structures"), None
    in_entity = target_type in entity_enum
    in_phrase = target_type in phrase_enum
    if in_entity and in_phrase:
        return ("entities" if _word_count(span) == 1 else "json_structures"), None
    if in_entity:
        return "entities", None
    if in_phrase:
        return "json_structures", None
    return None, (
        f"target_type {target_type!r} is not defined in this project's "
        f"entityType or jsonStructureType schema"
    )


def find_shared_type_field_violations(payload: Any) -> "list[dict[str, Any]]":
    """Detect spans of a SHARED type (currently just ``technology``) placed
    in the wrong field per the word-count routing rule:

      - single-word span → must live in ``entities.<type>``
      - multi-word span  → must live in ``json_structures.<type>``

    Also flags the cross-field collision case: the same span appearing in
    both ``entities.<type>`` AND ``json_structures.<type>`` within one row.

    Returned violation dicts have ``row_index``, ``span``, ``shared_type``,
    ``current_field``, ``correct_field``, and ``kind`` keys.
    """
    if not isinstance(payload, dict):
        return []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        row_index = r.get("row_index") if isinstance(r.get("row_index"), int) else 0
        output = r.get("output")
        if not isinstance(output, dict):
            continue
        entities = output.get("entities") if isinstance(output.get("entities"), dict) else {}
        phrases = output.get("json_structures") if isinstance(output.get("json_structures"), dict) else {}
        for typ in _SHARED_TYPES:
            in_entities = [s for s in (entities.get(typ) or []) if isinstance(s, str) and s.strip()]
            in_phrases = [s for s in (phrases.get(typ) or []) if isinstance(s, str) and s.strip()]
            both = set(in_entities) & set(in_phrases)
            for span in sorted(both):
                out.append({
                    "row_index": row_index, "span": span, "shared_type": typ,
                    "kind": "cross_field_collision",
                    "current_field": "both",
                    "correct_field": (
                        f"entities.{typ}" if _word_count(span) == 1 else f"json_structures.{typ}"
                    ),
                })
            for span in in_entities:
                if span in both:
                    continue
                if _word_count(span) > 1:
                    out.append({
                        "row_index": row_index, "span": span, "shared_type": typ,
                        "kind": "multiword_in_entities",
                        "current_field": f"entities.{typ}",
                        "correct_field": f"json_structures.{typ}",
                    })
            for span in in_phrases:
                if span in both:
                    continue
                if _word_count(span) == 1:
                    out.append({
                        "row_index": row_index, "span": span, "shared_type": typ,
                        "kind": "singleword_in_json_structures",
                        "current_field": f"json_structures.{typ}",
                        "correct_field": f"entities.{typ}",
                    })
    return out


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

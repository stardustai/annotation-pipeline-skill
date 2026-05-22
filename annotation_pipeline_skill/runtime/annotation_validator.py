"""AnnotationValidator — validates annotation payloads against task constraints.

Extracted from SubagentRuntime._check_annotation_validation (subagent_cycle.py)
to implement the Validator protocol from annotation_pipeline_skill/plugins/base.py.

The original method received a raw JSON *string* (final_text). This class
accepts an already-parsed *dict* (payload); JSON parsing is handled at the
call site (SubagentRuntime) before delegating here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from annotation_pipeline_skill.core.schema_validation import (
    SchemaValidationError,
    resolve_output_schema,
    validate_payload_against_task_schema,
)

if TYPE_CHECKING:
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore


class AnnotationValidator:
    """Implements the Validator protocol.

    Parameters
    ----------
    output_schema:
        An explicit JSON schema dict to validate against. When *None* the
        validator falls back to ``resolve_output_schema(task, self._store)``
        (inline schema in the task's source_ref, or project-level schema from
        the store). Pass ``None`` when you only want verbatim checking.
    store:
        Optional SqliteStore used for project-level schema resolution.
        Not required if ``output_schema`` is provided or if schema validation
        is not needed.
    """

    def __init__(
        self,
        output_schema: dict | None,
        store: "SqliteStore | None" = None,
    ) -> None:
        self._output_schema = output_schema
        self._store = store

    # ------------------------------------------------------------------
    # Validator protocol
    # ------------------------------------------------------------------

    def validate(self, task: object, payload: dict) -> dict | None:
        """Validate *payload* against task constraints.

        Returns ``None`` when the payload is valid.  Returns a ``dict`` with
        at least a ``"category"`` key describing the failure when invalid.

        The ``payload`` must already be parsed (dict); the caller is
        responsible for JSON parsing and the empty-string guard.
        """
        # Resolve schema: explicit override wins, then fall back to
        # resolve_output_schema which checks inline task schema then the
        # project-level file.
        if self._output_schema is not None:
            schema = self._output_schema
        else:
            schema = resolve_output_schema(task, self._store)  # type: ignore[arg-type]

        if schema is not None:
            # Strip discussion_replies before schema validation: side-channel
            # for QC dialogue, not part of the output schema.
            if isinstance(payload, dict):
                payload.pop("discussion_replies", None)
                rows = payload.get("rows")
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            row.pop("discussion_replies", None)

            try:
                validate_payload_against_task_schema(task, payload, store=self._store)  # type: ignore[arg-type]
            except SchemaValidationError as exc:
                return {
                    "category": "schema_invalid",
                    "message": f"Annotation result failed schema validation: {exc}",
                    "reason": "schema validation failed",
                    "target": {"errors": exc.errors},
                }

        # Enforce row coverage: all source row_ids must appear in the output.
        try:
            source_rows = task.source_ref["payload"]["rows"]  # type: ignore[union-attr]
            if isinstance(source_rows, list) and source_rows:
                source_ids = {
                    r["row_id"]
                    for r in source_rows
                    if isinstance(r, dict) and "row_id" in r
                }
                if source_ids:
                    ann_rows = (
                        payload.get("rows", []) if isinstance(payload, dict) else []
                    )
                    ann_ids = {
                        r["row_id"]
                        for r in ann_rows
                        if isinstance(r, dict) and "row_id" in r
                    }
                    missing = source_ids - ann_ids
                    if missing:
                        missing_sorted = sorted(missing)[:5]
                        return {
                            "category": "missing_rows",
                            "message": (
                                f"Annotation is missing {len(source_ids - ann_ids)} of "
                                f"{len(source_ids)} expected rows. "
                                f"First missing row_ids: {missing_sorted}. "
                                f"Every input row must appear in the output, even rows with no "
                                f"entities (emit them with empty dicts)."
                            ),
                            "reason": "missing rows in annotation output",
                        }
        except (KeyError, TypeError, AttributeError):
            pass

        # Verbatim span check: every annotated span must be a substring of
        # the corresponding row's input text.
        verbatim_failure = self.check_verbatim_spans(task, payload)
        if verbatim_failure is not None:
            return verbatim_failure

        # Cross-type entity collision check.
        from annotation_pipeline_skill.core.schema_validation import (
            find_cross_type_collisions,
            find_trailing_punctuation_spans,
        )
        collisions = find_cross_type_collisions(payload)
        if collisions:
            first = collisions[0]
            return {
                "category": "cross_type_collision",
                "message": (
                    f"Row {first['row_index']} entity span {first['span']!r} is tagged as "
                    f"both {first['types'][0]!r} and {first['types'][1]!r}. Pick one type "
                    f"per span — the schema allows separate keys but a single occurrence "
                    f"should resolve to a single entity type."
                ),
                "reason": "cross-type entity collision",
                "target": {
                    "row_index": first["row_index"],
                    "span": first["span"],
                    "types": first["types"],
                },
            }

        # Trailing-punctuation span boundary check.
        trailing = find_trailing_punctuation_spans(task, payload)
        if trailing:
            first = trailing[0]
            return {
                "category": "trailing_punctuation_span",
                "message": (
                    f"Row {first['row_index']} {first['field']} span {first['span']!r} ends "
                    f"with sentence-ending punctuation that should not be part of the entity. "
                    f"Re-emit as {first['trimmed']!r} — the trimmed form is also verbatim in "
                    f"input.text and that's where the entity boundary belongs."
                ),
                "reason": "trailing-punctuation span boundary",
                "target": {
                    "row_index": first["row_index"],
                    "field": first["field"],
                    "span": first["span"],
                    "trimmed": first["trimmed"],
                },
            }

        # Shared-type cross-field consistency check.
        from annotation_pipeline_skill.core.schema_validation import (
            find_shared_type_field_violations,
        )
        shared_violations = find_shared_type_field_violations(payload)
        if shared_violations:
            first = shared_violations[0]
            return {
                "category": "shared_type_wrong_field",
                "message": (
                    f"Row {first['row_index']} span {first['span']!r} (type "
                    f"{first['shared_type']!r}) is in {first['current_field']} but the "
                    f"word-count rule requires {first['correct_field']} (single-word "
                    f"goes to entities, multi-word goes to json_structures)."
                ),
                "reason": "shared-type field placement",
                "target": first,
            }

        return None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def check_verbatim_spans(self, task: object, payload: Any) -> dict | None:
        """Return a failure dict if any entity span is not verbatim in input.

        Wraps the shared ``find_verbatim_violations`` helper. Returns ``None``
        when all spans are verbatim substrings of their row's input text.
        """
        from annotation_pipeline_skill.core.schema_validation import find_verbatim_violations

        violations = find_verbatim_violations(task, payload)  # type: ignore[arg-type]
        if not violations:
            return None
        first = violations[0]
        return {
            "category": "non_verbatim_span",
            "message": (
                f"Row {first['row_index']} {first['field']}: span {first['span']!r} "
                f"is not a verbatim substring of the input text."
            ),
            "reason": "verbatim check failed",
            "target": first,
        }

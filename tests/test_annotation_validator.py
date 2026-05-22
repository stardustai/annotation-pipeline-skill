"""Tests for AnnotationValidator extracted from SubagentRuntime."""
import pytest
from annotation_pipeline_skill.runtime.annotation_validator import AnnotationValidator
from annotation_pipeline_skill.plugins.base import Validator


def _make_task(task_id="test-001", rows=None):
    """Return a minimal Task-like object for testing."""
    from annotation_pipeline_skill.core.models import Task

    return Task.new(
        task_id=task_id,
        pipeline_id="pipe",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "rows": rows
                or [{"row_index": 0, "input": "Apple is a company"}],
            },
        },
        modality="text",
        annotation_requirements={"annotation_types": ["extraction"]},
        metadata={},
    )


def test_validate_returns_none_for_valid_payload():
    """A payload with a verbatim span should pass validation when schema is None."""
    task = _make_task()
    validator = AnnotationValidator(output_schema=None)
    # Correct payload shape: rows[i].output.entities.<type> = [span, ...]
    payload = {
        "rows": [
            {
                "row_index": 0,
                "output": {
                    "entities": {"organization": ["Apple"]},
                },
            }
        ]
    }
    result = validator.validate(task, payload)
    assert result is None


def test_validate_catches_verbatim_violation():
    """A span that is not a verbatim substring of the input must fail."""
    task = _make_task(rows=[{"row_index": 0, "input": "Apple is a company"}])
    validator = AnnotationValidator(output_schema=None)
    # "Google" is NOT in "Apple is a company" — the verbatim check must catch it.
    payload = {
        "rows": [
            {
                "row_index": 0,
                "output": {
                    "entities": {"organization": ["Google"]},
                },
            }
        ]
    }
    result = validator.validate(task, payload)
    assert result is not None
    assert result.get("category") == "non_verbatim_span"


def test_validate_catches_verbatim_violation_in_json_structures():
    """Verbatim check applies to json_structures as well as entities."""
    task = _make_task(rows=[{"row_index": 0, "input": "Apple is a company"}])
    validator = AnnotationValidator(output_schema=None)
    payload = {
        "rows": [
            {
                "row_index": 0,
                "output": {
                    "json_structures": {"decision": ["HALLUCINATED"]},
                },
            }
        ]
    }
    result = validator.validate(task, payload)
    assert result is not None
    assert result.get("category") == "non_verbatim_span"


def test_validate_empty_payload_returns_none():
    """An empty rows list is valid (no spans to check)."""
    task = _make_task()
    validator = AnnotationValidator(output_schema=None)
    payload = {"rows": []}
    result = validator.validate(task, payload)
    assert result is None


def test_validator_implements_protocol():
    """AnnotationValidator must satisfy the Validator protocol."""
    v = AnnotationValidator(output_schema=None)
    assert isinstance(v, Validator)


def test_validate_with_schema_catches_schema_violation():
    """When output_schema is provided, schema violations are caught."""
    schema = {
        "type": "object",
        "required": ["rows"],
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["row_index", "output"],
                    "properties": {
                        "row_index": {"type": "integer"},
                        "output": {"type": "object"},
                    },
                },
            },
        },
    }
    task = _make_task()
    validator = AnnotationValidator(output_schema=schema)
    # Missing required "rows" key
    payload = {"wrong_key": []}
    result = validator.validate(task, payload)
    assert result is not None
    assert result.get("category") == "schema_invalid"


def test_validate_verbatim_exact_match_passes():
    """An exact verbatim match must pass."""
    task = _make_task(rows=[{"row_index": 0, "input": "Google is a tech company"}])
    validator = AnnotationValidator(output_schema=None)
    payload = {
        "rows": [
            {
                "row_index": 0,
                "output": {
                    "entities": {"organization": ["Google"]},
                    "json_structures": {"description": ["tech company"]},
                },
            }
        ]
    }
    result = validator.validate(task, payload)
    assert result is None

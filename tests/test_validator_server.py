"""Tests for the annotation-validator MCP server's pure check/lookup functions.

The stdio server wrapper is thin; these tests exercise the validation logic
(`check_annotation_draft`, `lookup_row_text`) directly so they don't depend on
the MCP transport.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.mcp.validator_server import (
    check_annotation_draft,
    lookup_row_text,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


# Minimal output_schema that the validator's schema check will accept for any
# {rows: [{row_index, row_id, output: {entities, json_structures}}, ...]} draft.
# Keeps tests focused on the row-level checks (verbatim/coverage/cross-type).
_MINIMAL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["rows"],
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["row_index", "row_id", "output"],
                "properties": {
                    "row_index": {"type": "integer"},
                    "row_id": {"type": "string"},
                    "output": {
                        "type": "object",
                        "properties": {
                            "entities": {"type": "object"},
                            "json_structures": {"type": "object"},
                        },
                    },
                },
            },
        }
    },
}


def _make_task(store: SqliteStore, *, task_id: str = "t-1") -> Task:
    """Persist a 3-row task with a minimal output_schema in the project root."""
    (store.root / "output_schema.json").write_text(
        json.dumps(_MINIMAL_SCHEMA), encoding="utf-8"
    )
    task = Task.new(
        task_id=task_id,
        pipeline_id="pipe",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "rows": [
                    {"row_index": 0, "row_id": "r0", "input": "Alpha launched in 2024."},
                    {"row_index": 1, "row_id": "r1", "input": "Bravo uses Python."},
                    {"row_index": 2, "row_id": "r2", "input": "Charlie shipped a feature."},
                ]
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    return task


def test_check_annotation_draft_returns_ok_on_clean_payload(tmp_path: Path) -> None:
    store = SqliteStore.open(tmp_path)
    _make_task(store)
    payload = {
        "rows": [
            {"row_index": 0, "row_id": "r0", "output": {"entities": {}, "json_structures": {}}},
            {"row_index": 1, "row_id": "r1", "output": {"entities": {}, "json_structures": {}}},
            {"row_index": 2, "row_id": "r2", "output": {"entities": {}, "json_structures": {}}},
        ]
    }
    result = check_annotation_draft(store, {"task_id": "t-1", "payload": payload})
    assert result["ok"] is True, result
    assert result["violations"] == {}
    assert "Submit" in result["next_action"] or "emit" in result["next_action"]


def test_check_annotation_draft_flags_verbatim_violation(tmp_path: Path) -> None:
    store = SqliteStore.open(tmp_path)
    _make_task(store)
    # Span "GoLang" is NOT a substring of row 1's input "Bravo uses Python."
    payload = {
        "rows": [
            {"row_index": 0, "row_id": "r0", "output": {"entities": {}, "json_structures": {}}},
            {
                "row_index": 1,
                "row_id": "r1",
                "output": {
                    "entities": {"technology": ["GoLang"]},
                    "json_structures": {},
                },
            },
            {"row_index": 2, "row_id": "r2", "output": {"entities": {}, "json_structures": {}}},
        ]
    }
    result = check_annotation_draft(store, {"task_id": "t-1", "payload": payload})
    assert result["ok"] is False
    assert "verbatim_violations" in result["violations"]
    vbs = result["violations"]["verbatim_violations"]
    assert any(v.get("span") == "GoLang" for v in vbs), vbs
    assert "Fix the listed" in result["next_action"]


def test_check_annotation_draft_flags_missing_row_coverage(tmp_path: Path) -> None:
    store = SqliteStore.open(tmp_path)
    _make_task(store)
    # Only emit 1 of the 3 source rows — the other two should appear in coverage_missing.
    payload = {
        "rows": [
            {"row_index": 0, "row_id": "r0", "output": {"entities": {}, "json_structures": {}}},
        ]
    }
    result = check_annotation_draft(store, {"task_id": "t-1", "payload": payload})
    assert result["ok"] is False
    assert set(result["violations"]["row_coverage_missing"]) == {"r1", "r2"}


def test_check_annotation_draft_unknown_task_returns_error(tmp_path: Path) -> None:
    store = SqliteStore.open(tmp_path)
    _make_task(store)
    result = check_annotation_draft(
        store, {"task_id": "does-not-exist", "payload": {"rows": []}}
    )
    assert result["ok"] is False
    assert "task not found" in result["error"]


def test_check_annotation_draft_rejects_malformed_payload(tmp_path: Path) -> None:
    store = SqliteStore.open(tmp_path)
    _make_task(store)
    result = check_annotation_draft(store, {"task_id": "t-1", "payload": "not an object"})
    assert result["ok"] is False
    assert "must be an object" in result["error"]


def test_lookup_row_text_returns_row_by_index(tmp_path: Path) -> None:
    store = SqliteStore.open(tmp_path)
    _make_task(store)
    result = lookup_row_text(store, {"task_id": "t-1", "row_index": 1})
    assert result["row_index"] == 1
    assert result["row_id"] == "r1"
    assert result["input"] == "Bravo uses Python."


def test_lookup_row_text_returns_row_by_id(tmp_path: Path) -> None:
    store = SqliteStore.open(tmp_path)
    _make_task(store)
    result = lookup_row_text(store, {"task_id": "t-1", "row_id": "r2"})
    assert result["row_index"] == 2
    assert result["row_id"] == "r2"
    assert "Charlie shipped" in result["input"]


def test_lookup_row_text_missing_row_returns_error(tmp_path: Path) -> None:
    store = SqliteStore.open(tmp_path)
    _make_task(store)
    result = lookup_row_text(store, {"task_id": "t-1", "row_index": 99})
    assert "error" in result
    assert "99" in result["error"]


def test_lookup_row_text_requires_index_or_id(tmp_path: Path) -> None:
    store = SqliteStore.open(tmp_path)
    _make_task(store)
    result = lookup_row_text(store, {"task_id": "t-1"})
    assert "specify" in result["error"]

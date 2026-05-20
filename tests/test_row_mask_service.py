"""Tests for RowMaskService and the filter_masked_rows pure helper."""
from __future__ import annotations

import pytest

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.services.row_mask_service import (
    RowMask,
    RowMaskService,
    apply_masks_to_task,
    filter_masked_rows,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


# ---------------------------------------------------------------------------
# RowMaskService tests
# ---------------------------------------------------------------------------


def test_apply_and_list_for_task(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = RowMaskService(store)

    mask = svc.apply(
        task_id="task-1",
        row_index=3,
        reason="duplicate",
        masked_by="dedup-bot",
        metadata={"source": "row_dedup"},
    )

    assert mask.task_id == "task-1"
    assert mask.row_index == 3
    assert mask.reason == "duplicate"
    assert mask.masked_by == "dedup-bot"
    assert mask.metadata == {"source": "row_dedup"}

    listed = svc.list_for_task("task-1")
    assert len(listed) == 1
    assert listed[0] == mask

    indices = svc.masked_indices_for_task("task-1")
    assert indices == {3}

    # Different task — should return empty set.
    assert svc.masked_indices_for_task("task-99") == set()


def test_apply_many_bulk_upserts(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = RowMaskService(store)

    count = svc.apply_many([
        {"task_id": "t-a", "row_index": 0, "reason": "dup", "masked_by": "bot"},
        {"task_id": "t-a", "row_index": 1, "reason": "dup", "masked_by": "bot", "metadata": {"x": 1}},
        {"task_id": "t-b", "row_index": 0, "reason": "dup", "masked_by": "bot"},
    ])
    assert count == 3

    assert svc.masked_indices_for_task("t-a") == {0, 1}
    assert svc.masked_indices_for_task("t-b") == {0}

    ta_masks = svc.list_for_task("t-a")
    assert len(ta_masks) == 2
    # apply_many orders by row_index in list_for_task
    assert ta_masks[0].row_index == 0
    assert ta_masks[1].row_index == 1
    assert ta_masks[1].metadata == {"x": 1}


def test_remove_returns_true_when_existed_false_otherwise(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = RowMaskService(store)

    svc.apply(task_id="t", row_index=5, reason="r", masked_by="b")

    # First remove — should succeed.
    assert svc.remove(task_id="t", row_index=5) is True

    # Second remove — row is gone, should return False.
    assert svc.remove(task_id="t", row_index=5) is False

    # Nonexistent task/row.
    assert svc.remove(task_id="no-such-task", row_index=0) is False


def test_list_for_project_joins_via_tasks(tmp_path):
    store = SqliteStore.open(tmp_path)

    # Seed two tasks in different pipelines.
    task_a = Task.new(task_id="task-pa", pipeline_id="pipeline-A", source_ref={"kind": "jsonl"})
    task_b = Task.new(task_id="task-pb", pipeline_id="pipeline-B", source_ref={"kind": "jsonl"})
    store.save_task(task_a)
    store.save_task(task_b)

    svc = RowMaskService(store)
    svc.apply(task_id="task-pa", row_index=0, reason="dup", masked_by="bot")
    svc.apply(task_id="task-pa", row_index=1, reason="dup", masked_by="bot")
    svc.apply(task_id="task-pb", row_index=0, reason="dup", masked_by="bot")

    # Only pipeline-A's masks.
    masks_a = svc.list_for_project("pipeline-A")
    assert len(masks_a) == 2
    assert all(m.task_id == "task-pa" for m in masks_a)

    # Only pipeline-B's masks.
    masks_b = svc.list_for_project("pipeline-B")
    assert len(masks_b) == 1
    assert masks_b[0].task_id == "task-pb"

    # Nonexistent project.
    assert svc.list_for_project("pipeline-X") == []


def test_masked_indices_by_task_bulk(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = RowMaskService(store)

    svc.apply_many([
        {"task_id": "t1", "row_index": 0, "reason": "r", "masked_by": "b"},
        {"task_id": "t1", "row_index": 2, "reason": "r", "masked_by": "b"},
        {"task_id": "t2", "row_index": 1, "reason": "r", "masked_by": "b"},
    ])

    result = svc.masked_indices_by_task(["t1", "t2", "t3"])
    assert result["t1"] == {0, 2}
    assert result["t2"] == {1}
    assert result["t3"] == set()

    # Empty input.
    assert svc.masked_indices_by_task([]) == {}


# ---------------------------------------------------------------------------
# filter_masked_rows pure helper tests
# ---------------------------------------------------------------------------


def test_filter_masked_rows_excludes_matching_row_index():
    payload = {
        "rows": [
            {"row_index": 0, "input": "keep me"},
            {"row_index": 1, "input": "mask me"},
            {"row_index": 2, "input": "keep me too"},
        ]
    }
    result = filter_masked_rows(payload, {1})
    assert result is not payload  # new dict returned
    assert len(result["rows"]) == 2
    assert result["rows"][0]["row_index"] == 0
    assert result["rows"][1]["row_index"] == 2
    # Other keys preserved.
    assert "rows" in result


def test_filter_masked_rows_noop_on_empty_set():
    payload = {
        "rows": [
            {"row_index": 0, "input": "hello"},
        ]
    }
    result = filter_masked_rows(payload, set())
    # Empty set → same reference returned unchanged.
    assert result is payload


def test_filter_masked_rows_noop_on_none_payload():
    assert filter_masked_rows(None, {0}) is None


def test_filter_masked_rows_noop_when_no_rows_key():
    payload = {"text": "no rows here"}
    result = filter_masked_rows(payload, {0})
    assert result is payload


def test_filter_masked_rows_noop_when_nothing_masked():
    payload = {"rows": [{"row_index": 5, "input": "x"}]}
    # masked_indices has index 99, not present in payload.
    result = filter_masked_rows(payload, {99})
    assert result is payload


def test_filter_masked_rows_works_on_annotation_output_shape():
    """Verify the helper works on annotation_result-shaped rows too."""
    payload = {
        "rows": [
            {"row_index": 0, "output": {"entities": {"org": ["Apple"]}}},
            {"row_index": 1, "output": {"entities": {"org": ["Banana"]}}},
        ]
    }
    result = filter_masked_rows(payload, {0})
    assert len(result["rows"]) == 1
    assert result["rows"][0]["row_index"] == 1


def test_apply_masks_to_task_filters_source_ref_rows(tmp_path):
    """apply_masks_to_task returns a Task whose source_ref.payload.rows
    omits rows present in row_masks. The original task object is left
    untouched (shallow copy semantics) so other readers that want the
    full payload are unaffected.
    """
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-1",
        pipeline_id="proj",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "rows": [
                    {"row_index": 0, "input": "alpha"},
                    {"row_index": 1, "input": "bravo"},
                    {"row_index": 2, "input": "charlie"},
                ],
            },
        },
    )
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)

    svc = RowMaskService(store)
    svc.apply(task_id="t-1", row_index=1, reason="dup", masked_by="test")

    masked_task = apply_masks_to_task(store, task)
    indices = [r["row_index"] for r in masked_task.source_ref["payload"]["rows"]]
    assert indices == [0, 2]
    # Original task unchanged
    assert [r["row_index"] for r in task.source_ref["payload"]["rows"]] == [0, 1, 2]


def test_apply_masks_to_task_noop_when_no_masks(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-2",
        pipeline_id="proj",
        source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": "x"}]}},
    )
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    # No masks applied; helper returns the input task unchanged
    result = apply_masks_to_task(store, task)
    assert result is task

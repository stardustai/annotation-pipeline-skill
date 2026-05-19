from annotation_pipeline_skill.similarity.extractors import canonical_task_text


def _task(rows):
    return type("T", (), {"source_ref": {"payload": {"rows": rows}}})()


def test_concatenates_row_inputs_with_newline():
    task = _task([
        {"row_index": 0, "input": "first row text"},
        {"row_index": 1, "input": "second row text"},
    ])
    assert canonical_task_text(task) == "first row text\nsecond row text"


def test_orders_by_row_index_not_list_position():
    task = _task([
        {"row_index": 5, "input": "later"},
        {"row_index": 1, "input": "earlier"},
    ])
    assert canonical_task_text(task) == "earlier\nlater"


def test_skips_non_string_inputs():
    task = _task([
        {"row_index": 0, "input": "ok"},
        {"row_index": 1, "input": None},
        {"row_index": 2, "input": "also ok"},
    ])
    assert canonical_task_text(task) == "ok\nalso ok"


def test_returns_empty_string_for_no_rows():
    assert canonical_task_text(_task([])) == ""
    assert canonical_task_text(type("T", (), {"source_ref": {}})()) == ""


def test_store_mask_excludes_masked_row(tmp_path):
    """When a real SqliteStore is passed, masked rows are excluded from the text."""
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.services.row_mask_service import RowMaskService
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="task-extract-1",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "rows": [
                    {"row_index": 0, "input": "keep this"},
                    {"row_index": 1, "input": "mask this"},
                    {"row_index": 2, "input": "keep this too"},
                ]
            },
        },
    )
    store.save_task(task)

    # Mask row_index=1.
    RowMaskService(store).apply(
        task_id="task-extract-1",
        row_index=1,
        reason="duplicate",
        masked_by="test",
    )

    # Without store= the masked row is still included.
    assert canonical_task_text(task) == "keep this\nmask this\nkeep this too"

    # With store= the masked row is excluded.
    result = canonical_task_text(task, store=store)
    assert result == "keep this\nkeep this too"

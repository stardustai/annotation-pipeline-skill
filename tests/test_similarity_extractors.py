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

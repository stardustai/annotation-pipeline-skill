import json

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.runtime.subagent_cycle import (
    _parse_llm_json,
    _repaired_rows_misaligned,
    _serialize_llm_json,
)


def _task() -> Task:
    rows = [
        {"row_id": "r0", "row_index": 0, "text": "张三来了"},
        {"row_id": "r1", "row_index": 1, "text": "李四来了"},
        {"row_id": "r2", "row_index": 2, "text": "王五来了"},
    ]
    return Task.new(task_id="t", pipeline_id="p", source_ref={"kind": "jsonl", "payload": {"rows": rows}})


def _row(i: int, name: str) -> dict:
    return {"row_id": f"r{i}", "row_index": i, "output": {"entities": {"person": [name]}, "json_structures": {}}}


CLEAN = json.dumps({"rows": [_row(0, "张三"), _row(1, "李四"), _row(2, "王五")]}, ensure_ascii=False)


def _output_first(i: int, name: str) -> str:
    # MiniMax writes `output` before the row keys.
    row = _row(i, name)
    return json.dumps({"output": row["output"], "row_id": row["row_id"], "row_index": row["row_index"]}, ensure_ascii=False)


# Row 0 lost the "}" that closes its `entities`. The lenient repair then closes `output`
# early and folds row 0's row_id / row_index into `output`, so the row has no keys and
# downstream alignment shifts its annotation onto the wrong row.
_ROW0 = _output_first(0, "张三")
MISSING_BRACE = (
    '{"rows": ['
    + ", ".join([_ROW0.replace("]}", "]", 1), _output_first(1, "李四"), _output_first(2, "王五")])
    + "]}"
)


def test_missing_brace_fixture_is_not_strict_json() -> None:
    assert MISSING_BRACE != CLEAN
    try:
        json.loads(MISSING_BRACE)
    except ValueError:
        return
    raise AssertionError("fixture should need a lenient repair")


def test_lenient_repair_swallows_the_row_keys() -> None:
    rows = _parse_llm_json(MISSING_BRACE)["rows"]
    assert "row_id" not in rows[0]
    assert _repaired_rows_misaligned(_task(), MISSING_BRACE, {"rows": rows}) is not None


def test_clean_output_is_not_flagged() -> None:
    assert _repaired_rows_misaligned(_task(), CLEAN, json.loads(CLEAN)) is None


def test_fenced_clean_output_is_not_flagged() -> None:
    text = "Here you go:\n```json\n" + CLEAN + "\n```"
    assert _repaired_rows_misaligned(_task(), text, json.loads(CLEAN)) is None


def test_clean_output_that_drops_a_row_is_left_to_auto_fill() -> None:
    dropped = json.dumps({"rows": [_row(0, "张三"), _row(2, "王五")]}, ensure_ascii=False)
    assert _repaired_rows_misaligned(_task(), dropped, json.loads(dropped)) is None


def test_strict_json_that_repeats_row_keys_is_flagged() -> None:
    # Row 1 written inside row 0's object: strict JSON, but with repeated keys.
    text = (
        '{"rows":[{"row_id":"r0","row_index":0,"output":{"entities":{"person":["张三"]}},'
        '"row_id":"r1","row_index":1,"output":{"entities":{"person":["李四"]}}},'
        '{"row_id":"r2","row_index":2,"output":{"entities":{"person":["王五"]}}}]}'
    )
    parsed = json.loads(text)
    assert _repaired_rows_misaligned(_task(), text, parsed) is not None


def test_a_repair_that_keeps_rows_aligned_is_accepted() -> None:
    text = CLEAN[:-1]  # truncated closing brace; the repair only has to append it
    parsed = json.loads(CLEAN)
    assert _repaired_rows_misaligned(_task(), text, parsed) is None


def test_serialize_does_not_paper_over_a_misaligned_repair() -> None:
    text, stripped, total = _serialize_llm_json(MISSING_BRACE, task=_task())
    # The raw text is returned untouched so validation reports the missing row and the
    # annotator retries, instead of an empty stub replacing the row's real annotation.
    assert text == MISSING_BRACE
    assert (stripped, total) == (0, 0)


def test_serialize_still_auto_fills_rows_dropped_from_clean_json() -> None:
    dropped = json.dumps({"rows": [_row(0, "张三"), _row(2, "王五")]}, ensure_ascii=False)
    text, _, _ = _serialize_llm_json(dropped, task=_task())
    rows = json.loads(text)["rows"]
    assert [r["row_id"] for r in rows] == ["r0", "r1", "r2"]

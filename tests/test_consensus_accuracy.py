"""Tests for consensus accuracy scoring (annotation_pipeline_skill.eval).

The judge is dependency-injected so these tests run without network access.
The core property under test: an error flagged in a MINORITY of passes is
filtered out, while an error flagged by a MAJORITY is kept and scored.
"""
from __future__ import annotations

import json

from annotation_pipeline_skill.eval.consensus_accuracy import (
    count_spans,
    parse_json_safe,
    run_consensus,
    run_single_pass,
)


def _rows():
    # Two rows; 5 spans total (Alice, Bob, Acme Corp, FFT, Beijing).
    return [
        {"task_id": "t-1", "row_index": 0, "_flat_line_no": 0,
         "input": "Alice and Bob work at Acme Corp using FFT.",
         "output": {"entities": {"person": ["Alice", "Bob"], "org": ["Acme Corp"]},
                    "json_structures": {"technology": ["FFT"]}}},
        {"task_id": "t-1", "row_index": 1, "_flat_line_no": 1,
         "input": "The Beijing office opened.",
         "output": {"entities": {"location": ["Beijing"]}}},
    ]


def _judge_from_script(script):
    """Return a judge fn that yields failure-lists from `script` per call.

    `script` is a list (one entry per call) of failure-dict lists. The judge is
    called once per chunk per pass; with a single chunk, calls map 1:1 to passes.
    """
    calls = {"i": 0}

    def judge(system, user):  # noqa: ARG001
        idx = calls["i"]
        calls["i"] += 1
        failures = script[idx] if idx < len(script) else []
        return json.dumps({"chunk": 0, "total_rows_checked": 2, "failures": failures})

    return judge


def test_count_spans():
    assert count_spans(_rows()[0]["output"]) == 4
    assert count_spans(_rows()[1]["output"]) == 1
    assert count_spans({}) == 0
    assert count_spans({"entities": {"person": ["", "  "]}}) == 0


def test_parse_json_safe_handles_fences_and_thinking():
    assert parse_json_safe('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_safe("<think>reasoning</think>{\"a\": 2}") == {"a": 2}
    assert parse_json_safe("garbage prefix {\"a\": 3} suffix") == {"a": 3}
    assert parse_json_safe("not json at all") is None


def test_single_pass_precision():
    rows = _rows()  # 5 spans
    fp = {"id": "t-1:0", "error_type": "false_positive", "span": "Bob",
          "fp_delta": 1, "fn_delta": 0}
    res = run_single_pass(rows, "RULES", _judge_from_script([[fp]]),
                          chunk_size=30, workers=1)
    assert res["fp"] == 1
    assert res["tp"] == 4  # 5 spans - 1 fp
    assert res["precision"] == round(4 / 5, 4)


def test_consensus_filters_minority_noise():
    """A span flagged in only 1 of 3 passes is noise and must be dropped."""
    rows = _rows()  # 5 spans
    noise = {"id": "t-1:0", "error_type": "false_positive", "span": "Alice",
             "fp_delta": 1, "fn_delta": 0}
    # Flagged once, clean twice.
    res = run_consensus(rows, "RULES", _judge_from_script([[noise], [], []]),
                        passes=3, chunk_size=30, workers=1)
    assert res.fp_flagged_any == 1   # it WAS flagged once
    assert res.fp == 0               # but did not reach 2/3 majority
    assert res.precision == 1.0
    assert res.threshold == 2


def test_consensus_keeps_majority_confirmed_error():
    """A span flagged in 2 of 3 passes is genuine and must be scored."""
    rows = _rows()  # 5 spans
    fp = {"id": "t-1:0", "error_type": "false_positive", "span": "Bob",
          "fp_delta": 1, "fn_delta": 0, "suggested_action": "Remove from person"}
    res = run_consensus(rows, "RULES", _judge_from_script([[fp], [fp], []]),
                        passes=3, chunk_size=30, workers=1)
    assert res.fp == 1
    assert res.tp == 4
    assert res.precision == round(4 / 5, 4)
    # The confirmed failure is surfaced for downstream fixing.
    assert any(f["span"] == "Bob" for f in res.confirmed_failures)


def test_consensus_wrong_type_votes_both_tallies():
    """wrong_type (fp_delta=1 AND fn_delta=1) counts against P and R."""
    rows = _rows()  # 5 spans
    wt = {"id": "t-1:1", "error_type": "wrong_type", "span": "Beijing",
          "fp_delta": 1, "fn_delta": 1}
    res = run_consensus(rows, "RULES", _judge_from_script([[wt], [wt], [wt]]),
                        passes=3, chunk_size=30, workers=1)
    assert res.fp == 1
    assert res.fn == 1
    assert res.tp == 4
    assert res.precision == round(4 / 5, 4)
    assert res.recall == round(4 / 5, 4)


def test_consensus_perfect_when_no_failures():
    rows = _rows()
    res = run_consensus(rows, "RULES", _judge_from_script([[], [], []]),
                        passes=3, chunk_size=30, workers=1)
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.fp == 0 and res.fn == 0
    assert res.meets(target_precision=0.98, target_f1=0.95)

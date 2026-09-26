from annotation_pipeline_skill.runtime.consensus import iter_span_items


def test_iter_span_items_flattens_entities_and_json():
    payload = {"rows": [
        {"row_index": 0, "output": {
            "entities": {"person": ["Alice"], "organization": ["ACME"]},
            "json_structures": {"task": ["ship it"]},
        }},
        {"row_index": 1, "output": {"entities": {"number": ["42"]}}},
    ]}
    items = set(iter_span_items(payload))
    assert items == {
        (0, "entities", "person", "Alice"),
        (0, "entities", "organization", "ACME"),
        (0, "json_structures", "task", "ship it"),
        (1, "entities", "number", "42"),
    }


def test_iter_span_items_tolerates_missing_keys():
    assert list(iter_span_items({})) == []
    assert list(iter_span_items({"rows": [{"row_index": 0}]})) == []


from annotation_pipeline_skill.runtime.consensus import build_consensus


def _row(ri, ents):
    return {"row_index": ri, "output": {"entities": ents}}


_SRC0 = [{"row_index": 0, "row_id": "r0", "input": "Alice at ACME uses Spark; Bob; Apple"}]


def test_unanimous_kept_one_sided_disagrees():
    a = {"rows": [_row(0, {"person": ["Alice"], "organization": ["ACME"]})]}
    b = {"rows": [_row(0, {"person": ["Alice"], "technology": ["Spark"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=2, source_rows=_SRC0)
    assert consensus["rows"][0]["output"]["entities"] == {"person": ["Alice"]}
    dis = {(d["field"], d["type"], d["span"], d["support"]) for d in disagree}
    assert dis == {
        ("entities", "organization", "ACME", 1),
        ("entities", "technology", "Spark", 1),
    }


def test_threshold_one_is_union():
    a = {"rows": [_row(0, {"person": ["Alice"]})]}
    b = {"rows": [_row(0, {"person": ["Bob"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=1, source_rows=_SRC0)
    assert sorted(consensus["rows"][0]["output"]["entities"]["person"]) == ["Alice", "Bob"]
    assert disagree == []


def test_type_conflict_same_span_surfaces_both_as_disagreements():
    a = {"rows": [_row(0, {"organization": ["Apple"]})]}
    b = {"rows": [_row(0, {"technology": ["Apple"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=2, source_rows=_SRC0)
    assert consensus["rows"][0]["output"].get("entities", {}) == {}
    types = {(d["type"], d["span"]) for d in disagree}
    assert types == {("organization", "Apple"), ("technology", "Apple")}


from annotation_pipeline_skill.runtime.consensus import align_rows_to_source, source_row_ids

# Real row shape from zh_chunks_v6-000005: source rows start at row_index 40.
_SRC = [
    {"row_index": 40, "row_id": "zh6-0022-1", "input": "Friday 和 ChatGPT"},
    {"row_index": 41, "row_id": "zh6-0023-0", "input": "阿里云 折扣"},
]


def _out(ents):
    return {"entities": ents, "json_structures": {}}


def test_source_row_ids_follow_source_order_and_identity():
    assert source_row_ids(_SRC) == {40: "zh6-0022-1", 41: "zh6-0023-0"}
    # A dataset that doesn't name/number its rows: position + row-<index>.
    assert source_row_ids([{"input": "a"}, {"input": "b"}]) == {0: "row-0", 1: "row-1"}


def test_build_consensus_emits_exactly_the_source_rows_with_source_ids():
    a = {"rows": [_row(40, {"product": ["Friday"]}), _row(41, {})]}
    b = {"rows": [_row(40, {"product": ["Friday"]}), _row(41, {})]}
    consensus, _ = build_consensus([a, b], keep_threshold=2, source_rows=_SRC)
    assert [(r["row_index"], r["row_id"]) for r in consensus["rows"]] == [
        (40, "zh6-0022-1"), (41, "zh6-0023-0")]
    # both required output fields present even on an empty row
    assert consensus["rows"][1]["output"] == {"entities": {}, "json_structures": {}}


def test_build_consensus_ignores_keyless_draft_row():
    # MiniMax-M2.7 emitted its first row as {"output": {...}} with no
    # row_index/row_id (copying the prompt's empty-row example). The old
    # consensus defaulted it to row_index 0 -> a phantom empty row 0 that is
    # not in the source batch -> schema_invalid (maxItems) -> human review.
    qwen = {"rows": [_row(40, {"product": ["Friday"]}), _row(41, {"organization": ["阿里云"]})]}
    minimax = {"rows": [{"output": _out({"product": ["Friday"]})},
                        _row(41, {"organization": ["阿里云"]})]}
    consensus, disagreements = build_consensus([qwen, minimax], keep_threshold=2, source_rows=_SRC)
    assert [r["row_index"] for r in consensus["rows"]] == [40, 41]
    assert all(d["row_index"] in (40, 41) for d in disagreements)


def test_build_consensus_ignores_row_outside_the_source_batch():
    # qwen occasionally appends an 11th row numbered past the batch
    # (row_index 480 on a 470..479 task).
    a = {"rows": [_row(40, {"product": ["Friday"]}), _row(42, {"product": ["ChatGPT"]})]}
    b = {"rows": [_row(40, {"product": ["Friday"]})]}
    consensus, disagreements = build_consensus([a, b], keep_threshold=1, source_rows=_SRC)
    assert [r["row_index"] for r in consensus["rows"]] == [40, 41]
    assert disagreements == []


def test_align_replaces_model_row_id_with_source_row_id():
    # zh_chunks_v6-000011: the arbiter, shown only row_index, wrote
    # row_id = str(row_index) ("40") -> every source row_id "missing".
    arbiter = {"rows": [
        {"row_index": 40, "row_id": "40", "output": _out({"product": ["Friday"]})},
        {"row_index": 41, "row_id": "41", "output": _out({})},
    ]}
    out = align_rows_to_source(arbiter, _SRC)
    assert [(r["row_index"], r["row_id"]) for r in out["rows"]] == [
        (40, "zh6-0022-1"), (41, "zh6-0023-0")]
    assert out["rows"][0]["output"]["entities"] == {"product": ["Friday"]}


def test_align_drops_rows_not_in_source():
    # The arbiter echoed the consensus' phantom row as {row_index: 0, row_id: "0"}.
    arbiter = {"rows": [
        {"row_index": 0, "row_id": "0", "output": _out({})},
        {"row_index": 40, "row_id": "40", "output": _out({})},
        {"row_id": "row-x", "output": _out({"product": ["Friday"]})},
        {"row_index": 41, "row_id": "41", "output": _out({})},
    ]}
    out = align_rows_to_source(arbiter, _SRC)
    assert [r["row_index"] for r in out["rows"]] == [40, 41]


def test_align_keeps_omitted_source_row_absent():
    # A source row the model left out is a real coverage failure: it must stay
    # missing so row-coverage validation reports it, not be filled silently.
    out = align_rows_to_source({"rows": [{"row_index": 41, "output": _out({})}]}, _SRC)
    assert [r["row_id"] for r in out["rows"]] == ["zh6-0023-0"]


def test_align_merges_duplicate_rows_emitted_by_arbiter():
    # Observed arbiter failure mode: each row emitted twice — one filled copy
    # plus one empty scaffold — yielding 2x rows that violate schema maxItems.
    payload = {"rows": [
        {"row_index": 40, "output": {"entities": {"location": ["Punjab"]}, "json_structures": {}}},
        {"row_index": 40, "output": {"entities": {}, "json_structures": {}}},
        {"row_index": 41, "output": {"entities": {"person": ["Alice"]}}},
        {"row_index": 41, "output": {"entities": {"person": ["Bob"], "org": ["ACME"]}}},
    ]}
    out = align_rows_to_source(payload, _SRC)
    assert [r["row_index"] for r in out["rows"]] == [40, 41]
    assert out["rows"][0]["output"]["entities"] == {"location": ["Punjab"]}
    assert out["rows"][1]["output"]["entities"] == {"org": ["ACME"], "person": ["Alice", "Bob"]}
    # schema requires BOTH fields present on every row's output, even when empty
    for r in out["rows"]:
        assert set(r["output"].keys()) == {"entities", "json_structures"}


from annotation_pipeline_skill.runtime.consensus import build_arbiter_merge_prompt


def test_merge_prompt_contains_drafts_and_disagreements():
    a = {"rows": [_row(0, {"person": ["Alice"], "organization": ["ACME"]})]}
    b = {"rows": [_row(0, {"person": ["Alice"]})]}
    src = [{"row_index": 0, "row_id": "r0", "input": "Alice at ACME"}]
    consensus, disagree = build_consensus([a, b], keep_threshold=2, source_rows=src)
    prompt = build_arbiter_merge_prompt(
        source_rows=src, consensus=consensus, disagreements=disagree,
    )
    assert "ACME" in prompt
    assert "Alice at ACME" in prompt
    assert "json" in prompt.lower()


def test_merge_prompt_shows_source_row_id_next_to_row_index():
    # The output schema requires row_id; an arbiter shown only row_index
    # invented row_id = str(row_index). Each row must carry its source row_id.
    import json as _json
    consensus, disagree = build_consensus(
        [{"rows": [_row(40, {"product": ["Friday"]})]}, {"rows": [_row(40, {})]}],
        keep_threshold=2, source_rows=_SRC)
    prompt = build_arbiter_merge_prompt(source_rows=_SRC, consensus=consensus, disagreements=disagree)
    rows = _json.loads(prompt[prompt.rindex("\n\n"):])
    assert [(r["row_index"], r["row_id"]) for r in rows] == [(40, "zh6-0022-1"), (41, "zh6-0023-0")]
    assert '"row_id":str' in prompt

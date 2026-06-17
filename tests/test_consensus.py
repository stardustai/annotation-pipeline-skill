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


def test_unanimous_kept_one_sided_disagrees():
    a = {"rows": [_row(0, {"person": ["Alice"], "organization": ["ACME"]})]}
    b = {"rows": [_row(0, {"person": ["Alice"], "technology": ["Spark"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=2)
    assert consensus["rows"][0]["output"]["entities"] == {"person": ["Alice"]}
    dis = {(d["field"], d["type"], d["span"], d["support"]) for d in disagree}
    assert dis == {
        ("entities", "organization", "ACME", 1),
        ("entities", "technology", "Spark", 1),
    }


def test_threshold_one_is_union():
    a = {"rows": [_row(0, {"person": ["Alice"]})]}
    b = {"rows": [_row(0, {"person": ["Bob"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=1)
    assert sorted(consensus["rows"][0]["output"]["entities"]["person"]) == ["Alice", "Bob"]
    assert disagree == []


def test_type_conflict_same_span_surfaces_both_as_disagreements():
    a = {"rows": [_row(0, {"organization": ["Apple"]})]}
    b = {"rows": [_row(0, {"technology": ["Apple"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=2)
    assert consensus["rows"][0]["output"].get("entities", {}) == {}
    types = {(d["type"], d["span"]) for d in disagree}
    assert types == {("organization", "Apple"), ("technology", "Apple")}


from annotation_pipeline_skill.runtime.consensus import coalesce_rows_by_index


def test_coalesce_merges_duplicate_rows_emitted_by_arbiter():
    # Observed arbiter failure mode: each row emitted twice — one filled copy
    # plus one empty scaffold — yielding 2x rows that violate schema maxItems.
    payload = {"rows": [
        {"row_index": 0, "row_id": "r0",
         "output": {"entities": {"location": ["Punjab"]}, "json_structures": {}}},
        {"row_index": 0, "row_id": "r0",
         "output": {"entities": {}, "json_structures": {}}},
        {"row_index": 1, "row_id": "r1",
         "output": {"entities": {"organization": ["AHR"]}, "json_structures": {}}},
        {"row_index": 1, "row_id": "r1", "output": {"entities": {}}},
    ]}
    out = coalesce_rows_by_index(payload)
    assert [r["row_index"] for r in out["rows"]] == [0, 1]
    assert out["rows"][0]["output"]["entities"] == {"location": ["Punjab"]}
    assert out["rows"][0]["row_id"] == "r0"
    assert out["rows"][1]["output"]["entities"] == {"organization": ["AHR"]}
    # schema requires BOTH fields present on every row's output, even when empty
    for r in out["rows"]:
        assert set(r["output"].keys()) == {"entities", "json_structures"}


def test_coalesce_empty_row_keeps_both_required_fields():
    # An all-empty row (no spans from any draft) must still carry both required
    # keys as {} — dropping them fails the output schema's `required` rule.
    payload = {"rows": [{"row_index": 3, "row_id": "r3", "output": {"entities": {}}}]}
    out = coalesce_rows_by_index(payload)
    assert out["rows"][0]["output"] == {"entities": {}, "json_structures": {}}


def test_coalesce_unions_spans_across_duplicate_rows():
    payload = {"rows": [
        {"row_index": 0, "output": {"entities": {"person": ["Alice"]}}},
        {"row_index": 0, "output": {"entities": {"person": ["Bob"], "org": ["ACME"]}}},
    ]}
    out = coalesce_rows_by_index(payload)
    assert len(out["rows"]) == 1
    assert out["rows"][0]["output"]["entities"] == {"org": ["ACME"], "person": ["Alice", "Bob"]}


def test_coalesce_noop_on_unique_rows():
    payload = {"rows": [
        {"row_index": 0, "output": {"entities": {"person": ["Alice"]}}},
        {"row_index": 1, "output": {"entities": {"org": ["ACME"]}}},
    ]}
    out = coalesce_rows_by_index(payload)
    assert [r["row_index"] for r in out["rows"]] == [0, 1]
    assert len(out["rows"]) == 2


from annotation_pipeline_skill.runtime.consensus import build_arbiter_merge_prompt


def test_merge_prompt_contains_drafts_and_disagreements():
    a = {"rows": [_row(0, {"person": ["Alice"], "organization": ["ACME"]})]}
    b = {"rows": [_row(0, {"person": ["Alice"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=2)
    prompt = build_arbiter_merge_prompt(
        row_inputs={0: "Alice at ACME"},
        consensus=consensus, disagreements=disagree,
    )
    assert "ACME" in prompt
    assert "Alice at ACME" in prompt
    assert "json" in prompt.lower()

from annotation_pipeline_skill.core.span_cleanup import (
    is_redacted_placeholder,
    clean_spans_in_place,
)


def test_money_mask_is_placeholder():
    assert is_redacted_placeholder("{$10000.00}")
    assert is_redacted_placeholder("{$300.00}")
    assert is_redacted_placeholder("{$.5}")
    # a real money string with currency text is NOT a mask
    assert not is_redacted_placeholder("$10,000")
    assert not is_redacted_placeholder("USD 300")


def test_xmask_is_placeholder():
    assert is_redacted_placeholder("XX/XX/XXXX")
    assert is_redacted_placeholder("XXXX")
    assert is_redacted_placeholder("XX XX")
    assert is_redacted_placeholder("XX/XX/2024")  # masked day/month, real year
    # single X or words containing X are NOT masks
    assert not is_redacted_placeholder("X")
    assert not is_redacted_placeholder("Xbox")
    assert not is_redacted_placeholder("Max")


def test_org_id_is_placeholder():
    assert is_redacted_placeholder("ORG727")
    assert is_redacted_placeholder("ORG36")
    # real orgs are not ORG\d+
    assert not is_redacted_placeholder("ORGANIZATION")
    assert not is_redacted_placeholder("Chase")


def _payload(entities):
    return {"rows": [{"row_index": 0, "row_id": "r0",
                      "output": {"entities": entities, "json_structures": {}}}]}


def test_clean_drops_placeholder_spans():
    p = _payload({"number": ["{$300.00}", "1500"], "time": ["XX/XX/XXXX"],
                  "organization": ["ORG727", "Chase"]})
    removed = clean_spans_in_place(p)
    assert removed == 3
    out = p["rows"][0]["output"]["entities"]
    assert out["number"] == ["1500"]
    assert "time" not in out          # emptied → key dropped
    assert out["organization"] == ["Chase"]


def test_clean_drops_alpha_nested_substrings():
    # FFT ⊂ FlashFFTConv (same type) → drop FFT; keep maximal
    p = _payload({"technology": ["FFT", "FlashFFTConv", "GNN", "GNNs"]})
    removed = clean_spans_in_place(p)
    assert removed == 2
    assert sorted(p["rows"][0]["output"]["entities"]["technology"]) == ["FlashFFTConv", "GNNs"]


def test_clean_never_drops_numeric_substring():
    # "5" is a substring of "1500" but purely numeric → must NOT be dropped
    p = _payload({"number": ["5", "1500"]})
    removed = clean_spans_in_place(p)
    assert removed == 0
    assert sorted(p["rows"][0]["output"]["entities"]["number"]) == ["1500", "5"]


def test_clean_preserves_required_output_keys():
    p = _payload({"time": ["XX/XX/XXXX"]})
    clean_spans_in_place(p)
    out = p["rows"][0]["output"]
    assert set(out.keys()) == {"entities", "json_structures"}  # both required keys remain


def test_nested_rule_is_entities_only():
    # In json_structures, a short phrase inside a longer one is two legit
    # annotations — must NOT be substring-dropped (would cost recall).
    p = {"rows": [{"row_index": 0, "row_id": "r0", "output": {
        "entities": {"technology": ["GNN", "GNNs"]},          # entities: dropped
        "json_structures": {"goal": ["improve recall", "improve recall on dense rows"]},
    }}]}
    removed = clean_spans_in_place(p)
    out = p["rows"][0]["output"]
    assert out["entities"]["technology"] == ["GNNs"]           # nested dropped
    assert sorted(out["json_structures"]["goal"]) == ["improve recall", "improve recall on dense rows"]
    assert removed == 1  # only the entities one


def test_placeholder_still_dropped_in_json_structures():
    p = {"rows": [{"row_index": 0, "row_id": "r0", "output": {
        "entities": {}, "json_structures": {"number": ["{$300.00}", "fifteen"]},
    }}]}
    removed = clean_spans_in_place(p)
    assert removed == 1
    assert p["rows"][0]["output"]["json_structures"]["number"] == ["fifteen"]


def test_company_retyped_from_technology_to_organization():
    p = _payload({"technology": ["google", "PyTorch", "Skype"], "organization": ["Acme"]})
    changed = clean_spans_in_place(p)
    ent = p["rows"][0]["output"]["entities"]
    assert ent["technology"] == ["PyTorch"]            # real tech kept
    assert set(ent["organization"]) == {"Acme", "google", "Skype"}  # companies moved
    assert changed == 2


def test_sensor_row_drops_bare_numbers():
    # >=3 bare-numeral number spans in a row -> telemetry -> drop them
    p = _payload({"number": ["19", "71.52", "0.3", "8"], "person": ["Ada"]})
    changed = clean_spans_in_place(p)
    ent = p["rows"][0]["output"]["entities"]
    assert "number" not in ent          # all 4 bare numbers dropped
    assert ent["person"] == ["Ada"]
    assert changed == 4


def test_sensor_guard_keeps_few_or_unit_numbers():
    # a prose row with 1-2 numbers, or unit-bearing numbers, is untouched
    p = _payload({"number": ["30", "10%"]})
    assert clean_spans_in_place(p) == 0
    assert p["rows"][0]["output"]["entities"]["number"] == ["30", "10%"]
    # 3 numbers but unit-bearing -> not bare -> kept
    p2 = _payload({"number": ["900mm", "10%", "£3485", "135°"]})
    assert clean_spans_in_place(p2) == 0


def test_version_strings_dropped_from_number():
    p = _payload({"number": ["3.13.0", "4.4.4", "10%"]})
    clean_spans_in_place(p)
    assert p["rows"][0]["output"]["entities"]["number"] == ["10%"]   # versions dropped, unit kept


def test_company_variants_retyped():
    p = _payload({"technology": ["you tube", "what's app", "tripadvisor"]})
    clean_spans_in_place(p)
    ent = p["rows"][0]["output"]["entities"]
    assert "technology" not in ent
    assert set(ent["organization"]) == {"you tube", "what's app", "tripadvisor"}


def test_clean_noop_on_clean_payload():
    p = _payload({"organization": ["Chase", "Equifax"], "number": ["1500"]})
    assert clean_spans_in_place(p) == 0

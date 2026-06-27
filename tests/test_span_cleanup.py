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
    # Use a VALID json_structures type (status) — 'number' is an entity type and
    # is now relocated to entities by key-normalization, so it can't double as
    # the json_structures example here.
    p = {"rows": [{"row_index": 0, "row_id": "r0", "output": {
        "entities": {}, "json_structures": {"status": ["{$300.00}", "still broken"]},
    }}]}
    removed = clean_spans_in_place(p)
    assert removed == 1
    assert p["rows"][0]["output"]["json_structures"]["status"] == ["still broken"]


def test_app_names_are_NOT_blanket_retyped_to_organization():
    # Policy (2026-06-19): the blanket technology->organization retype was removed.
    # In this app-review domain, app/product mentions stay `technology`; the
    # company-vs-app call is per-mention and left to the annotator. So clean_spans
    # must NOT move google/Skype out of technology.
    p = _payload({"technology": ["google", "PyTorch", "Skype"], "organization": ["Acme"]})
    clean_spans_in_place(p)
    ent = p["rows"][0]["output"]["entities"]
    assert set(ent["technology"]) == {"google", "PyTorch", "Skype"}  # untouched
    assert ent["organization"] == ["Acme"]


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


def test_app_name_variants_stay_technology():
    # Policy (2026-06-19): blanket app->organization retype removed; app/product
    # names (incl. spacing/nickname variants) stay `technology` unless the
    # annotator tags the company-entity sense.
    p = _payload({"technology": ["you tube", "what's app", "tripadvisor"]})
    clean_spans_in_place(p)
    ent = p["rows"][0]["output"]["entities"]
    assert set(ent["technology"]) == {"you tube", "what's app", "tripadvisor"}
    assert "organization" not in ent


def test_clean_noop_on_clean_payload():
    p = _payload({"organization": ["Chase", "Equifax"], "number": ["1500"]})
    assert clean_spans_in_place(p) == 0


# ── type-key normalization (2026-06-19) ─────────────────────────────────────
# Annotators mis-section type-keys (entity types under json_structures, or vice
# versa) and emit whitespace-prefixed keys, which fail schema validation and
# bounce tasks to Human Review even though the spans are fine. Regression for
# the 43-task HR backlog (16 of which were exactly this).
from annotation_pipeline_skill.core.span_cleanup import normalize_output_keys_in_place


def test_entity_type_key_under_json_structures_is_moved_to_entities():
    payload = {"rows": [{"row_index": 0, "output": {
        "entities": {"technology": ["ppsspp"]},
        "json_structures": {"document": ["ppsspp FAQ"], "status": ["works well"]},
    }}]}
    n = normalize_output_keys_in_place(payload)
    out = payload["rows"][0]["output"]
    assert out["entities"]["document"] == ["ppsspp FAQ"]      # moved into entities
    assert "document" not in out["json_structures"]           # gone from json
    assert out["json_structures"]["status"] == ["works well"] # untouched
    assert out["entities"]["technology"] == ["ppsspp"]
    assert n >= 1


def test_time_under_json_structures_is_moved_to_entities():
    payload = {"rows": [{"row_index": 0, "output": {
        "entities": {}, "json_structures": {"time": ["seven minutes"]},
    }}]}
    normalize_output_keys_in_place(payload)
    out = payload["rows"][0]["output"]
    assert out["entities"]["time"] == ["seven minutes"]
    assert "time" not in out["json_structures"]


def test_whitespace_prefixed_keys_are_stripped():
    payload = {"rows": [{"row_index": 0, "output": {
        "entities": {"\ntechnology": ["Android"]},
        "json_structures": {"\ndecision": ["we will ship"]},
    }}]}
    normalize_output_keys_in_place(payload)
    out = payload["rows"][0]["output"]
    assert out["entities"]["technology"] == ["Android"]
    assert out["json_structures"]["decision"] == ["we will ship"]
    assert "\ntechnology" not in out["entities"]


def test_normalization_merges_and_dedupes():
    # 'document' arrives in both sections; entities is the valid home.
    payload = {"rows": [{"row_index": 0, "output": {
        "entities": {"document": ["Exhibit B"]},
        "json_structures": {"document": ["Exhibit B", "Schedule 3.1"]},
    }}]}
    normalize_output_keys_in_place(payload)
    out = payload["rows"][0]["output"]
    assert sorted(out["entities"]["document"]) == ["Exhibit B", "Schedule 3.1"]
    assert "document" not in out["json_structures"]


def test_clean_spans_in_place_applies_key_normalization():
    # End-to-end: clean_spans_in_place must also normalize keys (single wiring point).
    payload = {"rows": [{"row_index": 0, "output": {
        "entities": {}, "json_structures": {"time": ["3 years"]},
    }}]}
    clean_spans_in_place(payload)
    out = payload["rows"][0]["output"]
    assert out["entities"]["time"] == ["3 years"]
    assert "time" not in out["json_structures"]


def test_strips_disallowed_top_level_payload_keys():
    # The answer schema allows only `rows` at the top level. Some imports leaked
    # auxiliary keys (e.g. `discussion_replies`) into the answer payload, which
    # fails root-level additionalProperties and blocks corrections. Strip them.
    payload = {
        "rows": [{"row_index": 0, "output": {"entities": {}, "json_structures": {}}}],
        "discussion_replies": [{"author": "x", "text": "hi"}],
    }
    n = normalize_output_keys_in_place(payload)
    assert "discussion_replies" not in payload
    assert "rows" in payload
    assert n >= 1


def test_clean_spans_strips_top_level_and_placeholder_together():
    payload = {
        "rows": [{"row_index": 0, "output": {
            "entities": {"organization": ["ORG616", "Chase"]}, "json_structures": {}}}],
        "discussion_replies": ["leaked"],
    }
    clean_spans_in_place(payload)
    assert "discussion_replies" not in payload                       # top-level stripped
    assert payload["rows"][0]["output"]["entities"]["organization"] == ["Chase"]  # ORG616 dropped


def test_top_level_strip_noop_when_only_rows():
    payload = {"rows": [{"row_index": 0, "output": {"entities": {}, "json_structures": {}}}]}
    n = normalize_output_keys_in_place(payload)
    assert set(payload.keys()) == {"rows"}
    assert n == 0


def test_technology_phrase_stays_in_json_structures():
    # 'technology' is valid in both; keep it where the annotator put it.
    payload = {"rows": [{"row_index": 0, "output": {
        "entities": {"technology": ["Python"]},
        "json_structures": {"technology": ["Large Language Models"]},
    }}]}
    normalize_output_keys_in_place(payload)
    out = payload["rows"][0]["output"]
    assert out["entities"]["technology"] == ["Python"]
    assert out["json_structures"]["technology"] == ["Large Language Models"]

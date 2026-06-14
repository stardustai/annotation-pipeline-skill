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

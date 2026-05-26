"""Tests for _coerce_to_enum_in_place cross-field rescue logic."""
from __future__ import annotations

import pytest

from annotation_pipeline_skill.runtime.subagent_cycle import _coerce_to_enum_in_place

# Minimal resolved schema that mirrors v3_initial_deployment's output_schema.
_SCHEMA = {
    "$defs": {
        "entityType": {
            "oneOf": [
                {"const": "person"},
                {"const": "organization"},
                {"const": "project"},
                {"const": "document"},
                {"const": "time"},
                {"const": "number"},
                {"const": "event"},
                {"const": "location"},
                {"const": "technology"},
                {"const": "entity"},
            ]
        },
        "jsonStructureType": {
            "oneOf": [
                {"const": "status"},
                {"const": "risk"},
                {"const": "goal"},
                {"const": "strategy"},
                {"const": "constraint"},
                {"const": "decision"},
                {"const": "task"},
                {"const": "preference"},
                {"const": "reason"},
                {"const": "technology"},
            ]
        },
    }
}


def _payload(entities: dict, json_structures: dict) -> dict:
    return {
        "rows": [
            {
                "row_index": 0,
                "row_id": "r0",
                "output": {
                    "entities": entities,
                    "json_structures": json_structures,
                },
            }
        ]
    }


def _out(payload: dict) -> dict:
    return payload["rows"][0]["output"]


class TestRescueJstTypeUnderEntities:
    def test_misrouted_jst_type_moved_to_json_structures(self):
        p = _payload(
            entities={"technology": ["API"], "risk": ["data loss"]},
            json_structures={},
        )
        dropped, rescued = _coerce_to_enum_in_place(p, _SCHEMA)
        out = _out(p)
        assert "risk" not in out["entities"]
        assert out["json_structures"]["risk"] == ["data loss"]
        assert not dropped
        assert rescued == {"entities/risk→json_structures": 1}

    def test_rescued_spans_extend_existing_target_list(self):
        p = _payload(
            entities={"risk": ["new risk"]},
            json_structures={"risk": ["existing risk"]},
        )
        dropped, rescued = _coerce_to_enum_in_place(p, _SCHEMA)
        out = _out(p)
        assert set(out["json_structures"]["risk"]) == {"existing risk", "new risk"}
        assert rescued == {"entities/risk→json_structures": 1}

    def test_all_jst_types_under_entities_are_rescued(self):
        jst_types = ["status", "risk", "goal", "strategy", "constraint",
                     "decision", "task", "preference", "reason"]
        entities = {t: [f"span for {t}"] for t in jst_types}
        p = _payload(entities=entities, json_structures={})
        dropped, rescued = _coerce_to_enum_in_place(p, _SCHEMA)
        out = _out(p)
        for t in jst_types:
            assert t not in out["entities"], f"{t} still in entities"
            assert out["json_structures"][t] == [f"span for {t}"]
        assert not dropped
        assert sum(rescued.values()) == len(jst_types)


class TestRescueEntityTypeUnderJsonStructures:
    def test_misrouted_entity_type_moved_to_entities(self):
        p = _payload(
            entities={},
            json_structures={"risk": ["serious issue"], "number": ["42"]},
        )
        dropped, rescued = _coerce_to_enum_in_place(p, _SCHEMA)
        out = _out(p)
        assert "number" not in out["json_structures"]
        assert out["entities"]["number"] == ["42"]
        assert out["json_structures"]["risk"] == ["serious issue"]
        assert not dropped
        assert rescued == {"json_structures/number→entities": 1}


class TestTrulyInventedTypesStillDropped:
    def test_invented_type_is_dropped_not_rescued(self):
        p = _payload(
            entities={"attribute": ["some attr"], "person": ["Alice"]},
            json_structures={},
        )
        dropped, rescued = _coerce_to_enum_in_place(p, _SCHEMA)
        out = _out(p)
        assert "attribute" not in out["entities"]
        assert "attribute" not in out.get("json_structures", {})
        assert dropped == {"entities/attribute": 1}
        assert not rescued

    def test_empty_span_list_not_counted(self):
        p = _payload(
            entities={"attribute": [], "person": ["Alice"]},
            json_structures={},
        )
        dropped, rescued = _coerce_to_enum_in_place(p, _SCHEMA)
        # Empty list: not counted, but type is still removed
        assert "attribute" not in _out(p)["entities"]
        assert not dropped
        assert not rescued


class TestTechnologyValidInBoth:
    def test_technology_in_entities_stays_put(self):
        p = _payload(
            entities={"technology": ["Python"]},
            json_structures={},
        )
        dropped, rescued = _coerce_to_enum_in_place(p, _SCHEMA)
        assert _out(p)["entities"]["technology"] == ["Python"]
        assert not dropped
        assert not rescued

    def test_technology_in_json_structures_stays_put(self):
        p = _payload(
            entities={},
            json_structures={"technology": ["PostgreSQL"], "risk": ["downtime"]},
        )
        dropped, rescued = _coerce_to_enum_in_place(p, _SCHEMA)
        assert _out(p)["json_structures"]["technology"] == ["PostgreSQL"]
        assert not dropped
        assert not rescued


class TestEdgeCases:
    def test_no_schema_is_noop(self):
        p = _payload(
            entities={"risk": ["bad thing"]},
            json_structures={},
        )
        dropped, rescued = _coerce_to_enum_in_place(p, None)
        assert _out(p)["entities"]["risk"] == ["bad thing"]
        assert not dropped
        assert not rescued

    def test_non_dict_payload_is_noop(self):
        dropped, rescued = _coerce_to_enum_in_place("not a dict", _SCHEMA)
        assert not dropped
        assert not rescued

    def test_multiple_rows_all_coerced(self):
        payload = {
            "rows": [
                {
                    "row_index": i,
                    "row_id": f"r{i}",
                    "output": {
                        "entities": {"risk": [f"risk {i}"]},
                        "json_structures": {},
                    },
                }
                for i in range(3)
            ]
        }
        dropped, rescued = _coerce_to_enum_in_place(payload, _SCHEMA)
        for row in payload["rows"]:
            assert "risk" not in row["output"]["entities"]
            assert row["output"]["json_structures"]["risk"] == [f"risk {row['row_index']}"]
        assert not dropped
        assert rescued == {"entities/risk→json_structures": 3}

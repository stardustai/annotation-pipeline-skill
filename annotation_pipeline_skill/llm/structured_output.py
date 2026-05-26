"""JSON Schema builders for OpenAI / vLLM Structured Outputs (strict: True).

vLLM strict mode requirements:
  - No $ref / $defs / anyOf / oneOf / allOf
  - additionalProperties: false at every object level
  - required lists every field that has no default

QC and arbiter output shapes are fixed; Pydantic models are defined here so
schema generation is automatic. The annotation shape is per-project (derived
from output_schema.json) and is built dynamically in build_annotation_strict_schema().
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


# ── QC Pydantic models ────────────────────────────────────────────────────────

class _QCFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_id: str = ""
    category: str
    message: str
    severity: Literal["info", "warning", "error", "blocking"] = "warning"
    suggested_action: str = "annotator_rerun"
    confidence: Literal["certain", "confident", "tentative", "unsure"]


class _QCFeedbackResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_id: str
    decision: str
    reason: str


class QCResponse(BaseModel):
    """Expected output shape for the QC subagent."""
    model_config = ConfigDict(extra="forbid")
    passed: bool
    message: str = ""
    failures: list[_QCFailure] = []
    feedback_resolution: list[_QCFeedbackResolution] = []
    consensus_acknowledgements: list[str] = []


# ── Arbiter Pydantic models ───────────────────────────────────────────────────

class _ArbiterVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feedback_id: str
    verdict: Literal["annotator", "qc", "neither"]
    confidence: Literal["certain", "confident", "tentative", "unsure"]
    reasoning: str


class _ArbiterVerdicts(BaseModel):
    """Fixed part of the arbiter response (verdicts only).

    corrected_annotation is appended by build_arbiter_strict_schema() because
    its shape is derived from the project output_schema.
    """
    model_config = ConfigDict(extra="forbid")
    verdicts: list[_ArbiterVerdict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve all $ref/$defs so the schema is self-contained."""
    defs: dict[str, Any] = schema.get("$defs", {})

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                return _resolve(defs[ref_name])
            return {k: _resolve(v) for k, v in obj.items() if k != "$defs"}
        if isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    return _resolve(schema)


def _pydantic_strict(model: type[BaseModel]) -> dict[str, Any]:
    """Inline all $defs in a Pydantic-generated schema for vLLM strict compat."""
    return _inline_refs(model.model_json_schema())


# ── Per-stage schema builders ─────────────────────────────────────────────────

def build_annotation_strict_schema(output_schema: dict[str, Any]) -> dict[str, Any]:
    """Build a vLLM-strict annotation schema from the project output_schema.

    Expands entityType / jsonStructureType oneOf enums into explicit optional
    properties — no $ref, no propertyNames, no oneOf.
    """
    defs = output_schema.get("$defs", {})

    def _names(key: str) -> list[str]:
        defn = defs.get(key, {})
        if isinstance(defn.get("oneOf"), list):
            return [x["const"] for x in defn["oneOf"] if isinstance(x.get("const"), str)]
        return [v for v in defn.get("enum", []) if isinstance(v, str)]

    entity_types = _names("entityType")
    json_structure_types = _names("jsonStructureType")
    span_list: dict[str, Any] = {"type": "array", "items": {"type": "string"}}

    def _type_map(names: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {n: span_list for n in names},
            "required": [],
            "additionalProperties": False,
        }

    row_output: dict[str, Any] = {
        "type": "object",
        "properties": {
            "entities": _type_map(entity_types),
            "json_structures": _type_map(json_structure_types),
        },
        "required": ["entities", "json_structures"],
        "additionalProperties": False,
    }
    row: dict[str, Any] = {
        "type": "object",
        "properties": {
            "row_index": {"type": "integer"},
            "row_id": {"type": "string"},
            "output": row_output,
        },
        "required": ["row_index", "row_id", "output"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": row}},
        "required": ["rows"],
        "additionalProperties": False,
    }


def build_qc_strict_schema() -> dict[str, Any]:
    """Return a vLLM-strict schema for the QC subagent response."""
    return _pydantic_strict(QCResponse)


def build_arbiter_strict_schema(output_schema: dict[str, Any]) -> dict[str, Any]:
    """Return a vLLM-strict schema for the arbiter response.

    verdicts is taken from the Pydantic model; corrected_annotation mirrors
    the annotation schema and is OPTIONAL (absent from required). When the
    arbiter has no correction it must OMIT the field — the runtime reads
    .get("corrected_annotation") as None either way, so runtime logic is
    unchanged. Arbiter instructions in strict mode say 'omit the field'
    instead of 'set it to null'.
    """
    base = _pydantic_strict(_ArbiterVerdicts)
    ann = build_annotation_strict_schema(output_schema)
    base["properties"]["corrected_annotation"] = ann
    # intentionally NOT added to required — omitted field == no correction
    return base


def make_json_schema_response_format(
    schema: dict[str, Any], *, name: str
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }

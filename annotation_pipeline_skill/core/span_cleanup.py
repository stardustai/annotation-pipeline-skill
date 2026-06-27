"""Deterministic, precision-safe span cleanup for NER annotations.

Removes two classes of annotator error the strict gate flags as precision
leaks, without ever touching real entities:

* REDACTED_PLACEHOLDER — anonymization masks tagged as entities:
  ``{$...}`` money masks, ``XX/XX/XXXX`` X-masks, ``ORG\\d+`` org ids.
  These are never real entities, so dropping them can only raise precision.
* NESTED_SPAN — a span that is a proper substring of a longer span of the
  SAME type in the SAME row (e.g. ``FFT`` ⊂ ``FlashFFTConv``). The guideline
  prefers the maximal span. Restricted to spans containing a letter so
  coincidental numeric substrings (``5`` ⊂ ``1500``) are never removed.

Both edits are removals only — no span text is ever rewritten and no type is
ever changed, so recall against the gold set is unaffected (the gate already
counts these as errors, not gold entities).
"""
from __future__ import annotations

import re
from typing import Any

_FIELDS = ("entities", "json_structures")

# Valid type-keys per section (mirror output_schema.json). `technology` is the
# only type valid in BOTH sections (single-word → entities, phrase → json).
_ENTITY_TYPES = frozenset({
    "person", "organization", "project", "document", "time", "number",
    "event", "location", "technology", "generic_entity",
})
_JSON_TYPES = frozenset({
    "status", "risk", "goal", "strategy", "constraint", "decision", "task",
    "preference", "reason", "technology",
})


def normalize_output_keys_in_place(payload: Any) -> int:
    """Fix mis-sectioned / whitespace type-keys so the result is schema-valid.

    Annotators systematically (a) put an ENTITY-type key (e.g. ``document``,
    ``time``) under ``json_structures`` — where it isn't a valid key — or vice
    versa, and (b) emit whitespace-prefixed keys like ``"\\ntechnology"``. Both
    fail schema validation and bounce the task to Human Review even though the
    spans are fine. This moves each key to the section where it's valid and
    strips surrounding whitespace, merging+de-duping lists. Keys valid in
    neither section are left in place (genuinely bad → schema still rejects).
    Returns the number of keys moved or stripped.
    """
    if not isinstance(payload, dict):
        return 0
    fixed = 0
    # Strip disallowed TOP-LEVEL payload keys. The answer schema allows only
    # `rows` at the root; some imports leaked auxiliary keys (e.g.
    # `discussion_replies`) into the answer payload, which fails root-level
    # additionalProperties and silently blocks every correction to the task.
    for key in [k for k in payload if k != "rows"]:
        del payload[key]
        fixed += 1
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        src_ent = output.get("entities") if isinstance(output.get("entities"), dict) else {}
        src_js = output.get("json_structures") if isinstance(output.get("json_structures"), dict) else {}
        new_ent: dict[str, list] = {}
        new_js: dict[str, list] = {}

        def _add(bucket: dict, key: str, vals) -> None:
            dst = bucket.setdefault(key, [])
            for v in (vals or []):
                if v not in dst:
                    dst.append(v)

        for section, src in (("entities", src_ent), ("json_structures", src_js)):
            for key, vals in src.items():
                if not isinstance(key, str):
                    continue
                k = key.strip()
                if k != key:
                    fixed += 1
                if k in _ENTITY_TYPES and (section == "entities" or k not in _JSON_TYPES):
                    if section != "entities":
                        fixed += 1
                    _add(new_ent, k, vals)
                elif k in _JSON_TYPES and (section == "json_structures" or k not in _ENTITY_TYPES):
                    if section != "json_structures":
                        fixed += 1
                    _add(new_js, k, vals)
                else:
                    # Valid in neither (or ambiguous like 'technology' which is
                    # kept in its current section): leave where it is.
                    _add(new_ent if section == "entities" else new_js, k, vals)
        output["entities"] = new_ent
        output["json_structures"] = new_js
    return fixed

_MONEY_MASK = re.compile(r"^\{\$[\d.,]*\}$")
_ORG_ID = re.compile(r"^ORG\d+$")
# Only X-runs / slashes / spaces / digits, with at least two consecutive X.
_XMASK_CHARS = re.compile(r"^[X/\s\d]+$")

# Bare numeral with no unit / currency / percent (e.g. "19", "71.52", "1,500").
_BARE_NUM = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
# Multi-part software version (e.g. "3.13.0", "4.4.4") — never a `number` entity.
_VERSION = re.compile(r"^\d+\.\d+\.\d+\S*$")
# Minimum bare-numeral `number` spans in one row to treat it as sensor/telemetry
# structured metadata (and drop them). Prose rows with a legit count or two stay
# untouched; sensor logs tag 5-8 bare values per row.
_SENSOR_MIN_BARE = 3

# Blanket technology->organization retype REMOVED (2026-06-19, operator policy):
# in this app-review domain the SAME name is `organization` when it denotes the
# company/legal entity and `technology` when it denotes the app/product/service
# ("Microsoft" the company vs "Microsoft Teams" the app; "google" the company vs
# the search app). That distinction is per-MENTION and cannot be made by a flat
# span->type list — the old blanket retype force-flipped every app-mention to
# `organization`, contradicting the project's apps=technology convention and
# generating ~90 audit deviations (google x82). The per-mention call now stays
# with the annotator/arbiter (guided by the guidelines). Kept empty (not deleted)
# so a genuinely-unambiguous non-app company could be re-added if ever needed.
_COMPANIES: frozenset[str] = frozenset()


def is_redacted_placeholder(span: str) -> bool:
    """True if ``span`` is an anonymization placeholder, not a real entity."""
    if not isinstance(span, str):
        return False
    s = span.strip()
    if not s:
        return False
    if _MONEY_MASK.match(s):
        return True
    if _ORG_ID.match(s):
        return True
    if "XX" in s and _XMASK_CHARS.match(s):
        return True
    return False


def _has_alpha(s: str) -> bool:
    return any(c.isalpha() for c in s)


def clean_spans_in_place(payload: Any) -> int:
    """Drop placeholder + alpha-nested spans from ``payload`` in place.

    Returns the number of spans removed. The ``entities`` / ``json_structures``
    keys are preserved on every row (schema requires them); type keys whose
    list becomes empty are dropped.
    """
    if not isinstance(payload, dict):
        return 0
    # First fix mis-sectioned / whitespace type-keys so the rest of the cleanup
    # (and downstream schema validation) sees a well-formed structure.
    normalize_output_keys_in_place(payload)
    removed = 0
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        for field in _FIELDS:
            buckets = output.get(field)
            if not isinstance(buckets, dict):
                continue
            for typ in list(buckets.keys()):
                spans = buckets.get(typ)
                if not isinstance(spans, list):
                    continue
                # 1. drop placeholders
                after_ph = []
                for sp in spans:
                    if isinstance(sp, str) and is_redacted_placeholder(sp):
                        removed += 1
                        continue
                    after_ph.append(sp)
                # 2. drop alpha spans that are a proper substring of a longer
                #    same-type span (prefer the maximal span). ENTITIES ONLY:
                #    in json_structures a short phrase inside a longer one is
                #    usually two legitimate annotations, not a duplicate, so
                #    substring-dropping there would silently cost recall.
                if field != "entities":
                    if after_ph:
                        buckets[typ] = after_ph
                    else:
                        del buckets[typ]
                    continue
                kept = []
                for sp in after_ph:
                    if (
                        isinstance(sp, str)
                        and _has_alpha(sp)
                        and any(
                            isinstance(o, str) and o != sp and sp in o
                            for o in after_ph
                        )
                    ):
                        removed += 1
                        continue
                    kept.append(sp)
                if kept:
                    buckets[typ] = kept
                else:
                    del buckets[typ]

        # 3 + 4: entity-level post-passes (need the whole row's entities).
        ents = output.get("entities")
        if isinstance(ents, dict):
            # 3. Company retype: named companies tagged `technology` -> `organization`.
            tech = ents.get("technology")
            if isinstance(tech, list):
                move = [s for s in tech if isinstance(s, str) and s.strip().lower() in _COMPANIES]
                if move:
                    ents["technology"] = [s for s in tech if s not in move]
                    org = ents.setdefault("organization", [])
                    for s in move:
                        if s not in org:
                            org.append(s)
                    removed += len(move)
                    if not ents["technology"]:
                        del ents["technology"]
            # 4. Sensor guard: a row with >= _SENSOR_MIN_BARE bare-numeral
            #    `number` spans is telemetry/structured metadata — drop the bare
            #    ones (keep any unit-bearing numbers).
            nums = ents.get("number")
            if isinstance(nums, list):
                # 4a. Drop multi-part version strings (never a number).
                after_ver = []
                for s in nums:
                    if isinstance(s, str) and _VERSION.match(s.strip()):
                        removed += 1
                        continue
                    after_ver.append(s)
                # 4b. Sensor guard: >= _SENSOR_MIN_BARE bare-numeral spans -> telemetry.
                bare = [s for s in after_ver if isinstance(s, str) and _BARE_NUM.match(s.strip())]
                if len(bare) >= _SENSOR_MIN_BARE:
                    bareset = set(bare)
                    after_ver = [s for s in after_ver if s not in bareset]
                    removed += len(bare)
                if after_ver:
                    ents["number"] = after_ver
                else:
                    ents.pop("number", None)
    return removed

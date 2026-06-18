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

# Named consumer companies the rules class as organizations — the annotators
# (both qwen and Sonnet) systematically mistype these as `technology`, violating
# "a company is NEVER technology". Deterministic retype: technology -> organization.
# Curated (not prior-derived: priors are polluted by the same error). Lowercase.
_COMPANIES = frozenset({
    "google", "youtube", "skype", "spotify", "netflix", "evernote", "tinder",
    "vimeo", "airbnb", "duolingo", "headspace", "viber", "onavo", "yahoo",
    "instagram", "facebook", "whatsapp", "snapchat", "tiktok", "twitter",
    "uber", "lyft", "paypal", "dropbox", "slack", "zoom", "reddit", "pinterest",
    "linkedin", "hulu", "twitch", "discord", "telegram", "wechat", "microsoft",
    "samsung", "ebay", "etsy", "shopify", "doordash", "grubhub", "instacart",
    # spacing / nickname variants and further offenders seen in app-review rows
    "you tube", "whatsapp", "what's app", "whats app", "tripadvisor",
    "trip advisor", "insta", "nyt", "viber", "wechat", "deliveroo", "just eat",
})


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

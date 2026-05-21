"""Shared wordfreq scoring helpers."""
from __future__ import annotations


def wordfreq_score(span: str) -> float:
    """Average Zipf frequency over the tokens of ``span``.

    Auto-detects CJK vs English based on whether the span contains
    CJK Unified Ideographs. Returns 0.0 for empty or untokenizable input.
    """
    if not span:
        return 0.0
    from wordfreq import zipf_frequency, tokenize

    lang = "zh" if any("一" <= ch <= "鿿" for ch in span) else "en"
    tokens = tokenize(span, lang)
    if not tokens:
        return 0.0
    return sum(zipf_frequency(t, lang) for t in tokens) / len(tokens)

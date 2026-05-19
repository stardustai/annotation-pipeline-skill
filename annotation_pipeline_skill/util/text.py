"""Small text utilities shared across services and the API layer."""

from __future__ import annotations

import re

# Token = one CJK ideograph / Japanese kana / Korean hangul char (each
# counts as a word in languages without whitespace word-delimiters),
# OR a run of non-whitespace ASCII / Latin / number / punctuation. The
# alternation handles mixed-script text correctly: regex tries the CJK
# branch first at every position so CJK runs are split per-character
# instead of greedily eaten by the \S+ branch.
_WORD_TOKEN_RE = re.compile(r"[一-鿿぀-ヿ가-힯]|\S+")


def truncate_to_words(text: str, max_words: int = 100) -> str:
    """Return ``text`` truncated at the boundary of the ``max_words``-th
    token.

    "Word" here means one of:
      - a single CJK ideograph, kana, or hangul character
      - a contiguous non-whitespace run of any other characters

    Preserves the original text including its interleaved whitespace
    up to that boundary; just chops at the boundary without inserting
    an ellipsis (callers can append "…" if they want).

    Returns the input unchanged when it has fewer than ``max_words``
    tokens or when ``max_words`` is non-positive.
    """
    if not text or max_words <= 0:
        return text
    count = 0
    last_end = 0
    for m in _WORD_TOKEN_RE.finditer(text):
        count += 1
        last_end = m.end()
        if count >= max_words:
            return text[:last_end]
    return text

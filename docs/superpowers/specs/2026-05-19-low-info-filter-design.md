# Low-Info Filter & Naming Alignment — Design Spec

**Date:** 2026-05-19  
**Status:** Approved

---

## Problem

1. **Naming mismatch.** The internal key `contested_spans` has always mapped to the UI tab "Divergent Annotations" (spans with split type distributions). `task_deviations` maps to "Task Deviations". The mismatch causes confusion in code and discussion.

2. **Noise in Divergent Entries.** Generic phrases ("very nice", "love it", "智能推荐系统") appear as divergent entries because annotators disagree on their type. These are low-information spans that should be bulk-set to `not_an_entity` rather than adjudicated one-by-one.

---

## Goals

- Rename `contested_spans` → `divergent_entries` everywhere (code + cache + frontend).
- Add `type_entropy` to each `divergent_entries` item for sorting.
- Add a new `low_info_entries` list (high-wordfreq spans) to the posterior audit output.
- New **Low Info Entries** tab in Posterior Audit UI with per-row and bulk `not_an_entity` actions.

---

## Non-Goals

- Changing the `task_deviations` structure or its UI tab.
- Auto-applying `not_an_entity` without operator confirmation.
- Training custom models or building corpus-specific IDF.

---

## Data Structure

`build_posterior_audit` returns:

```python
{
  "task_deviations": [...],          # unchanged

  "divergent_entries": [             # renamed from contested_spans
    {
      "span": str,
      "prior_total": int,
      "prior_distribution": dict[str, int],
      "top_share": float,
      "runner_up_share": float,
      "resolved_convention_type": str | None,
      "type_entropy": float,         # NEW: Shannon H from prior_distribution
    }
  ],

  "low_info_entries": [              # NEW
    {
      "span": str,
      "prior_total": int,
      "prior_distribution": dict[str, int],
      "wordfreq": float,             # Zipf avg over span tokens, 0–7
    }
  ]
}
```

---

## Scoring

### `type_entropy` (added to `divergent_entries`)

Shannon entropy of the type distribution:

```python
def _type_entropy(dist: dict[str, int]) -> float:
    total = sum(dist.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in dist.values() if c > 0)
```

Higher entropy = more annotator disagreement within the contested set.

### `wordfreq` (in `low_info_entries`)

Average Zipf frequency of tokens in the span using the `wordfreq` library (no training required, supports Chinese + English):

```python
from wordfreq import zipf_frequency, tokenize

def _wordfreq_score(span: str) -> float:
    lang = "zh" if any('一' <= c <= '鿿' for c in span) else "en"
    tokens = tokenize(span, lang)
    if not tokens:
        return 0.0
    return sum(zipf_frequency(t, lang) for t in tokens) / len(tokens)
```

Language detection: CJK characters → `"zh"`, otherwise `"en"`. No additional dependency needed.

Zipf scale reference:
- ≥ 6: extremely common ("very", "nice", "the")
- 4–6: common everyday words ("love", "update", "system")
- < 4: domain-specific or rare ("kubernetes", "tensorflow", "rooflights")

### `low_info_entries` generation

Derived from `divergent_entries`:
- Filter: **no active convention** AND `wordfreq >= 4.0`
- Sort: `wordfreq` descending

Threshold 4.0 is the initial default; can be made a project-level config later.

---

## Rename Scope

| Old | New | Where |
|-----|-----|-------|
| `contested_spans` | `divergent_entries` | `build_posterior_audit` return dict |
| `contested_spans` | `divergent_entries` | `_post_posterior_audit_retroactive_fix` cache surgery |
| `contested_spans` | `divergent_entries` | `PosteriorAuditPanel.tsx` (all references) |
| `contested_spans` | `divergent_entries` | `batch_apply_all_contested.py` cache read |

**Cache migration:** after rename, invalidate existing cached payloads by clearing the `posterior_audit_cache` table. Operator must trigger a rebuild.

---

## UI Changes

### Divergent Entries tab (existing)

- Add `type_entropy` column, sortable.
- No other changes.

### Low Info Entries tab (new)

Columns: **Span** | **Wordfreq** | **Distribution** | **Total** | **Action**

- Each row: `Set not_an_entity` button → calls existing `/api/posterior-audit/retroactive-fix` with `entity_type: null`.
- Top bar: multi-select checkboxes + `Set not_an_entity for selected (N)` bulk button.
- Default sort: `wordfreq` descending.
- Reuses the existing retroactive-fix progress/result display pattern from Divergent Entries.

---

## Dependencies

- `wordfreq` (pip) — multilingual, no model download, ~30 MB data files bundled.

---

## Testing

- Unit test `_type_entropy`: zero-count input, single-type, uniform split.
- Unit test `_wordfreq_score`: English phrase, Chinese phrase, OOV token (returns 0, average degrades gracefully).
- Integration: `build_posterior_audit` output contains all three keys; `low_info_entries` is a subset of `divergent_entries` items.
- Frontend: after cache clear + rebuild, all three keys present and rendered correctly.

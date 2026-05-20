# Low-Info Filter & Naming Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `contested_spans` → `divergent_entries` everywhere, add `type_entropy` to divergent entries, add `low_info_entries` (high-wordfreq spans) to posterior audit, and surface a new Low Info Entries tab in the UI with per-row and bulk `not_an_entity` actions.

**Architecture:** Backend computes `_type_entropy` and `_wordfreq_score` inside `build_posterior_audit`; `low_info_entries` is derived from `divergent_entries` (no active convention + wordfreq ≥ 4.0). Frontend adds a third subtab and reuses the existing retroactive-fix sweep pattern for Low Info actions.

**Tech Stack:** Python (`wordfreq>=3.0`, `math`), TypeScript/React (new `LowInfoTable` component), SQLite cache (clear required after rename).

---

## File Map

| File | Change |
|------|--------|
| `pyproject.toml` | Add `wordfreq>=3.0` dependency |
| `annotation_pipeline_skill/services/entity_statistics_service.py` | Rename method `contested_spans` → `divergent_entries` |
| `annotation_pipeline_skill/interfaces/api.py` | Add `_type_entropy`/`_wordfreq_score` helpers; rename callers; add `type_entropy` field; add `low_info_entries`; update cache surgery |
| `scripts/batch_apply_all_contested.py` | Update cache read key at Phase 2 |
| `web/src/types.ts` | Rename `ContestedSpan` → `DivergentEntry` + add fields; add `LowInfoEntry`; update `PosteriorAudit` |
| `web/src/components/PosteriorAuditPanel.tsx` | Update all refs; add `type_entropy` column; add `LowInfoTable` component + tab |
| `tests/test_entity_statistics_service.py` | Rename `test_contested_spans`; call `divergent_entries` |
| `tests/test_posterior_audit_api.py` | Update key names; assert `type_entropy`; assert `low_info_entries` |
| `tests/test_low_info_helpers.py` | New: unit tests for `_type_entropy` and `_wordfreq_score` |

---

### Task 1: Add `wordfreq` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"wordfreq>=3.0"` to the `dependencies` list (after `"httpx>=0.27"`):

```toml
dependencies = [
  "openai>=2.0",
  "pydantic>=2.0",
  "pyyaml>=6.0",
  "jsonschema>=4.0",
  "robust-json-parser>=0.1",
  "datasketch>=1.6",
  "numpy>=1.26",
  "httpx>=0.27",
  "wordfreq>=3.0",
  "umap-learn>=0.5",
  "hdbscan>=0.8",
  "matplotlib>=3.8",
]
```

- [ ] **Step 2: Install and verify**

```bash
pip install wordfreq
python -c "from wordfreq import zipf_frequency, tokenize; print(zipf_frequency('nice', 'en'))"
```

Expected output: a float around 5.x (common word).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add wordfreq>=3.0 dependency"
```

---

### Task 2: Rename `contested_spans` → `divergent_entries` in `entity_statistics_service.py`

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_statistics_service.py:235`
- Test: `tests/test_entity_statistics_service.py:66`

- [ ] **Step 1: Write the failing test**

In `tests/test_entity_statistics_service.py`, rename `test_contested_spans` and update the method call:

```python
def test_divergent_entries(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    # Contested: 13 org + 12 project + 5 tech (top=43%, runner-up=40%)
    for _ in range(13):
        svc.increment(project_id="p", span="Microsoft", entity_type="organization")
    for _ in range(12):
        svc.increment(project_id="p", span="Microsoft", entity_type="project")
    for _ in range(5):
        svc.increment(project_id="p", span="Microsoft", entity_type="technology")
    # Not contested: 9 org + 1 project (dominant > 80%)
    for _ in range(9):
        svc.increment(project_id="p", span="Apple", entity_type="organization")
    svc.increment(project_id="p", span="Apple", entity_type="project")

    entries = svc.divergent_entries(project_id="p")
    assert len(entries) == 1
    assert entries[0]["span"] == "Microsoft" or entries[0]["span"] == "microsoft"
    assert entries[0]["prior_total"] == 30
    assert entries[0]["prior_distribution"] == {"organization": 13, "project": 12, "technology": 5}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_entity_statistics_service.py::test_divergent_entries -v
```

Expected: FAIL — `AttributeError: 'EntityStatisticsService' object has no attribute 'divergent_entries'`

- [ ] **Step 3: Rename the method**

In `annotation_pipeline_skill/services/entity_statistics_service.py` at line 235, change `def contested_spans(` to `def divergent_entries(`:

```python
    def divergent_entries(self, *, project_id: str) -> list[dict[str, Any]]:
        """Return spans where the prior distribution has no clear winner.
        ...
```

(Keep the entire method body unchanged.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_entity_statistics_service.py::test_divergent_entries -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: no failures (the old `test_contested_spans` is gone; all other tests pass).

- [ ] **Step 6: Commit**

```bash
git add annotation_pipeline_skill/services/entity_statistics_service.py tests/test_entity_statistics_service.py
git commit -m "refactor: rename contested_spans → divergent_entries in EntityStatisticsService"
```

---

### Task 3: Add helpers + rename + compute `type_entropy` and `low_info_entries` in `api.py`

**Files:**
- Modify: `annotation_pipeline_skill/interfaces/api.py:117–225` and `1876–1882`
- Test: `tests/test_low_info_helpers.py` (new), `tests/test_posterior_audit_api.py`

- [ ] **Step 1: Write the failing unit tests for helpers**

Create `tests/test_low_info_helpers.py`:

```python
import math
import pytest


def test_type_entropy_empty():
    from annotation_pipeline_skill.interfaces.api import _type_entropy
    assert _type_entropy({}) == 0.0


def test_type_entropy_single_type():
    from annotation_pipeline_skill.interfaces.api import _type_entropy
    assert _type_entropy({"organization": 10}) == pytest.approx(0.0)


def test_type_entropy_uniform_split():
    from annotation_pipeline_skill.interfaces.api import _type_entropy
    # 50/50 split → H = 1.0 bit
    result = _type_entropy({"organization": 5, "project": 5})
    assert result == pytest.approx(1.0)


def test_type_entropy_three_way():
    from annotation_pipeline_skill.interfaces.api import _type_entropy
    # 3-way equal → H = log2(3) ≈ 1.585
    result = _type_entropy({"a": 1, "b": 1, "c": 1})
    assert result == pytest.approx(math.log2(3), rel=1e-6)


def test_wordfreq_score_english():
    from annotation_pipeline_skill.interfaces.api import _wordfreq_score
    score = _wordfreq_score("very nice")
    assert score >= 4.0  # "very" and "nice" are extremely common


def test_wordfreq_score_chinese():
    from annotation_pipeline_skill.interfaces.api import _wordfreq_score
    # 系统 (system) is a very common Chinese word
    score = _wordfreq_score("系统")
    assert score > 0.0


def test_wordfreq_score_empty():
    from annotation_pipeline_skill.interfaces.api import _wordfreq_score
    assert _wordfreq_score("") == 0.0


def test_wordfreq_score_oov():
    from annotation_pipeline_skill.interfaces.api import _wordfreq_score
    # A truly rare/invented token returns 0 from zipf_frequency; average degrades gracefully.
    score = _wordfreq_score("xyzzyqwerty")
    assert score >= 0.0  # no crash, some score
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_low_info_helpers.py -v
```

Expected: FAIL — `ImportError: cannot import name '_type_entropy' from 'annotation_pipeline_skill.interfaces.api'`

- [ ] **Step 3: Add the helper functions to `api.py`**

Insert the following two functions after the `build_posterior_audit` function (after line 225, before `find_typical_text_for_span`):

```python
def _type_entropy(dist: dict[str, int]) -> float:
    import math as _math
    total = sum(dist.values())
    if not total:
        return 0.0
    return -sum((c / total) * _math.log2(c / total) for c in dist.values() if c > 0)


def _wordfreq_score(span: str) -> float:
    from wordfreq import zipf_frequency, tokenize
    lang = "zh" if any("一" <= ch <= "鿿" for ch in span) else "en"
    tokens = tokenize(span, lang)
    if not tokens:
        return 0.0
    return sum(zipf_frequency(t, lang) for t in tokens) / len(tokens)
```

- [ ] **Step 4: Verify helpers pass**

```bash
pytest tests/test_low_info_helpers.py -v
```

Expected: all PASS

- [ ] **Step 5: Write the failing integration test**

In `tests/test_posterior_audit_api.py`, update the test function name and assertions:

```python
def test_posterior_audit_returns_task_deviations_and_divergent_entries(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)

    # Build prior: 12 Apple → organization (dominant, eligible)
    for _ in range(12):
        svc.increment(project_id="p", span="Apple", entity_type="organization")
    # Divergent: Microsoft has 13/12/5
    for _ in range(13):
        svc.increment(project_id="p", span="Microsoft", entity_type="organization")
    for _ in range(12):
        svc.increment(project_id="p", span="Microsoft", entity_type="project")
    for _ in range(5):
        svc.increment(project_id="p", span="Microsoft", entity_type="technology")

    # Create an accepted task whose annotation tags Apple as technology (diverges from prior).
    task = Task.new(
        task_id="t-dev", pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "Apple"}],
        }},
    )
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    rel = "artifact_payloads/t-dev/final.json"
    abs_path = store.root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(json.dumps({"text": json.dumps({
        "rows": [{"row_index": 0,
                  "output": {"entities": {"technology": ["Apple"]}}}]
    })}))
    store.append_artifact(ArtifactRef.new(
        task_id="t-dev", kind="annotation_result", path=rel,
        content_type="application/json",
    ))

    from annotation_pipeline_skill.interfaces.api import build_posterior_audit
    payload = build_posterior_audit(store, project_id="p")

    assert any(d["span"] == "Apple" and d["current_type"] == "technology"
               for d in payload["task_deviations"])
    # divergent_entries entries come from entity_statistics (span stored as lower-case).
    assert any(c["span"].lower() == "microsoft"
               for c in payload["divergent_entries"])
    # Each divergent entry must have type_entropy >= 0.
    for entry in payload["divergent_entries"]:
        assert "type_entropy" in entry
        assert entry["type_entropy"] >= 0.0
    # low_info_entries must be present (may be empty for these spans).
    assert "low_info_entries" in payload
    assert isinstance(payload["low_info_entries"], list)
```

- [ ] **Step 6: Run to verify failure**

```bash
pytest tests/test_posterior_audit_api.py::test_posterior_audit_returns_task_deviations_and_divergent_entries -v
```

Expected: FAIL — `KeyError: 'divergent_entries'` (old key `contested_spans` in return dict)

- [ ] **Step 7: Update `build_posterior_audit` in `api.py`**

Replace lines 215–225 of `api.py`:

```python
    contested_all = svc.contested_spans(project_id=project_id)
    contested = []
    for c in contested_all:
        conv_type = convention_index.get(c.get("span", "").lower())
        if conv_type is not None:
            c = {**c, "resolved_convention_type": conv_type}
        contested.append(c)
    return {
        "task_deviations": deviations,
        "contested_spans": contested,
    }
```

with:

```python
    divergent_all = svc.divergent_entries(project_id=project_id)
    divergent_entries = []
    for c in divergent_all:
        conv_type = convention_index.get(c.get("span", "").lower())
        entry = {**c, "type_entropy": _type_entropy(c.get("prior_distribution", {}))}
        if conv_type is not None:
            entry = {**entry, "resolved_convention_type": conv_type}
        divergent_entries.append(entry)

    # low_info_entries: divergent spans with no active convention and high wordfreq
    LOW_INFO_THRESHOLD = 4.0
    low_info_entries = []
    for entry in divergent_entries:
        if entry.get("resolved_convention_type"):
            continue
        wf = _wordfreq_score(entry["span"])
        if wf >= LOW_INFO_THRESHOLD:
            low_info_entries.append({
                "span": entry["span"],
                "prior_total": entry["prior_total"],
                "prior_distribution": entry["prior_distribution"],
                "wordfreq": round(wf, 3),
            })
    low_info_entries.sort(key=lambda r: r["wordfreq"], reverse=True)

    return {
        "task_deviations": deviations,
        "divergent_entries": divergent_entries,
        "low_info_entries": low_info_entries,
    }
```

- [ ] **Step 8: Run integration test to verify it passes**

```bash
pytest tests/test_posterior_audit_api.py::test_posterior_audit_returns_task_deviations_and_divergent_entries -v
```

Expected: PASS

- [ ] **Step 9: Update cache surgery in `_post_posterior_audit_retroactive_fix`**

In `api.py` at lines 1876–1882, replace:

```python
                contested = cache_payload.get("contested_spans", [])
                kept_contested = [
                    c for c in contested
                    if (c.get("span") or "").lower() != span_lower
                ]
                cache_payload["task_deviations"] = kept_devs
                cache_payload["contested_spans"] = kept_contested
```

with:

```python
                divergent = cache_payload.get("divergent_entries", [])
                kept_divergent = [
                    c for c in divergent
                    if (c.get("span") or "").lower() != span_lower
                ]
                low_info = cache_payload.get("low_info_entries", [])
                kept_low_info = [
                    c for c in low_info
                    if (c.get("span") or "").lower() != span_lower
                ]
                cache_payload["task_deviations"] = kept_devs
                cache_payload["divergent_entries"] = kept_divergent
                cache_payload["low_info_entries"] = kept_low_info
```

- [ ] **Step 10: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 11: Commit**

```bash
git add annotation_pipeline_skill/interfaces/api.py \
        tests/test_low_info_helpers.py \
        tests/test_posterior_audit_api.py
git commit -m "feat(posterior-audit): add _type_entropy/_wordfreq_score, divergent_entries, low_info_entries"
```

---

### Task 4: Update `batch_apply_all_contested.py` Phase 2 cache read

**Files:**
- Modify: `scripts/batch_apply_all_contested.py:213`

- [ ] **Step 1: Update the cache key**

In `scripts/batch_apply_all_contested.py` at line 213, replace:

```python
    contested = cache_payload.get("contested_spans", [])
```

with:

```python
    contested = cache_payload.get("divergent_entries", [])
```

- [ ] **Step 2: Verify no test failures**

```bash
pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add scripts/batch_apply_all_contested.py
git commit -m "fix(batch-script): read divergent_entries key from cache (contested_spans renamed)"
```

---

### Task 5: Update TypeScript types in `web/src/types.ts`

**Files:**
- Modify: `web/src/types.ts:301–323`

- [ ] **Step 1: Replace the old types**

Replace lines 301–323 of `web/src/types.ts`:

```typescript
export type TaskDeviation = {
  task_id: string;
  row_index: number;
  span: string;
  current_type: string;
  prior_dominant_type: string;
  prior_distribution: Record<string, number>;
  prior_total: number;
};

export type ContestedSpan = {
  span: string;
  prior_total: number;
  prior_distribution: Record<string, number>;
  top_share: number;
  runner_up_share: number;
  resolved_convention_type?: string;
};

export type PosteriorAudit = {
  task_deviations: TaskDeviation[];
  contested_spans: ContestedSpan[];
};
```

with:

```typescript
export type TaskDeviation = {
  task_id: string;
  row_index: number;
  span: string;
  current_type: string;
  prior_dominant_type: string;
  prior_distribution: Record<string, number>;
  prior_total: number;
};

export type DivergentEntry = {
  span: string;
  prior_total: number;
  prior_distribution: Record<string, number>;
  top_share: number;
  runner_up_share: number;
  type_entropy: number;
  resolved_convention_type?: string;
};

export type LowInfoEntry = {
  span: string;
  prior_total: number;
  prior_distribution: Record<string, number>;
  wordfreq: number;
};

export type PosteriorAudit = {
  task_deviations: TaskDeviation[];
  divergent_entries: DivergentEntry[];
  low_info_entries: LowInfoEntry[];
};
```

- [ ] **Step 2: Build TypeScript to verify no type errors**

```bash
cd web && npx tsc --noEmit 2>&1 | head -30
```

Expected: errors from `PosteriorAuditPanel.tsx` (uses old names) — that's expected; we'll fix in the next task.

- [ ] **Step 3: Commit**

```bash
git add web/src/types.ts
git commit -m "feat(types): DivergentEntry + LowInfoEntry + updated PosteriorAudit type"
```

---

### Task 6: Update `PosteriorAuditPanel.tsx`

**Files:**
- Modify: `web/src/components/PosteriorAuditPanel.tsx`

This task has three parts: (A) rename all refs + add type_entropy column to existing ContestedTable, (B) add LowInfoTable component, (C) wire up the third subtab.

#### Part A: Rename refs + add type_entropy column

- [ ] **Step 1: Update imports and top-level variable**

At line 2, change:
```typescript
import type { PosteriorAudit, TaskDeviation, ContestedSpan } from "../types";
```
to:
```typescript
import type { PosteriorAudit, TaskDeviation, DivergentEntry, LowInfoEntry } from "../types";
```

At line 36, change:
```typescript
type Subtab = "deviations" | "contested";
```
to:
```typescript
type Subtab = "deviations" | "contested" | "low_info";
```

At line 183, change:
```typescript
  const contested = payload?.contested_spans ?? [];
```
to:
```typescript
  const contested = payload?.divergent_entries ?? [];
  const lowInfo = payload?.low_info_entries ?? [];
```

- [ ] **Step 2: Update the empty-state check (lines 252–257)**

Change:
```typescript
      {cachedExists &&
       deviations.length === 0 &&
       contested.length === 0 ? (
```
to:
```typescript
      {cachedExists &&
       deviations.length === 0 &&
       contested.length === 0 &&
       lowInfo.length === 0 ? (
```

- [ ] **Step 3: Update tab visibility condition (line 259)**

Change:
```typescript
      {cachedExists && (deviations.length > 0 || contested.length > 0) ? (
```
to:
```typescript
      {cachedExists && (deviations.length > 0 || contested.length > 0 || lowInfo.length > 0) ? (
```

- [ ] **Step 4: Add Low Info Entries tab button (after the Divergent annotations tab button, after line 284)**

Add a third button in the `<nav>` block:
```tsx
              <button
                className={subtab === "low_info" ? "view-tab selected" : "view-tab"}
                type="button"
                onClick={() => { setSubtab("low_info"); setFilter(""); }}
              >
                Low info entries ({lowInfo.length})
              </button>
```

- [ ] **Step 5: Update filter placeholder for the new tab (line 289)**

Change:
```typescript
              placeholder={
                subtab === "deviations" ? "Filter task / span / type…" : "Filter span / type…"
              }
```
to:
```typescript
              placeholder={
                subtab === "deviations" ? "Filter task / span / type…" : "Filter span…"
              }
```

- [ ] **Step 6: Hide "Save as convention" checkbox on low_info subtab (line 295)**

Change:
```typescript
            {subtab === "contested" ? (
```
to:
```typescript
            {subtab === "contested" ? (
```
(No change needed here — the checkbox already only shows on `"contested"`, which is correct.)

- [ ] **Step 7: Update `ContestedTable` prop type (line 997)**

Change:
```typescript
  items: ContestedSpan[];
```
to:
```typescript
  items: DivergentEntry[];
```

And in the function signature at line 988:
```typescript
function ContestedTable({
  items,
  ...
}: {
  items: DivergentEntry[];
```

- [ ] **Step 8: Add `type_entropy` column header to ContestedTable (after "Total" column, around line 1308)**

Change the `<thead>` of ContestedTable from:
```tsx
              <th style={TH_FIRST}>Span</th>
              <th style={TH}>Total</th>
              <th style={{ ...TH, width: "20%" }}>Sample text</th>
```
to:
```tsx
              <th style={TH_FIRST}>Span</th>
              <th style={TH}>Total</th>
              <th style={{ ...TH, width: "6%" }} title="Shannon entropy of type distribution (higher = more disagreement)">Entropy</th>
              <th style={{ ...TH, width: "18%" }}>Sample text</th>
```

- [ ] **Step 9: Add `type_entropy` cell to each ContestedTable row**

In the `<tbody>` row after the `<td style={TD}>{c.prior_total}</td>` cell, add:
```tsx
                  <td style={{ ...TD, fontSize: "0.8rem", color: "var(--muted, #6b7280)" }}>
                    {c.type_entropy.toFixed(2)}
                  </td>
```

- [ ] **Step 10: Also add usage of `lowInfo` in the subtab render**

After the `{subtab === "contested" ? (...)  : null}` block (around line 380), add:

```tsx
          {subtab === "low_info" ? (
            <>
              <div style={FORMULA_BLOCK_STYLE}>
                <strong>Low info entries</strong> — divergent spans with no active convention whose
                tokens score ≥&nbsp;<code>4.0</code> on the Zipf frequency scale (common everyday
                words). Set to <code>not_an_entity</code> in bulk rather than adjudicating one-by-one.
                <br />
                <span style={{ color: "var(--muted, #6b7280)" }}>
                  Scored via <code>wordfreq</code> (multilingual; Zipf 0–7, 6+ = "the / very / nice").
                </span>
              </div>
              {lowInfo.length > 0 ? (
                <LowInfoTable
                  items={lowInfo}
                  projectId={projectId!}
                  storeKey={storeKey ?? null}
                  onAfterFix={reloadCache}
                  externalFilter={filter}
                />
              ) : (
                <p className="runtime-muted">No low-info spans above threshold.</p>
              )}
            </>
          ) : null}
```

- [ ] **Step 11: Build to check progress**

```bash
cd web && npx tsc --noEmit 2>&1 | head -30
```

Expected: errors about `LowInfoTable` not defined — that's fine, we add it next.

#### Part B: Add `LowInfoTable` component

- [ ] **Step 12: Add `LowInfoTable` at the end of the file**

Append the following component after the closing brace of `ContestedTable`:

```tsx
const LOW_INFO_PAGE_SIZE = 30;

function LowInfoTable({
  items,
  projectId,
  storeKey,
  onAfterFix,
  externalFilter,
}: {
  items: LowInfoEntry[];
  projectId: string;
  storeKey: string | null;
  onAfterFix?: () => void;
  externalFilter?: string;
}): React.ReactElement {
  const filter = externalFilter ?? "";
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [rowStatus, setRowStatus] = useState<Record<string, string>>({});
  const [bulkRunning, setBulkRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const lower = filter.trim().toLowerCase();
  const filtered = lower
    ? items.filter((c) => c.span.toLowerCase().includes(lower))
    : items;
  useEffect(() => { setPage(0); }, [filter]);
  useEffect(() => {
    const maxPage = Math.max(0, Math.ceil(filtered.length / LOW_INFO_PAGE_SIZE) - 1);
    if (page > maxPage) setPage(maxPage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtered.length]);
  const visible = filtered.slice(page * LOW_INFO_PAGE_SIZE, (page + 1) * LOW_INFO_PAGE_SIZE);

  async function applyNotAnEntity(span: string) {
    setRowStatus((s) => ({ ...s, [span]: "submitting" }));
    try {
      const storeQ = storeKey ? `?store=${encodeURIComponent(storeKey)}` : "";
      const baseBody = {
        project_id: projectId,
        span,
        entity_type: "not_an_entity",
        actor: "low_info_ui",
        set_convention: false,
      };
      const BATCH = 10;
      const initialResp = await fetch(`/api/posterior-audit/retroactive-fix${storeQ}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...baseBody, batch_size: BATCH }),
      });
      if (!initialResp.ok) {
        const txt = await initialResp.text();
        throw new Error(`HTTP ${initialResp.status}: ${txt.slice(0, 200)}`);
      }
      const initialData = (await initialResp.json()) as {
        fixed: number; skipped: number;
        errors: { task_id: string; reason: string }[];
        candidate_task_ids: string[] | null;
      };
      const allCandidates = initialData.candidate_task_ids ?? [];
      let cursor = initialData.fixed + initialData.errors.length;
      while (cursor < allCandidates.length) {
        const slice = allCandidates.slice(cursor, cursor + BATCH);
        const r = await fetch(`/api/posterior-audit/retroactive-fix${storeQ}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ...baseBody, task_ids: slice, batch_size: BATCH }),
        });
        if (!r.ok) {
          const txt = await r.text();
          throw new Error(`HTTP ${r.status}: ${txt.slice(0, 200)}`);
        }
        cursor += slice.length;
      }
      try {
        await fetch(`/api/entity-statistics/recount${storeQ}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ project_id: projectId, span }),
        });
      } catch { /* best-effort */ }
      setRowStatus((s) => ({ ...s, [span]: "done" }));
    } catch (e) {
      setRowStatus((s) => ({
        ...s,
        [span]: `error: ${e instanceof Error ? e.message : String(e)}`,
      }));
    }
  }

  async function applyBulk() {
    if (selected.size === 0) return;
    if (!window.confirm(
      `Set ${selected.size} span(s) to not_an_entity? This patches all matching accepted task annotations.`,
    )) return;
    setBulkRunning(true);
    setError(null);
    const spans = Array.from(selected);
    for (const span of spans) {
      await applyNotAnEntity(span);
    }
    setBulkRunning(false);
    setSelected(new Set());
    onAfterFix?.();
  }

  function toggleSelect(span: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(span)) next.delete(span);
      else next.add(span);
      return next;
    });
  }

  function toggleAll() {
    if (visible.every((c) => selected.has(c.span))) {
      setSelected((prev) => {
        const next = new Set(prev);
        for (const c of visible) next.delete(c.span);
        return next;
      });
    } else {
      setSelected((prev) => {
        const next = new Set(prev);
        for (const c of visible) next.add(c.span);
        return next;
      });
    }
  }

  const allVisibleSelected = visible.length > 0 && visible.every((c) => selected.has(c.span));

  return (
    <>
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", margin: "0.4rem 0", fontSize: "0.8rem" }}>
        <span style={{ color: "var(--muted, #6b7280)" }}>
          {filter ? `${filtered.length} of ${items.length}` : `${items.length} total`}
          {selected.size > 0 ? ` · ${selected.size} selected` : ""}
        </span>
        {selected.size > 0 ? (
          <button
            type="button"
            disabled={bulkRunning}
            onClick={applyBulk}
            style={{
              fontSize: "0.8rem",
              background: "var(--danger, #b91c1c)",
              color: "white",
              border: "none",
              padding: "2px 10px",
              borderRadius: "4px",
              cursor: bulkRunning ? "wait" : "pointer",
              opacity: bulkRunning ? 0.7 : 1,
            }}
          >
            {bulkRunning ? "Applying…" : `Set not_an_entity for selected (${selected.size})`}
          </button>
        ) : null}
      </div>
      {error ? <div className="notice compact">{error}</div> : null}
      <Pagination
        total={filtered.length}
        page={page}
        pageSize={LOW_INFO_PAGE_SIZE}
        onPageChange={setPage}
      />
      <div className="runtime-card">
        <table style={TABLE_STYLE}>
          <thead>
            <tr style={THEAD_ROW}>
              <th style={{ ...TH_FIRST, width: "3%" }}>
                <input
                  type="checkbox"
                  checked={allVisibleSelected}
                  onChange={toggleAll}
                  style={{ margin: 0, cursor: "pointer" }}
                  title="Select/deselect all visible rows"
                />
              </th>
              <th style={TH_FIRST}>Span</th>
              <th style={{ ...TH, width: "8%" }} title="Average Zipf frequency of tokens (0–7 scale; ≥6 = 'the/very/nice')">Wordfreq</th>
              <th style={{ ...TH, width: "22%" }}>Distribution</th>
              <th style={{ ...TH, width: "8%" }}>Total</th>
              <th style={{ ...TH, width: "20%" }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((c) => {
              const status = rowStatus[c.span];
              const isDone = status === "done";
              const isSubmitting = status === "submitting";
              const isErr = status?.startsWith("error:");
              return (
                <tr key={c.span} style={{ ...TR, opacity: isDone ? 0.5 : 1 }}>
                  <td style={{ ...TD, width: "3%" }}>
                    <input
                      type="checkbox"
                      checked={selected.has(c.span)}
                      onChange={() => toggleSelect(c.span)}
                      disabled={isDone || isSubmitting}
                      style={{ margin: 0, cursor: "pointer" }}
                    />
                  </td>
                  <td style={{ ...TD_MONO }}>{c.span}</td>
                  <td style={{ ...TD, fontVariantNumeric: "tabular-nums" }}>{c.wordfreq.toFixed(2)}</td>
                  <td style={TD}>
                    <DistributionBar distribution={c.prior_distribution} total={c.prior_total} />
                  </td>
                  <td style={TD}>{c.prior_total}</td>
                  <td style={TD}>
                    {isDone ? (
                      <span style={{ fontSize: "0.8rem", color: "var(--success, #047857)" }}>✓ done</span>
                    ) : isErr ? (
                      <span style={{ fontSize: "0.75rem", color: "var(--danger, #b91c1c)" }}>{status?.slice(7)}</span>
                    ) : (
                      <button
                        type="button"
                        disabled={isSubmitting}
                        onClick={() => applyNotAnEntity(c.span).then(() => onAfterFix?.())}
                        style={{
                          fontSize: "0.8rem",
                          background: "var(--danger, #b91c1c)",
                          color: "white",
                          border: "none",
                          padding: "2px 8px",
                          borderRadius: "4px",
                          cursor: isSubmitting ? "wait" : "pointer",
                          opacity: isSubmitting ? 0.7 : 1,
                        }}
                      >
                        {isSubmitting ? "Applying…" : "Set not_an_entity"}
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
```

- [ ] **Step 13: Build TypeScript to verify no errors**

```bash
cd web && npx tsc --noEmit 2>&1 | head -30
```

Expected: no errors (or only pre-existing unrelated errors).

- [ ] **Step 14: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 15: Commit**

```bash
git add web/src/components/PosteriorAuditPanel.tsx web/src/types.ts
git commit -m "feat(ui): divergent_entries rename + type_entropy column + Low Info Entries tab"
```

---

### Task 7: Clear cache + smoke test

**Files:** none (operational step)

- [ ] **Step 1: Clear the posterior_audit_cache table**

After deploying the rename, clear the stale cache so the old `contested_spans` key is gone:

```bash
python -c "
from pathlib import Path
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
store = SqliteStore.open(Path('projects/v3_initial_deployment/.annotation-pipeline'))
store._conn.execute('DELETE FROM posterior_audit_cache')
store._conn.commit()
print('Cache cleared.')
"
```

Expected output: `Cache cleared.`

- [ ] **Step 2: Trigger a rebuild in the UI**

Open Posterior Audit → click **Check** → wait for completion.

Expected: all three keys present in response — Task Deviations tab, Divergent Entries tab (with Entropy column), Low Info Entries tab (if any spans score ≥ 4.0).

- [ ] **Step 3: Verify Low Info Entries tab**

Confirm:
- Each row shows Span, Wordfreq, Distribution, Total, Action columns.
- "Set not_an_entity" button fires and updates row to "✓ done".
- Checkboxes + bulk button work for multi-select.

- [ ] **Step 4: Final commit (if any cleanup needed)**

```bash
git add -p
git commit -m "chore: post-rename cache clear + smoke-test verified"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|-----------------|------|
| Rename `contested_spans` → `divergent_entries` everywhere | Tasks 2, 3, 4, 5, 6 |
| Add `type_entropy` to `divergent_entries` items | Task 3 step 7 |
| Add `low_info_entries` list (high-wordfreq, no convention) | Task 3 step 7 |
| `wordfreq` library dependency | Task 1 |
| Cache surgery uses new key names | Task 3 step 9 |
| Clear `posterior_audit_cache` after rename | Task 7 step 1 |
| Divergent Entries tab: add `type_entropy` sortable column | Task 6 steps 8–9 |
| Low Info Entries tab: per-row + bulk `not_an_entity` | Task 6 Part B |
| Unit tests: `_type_entropy`, `_wordfreq_score` | Task 3 steps 1–4 |
| Integration test: all three keys in output | Task 3 steps 5–8 |
| `batch_apply_all_contested.py` Phase 2 reads new key | Task 4 |

### No placeholders: confirmed — all steps include exact code.

### Type consistency: `DivergentEntry` defined in Task 5 and used consistently in Task 6; `_type_entropy` and `_wordfreq_score` defined in Task 3 and tested in Task 3.

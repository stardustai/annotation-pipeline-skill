# Annotation Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `check_past_experience(entry)` MCP tool that returns per-span convention, type distribution, diversity-selected sentence-level examples, and wordfreq meta to annotator/QC/arbiter subagents — sourced entirely from the existing `entity_conventions` table (extended with row trace fields).

**Architecture:** A single-tool stdio MCP server (`annotation_pipeline_skill.mcp.kb_server`) is launched by Claude CLI via `--mcp-config`. The tool queries `EntityConventionService`, groups proposals by type, selects ≤3 diverse `context_snippet` strings per type via MinHash + farthest-first, and pairs the result with Zipf wordfreq metadata. LLM provider switching already works via the `LLMProfile.base_url` profile field — `isolated_claude_home` injects it into the subprocess environment in isolation (no shell state changed, no effect on the parent process). This plan does not require the operator to manage any environment variables.

**Tech Stack:** Python 3.13, SQLite (existing `entity_conventions`), `mcp` Python SDK (new dep), `jieba` (new dep, lazy-loaded only when CJK detected), `wordfreq` (existing dep), `datasketch` (existing dep for MinHash).

**Spec:** `docs/superpowers/specs/2026-05-19-annotation-knowledge-base-design.md`

---

## File Structure

**New files:**
- `annotation_pipeline_skill/text/__init__.py` — package marker
- `annotation_pipeline_skill/text/wordfreq_utils.py` — `wordfreq_score()`, promoted from `api.py`
- `annotation_pipeline_skill/similarity/diverse.py` — `select_diverse_examples()` farthest-first sampler
- `annotation_pipeline_skill/mcp/__init__.py` — package marker
- `annotation_pipeline_skill/mcp/check_past_experience.py` — pure-function tool logic
- `annotation_pipeline_skill/mcp/kb_server.py` — stdio MCP server entry point
- `tests/test_text_wordfreq_utils.py`
- `tests/test_similarity_diverse.py`
- `tests/test_entity_convention_proposals_schema.py`
- `tests/test_mcp_check_past_experience.py`
- `tests/test_mcp_kb_server.py`
- `tests/test_llm_profiles_mcp.py`
- `tests/test_local_cli_claude_mcp.py`

**Modified files:**
- `annotation_pipeline_skill/similarity/minhash.py` — add CJK fallback to `shingle()`
- `annotation_pipeline_skill/interfaces/api.py` — switch to `wordfreq_utils.wordfreq_score()`; update 2 `record_decision` call sites
- `annotation_pipeline_skill/services/entity_convention_service.py` — extend `record_decision()` signature; persist `row_id` + `context_snippet` in proposals
- `annotation_pipeline_skill/services/human_review_service.py` — update 2 `record_decision` call sites
- `annotation_pipeline_skill/runtime/subagent_cycle.py` — update 1 `record_decision` call site
- `annotation_pipeline_skill/llm/profiles.py` — add `mcp_servers`, `strict_mcp_config`, `disallowed_tools` fields to `LLMProfile` (the existing `base_url` profile field already handles provider switching with subprocess-level isolation)
- `annotation_pipeline_skill/llm/local_cli.py` — extend `build_claude_command()` to accept MCP flags; `_generate_claude()` materializes the per-invocation `mcp-config.json` inside the `isolated_claude_home` directory
- `tests/test_similarity_minhash.py` — add CJK case tests
- `pyproject.toml` — add `mcp` and `jieba` dependencies

---

## Task 1: Promote `wordfreq_score` to shared module

**Files:**
- Create: `annotation_pipeline_skill/text/__init__.py`
- Create: `annotation_pipeline_skill/text/wordfreq_utils.py`
- Modify: `annotation_pipeline_skill/interfaces/api.py:261-267`
- Test: `tests/test_text_wordfreq_utils.py`

- [ ] **Step 1.1: Write failing test for `wordfreq_score`**

Create `tests/test_text_wordfreq_utils.py`:

```python
from annotation_pipeline_skill.text.wordfreq_utils import wordfreq_score


def test_wordfreq_score_high_for_generic_english_word():
    score = wordfreq_score("the")
    assert score > 7.0  # zipf 7+ for ultra-common words


def test_wordfreq_score_low_for_proper_noun():
    score = wordfreq_score("Substack")
    assert score < 4.5


def test_wordfreq_score_handles_cjk():
    score = wordfreq_score("苹果")
    assert score > 4.0  # 'apple' in Chinese is common


def test_wordfreq_score_empty_returns_zero():
    assert wordfreq_score("") == 0.0


def test_wordfreq_score_averages_multi_token_span():
    multi = wordfreq_score("the cat")
    the_score = wordfreq_score("the")
    cat_score = wordfreq_score("cat")
    assert abs(multi - (the_score + cat_score) / 2) < 0.01
```

- [ ] **Step 1.2: Run test, confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_text_wordfreq_utils.py -v`
Expected: `ModuleNotFoundError: No module named 'annotation_pipeline_skill.text'`.

- [ ] **Step 1.3: Create the module**

Create `annotation_pipeline_skill/text/__init__.py` (empty).

Create `annotation_pipeline_skill/text/wordfreq_utils.py`:

```python
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
```

- [ ] **Step 1.4: Replace inline copy in `api.py`**

Open `annotation_pipeline_skill/interfaces/api.py`, find the `_wordfreq_score` function (around line 261). Replace its body to delegate:

```python
def _wordfreq_score(span: str) -> float:
    from annotation_pipeline_skill.text.wordfreq_utils import wordfreq_score
    return wordfreq_score(span)
```

(Keep the private name `_wordfreq_score` to avoid touching its callers in this task.)

- [ ] **Step 1.5: Run new test and existing api tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_text_wordfreq_utils.py tests/test_dashboard_api_row_dedup.py -v`
Expected: all green. The api tests verify we didn't break the `_wordfreq_score` delegation.

- [ ] **Step 1.6: Commit**

```bash
git add annotation_pipeline_skill/text/ annotation_pipeline_skill/interfaces/api.py tests/test_text_wordfreq_utils.py
git commit -m "refactor(text): extract wordfreq_score into shared utility module"
```

---

## Task 2: Add CJK fallback to `shingle()`

**Files:**
- Modify: `annotation_pipeline_skill/similarity/minhash.py:21-34`
- Test: `tests/test_similarity_minhash.py`

- [ ] **Step 2.1: Write failing tests for CJK behavior**

Append to `tests/test_similarity_minhash.py`:

```python
import re
from annotation_pipeline_skill.similarity.minhash import shingle


def test_shingle_pure_ascii_unchanged_by_cjk_gate():
    text = "Telegram is faster than Facebook on my Redmi 3S"
    grams = shingle(text, n=3)
    # 9 tokens → 7 trigrams
    assert len(grams) == 7
    assert "telegram is faster" in grams
    assert "my redmi 3s" in grams


def test_shingle_cjk_uses_jieba_path():
    text = "苹果的客户支持昨天帮我处理了退款问题"
    grams = shingle(text, n=3)
    # Should NOT be the degenerate single-shingle result.
    assert len(grams) > 1
    # Should contain semantically meaningful 3-grams of jieba tokens.
    assert any("客户" in g for g in grams)


def test_shingle_mixed_cjk_ascii_uses_jieba_path():
    text = "TalkBack 在 Android 10 上经常崩溃"
    grams = shingle(text, n=3)
    # Mixed CJK + ASCII → jieba path produces more granular split than
    # whitespace alone (which keeps '上经常崩溃' as one token).
    joined = " | ".join(grams)
    assert "经常" in joined
    assert "崩溃" in joined


def test_shingle_empty_string_returns_empty_set():
    assert shingle("", n=3) == set()


def test_shingle_short_cjk_returns_singleton():
    grams = shingle("苹果", n=3)
    assert grams == {"苹果"}
```

- [ ] **Step 2.2: Run tests, confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_similarity_minhash.py -v -k "cjk or pure_ascii_unchanged"`
Expected: `test_shingle_cjk_uses_jieba_path` and `test_shingle_mixed_cjk_ascii_uses_jieba_path` fail (the current implementation degenerates CJK to a single shingle).

- [ ] **Step 2.3: Implement CJK gate**

In `annotation_pipeline_skill/similarity/minhash.py`, find:

```python
_WHITESPACE_RE = re.compile(r"\s+")


def shingle(text: str, n: int = 5) -> set[str]:
    """Word-level n-gram shingle set. Lowercases and collapses runs of
    whitespace so trivially-different spacing doesn't perturb the
    fingerprint."""
    if not text:
        return set()
    normalized = _WHITESPACE_RE.sub(" ", text.lower()).strip()
    tokens = normalized.split(" ")
    if len(tokens) < n:
        return {normalized} if normalized else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
```

Replace with:

```python
_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def shingle(text: str, n: int = 5) -> set[str]:
    """Word-level n-gram shingle set. Lowercases and collapses runs of
    whitespace so trivially-different spacing doesn't perturb the
    fingerprint.

    CJK fallback: when the input contains CJK Unified Ideographs (or
    Extension A), tokens are produced by jieba word segmentation instead
    of whitespace splitting. This rescues CJK rows from degenerating to
    a single shingle (which makes Jaccard binary and useless for
    diversity ranking or near-duplicate clustering). Pure-ASCII inputs
    are unaffected — the jieba path is skipped entirely so existing
    row_dedup behaviour is preserved.
    """
    if not text:
        return set()
    normalized = _WHITESPACE_RE.sub(" ", text.lower()).strip()

    if _CJK_RE.search(normalized):
        # Lazy import: ASCII-only projects never pay the jieba load cost.
        import jieba
        tokens = [t for t in jieba.cut(normalized) if t.strip()]
    else:
        tokens = normalized.split(" ")

    if len(tokens) < n:
        return {normalized} if normalized else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
```

- [ ] **Step 2.4: Run tests, confirm pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_similarity_minhash.py -v`
Expected: all green. If `jieba` is not installed, the test will fail at `import jieba` — that's expected; the dependency add happens in Task 10. Skip these tests temporarily by running `pytest -k "not cjk"` until Task 10 completes if needed. **However**, the recommended path is to add the dependency now since later tasks also need it:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv add jieba
```

Then re-run.

- [ ] **Step 2.5: Commit**

```bash
git add annotation_pipeline_skill/similarity/minhash.py tests/test_similarity_minhash.py pyproject.toml uv.lock
git commit -m "feat(similarity): add CJK jieba fallback to shingle()"
```

---

## Task 3: Farthest-first diverse-example sampler

**Files:**
- Create: `annotation_pipeline_skill/similarity/diverse.py`
- Test: `tests/test_similarity_diverse.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_similarity_diverse.py`:

```python
from annotation_pipeline_skill.similarity.diverse import select_diverse_examples


def test_returns_input_when_smaller_than_k():
    snippets = ["a", "b"]
    out = select_diverse_examples(snippets, k=3)
    assert out == snippets


def test_returns_k_when_input_larger():
    snippets = [
        "Apple customer support helped me yesterday",
        "Apple customer service was great today",
        "My iPad from Apple broke last week",
        "Apple announced new privacy policy for developers",
    ]
    out = select_diverse_examples(snippets, k=3)
    assert len(out) == 3
    # All returned snippets must be from the input.
    assert all(s in snippets for s in out)


def test_picks_dissimilar_pair_over_near_duplicates():
    snippets = [
        "Apple customer support helped me yesterday",
        "Apple customer support helped me yesterday afternoon",
        "Apple customer support helped me today morning",
        "My iPad from Apple broke last week badly",
    ]
    out = select_diverse_examples(snippets, k=2)
    # Greedy farthest-first should NOT return the two near-duplicates
    # from the top of the list as its 2-element answer.
    assert not (
        out[0].startswith("Apple customer support helped me yesterday")
        and out[1].startswith("Apple customer support helped me yesterday")
    )


def test_deterministic_for_same_input():
    snippets = [
        "Apple customer support helped me yesterday",
        "Apple customer service was great today",
        "My iPad from Apple broke last week",
        "Apple announced new privacy policy for developers",
    ]
    a = select_diverse_examples(snippets, k=3)
    b = select_diverse_examples(snippets, k=3)
    assert a == b


def test_deduplicates_identical_snippets():
    snippets = ["dup", "dup", "dup", "different one entirely"]
    out = select_diverse_examples(snippets, k=3)
    # After dedup only 2 distinct snippets exist.
    assert sorted(out) == sorted(["dup", "different one entirely"])
```

- [ ] **Step 3.2: Run tests, confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_similarity_diverse.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement the sampler**

Create `annotation_pipeline_skill/similarity/diverse.py`:

```python
"""Farthest-first diverse-example selection over context snippets.

Used by the annotation knowledge base MCP tool to surface up to k
representative snippets per (span, type) bucket: snippets that are
maximally dissimilar to each other, so an LLM agent sees the breadth
of past contexts rather than near-duplicates.
"""
from __future__ import annotations

from datasketch import MinHash

from annotation_pipeline_skill.similarity.minhash import shingle


_NUM_PERM = 64  # Lower than minhash.py default (128) — we operate on
                # short snippets and only do small pairwise comparisons,
                # so 64 is plenty and ~2x faster to build.


def _build_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=_NUM_PERM)
    for s in shingle(text, n=3):
        m.update(s.encode("utf-8"))
    return m


def select_diverse_examples(snippets: list[str], k: int = 3) -> list[str]:
    """Select up to k snippets that maximize pairwise dissimilarity.

    Algorithm: farthest-first traversal. Seed with the lexicographically
    smallest snippet (deterministic), then repeatedly add the snippet
    whose minimum Jaccard similarity to the already-selected set is
    lowest (i.e., is farthest from the selected set).
    """
    # Deduplicate while preserving original ordering for tie-break stability.
    deduped: list[str] = []
    seen: set[str] = set()
    for s in snippets:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    if len(deduped) <= k:
        return deduped

    minhashes = [_build_minhash(s) for s in deduped]
    seed_idx = min(range(len(deduped)), key=lambda i: deduped[i])
    selected: list[int] = [seed_idx]

    while len(selected) < k:
        best_idx, best_distance = -1, -1.0
        for i in range(len(deduped)):
            if i in selected:
                continue
            # Distance to the SET = 1 - max(similarity to any selected).
            max_sim = max(minhashes[i].jaccard(minhashes[j]) for j in selected)
            distance = 1.0 - max_sim
            if distance > best_distance or (
                distance == best_distance and (best_idx == -1 or deduped[i] < deduped[best_idx])
            ):
                best_distance, best_idx = distance, i
        selected.append(best_idx)

    return [deduped[i] for i in selected]
```

- [ ] **Step 3.4: Run tests, confirm pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_similarity_diverse.py -v`
Expected: all green.

- [ ] **Step 3.5: Commit**

```bash
git add annotation_pipeline_skill/similarity/diverse.py tests/test_similarity_diverse.py
git commit -m "feat(similarity): add farthest-first diverse-example sampler"
```

---

## Task 4: Extend `entity_conventions.proposals` schema and `record_decision()` signature

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py:61-163`
- Test: `tests/test_entity_convention_proposals_schema.py`

- [ ] **Step 4.1: Write failing tests for the new fields**

Create `tests/test_entity_convention_proposals_schema.py`:

```python
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.sqlite"
        s = SqliteStore(path)
        yield s


def test_record_decision_persists_row_id_in_proposal(store):
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="qc_consensus",
        task_id="task_019",
        row_id="row_18452",
        row_content="Crashes on Android 10 sometimes",
    )
    rows = list(store._conn.execute("SELECT proposals_json FROM entity_conventions"))
    proposals = json.loads(rows[0][0])
    assert proposals[0]["row_id"] == "row_18452"


def test_record_decision_persists_context_snippet_when_row_content_given(store):
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="qc_consensus",
        task_id="task_019",
        row_id="row_18452",
        row_content="The app keeps crashing on my Android phone every few hours",
    )
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    snippet = proposals[0]["context_snippet"]
    assert snippet is not None
    assert "Android" in snippet
    # Snippet should be windowed (not the full row) when content < 200 chars
    # the whole thing fits, which is fine — just verify it includes Android.


def test_record_decision_without_row_content_leaves_snippet_none(store):
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="declared:operator",
    )
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    assert proposals[0]["row_id"] is None
    assert proposals[0].get("context_snippet") is None


def test_snippet_window_truncates_long_rows(store):
    svc = EntityConventionService(store)
    long_row = "padding " * 50 + "Android" + " padding" * 50  # ~750 chars
    svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="qc_consensus",
        task_id="task_019",
        row_id="row_99",
        row_content=long_row,
    )
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    snippet = proposals[0]["context_snippet"]
    # Window is span ± 80 chars → max 200 chars including the span itself.
    assert len(snippet) <= 200
    assert "Android" in snippet
    # The snippet was truncated, so the leading/trailing "padding" tokens
    # past the window should be absent.
    assert snippet.count("padding") < long_row.count("padding")


def test_legacy_call_signature_still_works(store):
    """Existing call sites that don't pass row_id/row_content must keep working."""
    svc = EntityConventionService(store)
    conv = svc.record_decision(
        project_id="proj1",
        span="Android",
        entity_type="technology",
        source="qc_consensus",
        task_id="task_019",
    )
    assert conv.entity_type == "technology"
    proposals = json.loads(
        next(store._conn.execute("SELECT proposals_json FROM entity_conventions"))[0]
    )
    assert proposals[0]["row_id"] is None
    assert proposals[0].get("context_snippet") is None
```

- [ ] **Step 4.2: Run tests, confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_entity_convention_proposals_schema.py -v`
Expected: all tests fail with `TypeError: record_decision() got an unexpected keyword argument 'row_id'` (except `test_legacy_call_signature_still_works` which currently passes — confirm it still passes after the change).

- [ ] **Step 4.3: Add a `_build_context_snippet()` helper**

In `annotation_pipeline_skill/services/entity_convention_service.py`, **after the existing imports** (before the `@dataclass(frozen=True)` line), insert:

```python
_SNIPPET_WINDOW = 80  # chars before and after the span hit


def _build_context_snippet(span: str, row_content: str | None) -> str | None:
    """Build a ~200-char window around the first case-insensitive
    occurrence of ``span`` in ``row_content``. Returns ``None`` when no
    row_content is provided or the span isn't found.
    """
    if not row_content:
        return None
    hit = row_content.lower().find(span.lower())
    if hit < 0:
        # Span not present in row_content (e.g., normalization mismatch);
        # still surface the row as evidence by returning a head window.
        return row_content[: _SNIPPET_WINDOW * 2].strip()
    start = max(0, hit - _SNIPPET_WINDOW)
    end = min(len(row_content), hit + len(span) + _SNIPPET_WINDOW)
    snippet = row_content[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(row_content) else ""
    return f"{prefix}{snippet}{suffix}"
```

- [ ] **Step 4.4: Extend `record_decision()` signature and proposal payload**

Replace the existing `record_decision` method header and the `proposal` dict construction (around line 61-87):

```python
    def record_decision(
        self,
        *,
        project_id: str,
        span: str,
        entity_type: str,
        source: str,
        task_id: str | None = None,
        row_id: str | None = None,
        row_content: str | None = None,
        notes: str | None = None,
    ) -> EntityConvention:
        """Upsert a convention. Rules:
        - first time → insert as 'active'
        - same type re-affirmed → bump evidence_count, append proposal
        - different type → mark 'disputed', append proposal
        - already 'disputed' → just append proposal (do not silently re-activate)
        """
        if not span or not entity_type:
            raise ValueError("span and entity_type are required")
        span_lower = span.strip().lower()
        now = datetime.now(timezone.utc)
        proposal = {
            "type": entity_type,
            "source": source,
            "task_id": task_id,
            "row_id": row_id,
            "context_snippet": _build_context_snippet(span, row_content),
            "notes": notes,
            "at": now.isoformat(),
        }
```

The rest of the method (the SELECT, INSERT, UPDATE logic) is unchanged.

- [ ] **Step 4.5: Run tests, confirm pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_entity_convention_proposals_schema.py -v`
Expected: all 5 tests pass.

- [ ] **Step 4.6: Verify no regressions in surrounding tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/ -v -k "convention or entity" --no-header`
Expected: all green. Existing convention/entity tests don't pass `row_id`/`row_content` and rely on defaults — Task 4.4 keeps them at `None`.

- [ ] **Step 4.7: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py tests/test_entity_convention_proposals_schema.py
git commit -m "feat(conventions): persist row_id and context_snippet in proposals"
```

---

## Task 5: Update all `record_decision()` call sites to pass trace data

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py:798-813` (`_record_conventions_from_qc_consensus` loop body — line numbers may drift)
- Modify: `annotation_pipeline_skill/services/human_review_service.py:432-450` (`hr_correction` site) and `~609-616` (`posterior_audit_operator` site — leave as-is)
- Modify: `annotation_pipeline_skill/interfaces/api.py:~1429` (Set Convention endpoint — operator declarations, leave as-is unless request body adds row_id) and `~2079` (the per-row decision loop — thread row data through)
- Test: `tests/test_subagent_cycle.py` (existing) — extend
- Test: `tests/test_dashboard_api_row_dedup.py` (existing) — extend

> **Line numbers drift over time.** Before editing, run `grep -n "record_decision" annotation_pipeline_skill/<file>.py` to locate the current line. The structural pattern (kwargs to `record_decision(...)`) is stable even when line numbers shift.

> **Why this task exists:** Without these updates, the new `context_snippet` field stays `None` everywhere and the MCP tool's `examples_by_type` will be permanently empty. Each site needs to find out what `row_id` and `row_content` mean *for that call site* and pass them through.

- [ ] **Step 5.1: Inspect each call site to identify available row data**

For each site, run the file open to the line and read 30 lines of surrounding context:

```bash
sed -n '780,810p' /home/derek/Projects/annotation-pipeline-skill/annotation_pipeline_skill/runtime/subagent_cycle.py
sed -n '425,455p' /home/derek/Projects/annotation-pipeline-skill/annotation_pipeline_skill/services/human_review_service.py
sed -n '595,625p' /home/derek/Projects/annotation-pipeline-skill/annotation_pipeline_skill/services/human_review_service.py
sed -n '1410,1440p' /home/derek/Projects/annotation-pipeline-skill/annotation_pipeline_skill/interfaces/api.py
sed -n '2070,2100p' /home/derek/Projects/annotation-pipeline-skill/annotation_pipeline_skill/interfaces/api.py
```

Note: each site already has access to `task` (with `.task_id`) and the decisions list. The `row_id` and `row_content` need to be located in the surrounding code — typically the annotation payload has `rows: [{row_id, content, …}]`, and span decisions come from `extract_entity_type_decisions(prelabel, current)` which yields `(span, type)` tuples *without* row pointers.

**The pragmatic plan:** for spans that came from a specific row, find that row in the payload. For now, **enrich `extract_entity_type_decisions` once** to also emit the row info, then thread the new fields through.

- [ ] **Step 5.2: Find and read `extract_entity_type_decisions`**

```bash
grep -rn "def extract_entity_type_decisions" /home/derek/Projects/annotation-pipeline-skill/annotation_pipeline_skill/ | head -3
```

Open the file containing it and add a new helper next to it: `extract_entity_type_decisions_with_row` returning `list[tuple[str, str, str | None, str | None]]` (span, type, row_id, row_content). Keep the original helper intact for backward compat.

- [ ] **Step 5.3: Write failing tests for the new helper**

Add to the test file that exercises the existing helper (`grep -rn "extract_entity_type_decisions" tests/` to locate). Append:

```python
def test_extract_entity_type_decisions_with_row_links_rows():
    payload = {
        "rows": [
            {"row_id": "r1", "content": "Crashes on Android 10",
             "entities": [{"text": "Android", "entity_type": "technology"}]},
            {"row_id": "r2", "content": "PicsArt edits missing",
             "entities": [{"text": "PicsArt", "entity_type": "technology"}]},
        ]
    }
    from annotation_pipeline_skill.<existing_module> import extract_entity_type_decisions_with_row
    out = extract_entity_type_decisions_with_row(payload)
    assert ("Android", "technology", "r1", "Crashes on Android 10") in out
    assert ("PicsArt", "technology", "r2", "PicsArt edits missing") in out
```

Replace `<existing_module>` with the actual module path discovered in Step 5.2.

- [ ] **Step 5.4: Implement `extract_entity_type_decisions_with_row`**

Locate the existing `extract_entity_type_decisions` implementation. Mirror its structure but additionally carry `row_id` and the row's `content`/`text` field forward. Example template:

```python
def extract_entity_type_decisions_with_row(payload):
    """Like extract_entity_type_decisions but also yields the source row.

    Returns list of (span, entity_type, row_id, row_content) tuples.
    row_id and row_content are None if the payload didn't carry per-row
    information (e.g., flat top-level entities list).
    """
    out: list[tuple[str, str, str | None, str | None]] = []
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_id = row.get("row_id")
            row_content = row.get("content") or row.get("text")
            for ent in row.get("entities", []) or []:
                if not isinstance(ent, dict):
                    continue
                span = ent.get("text") or ent.get("span")
                etype = ent.get("entity_type") or ent.get("type")
                if isinstance(span, str) and span.strip() and isinstance(etype, str) and etype.strip():
                    out.append((span, etype, row_id, row_content))
    else:
        # Fallback: payload has no rows[], use the legacy flat extractor
        for span, etype in extract_entity_type_decisions(payload, {}):
            out.append((span, etype, None, None))
    return out
```

Adapt field names to match the actual schema observed in Step 5.2.

- [ ] **Step 5.5: Run new helper test, confirm pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest <test_file>::test_extract_entity_type_decisions_with_row_links_rows -v`

Expected: PASS.

- [ ] **Step 5.6: Update call site in `subagent_cycle.py`**

Change the loop body in `_record_conventions_from_qc_consensus` (currently around line 798-813 — confirm with `grep -n record_decision`) from:

```python
            decisions = extract_entity_type_decisions(prelabel or {}, current)
            if not decisions:
                return
            svc = EntityConventionService(self.store)
            for span, entity_type in decisions:
                try:
                    svc.record_decision(
                        project_id=task.pipeline_id,
                        span=span,
                        entity_type=entity_type,
                        source="qc_consensus",
                        task_id=task.task_id,
                    )
                except Exception:  # noqa: BLE001
                    continue
```

to:

```python
            decisions = extract_entity_type_decisions_with_row(current or prelabel or {})
            if not decisions:
                return
            svc = EntityConventionService(self.store)
            for span, entity_type, row_id, row_content in decisions:
                try:
                    svc.record_decision(
                        project_id=task.pipeline_id,
                        span=span,
                        entity_type=entity_type,
                        source="qc_consensus",
                        task_id=task.task_id,
                        row_id=row_id,
                        row_content=row_content,
                    )
                except Exception:  # noqa: BLE001
                    continue
```

Add the new import at the top of the file alongside the existing `extract_entity_type_decisions` import.

- [ ] **Step 5.7: Update call site in `human_review_service.py:441-450`**

Same pattern. Change:

```python
        svc = EntityConventionService(self.store)
        for span, entity_type in decisions:
            try:
                svc.record_decision(
                    project_id=task.pipeline_id,
                    span=span,
                    entity_type=entity_type,
                    source=f"hr_correction:{actor}",
                    task_id=task.task_id,
                )
            except (ValueError, TypeError):
                continue
```

to use `decisions_with_row` (call `extract_entity_type_decisions_with_row(...)` above this loop, in place of the existing `extract_entity_type_decisions(...)` call) and forward `row_id` + `row_content`.

- [ ] **Step 5.8: Update call site in `human_review_service.py:609-616`**

This is the **posterior_audit_operator** declaration — operator clicked a button in the UI, so there's typically no row_id available *at that level*. Leave this site as-is (just add explicit `row_id=None, row_content=None` for documentary clarity if you like, but it's not required since the defaults are `None`).

- [ ] **Step 5.9: Update call sites in `api.py`**

Two sites (lines 1427 and 2082). Read the surrounding code to determine whether row information is available:

- **Site 1 (line ~1427)** — `Set Convention` API endpoint, operator-declared via UI. No row_id in the request body unless it's added. Leave defaults (`None`).
- **Site 2 (line ~2082)** — depends on context; if it iterates over per-row decisions, thread `row_id` + `row_content` through.

If a site has no clean way to obtain the row, leave the defaults; the convention is still recorded, just without trace data. The MCP tool gracefully handles missing `context_snippet`.

- [ ] **Step 5.10: Extend existing tests to assert trace data flows through**

In `tests/test_subagent_cycle.py`, find the test that exercises `_record_conventions_from_qc_consensus` (search for `record_conventions_from_qc`, `record_decision`, or `qc_consensus`). Augment one such test to assert the resulting proposal carries the expected `row_id` and `context_snippet`. Pattern:

```python
def test_qc_consensus_records_row_trace_in_proposal(tmp_path):
    # ... existing setup ...
    cycle._record_conventions_from_qc_consensus(task, ...)
    rows = list(store._conn.execute("SELECT proposals_json FROM entity_conventions"))
    proposals = json.loads(rows[0][0])
    assert proposals[0]["row_id"] == "row_18452"
    assert "Android" in (proposals[0]["context_snippet"] or "")
```

Use the actual fixture/setup pattern already in the file.

- [ ] **Step 5.11: Run the touched test files**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_subagent_cycle.py tests/test_dashboard_api_row_dedup.py -v`
Expected: all green, including the new assertion(s).

- [ ] **Step 5.12: Run the full test suite to catch regressions**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest -q`
Expected: green.

- [ ] **Step 5.13: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py \
        annotation_pipeline_skill/services/human_review_service.py \
        annotation_pipeline_skill/interfaces/api.py \
        annotation_pipeline_skill/<module_containing_extractor>.py \
        tests/test_subagent_cycle.py tests/<extractor_test_file>.py
git commit -m "feat(conventions): thread row_id and row_content through QC/HR record_decision sites"
```

---

## Task 6: `check_past_experience` tool logic (pure function)

**Files:**
- Create: `annotation_pipeline_skill/mcp/__init__.py`
- Create: `annotation_pipeline_skill/mcp/check_past_experience.py`
- Test: `tests/test_mcp_check_past_experience.py`

> This task implements the tool as a **pure function** taking a store and a span, returning the response dict. The MCP server wrapper (Task 7) wires this into the protocol layer. Separating them means the logic is unit-testable without spinning up a subprocess.

- [ ] **Step 6.1: Write failing tests**

Create `tests/test_mcp_check_past_experience.py`:

```python
import json
import tempfile
from pathlib import Path

import pytest

from annotation_pipeline_skill.mcp.check_past_experience import check_past_experience
from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = SqliteStore(Path(tmpdir) / "t.sqlite")
        yield s


def _seed(svc, project, span, etype, source, task_id, row_id, row_content):
    svc.record_decision(
        project_id=project, span=span, entity_type=etype,
        source=source, task_id=task_id, row_id=row_id, row_content=row_content,
    )


def test_unknown_span_returns_none_shape(store):
    result = check_past_experience(store, project_id="p1", entry="NeverSeen")
    assert result["entry"] == "NeverSeen"
    assert result["convention"]["status"] == "none"
    assert result["convention"]["evidence_count"] == 0
    assert result["distribution"] == {}
    assert result["examples_by_type"] == {}
    assert "wordfreq_zipf" in result["meta"]


def test_active_convention_returns_examples(store):
    svc = EntityConventionService(store)
    for i in range(5):
        # Use a declared:operator source for the first to bypass min-evidence,
        # OR loop enough times to clear the threshold. Loop 5 times here.
        _seed(svc, "p1", "Android", "technology", "qc_consensus",
              f"task_{i}", f"row_{i}", f"Crashes on Android 10 ({i})")
    result = check_past_experience(store, project_id="p1", entry="Android")
    assert result["convention"]["status"] == "active"
    assert result["convention"]["type"] == "technology"
    assert result["convention"]["evidence_count"] == 5
    assert result["distribution"] == {"technology": 5}
    assert "technology" in result["examples_by_type"]
    # ≤ 3 examples per type.
    assert len(result["examples_by_type"]["technology"]) <= 3
    # Examples carry trace prefix.
    assert all(
        s.startswith("[task_") and "/row_" in s
        for s in result["examples_by_type"]["technology"]
    )


def test_disputed_returns_examples_per_type(store):
    svc = EntityConventionService(store)
    _seed(svc, "p1", "Apple", "organization", "qc_consensus",
          "task_a", "row_1", "Apple's customer support helped me yesterday")
    _seed(svc, "p1", "Apple", "organization", "qc_consensus",
          "task_b", "row_2", "Apple announced a new privacy policy")
    _seed(svc, "p1", "Apple", "product", "qc_consensus",
          "task_c", "row_3", "My Apple iPad keeps crashing on updates")
    result = check_past_experience(store, project_id="p1", entry="Apple")
    assert result["convention"]["status"] == "disputed"
    assert result["convention"]["type"] is None
    assert result["distribution"] == {"organization": 2, "product": 1}
    assert set(result["examples_by_type"].keys()) == {"organization", "product"}


def test_skips_proposals_without_context_snippet(store):
    """Operator declarations have no row → no example available; they
    still count toward evidence and distribution, but examples_by_type
    only contains the buckets that DO have snippets."""
    svc = EntityConventionService(store)
    # Operator declared, no row_content → no snippet.
    _seed(svc, "p1", "Apple", "organization", "declared:operator",
          None, None, None)
    # QC consensus with row_content → snippet exists.
    _seed(svc, "p1", "Apple", "organization", "qc_consensus",
          "task_a", "row_1", "Apple's customer support helped me")
    result = check_past_experience(store, project_id="p1", entry="Apple")
    assert result["distribution"]["organization"] == 2
    # Only one snippet → only one example.
    assert len(result["examples_by_type"]["organization"]) == 1


def test_generic_word_flag_for_high_freq_low_evidence(store):
    """'the' has Zipf ~7+ but no evidence → generic_word should be True."""
    result = check_past_experience(store, project_id="p1", entry="the")
    assert result["meta"]["wordfreq_zipf"] > 5.0
    assert result["meta"]["generic_word"] is True


def test_generic_word_flag_false_when_evidence_count_high(store):
    svc = EntityConventionService(store)
    for i in range(6):
        _seed(svc, "p1", "the", "project", "declared:operator", None, None, None)
    result = check_past_experience(store, project_id="p1", entry="the")
    # Still high zipf, but evidence_count >= 5 → don't flag as generic.
    assert result["meta"]["generic_word"] is False


def test_empty_entry_returns_error(store):
    with pytest.raises(ValueError):
        check_past_experience(store, project_id="p1", entry="")
```

- [ ] **Step 6.2: Run tests, confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_mcp_check_past_experience.py -v`
Expected: `ModuleNotFoundError: No module named 'annotation_pipeline_skill.mcp'`.

- [ ] **Step 6.3: Implement the tool**

Create `annotation_pipeline_skill/mcp/__init__.py` (empty file).

Create `annotation_pipeline_skill/mcp/check_past_experience.py`:

```python
"""Pure-function implementation of the check_past_experience MCP tool.

The MCP server wrapper (kb_server.py) is thin — it forwards the
JSON-RPC arguments here and serializes the returned dict back over
stdio. Keeping the logic separate makes it unit-testable without
launching a subprocess.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.similarity.diverse import select_diverse_examples
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.text.wordfreq_utils import wordfreq_score


_MAX_EXAMPLES_PER_TYPE = 3
_GENERIC_WORD_ZIPF = 5.0
_GENERIC_WORD_MIN_EVIDENCE = 5


def check_past_experience(
    store: SqliteStore,
    *,
    project_id: str,
    entry: str,
) -> dict[str, Any]:
    """Return past-annotation evidence for a candidate span.

    Output shape — see
    docs/superpowers/specs/2026-05-19-annotation-knowledge-base-design.md
    section "Tool Contract" for field semantics.
    """
    if not entry or not entry.strip():
        raise ValueError("entry is required")

    span_lower = entry.strip().lower()
    row = store._conn.execute(
        "SELECT convention_id, entity_type, status, evidence_count, proposals_json "
        "FROM entity_conventions WHERE project_id=? AND span_lower=?",
        (project_id, span_lower),
    ).fetchone()

    zipf = wordfreq_score(entry)

    if row is None:
        return {
            "entry": entry,
            "convention": {"status": "none", "type": None, "evidence_count": 0},
            "distribution": {},
            "examples_by_type": {},
            "meta": {
                "wordfreq_zipf": round(zipf, 3),
                "generic_word": zipf >= _GENERIC_WORD_ZIPF,
            },
        }

    proposals = json.loads(row["proposals_json"] or "[]")

    # Distribution counts every proposal by its declared type.
    distribution = Counter(
        p["type"] for p in proposals
        if isinstance(p, dict) and isinstance(p.get("type"), str)
    )

    # Group context snippets by type, formatted with trace prefix.
    snippets_by_type: dict[str, list[str]] = {}
    for p in proposals:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type")
        snippet = p.get("context_snippet")
        if not (isinstance(ptype, str) and isinstance(snippet, str) and snippet.strip()):
            continue
        task_id = p.get("task_id") or "?"
        row_id = p.get("row_id") or "?"
        formatted = f"[{task_id}/{row_id}] {snippet}"
        snippets_by_type.setdefault(ptype, []).append(formatted)

    examples_by_type = {
        ptype: select_diverse_examples(snippets, k=_MAX_EXAMPLES_PER_TYPE)
        for ptype, snippets in snippets_by_type.items()
    }

    evidence_count = row["evidence_count"]
    return {
        "entry": entry,
        "convention": {
            "status": row["status"],
            "type": row["entity_type"],
            "evidence_count": evidence_count,
        },
        "distribution": dict(distribution),
        "examples_by_type": examples_by_type,
        "meta": {
            "wordfreq_zipf": round(zipf, 3),
            "generic_word": (
                zipf >= _GENERIC_WORD_ZIPF and evidence_count < _GENERIC_WORD_MIN_EVIDENCE
            ),
        },
    }
```

- [ ] **Step 6.4: Run tests, confirm pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_mcp_check_past_experience.py -v`
Expected: all 7 tests pass.

- [ ] **Step 6.5: Commit**

```bash
git add annotation_pipeline_skill/mcp/__init__.py \
        annotation_pipeline_skill/mcp/check_past_experience.py \
        tests/test_mcp_check_past_experience.py
git commit -m "feat(mcp): pure-function check_past_experience tool logic"
```

---

## Task 7: MCP stdio server wrapper

**Files:**
- Create: `annotation_pipeline_skill/mcp/kb_server.py`
- Test: `tests/test_mcp_kb_server.py`

- [ ] **Step 7.1: Add `mcp` SDK dependency**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv add mcp
```

Verify `pyproject.toml` now contains an `mcp` entry under `dependencies`.

- [ ] **Step 7.2: Write failing smoke test**

Create `tests/test_mcp_kb_server.py`:

```python
"""Stdio smoke test for the annotation-kb MCP server.

Spawns the server as a subprocess and exchanges JSON-RPC messages
according to the MCP protocol. The full MCP spec includes initialize
+ tools/list + tools/call handshakes — this test exercises the
tools/list + tools/call path.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def project_with_convention(tmp_path):
    db_path = tmp_path / "db.sqlite"
    store = SqliteStore(db_path)
    svc = EntityConventionService(store)
    for i in range(5):
        svc.record_decision(
            project_id="proj_demo", span="Android", entity_type="technology",
            source="qc_consensus", task_id=f"task_{i}", row_id=f"row_{i}",
            row_content=f"Crashes on Android 10 in case {i}",
        )
    store._conn.commit()
    return db_path


def _rpc(proc, msg):
    proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
    proc.stdin.flush()


def _recv(proc, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            return json.loads(line.decode("utf-8"))
    raise TimeoutError("no response")


def test_mcp_server_lists_check_past_experience_tool(project_with_convention):
    proc = subprocess.Popen(
        [sys.executable, "-m", "annotation_pipeline_skill.mcp.kb_server",
         "--db-path", str(project_with_convention),
         "--project-id", "proj_demo"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Initialize.
        _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
        init_resp = _recv(proc)
        assert init_resp["id"] == 1
        # Notify initialized.
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        # List tools.
        _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        list_resp = _recv(proc)
        names = [t["name"] for t in list_resp["result"]["tools"]]
        assert "check_past_experience" in names
    finally:
        proc.kill()
        proc.wait(timeout=2)


def test_mcp_server_calls_check_past_experience(project_with_convention):
    proc = subprocess.Popen(
        [sys.executable, "-m", "annotation_pipeline_skill.mcp.kb_server",
         "--db-path", str(project_with_convention),
         "--project-id", "proj_demo"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
        _recv(proc)
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        _rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "check_past_experience",
                               "arguments": {"entry": "Android"}}})
        call_resp = _recv(proc)
        content = call_resp["result"]["content"][0]["text"]
        payload = json.loads(content)
        assert payload["entry"] == "Android"
        assert payload["convention"]["type"] == "technology"
        assert payload["convention"]["evidence_count"] == 5
    finally:
        proc.kill()
        proc.wait(timeout=2)
```

- [ ] **Step 7.3: Run tests, confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_mcp_kb_server.py -v`
Expected: `ModuleNotFoundError: No module named 'annotation_pipeline_skill.mcp.kb_server'`.

- [ ] **Step 7.4: Implement the MCP server**

Create `annotation_pipeline_skill/mcp/kb_server.py`:

```python
"""Stdio MCP server exposing the annotation knowledge base tool.

Launched by Claude CLI via --mcp-config. The server holds a read-only
SQLite connection to the project DB and exposes a single tool,
check_past_experience.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from annotation_pipeline_skill.mcp.check_past_experience import check_past_experience
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


logger = logging.getLogger("annotation_kb_mcp")


def build_server(*, db_path: Path, project_id: str) -> Server:
    server: Server = Server("annotation-kb")
    store = SqliteStore(db_path)

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="check_past_experience",
                description=(
                    "Query the project's annotation history for a candidate "
                    "entity/span. Returns the current convention (if any), "
                    "the distribution of past type proposals, up to 3 "
                    "diverse sentence-level examples per type, and a "
                    "wordfreq Zipf score. Use this BEFORE deciding the "
                    "type of an ambiguous or unfamiliar span — past "
                    "decisions and concrete row examples beat statistical "
                    "summaries for in-context generalization."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entry": {
                            "type": "string",
                            "description": "The candidate span text (case-insensitive lookup).",
                        },
                    },
                    "required": ["entry"],
                },
            )
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name != "check_past_experience":
            raise ValueError(f"unknown tool: {name}")
        entry = arguments.get("entry", "")
        try:
            result = check_past_experience(store, project_id=project_id, entry=entry)
        except ValueError as exc:
            payload = {"error": str(exc)}
        else:
            payload = result
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="annotation-kb-mcp-server")
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    server = build_server(db_path=args.db_path, project_id=args.project_id)

    async def _run() -> None:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.5: Run tests, confirm pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_mcp_kb_server.py -v`
Expected: both tests pass. If they hang, the most likely cause is a different MCP SDK version exposing a different API — inspect the installed `mcp` package version and adjust imports (the SDK has been stable enough that the symbols above should match recent versions; if not, run `uv run python -c "import mcp; help(mcp.server)"` for guidance).

- [ ] **Step 7.6: Commit**

```bash
git add annotation_pipeline_skill/mcp/kb_server.py \
        tests/test_mcp_kb_server.py pyproject.toml uv.lock
git commit -m "feat(mcp): stdio server exposing check_past_experience"
```

---

## Task 8: LLMProfile schema — add `mcp_servers`, `disallowed_tools`, `strict_mcp_config`

> **Schema note:** The project already uses a flat `runtime: claude_cli | codex_cli | …` schema. LLM provider switching already works via the existing `base_url` profile field (`isolated_claude_home` injects it into the subprocess in isolation — the operator never touches an env var). This task only adds **three** new fields.

**Files:**
- Modify: `annotation_pipeline_skill/llm/profiles.py:34-65, 106-…`
- Test: `tests/test_llm_profiles_mcp.py`

- [ ] **Step 8.1: Write failing tests**

Create `tests/test_llm_profiles_mcp.py`:

```python
from pathlib import Path

import pytest

from annotation_pipeline_skill.llm.profiles import (
    load_llm_registry,
    ProfileValidationError,
)


def _write_yaml(tmp: Path, body: str) -> Path:
    path = tmp / "llm_profiles.yaml"
    path.write_text(body)
    return path


def test_profile_parses_mcp_servers(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  annotator_claude_kb:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    mcp_servers:
      - name: annotation-kb
        command: python
        args: ["-m", "annotation_pipeline_skill.mcp.kb_server"]
    strict_mcp_config: true
    disallowed_tools: ["Bash", "Edit", "Write"]
""")
    reg = load_llm_registry(path)
    profile = reg.profiles["annotator_claude_kb"]
    assert profile.mcp_servers == [
        {"name": "annotation-kb", "command": "python",
         "args": ["-m", "annotation_pipeline_skill.mcp.kb_server"]}
    ]
    assert profile.strict_mcp_config is True
    assert profile.disallowed_tools == ["Bash", "Edit", "Write"]


def test_profile_mcp_fields_optional(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  classic:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
""")
    reg = load_llm_registry(path)
    p = reg.profiles["classic"]
    assert p.mcp_servers is None
    assert p.strict_mcp_config is None
    assert p.disallowed_tools is None


def test_profile_rejects_malformed_mcp_servers(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  bad:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    mcp_servers: "not a list"
""")
    with pytest.raises(ProfileValidationError):
        load_llm_registry(path)
```

- [ ] **Step 8.2: Run tests, confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_llm_profiles_mcp.py -v`
Expected: failures — `mcp_servers` etc. are not yet fields.

- [ ] **Step 8.3: Extend `LLMProfile` dataclass**

In `annotation_pipeline_skill/llm/profiles.py`, find the `@dataclass(frozen=True) class LLMProfile:` definition. Append these fields after `disable_continuity` and before the `resolve_api_key` method (so they remain part of the dataclass):

```python
    mcp_servers: list[dict] | None = None
    strict_mcp_config: bool | None = None
    disallowed_tools: list[str] | None = None
```

- [ ] **Step 8.4: Parse the new fields in `_parse_profile`**

In `annotation_pipeline_skill/llm/profiles.py`, find `_parse_profile`. After the existing field extractions, add (matching the existing kwarg style):

```python
        mcp_servers=_optional_mcp_servers(raw.get("mcp_servers"), f"profile {name} mcp_servers"),
        strict_mcp_config=_optional_bool(raw.get("strict_mcp_config"), f"profile {name} strict_mcp_config"),
        disallowed_tools=_optional_string_list(raw.get("disallowed_tools"), f"profile {name} disallowed_tools"),
```

Append these helpers at the end of the file (after the last `_optional_*` helper):

```python
def _optional_mcp_servers(value: object, label: str) -> list[dict] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ProfileValidationError(f"{label} must be a list")
    out: list[dict] = []
    for i, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ProfileValidationError(f"{label}[{i}] must be a mapping")
        name = entry.get("name")
        command = entry.get("command")
        args = entry.get("args", [])
        if not isinstance(name, str) or not name.strip():
            raise ProfileValidationError(f"{label}[{i}].name must be a non-empty string")
        if not isinstance(command, str) or not command.strip():
            raise ProfileValidationError(f"{label}[{i}].command must be a non-empty string")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ProfileValidationError(f"{label}[{i}].args must be a list of strings")
        out.append({"name": name, "command": command, "args": list(args)})
    return out


def _optional_string_list(value: object, label: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ProfileValidationError(f"{label} must be a list of strings")
    return list(value)
```

- [ ] **Step 8.5: Run tests, confirm pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_llm_profiles_mcp.py -v`
Expected: all green.

- [ ] **Step 8.6: Run existing profile tests for regression**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_provider_config_api.py tests/test_provider_cli.py -v`
Expected: green. New fields are optional and default to `None` so existing profiles parse unchanged.

- [ ] **Step 8.7: Commit**

```bash
git add annotation_pipeline_skill/llm/profiles.py tests/test_llm_profiles_mcp.py
git commit -m "feat(profiles): add mcp_servers, strict_mcp_config, disallowed_tools"
```

---

## Task 9: Wire MCP config into Claude CLI invocation

> **Schema note:** `isolated_claude_home` already handles base_url and api_key injection into the subprocess in isolation. This task does **not** touch env composition — it only adds MCP-related CLI flags and the per-invocation `mcp-config.json` materialization. The tmp config file's lifetime is managed **inside** the `isolated_claude_home` context (so it's cleaned up at the same time as the isolated home root).

**Files:**
- Modify: `annotation_pipeline_skill/llm/local_cli.py:123-153` (`build_claude_command`)
- Modify: `annotation_pipeline_skill/llm/local_cli.py:419-…` (`_generate_claude`)
- Test: `tests/test_local_cli_claude_mcp.py`

- [ ] **Step 9.1: Write failing tests for the new command composition**

Create `tests/test_local_cli_claude_mcp.py`:

```python
from pathlib import Path

from annotation_pipeline_skill.llm.local_cli import build_claude_command


def test_build_claude_command_without_mcp_unchanged():
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
    )
    assert "--mcp-config" not in cmd
    assert "--strict-mcp-config" not in cmd
    assert "--disallowedTools" not in cmd
    # Existing flags must still be there.
    assert "--bare" in cmd
    assert "--no-session-persistence" in cmd


def test_build_claude_command_includes_mcp_config_path(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text("{}")
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        mcp_config_path=cfg_path, strict_mcp_config=True,
        disallowed_tools=["Bash", "Edit"],
    )
    assert "--mcp-config" in cmd
    assert str(cfg_path) in cmd
    assert "--strict-mcp-config" in cmd
    assert "--disallowedTools" in cmd
    assert "Bash,Edit" in cmd


def test_build_claude_command_disallowed_tools_only(tmp_path):
    """disallowed_tools without mcp_config is also valid (lock down tools)."""
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        disallowed_tools=["Bash"],
    )
    assert "--disallowedTools" in cmd
    assert "Bash" in cmd
```

- [ ] **Step 9.2: Run tests, confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_local_cli_claude_mcp.py -v`
Expected: `TypeError: build_claude_command() got an unexpected keyword argument 'mcp_config_path'`.

- [ ] **Step 9.3: Extend `build_claude_command`**

In `annotation_pipeline_skill/llm/local_cli.py`, find the existing `build_claude_command` (around line 123). Update its signature and body to accept the new MCP-related kwargs. The current implementation is:

```python
def build_claude_command(
    *,
    binary: str,
    model: str,
    permission_mode: str | None,
    session_id: str | None = None,
) -> list[str]:
    # --bare: ...
    command = [binary, "--bare", "-p"]
    if session_id:
        command.extend(["--resume", session_id])
    else:
        command.append("--no-session-persistence")
    command.extend([
        "--verbose",
        "--output-format",
        "stream-json",
        "--model",
        model,
    ])
    if permission_mode:
        command.extend(["--permission-mode", permission_mode])
    command.append("-")
    return command
```

Replace it with:

```python
def build_claude_command(
    *,
    binary: str,
    model: str,
    permission_mode: str | None,
    session_id: str | None = None,
    mcp_config_path: Path | None = None,
    strict_mcp_config: bool = False,
    disallowed_tools: list[str] | None = None,
) -> list[str]:
    # --bare: never read OAuth / keychain / ~/.claude credentials. Auth is
    # strictly ANTHROPIC_API_KEY (no token writeback can clobber real creds).
    # Also skips hooks, auto-memory, CLAUDE.md auto-discovery, background
    # prefetches — exactly the surface we don't want in a worker.
    command = [binary, "--bare", "-p"]
    if session_id:
        command.extend(["--resume", session_id])
    else:
        command.append("--no-session-persistence")
    command.extend([
        "--verbose",
        "--output-format",
        "stream-json",
        "--model",
        model,
    ])
    if permission_mode:
        command.extend(["--permission-mode", permission_mode])
    if mcp_config_path is not None:
        command.extend(["--mcp-config", str(mcp_config_path)])
        if strict_mcp_config:
            command.append("--strict-mcp-config")
    if disallowed_tools:
        command.extend(["--disallowedTools", ",".join(disallowed_tools)])
    command.append("-")
    return command
```

`from pathlib import Path` is already imported at the top of `local_cli.py`.

- [ ] **Step 9.4: Run new command tests, confirm pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_local_cli_claude_mcp.py -v`
Expected: all 3 tests pass.

- [ ] **Step 9.5: Wire profile fields into `_generate_claude`**

Open `annotation_pipeline_skill/llm/local_cli.py` and locate `_generate_claude` (around line 419). Inside it, **two surgical changes** are needed; everything else stays as-is.

Change A — pass new kwargs to `build_claude_command`. Find this block:

```python
        command = build_claude_command(
            binary="claude",
            model=self.profile.model,
            permission_mode=self.profile.permission_mode,
            session_id=session_id,
        )
```

Replace with:

```python
        command = build_claude_command(
            binary="claude",
            model=self.profile.model,
            permission_mode=self.profile.permission_mode,
            session_id=session_id,
            mcp_config_path=None,  # set inside the isolated_claude_home block below
            strict_mcp_config=bool(self.profile.strict_mcp_config),
            disallowed_tools=self.profile.disallowed_tools,
        )
```

Change B — materialize the per-invocation `mcp-config.json` **inside the `isolated_claude_home` context**, then rebuild the command with the resolved path. Find this block:

```python
        with isolated_claude_home(
            {**os.environ, **request.env},
            home_id=home_id,
            provider_api_key=api_key,
            provider_base_url=self.profile.base_url,
        ) as (env, _home, resolved_home_id):
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(request.cwd) if request.cwd else None,
                env=env,
                ...
            )
```

Replace the *interior* of the `with` block (everything from the `with` line down to the `await asyncio.wait_for(...)` call) with:

```python
        with isolated_claude_home(
            {**os.environ, **request.env},
            home_id=home_id,
            provider_api_key=api_key,
            provider_base_url=self.profile.base_url,
        ) as (env, _home, resolved_home_id):
            # Materialize the per-invocation mcp-config.json INSIDE the
            # isolated home so it's automatically cleaned up with the home
            # (no separate try/finally).
            mcp_servers = self.profile.mcp_servers or []
            if mcp_servers:
                import json as _json
                mcp_payload = {
                    "mcpServers": {
                        s["name"]: {"command": s["command"], "args": s["args"]}
                        for s in mcp_servers
                    }
                }
                mcp_config_path = _home / "mcp-config.json"
                mcp_config_path.write_text(_json.dumps(mcp_payload), encoding="utf-8")
                # Rebuild the command now that we have a real path.
                command = build_claude_command(
                    binary="claude",
                    model=self.profile.model,
                    permission_mode=self.profile.permission_mode,
                    session_id=session_id,
                    mcp_config_path=mcp_config_path,
                    strict_mcp_config=bool(self.profile.strict_mcp_config),
                    disallowed_tools=self.profile.disallowed_tools,
                )
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(request.cwd) if request.cwd else None,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_die_with_parent,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode("utf-8")),
                    timeout=self.profile.timeout_seconds,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                process.kill()
                try:
                    await process.wait()
                except Exception:  # noqa: BLE001
                    pass
                raise
```

The rest of `_generate_claude` (parsing stdout, building the `LLMGenerateResult`) stays unchanged.

> **Why this is the right shape:** when `mcp_servers` is unset the inner block is a single `if` guard that does nothing — existing behaviour preserved exactly. When it's set, the config file lives inside the isolated home directory which `isolated_claude_home` is already responsible for managing, so no extra try/finally for tmp file cleanup is needed.

- [ ] **Step 9.6: Run all Claude CLI-touching tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/ -q -k "claude or local_cli or provider"`
Expected: green.

- [ ] **Step 9.7: Run the full test suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest -q`
Expected: green.

- [ ] **Step 9.8: Commit**

```bash
git add annotation_pipeline_skill/llm/local_cli.py tests/test_local_cli_claude_mcp.py
git commit -m "feat(local_cli): materialize MCP config inside isolated home for Claude CLI launches"
```

---

## Task 10: End-to-end manual verification (gated, not run in CI)

> This task is a manual verification step, not an automated test. Its purpose is to confirm the full flow with a real Claude CLI binary against a fixture project.

**Files:** none modified — this is a documented run-through.

- [ ] **Step 10.1: Create a fixture project**

```bash
cd /tmp
rm -rf annotation-kb-fixture
annotation-pipeline init --project-root /tmp/annotation-kb-fixture
```

- [ ] **Step 10.2: Seed conventions with row data**

Run the following Python one-liner to populate `entity_conventions` with sample data:

```bash
cd /home/derek/Projects/annotation-pipeline-skill
UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python -c "
from pathlib import Path
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.services.entity_convention_service import EntityConventionService

store = SqliteStore(Path('/tmp/annotation-kb-fixture/.annotation-pipeline/db.sqlite'))
svc = EntityConventionService(store)
for i in range(5):
    svc.record_decision(
        project_id='memory-ner-v2', span='Android', entity_type='technology',
        source='qc_consensus', task_id=f'task_{i}', row_id=f'row_{i}',
        row_content=f'Crashes on Android 10 in scenario {i}',
    )
print('seeded')
"
```

- [ ] **Step 10.3: Add a profile to `llm_profiles.yaml`**

Bootstrap the file from the example template if it doesn't exist:

```bash
cp /tmp/annotation-kb-fixture/.annotation-pipeline/llm_profiles.example.yaml \
   /tmp/annotation-kb-fixture/.annotation-pipeline/llm_profiles.yaml
```

Then edit it and add a new profile **alongside** the existing ones (note the flat `runtime` schema — no `provider`/`cli_kind` fields):

```yaml
profiles:
  # ... existing profiles ...
  annotator_claude_kb:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    permission_mode: bypassPermissions  # required for MCP tools in --print mode
    mcp_servers:
      - name: annotation-kb
        command: python
        args:
          - -m
          - annotation_pipeline_skill.mcp.kb_server
          - --db-path
          - /tmp/annotation-kb-fixture/.annotation-pipeline/db.sqlite
          - --project-id
          - memory-ner-v2
    strict_mcp_config: true
    disallowed_tools: ["Bash", "Edit", "Write"]
```

- [ ] **Step 10.4: Run a real subagent cycle through the runtime**

Bind the new `annotator_claude_kb` profile to the `annotation` stage target in the project's runtime configuration, then trigger a single subagent cycle against a fixture task:

```bash
annotation-pipeline runtime once --project-root /tmp/annotation-kb-fixture
```

The runtime composes `claude` with `--bare`, `--mcp-config <path-inside-isolated-home>`, `--strict-mcp-config`, `--disallowedTools=Bash,Edit,Write`, and `isolated_claude_home` injects `base_url` + `api_key` into the subprocess environment in isolation. The operator does **not** export or set anything in the shell — the parent process's env is untouched.

Inspect the subagent's transcript (under the project's artifacts/ directory) and confirm the agent invoked `check_past_experience` with `entry="Android"` and received a response showing `convention.type == "technology"` and 5 evidence proposals.

> Going through the runtime is the **only** verification path in this plan: it tests the production launch sequence (isolated home, env injection, MCP config materialization, tool-call round-trip) end-to-end without requiring the operator to manage shell state.

- [ ] **Step 10.5: Update CHANGELOG**

Edit `CHANGELOG.md` and add an entry under the next unreleased version:

```markdown
### Added
- `check_past_experience` MCP tool for annotator/QC/arbiter subagents: returns convention status, type distribution, diverse sentence-level examples, and wordfreq metadata for a candidate span. Wired via `--mcp-config` on Claude CLI subagents.
- `LLMProfile` schema: `mcp_servers`, `strict_mcp_config`, `disallowed_tools` for declaring MCP servers and locking down built-in tools per profile. LLM provider switching continues to use the existing `base_url` field — `isolated_claude_home` injects it into each subagent subprocess in isolation, leaving the operator's shell untouched.
- CJK fallback in `similarity.shingle()`: rows containing CJK characters are segmented with jieba instead of degenerating to a single shingle, improving both KB diversity sampling and existing row_dedup precision on Chinese text.

### Changed
- `EntityConventionService.record_decision()` accepts optional `row_id` and `row_content`; proposals now carry a `context_snippet` window built from the row content (the QC/HR call sites are updated; operator-declared sites leave `row_id` and `row_content` unset).
```

- [ ] **Step 10.6: Commit CHANGELOG**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): annotation knowledge base MCP tool + LLM profile switching"
```

---

## Self-Review Checklist

Cross-check against the spec sections:

| Spec Section | Implementing Task(s) |
|---|---|
| Tool Contract | Task 6 (logic) + Task 7 (MCP wrapper) |
| Schema Extension (proposals row_id, context_snippet) | Task 4 |
| `record_decision` signature change | Task 4 |
| Call site updates | Task 5 |
| Diversity Sampling (MinHash + farthest-first) | Task 3 |
| CJK Shingle Fallback | Task 2 |
| Shared `wordfreq_score` utility | Task 1 |
| MCP server (kb_server.py) | Task 7 |
| `mcp_servers` / `strict_mcp_config` / `disallowed_tools` profile fields | Task 8 |
| `build_claude_command` MCP wiring | Task 9 |
| LLM switching via profile `base_url` (isolated subprocess env, no shell pollution) | Already in place via `LLMProfile.base_url` + `isolated_claude_home` — no new work required |
| Tests (unit + integration smoke) | Tasks 1, 2, 3, 4, 6, 7, 8, 9 |
| Manual Claude CLI verification | Task 10 |

**Spec gaps identified:** none — every spec section maps to at least one task.

**Type consistency review:**
- `record_decision()` kwargs (`row_id: str | None`, `row_content: str | None`) consistent in Tasks 4, 5, 6.
- `mcp_servers` shape (`list[dict]` with `name`/`command`/`args`) consistent in Tasks 8, 9.
- `check_past_experience` return-dict shape consistent between Task 6 logic and Task 7 server passthrough.
- `build_claude_command` signature (Task 9.3) matches the call sites in `_generate_claude` (Task 9.5).
- LLMProfile schema fields (`mcp_servers`, `strict_mcp_config`, `disallowed_tools`) consistent in Tasks 8 (schema), 9 (consumption), and 10 (example yaml).
- All Task 10 yaml examples use the current flat `runtime: claude_cli` schema (no `provider`/`cli_kind`).

No placeholders detected.

---

## Revision Notes (2026-05-19, post-rebase)

The original plan was written against an older LLMProfile schema (`provider: local_cli` + `cli_kind: claude` + `cli_binary: claude`). The codebase has since migrated to a flat `runtime: claude_cli | codex_cli | …` schema with `base_url` as a required field and `isolated_claude_home` already translating `base_url` → `ANTHROPIC_BASE_URL`. Tasks 8, 9, and 10 were updated accordingly:

- **Removed `extra_env` field** — `base_url` is already a `LLMProfile` field, and `isolated_claude_home` already injects it into the subprocess in isolation. No env-merge work needed.
- **Task 9 mcp-config.json materialization moved inside `isolated_claude_home`** — the tmp file lives at `<isolated_home>/mcp-config.json` and is cleaned up with the home. No separate try/finally.
- **All yaml examples in Task 10 rewritten** to the flat `runtime:` schema.
- **Task 5 line-number guidance softened** — added `grep -n record_decision` step to locate current line numbers rather than baking specific line numbers into the plan.
- **Task 10 standalone smoke test removed** — the only verification path is now through the runtime, so the operator never touches shell env state. The parent process's environment remains untouched throughout the production launch sequence (subprocess env injection in `isolated_claude_home` does not modify `os.environ` or the user's shell).

Tasks 1, 2, 3, 4, 6, 7 are unchanged — they touch modules (`text/`, `similarity/minhash.py`, `similarity/diverse.py`, `entity_convention_service.py`, new `mcp/` package) that the recent refactor did not affect.

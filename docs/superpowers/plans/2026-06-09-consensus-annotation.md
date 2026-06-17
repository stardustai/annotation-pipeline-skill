# Consensus (N-way Duplicate) Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-driven `replicas: N` annotation mode where the pipeline runs N independent annotators per task, keeps spans agreed by ≥K of them, and routes the rest to an arbiter that resolves conflicts and fills gaps — instead of the current single-annotator flow.

**Architecture:** A new pure module (`runtime/consensus.py`) computes a consensus annotation + a disagreement list from N drafts. A new `AnnotationConfig` (parsed from `workflow.yaml` `stages.annotation`) carries `replicas / targets / keep_threshold / on_disagree / arbiter_target`. `SubagentRuntime` gains a `_run_consensus_annotation` path used only when `replicas > 1`; `replicas == 1` keeps the existing single-annotation flow byte-for-byte. The arbiter step reuses the existing `_generate_async` LLM machinery with a dedicated merge prompt.

**Tech Stack:** Python 3.13, dataclasses, asyncio, pytest. SQLite store. Existing `SubagentRuntime` in `annotation_pipeline_skill/runtime/subagent_cycle.py`.

**Empirical basis (why this design):** On 24 v5 tasks, every dual+arbiter configuration scored F1 0.97–0.99 regardless of model strength (qwen+Haiku+qwen-arbiter = 0.971; qwen+M3 = 0.994), vs single annotators 0.93–0.98. The *structure* (two drafts → high-recall union → cheap selection) carries the quality; model strength buys reliability + the last point. Sweet spot is N=2, K=2 (keep unanimous, arbitrate the rest). See `scratch/ab/` for the raw runs.

**Out of scope (separate effort):** Re-annotating the existing 2788 accepted qwen tasks offline (a one-shot script `scratch/ab_weakarb.py`-style batch). This plan only adds the *online pipeline* mode.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `annotation_pipeline_skill/runtime/consensus.py` | Pure consensus logic: N drafts → (consensus payload, disagreements); arbiter merge prompt builder | Create |
| `annotation_pipeline_skill/core/runtime.py` | Add `AnnotationConfig` dataclass + parsing | Modify |
| `annotation_pipeline_skill/config/loader.py` | Parse `stages.annotation` into `AnnotationConfig`; attach to `ProjectConfig` | Modify |
| `annotation_pipeline_skill/runtime/subagent_cycle.py` | `_run_consensus_annotation` path + dispatch in `_run_task`; accept `annotation_config` | Modify |
| `annotation_pipeline_skill/interfaces/cli.py` | Thread `AnnotationConfig` from project config into `SubagentRuntime` | Modify |
| `tests/test_consensus.py` | Unit tests for consensus + merge-prompt logic | Create |
| `tests/test_annotation_config.py` | Config parse/validate tests | Create |
| `tests/test_consensus_annotation_runtime.py` | Runtime integration test with stub clients | Create |

**Annotation payload shape** (used throughout): a parsed annotation is `{"rows": [{"row_index": int, "output": {"entities": {type: [span,...]}, "json_structures": {type: [span,...]}}}, ...]}`. On disk the artifact payload is `{"text": "<json string of the above>"}`.

---

## Phase 1 — Config

### Task 1: `AnnotationConfig` dataclass

**Files:**
- Modify: `annotation_pipeline_skill/core/runtime.py` (add near the top, after `RuntimeConfig`)
- Test: `tests/test_annotation_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_annotation_config.py
import pytest
from annotation_pipeline_skill.core.runtime import AnnotationConfig


def test_defaults_to_single_annotation():
    c = AnnotationConfig.from_dict({})
    assert c.replicas == 1
    assert c.targets == ["annotation"]
    assert c.keep_threshold == 1
    assert c.on_disagree == "arbiter"
    assert c.arbiter_target == "arbiter"


def test_dual_explicit_targets():
    c = AnnotationConfig.from_dict({
        "replicas": 2,
        "targets": ["annotator_a", "annotator_b"],
        "keep_threshold": 2,
        "arbiter_target": "arbiter",
    })
    assert c.replicas == 2
    assert c.targets == ["annotator_a", "annotator_b"]
    assert c.keep_threshold == 2


def test_single_target_broadcast_to_n_replicas():
    # one target + replicas: N → run the same target N times
    c = AnnotationConfig.from_dict({"replicas": 3, "targets": ["annotation"]})
    assert c.targets == ["annotation", "annotation", "annotation"]
    assert c.replicas == 3


def test_keep_threshold_defaults_to_replicas():
    c = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"]})
    assert c.keep_threshold == 2  # unanimous by default
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_annotation_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'AnnotationConfig'`

- [ ] **Step 3: Implement `AnnotationConfig`**

Add to `annotation_pipeline_skill/core/runtime.py` (after the `RuntimeConfig` class):

```python
@dataclass(frozen=True)
class AnnotationConfig:
    """Annotation-stage topology, parsed from workflow.yaml `stages.annotation`.

    replicas == 1 reproduces the legacy single-annotator flow exactly.
    replicas > 1 runs N annotators (one per entry in `targets`), keeps spans
    agreed by >= keep_threshold of them, and routes the rest to `arbiter_target`.
    """
    replicas: int = 1
    targets: list[str] = field(default_factory=lambda: ["annotation"])
    keep_threshold: int = 1
    on_disagree: str = "arbiter"   # "arbiter" (resolve+fill) | "drop" (skip below-threshold)
    arbiter_target: str = "arbiter"

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationConfig":
        data = data or {}
        replicas = int(data.get("replicas", 1))
        targets = list(data.get("targets") or ["annotation"])
        # Broadcast a single target to N replicas (same model run N times).
        if len(targets) == 1 and replicas > 1:
            targets = targets * replicas
        keep_threshold = int(data.get("keep_threshold", replicas))
        return cls(
            replicas=replicas,
            targets=targets,
            keep_threshold=keep_threshold,
            on_disagree=str(data.get("on_disagree", "arbiter")),
            arbiter_target=str(data.get("arbiter_target", "arbiter")),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_annotation_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/core/runtime.py tests/test_annotation_config.py
git commit -m "feat(config): add AnnotationConfig for N-way consensus annotation"
```

### Task 2: Validate `AnnotationConfig`

**Files:**
- Modify: `annotation_pipeline_skill/core/runtime.py` (add `validate()` to `AnnotationConfig`)
- Test: `tests/test_annotation_config.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_annotation_config.py
def test_validate_rejects_bad_threshold():
    c = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 3})
    with pytest.raises(ValueError, match="keep_threshold"):
        c.validate()


def test_validate_rejects_target_count_mismatch():
    c = AnnotationConfig.from_dict({"replicas": 3, "targets": ["a", "b"]})
    with pytest.raises(ValueError, match="targets"):
        c.validate()


def test_validate_accepts_single():
    AnnotationConfig.from_dict({}).validate()  # no raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_annotation_config.py::test_validate_rejects_bad_threshold -v`
Expected: FAIL — `AttributeError: 'AnnotationConfig' object has no attribute 'validate'`

- [ ] **Step 3: Implement `validate()`**

Add as a method on `AnnotationConfig`:

```python
    def validate(self) -> None:
        if self.replicas < 1:
            raise ValueError(f"annotation.replicas must be >= 1, got {self.replicas}")
        if len(self.targets) != self.replicas:
            raise ValueError(
                f"annotation.targets must list exactly replicas={self.replicas} entries, "
                f"got {len(self.targets)}: {self.targets}"
            )
        if not (1 <= self.keep_threshold <= self.replicas):
            raise ValueError(
                f"annotation.keep_threshold must be in [1, replicas={self.replicas}], "
                f"got {self.keep_threshold}"
            )
        if self.on_disagree not in {"arbiter", "drop"}:
            raise ValueError(f"annotation.on_disagree must be 'arbiter' or 'drop', got {self.on_disagree!r}")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_annotation_config.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/core/runtime.py tests/test_annotation_config.py
git commit -m "feat(config): validate AnnotationConfig invariants"
```

### Task 3: Parse `stages.annotation` in the config loader

**Files:**
- Modify: `annotation_pipeline_skill/config/loader.py:build_project_config_from_data`
- Modify: `annotation_pipeline_skill/core/runtime.py` (import `field` already present; ensure `AnnotationConfig` exported)
- Test: `tests/test_annotation_config.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_annotation_config.py
from annotation_pipeline_skill.config.loader import build_project_config_from_data


def test_loader_parses_stages_annotation():
    cfg = build_project_config_from_data(
        annotators_data={}, external_data={}, callbacks_data={},
        workflow_data={"stages": {"annotation": {"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2}}},
    )
    assert cfg.annotation.replicas == 2
    assert cfg.annotation.targets == ["a", "b"]


def test_loader_defaults_single_when_absent():
    cfg = build_project_config_from_data(
        annotators_data={}, external_data={}, callbacks_data={}, workflow_data={},
    )
    assert cfg.annotation.replicas == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_annotation_config.py::test_loader_parses_stages_annotation -v`
Expected: FAIL — `AttributeError: 'ProjectConfig' object has no attribute 'annotation'`

- [ ] **Step 3: Implement**

In `annotation_pipeline_skill/config/loader.py`, find the `ProjectConfig` dataclass definition (search `class ProjectConfig`) and add a field:

```python
    annotation: "AnnotationConfig" = field(default_factory=lambda: AnnotationConfig())
```

Add the import at the top of `loader.py`:

```python
from annotation_pipeline_skill.core.runtime import AnnotationConfig, RuntimeConfig
```

In `build_project_config_from_data`, add `annotation=` to the `ProjectConfig(...)` call:

```python
    return ProjectConfig(
        annotators=_load_annotators(annotators_data.get("annotators", {})),
        external_tasks=external_data.get("external_tasks", {}),
        callbacks=callbacks_data.get("callbacks", {}),
        workflow=workflow_data,
        runtime=RuntimeConfig.from_dict(workflow_data.get("runtime") or {}),
        annotation=AnnotationConfig.from_dict(
            (workflow_data.get("stages") or {}).get("annotation") or {}
        ),
    )
```

(If `ProjectConfig` is a frozen dataclass and `field` is not imported there, add `from dataclasses import field` to its module.)

- [ ] **Step 4: Run to verify pass + no regressions in config tests**

Run: `.venv/bin/python -m pytest tests/test_annotation_config.py -q && .venv/bin/python -m pytest -q -k "config or loader"`
Expected: PASS; no new failures.

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/config/loader.py annotation_pipeline_skill/core/runtime.py tests/test_annotation_config.py
git commit -m "feat(config): load stages.annotation into ProjectConfig.annotation"
```

---

## Phase 2 — Pure consensus logic

### Task 4: Iterate span items from an annotation payload

**Files:**
- Create: `annotation_pipeline_skill/runtime/consensus.py`
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_consensus.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_consensus.py -v`
Expected: FAIL — `ModuleNotFoundError: ... consensus`

- [ ] **Step 3: Implement**

```python
# annotation_pipeline_skill/runtime/consensus.py
"""Pure consensus logic for N-way (duplicate) annotation.

Given N independent annotation drafts of the same task, compute:
  - a consensus payload: spans agreed by >= keep_threshold drafts
  - a disagreement list: spans present in some-but-fewer drafts (for the arbiter)

No I/O, no LLM calls — fully unit-testable. The runtime layer wires this to
the actual annotators + arbiter.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterator

SpanItem = tuple[int, str, str, str]  # (row_index, field, type, span)
_FIELDS = ("entities", "json_structures")


def iter_span_items(payload: dict) -> Iterator[SpanItem]:
    """Yield (row_index, field, type, span) for every span in a parsed
    annotation payload {"rows": [{"row_index", "output": {...}}]}."""
    if not isinstance(payload, dict):
        return
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        ri = row.get("row_index", 0)
        out = row.get("output") or {}
        if not isinstance(out, dict):
            continue
        for field in _FIELDS:
            buckets = out.get(field) or {}
            if not isinstance(buckets, dict):
                continue
            for typ, spans in buckets.items():
                if not isinstance(spans, list):
                    continue
                for span in spans:
                    if isinstance(span, str) and span:
                        yield (ri, field, typ, span)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_consensus.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/runtime/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): iter_span_items flatten helper"
```

### Task 5: `build_consensus`

**Files:**
- Modify: `annotation_pipeline_skill/runtime/consensus.py`
- Test: `tests/test_consensus.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consensus.py
from annotation_pipeline_skill.runtime.consensus import build_consensus


def _row(ri, ents):
    return {"row_index": ri, "output": {"entities": ents}}


def test_unanimous_kept_one_sided_disagrees():
    a = {"rows": [_row(0, {"person": ["Alice"], "organization": ["ACME"]})]}
    b = {"rows": [_row(0, {"person": ["Alice"], "technology": ["Spark"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=2)
    # Alice in both -> consensus; ACME (only A) + Spark (only B) -> disagreements
    assert consensus["rows"][0]["output"]["entities"] == {"person": ["Alice"]}
    dis = {(d["field"], d["type"], d["span"], d["support"]) for d in disagree}
    assert dis == {
        ("entities", "organization", "ACME", 1),
        ("entities", "technology", "Spark", 1),
    }


def test_threshold_one_is_union():
    a = {"rows": [_row(0, {"person": ["Alice"]})]}
    b = {"rows": [_row(0, {"person": ["Bob"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=1)
    assert sorted(consensus["rows"][0]["output"]["entities"]["person"]) == ["Alice", "Bob"]
    assert disagree == []


def test_type_conflict_same_span_surfaces_both_as_disagreements():
    a = {"rows": [_row(0, {"organization": ["Apple"]})]}
    b = {"rows": [_row(0, {"technology": ["Apple"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=2)
    assert consensus["rows"][0]["output"].get("entities", {}) == {}
    types = {(d["type"], d["span"]) for d in disagree}
    assert types == {("organization", "Apple"), ("technology", "Apple")}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_consensus.py -k build_consensus -v` (or the new test names)
Expected: FAIL — `ImportError: cannot import name 'build_consensus'`

- [ ] **Step 3: Implement**

Append to `consensus.py`:

```python
def build_consensus(drafts: list[dict], keep_threshold: int) -> tuple[dict, list[dict]]:
    """Return (consensus_payload, disagreements).

    consensus_payload: same {"rows":[...]} shape, containing only spans whose
      (row, field, type, span) support count is >= keep_threshold.
    disagreements: list of {"row_index","field","type","span","support"} for
      items with 0 < support < keep_threshold.
    """
    counts: Counter[SpanItem] = Counter()
    for draft in drafts:
        # de-dup within a single draft so a draft can't vote twice
        counts.update(set(iter_span_items(draft)))

    kept: list[SpanItem] = []
    disagreements: list[dict] = []
    for item, support in counts.items():
        if support >= keep_threshold:
            kept.append(item)
        else:
            ri, field, typ, span = item
            disagreements.append(
                {"row_index": ri, "field": field, "type": typ, "span": span, "support": support}
            )

    # Rebuild a payload from kept items, preserving row order seen across drafts.
    row_order: list[int] = []
    seen_rows: set[int] = set()
    for draft in drafts:
        for row in draft.get("rows") or []:
            ri = row.get("row_index", 0) if isinstance(row, dict) else 0
            if ri not in seen_rows:
                seen_rows.add(ri); row_order.append(ri)
    by_row: dict[int, dict] = {ri: {"entities": {}, "json_structures": {}} for ri in row_order}
    for ri, field, typ, span in kept:
        by_row.setdefault(ri, {"entities": {}, "json_structures": {}})[field].setdefault(typ, []).append(span)
    # strip empty fields/buckets for a clean payload
    rows_out = []
    for ri in row_order:
        out = {f: {t: s for t, s in by_row[ri][f].items() if s} for f in _FIELDS}
        out = {f: v for f, v in out.items() if v}
        rows_out.append({"row_index": ri, "output": out})
    return {"rows": rows_out}, disagreements
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_consensus.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/runtime/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): build_consensus (keep >=K, surface disagreements)"
```

### Task 6: Arbiter merge prompt builder

**Files:**
- Modify: `annotation_pipeline_skill/runtime/consensus.py`
- Test: `tests/test_consensus.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consensus.py
from annotation_pipeline_skill.runtime.consensus import build_arbiter_merge_prompt


def test_merge_prompt_contains_drafts_and_disagreements():
    a = {"rows": [_row(0, {"person": ["Alice"], "organization": ["ACME"]})]}
    b = {"rows": [_row(0, {"person": ["Alice"]})]}
    consensus, disagree = build_consensus([a, b], keep_threshold=2)
    prompt = build_arbiter_merge_prompt(
        row_inputs={0: "Alice at ACME"}, drafts=[a, b],
        consensus=consensus, disagreements=disagree,
    )
    assert "ACME" in prompt          # the disputed span is shown
    assert "Alice at ACME" in prompt  # the row input is shown
    assert "json" in prompt.lower()   # asks for JSON output
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_consensus.py -k merge_prompt -v`
Expected: FAIL — `ImportError: cannot import name 'build_arbiter_merge_prompt'`

- [ ] **Step 3: Implement**

Append to `consensus.py`:

```python
import json as _json


def build_arbiter_merge_prompt(
    *, row_inputs: dict[int, str], drafts: list[dict],
    consensus: dict, disagreements: list[dict],
) -> str:
    """Build the user prompt for the arbiter merge call. The arbiter receives the
    per-row input text, the already-agreed consensus, and the disputed spans, and
    must return the FINAL annotation: keep consensus, resolve each disagreement per
    the rules (选择题), and add only clearly-required missed spans (补漏)."""
    rows = []
    for row in consensus.get("rows") or []:
        ri = row.get("row_index", 0)
        rows.append({
            "row_index": ri,
            "input": row_inputs.get(ri, ""),
            "agreed": row.get("output", {}),
            "disputed": [d for d in disagreements if d["row_index"] == ri],
        })
    return (
        "你是标注仲裁器(arbiter)。每行给出 input、已一致的 agreed 标注、以及 disputed(只有部分草稿标了的 span)。\n"
        "产出每行正确的最终标注:\n"
        "- 保留 agreed。\n"
        "- 对 disputed 做选择题:按规则选对的 type;不该标的删。\n"
        "- 补漏:规则明确要求但所有草稿都漏的 span 才补(verbatim)。\n"
        "- 每个 span 必须是该行 input 的 verbatim 子串。\n\n"
        "严格输出 JSON:{\"rows\":[{\"row_index\":int,\"output\":{\"entities\":{...},\"json_structures\":{...}}}]}\n\n"
        + _json.dumps(rows, ensure_ascii=False)
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_consensus.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/runtime/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): build_arbiter_merge_prompt"
```

---

## Phase 3 — Runtime integration

### Task 7: Accept `annotation_config` in `SubagentRuntime`

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py:SubagentRuntime.__init__`
- Test: `tests/test_consensus_annotation_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_consensus_annotation_runtime.py
from annotation_pipeline_skill.core.runtime import AnnotationConfig
from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def test_runtime_defaults_to_single_annotation(tmp_path):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    rt = SubagentRuntime(store, client_factory=lambda t: None)
    assert rt.annotation_config.replicas == 1


def test_runtime_accepts_annotation_config(tmp_path):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2})
    rt = SubagentRuntime(store, client_factory=lambda t: None, annotation_config=cfg)
    assert rt.annotation_config.replicas == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py -v`
Expected: FAIL — `AttributeError: 'SubagentRuntime' object has no attribute 'annotation_config'`

- [ ] **Step 3: Implement**

In `SubagentRuntime.__init__` (signature around line 198), add the kwarg and store it. Add to the parameter list (after `structured_output_targets`):

```python
        annotation_config: "AnnotationConfig | None" = None,
```

Add the import at the top of `subagent_cycle.py` (with the other `core.runtime` imports):

```python
from annotation_pipeline_skill.core.runtime import AnnotationConfig
```

In the `__init__` body (near `self._structured_output_targets = structured_output_targets`):

```python
        self.annotation_config = annotation_config or AnnotationConfig()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_consensus_annotation_runtime.py
git commit -m "feat(runtime): SubagentRuntime accepts annotation_config (default single)"
```

### Task 8: `_run_consensus_annotation` orchestration

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py` (new method on `SubagentRuntime`)
- Test: `tests/test_consensus_annotation_runtime.py` (append)

This method runs the N annotators, builds the consensus, runs the arbiter on disagreements, writes a single final annotation artifact, then hands off to the existing `_run_validation_and_qc`. It deliberately reuses `_generate_async`, `_write_stage_artifact`, `_serialize_llm_json`, and `_run_validation_and_qc`.

- [ ] **Step 1: Write the failing test (stub clients produce 2 drafts + arbiter)**

```python
# append to tests/test_consensus_annotation_runtime.py
import asyncio, json
from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus


class _StubClient:
    """Returns a canned annotation JSON for whichever target built it."""
    def __init__(self, payload_text): self._t = payload_text
    async def generate(self, request):
        from annotation_pipeline_skill.llm.client import LLMGenerateResult
        return LLMGenerateResult(final_text=self._t, provider="stub", model="stub",
                                 raw_response={}, usage={}, diagnostics={})


def _ann(person_rows):
    return json.dumps({"rows": [{"row_index": i, "output": {"entities": {"person": p}}}
                                for i, p in enumerate(person_rows)]})


def test_consensus_two_drafts_produces_final_artifact(tmp_path):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    # task with one row whose input contains both names
    t = Task.new(task_id="t1", pipeline_id="p",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)

    canned = {
        "a": _ann([["Alice", "Bob"]]),   # draft A: both
        "b": _ann([["Alice"]]),           # draft B: only Alice
        "arbiter": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"person": ["Alice", "Bob"]}}}]}),
    }
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2,
                                      "arbiter_target": "arbiter"})
    rt = SubagentRuntime(store, client_factory=lambda target: _StubClient(canned[target]),
                         annotation_config=cfg)

    # run only the consensus-annotation step (not full QC) and inspect the artifact
    task = store.load_task("t1")
    asyncio.run(rt._produce_consensus_annotation(task))

    from annotation_pipeline_skill.services.entity_statistics_service import _load_latest_annotation
    final = _load_latest_annotation(store, "t1")
    persons = final["rows"][0]["output"]["entities"]["person"]
    assert set(persons) == {"Alice", "Bob"}  # Alice agreed; Bob recovered by arbiter
```

> Note: this test targets a focused helper `_produce_consensus_annotation(task)` that returns the written annotation artifact (without running QC), so the consensus + arbiter behaviour is testable in isolation. `_run_consensus_annotation` (Task 9) wraps it + the QC handoff.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py::test_consensus_two_drafts_produces_final_artifact -v`
Expected: FAIL — `AttributeError: ... '_produce_consensus_annotation'`

- [ ] **Step 3: Implement `_produce_consensus_annotation`**

Add to `SubagentRuntime` (place near `_run_task`). It reuses helpers confirmed in the codebase: `_load_guideline`, `_annotation_prompt`, `_annotation_instructions` (module fn), `resolve_output_schema`, `_build_response_format`, `_generate_async`, `_serialize_llm_json`, `_next_attempt_id`, `_write_stage_artifact`. Span-input map comes from `task.source_ref["payload"]["rows"]`.

```python
    async def _produce_consensus_annotation(self, task: "Task"):
        """Run N annotators, build consensus, arbitrate disagreements, and write a
        single final annotation_result artifact. Returns (artifact, attempt_id, text)."""
        import asyncio
        from annotation_pipeline_skill.core.schema_validation import resolve_output_schema
        from annotation_pipeline_skill.runtime import consensus as _consensus

        cfg = self.annotation_config
        guideline = self._load_guideline(task)
        schema = resolve_output_schema(task, self.store)
        user_prompt = self._annotation_prompt(task)
        instr = _annotation_instructions(task, guideline=guideline, output_schema=schema)

        async def _one(target: str):
            res = await self._generate_async(target, LLMGenerateRequest(
                instructions=instr, prompt=user_prompt,
                response_format=self._build_response_format(target, stage="annotation", output_schema=schema),
                task_id=task.task_id))
            cleaned, _, _ = _serialize_llm_json(res.final_text, task=task)
            return json.loads(cleaned)

        drafts = await asyncio.gather(*[_one(t) for t in cfg.targets])
        consensus, disagreements = _consensus.build_consensus(drafts, cfg.keep_threshold)

        final_payload = consensus
        if disagreements and cfg.on_disagree == "arbiter":
            src_rows = (task.source_ref or {}).get("payload", {}).get("rows", []) if isinstance(task.source_ref, dict) else []
            row_inputs = {r.get("row_index", i): str(r.get("input") or r.get("text") or "")
                          for i, r in enumerate(src_rows) if isinstance(r, dict)}
            merge_prompt = _consensus.build_arbiter_merge_prompt(
                row_inputs=row_inputs, drafts=drafts, consensus=consensus, disagreements=disagreements)
            arb = await self._generate_async(cfg.arbiter_target, LLMGenerateRequest(
                instructions=instr, prompt=merge_prompt,
                response_format=self._build_response_format(cfg.arbiter_target, stage="annotation", output_schema=schema),
                task_id=task.task_id))
            cleaned, _, _ = _serialize_llm_json(arb.final_text, task=task)
            final_payload = json.loads(cleaned)

        attempt_id = self._next_attempt_id(task)
        text = json.dumps(final_payload, ensure_ascii=False, sort_keys=True)
        # Wrap a minimal result object for _write_stage_artifact's expectations.
        from annotation_pipeline_skill.llm.client import LLMGenerateResult
        result = LLMGenerateResult(final_text=text, provider="consensus",
                                   model=",".join(cfg.targets), raw_response={}, usage={}, diagnostics={})
        artifact = self._write_stage_artifact(task, result, kind="annotation_result",
                                              attempt_id=attempt_id, payload={"text": text})
        return artifact, attempt_id, text
```

> If `_write_stage_artifact`'s real signature differs from `(task, result, *, kind, attempt_id, payload)` (confirm at `subagent_cycle.py:3547`), adapt the call — it is used identically at `subagent_cycle.py:459`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_consensus_annotation_runtime.py
git commit -m "feat(runtime): _produce_consensus_annotation (N annotate -> consensus -> arbiter)"
```

### Task 9: Dispatch consensus path from `_run_task`

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py:_run_task`
- Test: `tests/test_consensus_annotation_runtime.py` (append)

- [ ] **Step 1: Write the failing test (full task → ACCEPTED via consensus, QC stubbed to accept)**

```python
# append to tests/test_consensus_annotation_runtime.py
def test_run_task_consensus_reaches_qc(tmp_path, monkeypatch):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t2", pipeline_id="p",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    canned = {"a": _ann([["Alice", "Bob"]]), "b": _ann([["Alice"]]),
              "arbiter": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"person": ["Alice", "Bob"]}}}]})}
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2, "arbiter_target": "arbiter"})
    rt = SubagentRuntime(store, client_factory=lambda target: _StubClient(canned[target]), annotation_config=cfg)

    # stub the QC handoff so the test focuses on the dispatch + artifact, not QC internals
    called = {}
    async def fake_qc(task, artifact, attempt_id, text):
        called["ok"] = True
    monkeypatch.setattr(rt, "_run_validation_and_qc", fake_qc)

    asyncio.run(rt._run_task(store.load_task("t2"), "annotation"))
    assert called.get("ok") is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py::test_run_task_consensus_reaches_qc -v`
Expected: FAIL — `_run_validation_and_qc` not called (single-path runs instead, using a target named "annotation" that the stub_factory KeyErrors on) OR the assertion fails.

- [ ] **Step 3: Implement the dispatch**

In `_run_task`, immediately after the early-return special cases (after the prelabeled block ends, before `guideline = self._load_guideline(task)` at line 364), insert:

```python
        if self.annotation_config.replicas > 1 and task.status in (TaskStatus.PENDING, TaskStatus.ANNOTATING):
            self._transition(task, TaskStatus.ANNOTATING,
                             reason=f"consensus annotation: {self.annotation_config.replicas} replicas",
                             stage="annotation", attempt_id=self._next_attempt_id(task))
            artifact, attempt_id, text = await self._produce_consensus_annotation(task)
            await self._run_validation_and_qc(task, artifact, attempt_id, text)
            return
```

> This guard runs the consensus path only when configured (`replicas > 1`); `replicas == 1` falls through to the unchanged single-annotation code below it.

- [ ] **Step 4: Run to verify pass + backward-compat regression**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py -v && .venv/bin/python -m pytest tests/test_subagent_cycle.py -q`
Expected: new tests PASS; `test_subagent_cycle.py` unchanged (still passing as before — single path untouched).

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_consensus_annotation_runtime.py
git commit -m "feat(runtime): dispatch consensus annotation path when replicas>1"
```

---

## Phase 4 — Wiring + docs

### Task 10: Thread `AnnotationConfig` from project config into the runtime

**Files:**
- Modify: `annotation_pipeline_skill/interfaces/cli.py:_build_runtime_scheduler` (and `LocalRuntimeScheduler` if it constructs the `SubagentRuntime`)
- Test: `tests/test_consensus_annotation_runtime.py` (append a wiring test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consensus_annotation_runtime.py
def test_scheduler_threads_annotation_config(tmp_path):
    from annotation_pipeline_skill.runtime.local_scheduler import LocalRuntimeScheduler
    from annotation_pipeline_skill.core.runtime import RuntimeConfig
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2})
    sched = LocalRuntimeScheduler(store=store, client_factory=lambda t: None,
                                  config=RuntimeConfig(), annotation_config=cfg)
    assert sched.annotation_config.replicas == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py::test_scheduler_threads_annotation_config -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'annotation_config'`

- [ ] **Step 3: Implement**

In `LocalRuntimeScheduler.__init__` (`annotation_pipeline_skill/runtime/local_scheduler.py`), add a kwarg `annotation_config=None`, store `self.annotation_config = annotation_config`, and pass it through where it constructs `SubagentRuntime` (around `local_scheduler.py:440`):

```python
        runtime = self._runtime_override or SubagentRuntime(
            store=self.store,
            client_factory=self.client_factory,
            max_qc_rounds=self.config.max_qc_rounds,
            config=self.config,
            structured_output_targets=self._structured_output_targets(),
            annotation_config=self.annotation_config,
        )
```

In `cli.py:_build_runtime_scheduler`, load the project config and pass its `.annotation`:

```python
    from annotation_pipeline_skill.config.loader import load_project_config
    proj = load_project_config(context.project_root)
    proj.annotation.validate()
    return LocalRuntimeScheduler(
        store=context.store,
        client_factory=lambda target: _build_client(context.registry.resolve(target)),
        registry=context.registry,
        client_builder=_build_client,
        config=runtime_config,
        profiles_yaml_path=profiles_yaml_path,
        annotation_config=proj.annotation,
    )
```

- [ ] **Step 4: Run to verify pass + full regression**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py -v && .venv/bin/python -m pytest -q`
Expected: new tests PASS; only the known pre-existing failures remain (test_cli x2, dashboard_api x2, local_cli_client x2, prior_verifier — unchanged count).

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/interfaces/cli.py annotation_pipeline_skill/runtime/local_scheduler.py tests/test_consensus_annotation_runtime.py
git commit -m "feat(runtime): thread AnnotationConfig from workflow.yaml into the scheduler"
```

### Task 11: Document the config + add an example

**Files:**
- Modify: `annotation_pipeline_skill/interfaces/cli.py` `CONFIG_FILES["workflow.yaml"]` seed (search `"workflow.yaml": """`) — add a commented `stages.annotation` block
- Create: `docs/consensus-annotation.md`

- [ ] **Step 1: Add the commented example to the seeded workflow.yaml**

In `cli.py`, in the `CONFIG_FILES["workflow.yaml"]` string, under `stages:` add:

```yaml
  annotation:
    target: annotation
    # --- N-way consensus annotation (optional; omit for single-annotator) ---
    # replicas: 2                       # run N annotators per task
    # targets: [annotator_a, annotator_b]   # one target per replica (or 1 target, broadcast to N)
    # keep_threshold: 2                 # keep spans agreed by >=K of N (2 = unanimous)
    # on_disagree: arbiter              # arbiter | drop
    # arbiter_target: arbiter           # resolves disagreements + 补漏
```

- [ ] **Step 2: Write `docs/consensus-annotation.md`**

```markdown
# Consensus (N-way) Annotation

Set `stages.annotation.replicas: N` in `workflow.yaml` to run N independent
annotators per task. Spans agreed by `keep_threshold` of them are kept; the rest
go to `arbiter_target`, which resolves conflicts and fills clear gaps.

Empirically (v5, 24 tasks): N=2 with a reliable arbiter lands F1 ~0.98–0.99 and is
robust to annotator strength — two cheap models + arbiter ≈ two strong models.
N=1 is the legacy single-annotator flow. N>2 mostly adds cost.

Example:
```yaml
stages:
  annotation:
    replicas: 2
    targets: [qwen3.6-35b-a3b, MiniMax-M3]
    keep_threshold: 2
    arbiter_target: MiniMax-M3
```
```

- [ ] **Step 3: Commit**

```bash
git add annotation_pipeline_skill/interfaces/cli.py docs/consensus-annotation.md
git commit -m "docs: document N-way consensus annotation config"
```

### Task 12: Final integration sweep + review

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green except the documented known-7 pre-existing failures (no new failures, and the count of failures is exactly 7).

- [ ] **Step 2: Manual smoke (config parse on the real project)**

Run:
```bash
.venv/bin/python -c "
from annotation_pipeline_skill.config.loader import load_project_config
c = load_project_config('projects/v5_ner_phrase')
print('replicas=', c.annotation.replicas, 'targets=', c.annotation.targets)
c.annotation.validate(); print('valid')
"
```
Expected: prints `replicas= 1 targets= ['annotation']` then `valid` (unchanged default, since v5's workflow.yaml has no `replicas`).

- [ ] **Step 3: Dispatch a fresh review subagent** over the whole diff for correctness (consensus edge cases, backward-compat of the single path, artifact shape) using superpowers:requesting-code-review.

- [ ] **Step 4: Commit any review fixes, then finish the branch** via superpowers:finishing-a-development-branch.

---

## Phase 5 — Direct-accept (no-QC) dual pipeline + end-to-end run

Rationale: the experiments showed the **arbiter is the quality gate** — a dual-annotation + arbiter merge already lands F1 ~0.98. So we want a mode where, after consensus+arbiter, the task goes **straight to ACCEPTED with no separate QC stage**. This phase adds that toggle and then **actually runs a dual+arbiter (no-QC) pipeline end-to-end** on real tasks to prove it works (跑通).

### Task 13: `accept_directly` config (dual+arbiter replaces QC)

**Files:**
- Modify: `annotation_pipeline_skill/core/runtime.py:AnnotationConfig`
- Test: `tests/test_annotation_config.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_annotation_config.py
def test_accept_directly_defaults_false():
    assert AnnotationConfig.from_dict({}).accept_directly is False


def test_accept_directly_parsed():
    c = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "accept_directly": True})
    assert c.accept_directly is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_annotation_config.py::test_accept_directly_parsed -v`
Expected: FAIL — `AttributeError: 'AnnotationConfig' object has no attribute 'accept_directly'`

- [ ] **Step 3: Implement**

Add the field to `AnnotationConfig` (after `arbiter_target`):

```python
    accept_directly: bool = False   # True: after consensus+arbiter, ACCEPT without a QC stage
```

In `AnnotationConfig.from_dict`, add to the `cls(...)` call:

```python
            accept_directly=bool(data.get("accept_directly", False)),
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_annotation_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/core/runtime.py tests/test_annotation_config.py
git commit -m "feat(config): accept_directly toggle (dual+arbiter replaces QC)"
```

### Task 14: Wire `accept_directly` into the consensus dispatch

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py:_run_task` (the consensus branch from Task 9)
- Test: `tests/test_consensus_annotation_runtime.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consensus_annotation_runtime.py
def test_accept_directly_skips_qc_and_accepts(tmp_path, monkeypatch):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t3", pipeline_id="p",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    canned = {"a": _ann([["Alice", "Bob"]]), "b": _ann([["Alice"]]),
              "arbiter": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"person": ["Alice", "Bob"]}}}]})}
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2,
                                      "arbiter_target": "arbiter", "accept_directly": True})
    rt = SubagentRuntime(store, client_factory=lambda target: _StubClient(canned[target]), annotation_config=cfg)

    qc_called = {"n": 0}
    async def fake_qc(*a, **k): qc_called["n"] += 1
    monkeypatch.setattr(rt, "_run_validation_and_qc", fake_qc)

    asyncio.run(rt._run_task(store.load_task("t3"), "annotation"))
    assert qc_called["n"] == 0                              # QC was skipped
    assert store.load_task("t3").status is TaskStatus.ACCEPTED
    from annotation_pipeline_skill.services.entity_statistics_service import _load_latest_annotation
    assert set(_load_latest_annotation(store, "t3")["rows"][0]["output"]["entities"]["person"]) == {"Alice", "Bob"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py::test_accept_directly_skips_qc_and_accepts -v`
Expected: FAIL — task ends in QC/ANNOTATING, not ACCEPTED (Task 9's branch always calls `_run_validation_and_qc`).

- [ ] **Step 3: Implement**

Replace the consensus branch body added in Task 9 with the `accept_directly` fork:

```python
        if self.annotation_config.replicas > 1 and task.status in (TaskStatus.PENDING, TaskStatus.ANNOTATING):
            self._transition(task, TaskStatus.ANNOTATING,
                             reason=f"consensus annotation: {self.annotation_config.replicas} replicas",
                             stage="annotation", attempt_id=self._next_attempt_id(task))
            artifact, attempt_id, text = await self._produce_consensus_annotation(task)
            if self.annotation_config.accept_directly:
                self._transition(task, TaskStatus.ACCEPTED,
                                 reason="dual annotation + arbiter consensus (no QC stage)",
                                 stage="annotation", attempt_id=attempt_id)
            else:
                await self._run_validation_and_qc(task, artifact, attempt_id, text)
            return
```

- [ ] **Step 4: Run to verify pass + regression**

Run: `.venv/bin/python -m pytest tests/test_consensus_annotation_runtime.py -v && .venv/bin/python -m pytest tests/test_subagent_cycle.py -q`
Expected: new test PASS; single-path tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_consensus_annotation_runtime.py
git commit -m "feat(runtime): accept_directly path — dual+arbiter accepts without QC"
```

### Task 15: Configure a dual+arbiter (no-QC) pipeline and run it end-to-end (跑通)

**Goal:** prove the whole thing works on real tasks + real models, not just stubs. Uses a throwaway clone project so it never touches `v5_ner_phrase` production data.

**Files:**
- Create: `scratch/dualpipe_smoke.py` (runnable smoke; scratch is never committed)

- [ ] **Step 1: Stand up a throwaway project with a dual+arbiter, no-QC workflow**

Run:
```bash
cd /home/derek/Projects/annotation-pipeline-skill
.venv/bin/annotation-pipeline init --project-root projects/dualpipe_smoke --pipeline-id dualpipe_smoke 2>/dev/null || true
cat > projects/dualpipe_smoke/.annotation-pipeline/workflow.yaml <<'YAML'
stages:
  annotation:
    replicas: 2
    targets: [qwen3.6-35b-a3b, MiniMax-M3]
    keep_threshold: 2
    on_disagree: arbiter
    arbiter_target: MiniMax-M3
    accept_directly: true
runtime:
  max_concurrent_tasks: 2
  max_qc_rounds: 1
YAML
echo "configured dual+arbiter no-QC workflow"
```

- [ ] **Step 2: Seed 3 tiny tasks + run the consensus pipeline directly**

Create `scratch/dualpipe_smoke.py`:

```python
import asyncio, json
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.config.loader import load_project_config
from annotation_pipeline_skill.llm.profiles import load_llm_registry
from annotation_pipeline_skill.llm.local_cli import LocalCLIClient
from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime

ROOT = "projects/dualpipe_smoke"
store = SqliteStore.open(f"{ROOT}/.annotation-pipeline")
proj = load_project_config(ROOT); proj.annotation.validate()
reg = load_llm_registry("projects/llm_profiles.yaml")
rt = SubagentRuntime(store, client_factory=lambda t: LocalCLIClient(reg.resolve(t), store=store, project_id="dualpipe_smoke"),
                     annotation_config=proj.annotation)

INPUTS = ["Alice from ACME shipped the release in Q1 2026.",
          "The CVE-2021-44228 bug hit Apache HTTP Server hard.",
          "Bob said the latency dropped to 200ms after the fix."]
for i, txt in enumerate(INPUTS):
    t = Task.new(task_id=f"dp-{i}", pipeline_id="dualpipe_smoke",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": txt}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)

for i in range(len(INPUTS)):
    asyncio.run(rt._run_task(store.load_task(f"dp-{i}"), "annotation"))
    t = store.load_task(f"dp-{i}")
    from annotation_pipeline_skill.services.entity_statistics_service import _load_latest_annotation
    ann = _load_latest_annotation(store, f"dp-{i}")
    print(f"dp-{i}: status={t.status.value} ann={json.dumps(ann.get('rows',[{}])[0].get('output',{}), ensure_ascii=False)}")
```

Run: `.venv/bin/python scratch/dualpipe_smoke.py`

- [ ] **Step 3: Verify the run "跑通"**

Expected: each line prints `status=accepted` and a non-empty merged annotation, e.g.:
```
dp-0: status=accepted ann={"entities": {"person": ["Alice"], "organization": ["ACME"], "time": ["Q1 2026"]}}
dp-1: status=accepted ann={"entities": {"document": ["CVE-2021-44228"], ...}}
dp-2: status=accepted ann={"entities": {"person": ["Bob"], "number": ["200ms"]}}
```
Pass criteria: all 3 reach `accepted`, **no QC attempts** recorded (`store.list_attempts` has only annotation-stage attempts), and the annotations are non-empty + verbatim. If a target is down, swap `MiniMax-M3`/`qwen3.6-35b-a3b` in the workflow.yaml for any two reachable profiles (the pipeline is model-agnostic).

- [ ] **Step 4: Tear down the throwaway project**

Run: `rm -rf projects/dualpipe_smoke && echo "cleaned up"`

- [ ] **Step 5: Commit (no scratch/no project — just confirm nothing committed)**

```bash
git status --short   # expect: no tracked changes from this task (scratch/ + projects/ are gitignored)
```

---

## Self-Review

**Spec coverage:**
- `replicas: N` config → Tasks 1–3, 10, 11. ✓
- N independent annotators → Task 8 (`asyncio.gather` over `cfg.targets`). ✓
- keep ≥K agreement → Tasks 5 (`build_consensus`) + 8. ✓
- disagreements → arbiter (选择题 + 补漏) → Tasks 6, 8. ✓
- agreement auto-keep / replicas=1 unchanged → Task 9 guard + Task 12 smoke. ✓
- config-switchable (not hardcoded) → Tasks 3, 10, 11. ✓
- dual+arbiter **without QC** (arbiter is the gate) → Tasks 13, 14 (`accept_directly`). ✓
- **end-to-end run (跑通)** of a dual+arbiter no-QC pipeline on real tasks → Task 15. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — every code step shows full code. Two explicit "confirm the real signature at line X" notes (`_write_stage_artifact`, `ProjectConfig` field) are integration verifications against existing code, not placeholders; the call shape is pinned to the existing call site at `subagent_cycle.py:459`.

**Type consistency:** `AnnotationConfig` fields (`replicas/targets/keep_threshold/on_disagree/arbiter_target`) identical across Tasks 1, 2, 3, 7, 8, 10. `build_consensus(drafts, keep_threshold) -> (payload, disagreements)` and disagreement dict keys (`row_index/field/type/span/support`) consistent across Tasks 5, 6, 8. `_produce_consensus_annotation(task) -> (artifact, attempt_id, text)` matches its use in Task 9.

**Known risk to verify during execution:** `LLMGenerateResult` construction in Task 8 — confirm its real field names at `annotation_pipeline_skill/llm/client.py` (the test stub uses the same constructor, so a mismatch fails fast in Task 8 Step 4, not silently).
</content>

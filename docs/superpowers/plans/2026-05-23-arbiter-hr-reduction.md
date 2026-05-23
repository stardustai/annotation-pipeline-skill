# Arbiter HR-Rate Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce HR escalation rate from ~19% toward Phase 1 target (<10%) via three targeted arbiter improvements.

**Architecture:** Three independent changes to `SubagentRuntime` in `subagent_cycle.py` and `LocalScheduler` in `local_scheduler.py`. P3 removes the row-index filter so arbiter always sees full input text. P2 filters disputed items to the latest QC round only, eliminating stale feedback. The second-arbiter-for-uncertain feature mirrors the existing `prior_verifier_first_arbiter_divergent` flag pattern: set a flag when the first arbiter is uncertain, leave the task in ARBITRATING, and let the scheduler route it to a `_resolve_uncertain_arbiter_async` method that invokes `arbiter_secondary`; only escalate to HR if the second arbiter is also uncertain.

**Tech Stack:** Python, SQLite (`SqliteStore`), pytest, existing `SubagentRuntime` / `LocalScheduler` / `FeedbackRecord` / `Attempt` models.

---

## File map

| File | Change |
|------|--------|
| `annotation_pipeline_skill/runtime/subagent_cycle.py` | P3: 2-line slim-input fix; P2: latest-round filter; second-arbiter: `target_name` param on `_arbitrate_and_apply`, flag-setting at 2 HR sites, new `_resolve_uncertain_arbiter` + `_resolve_uncertain_arbiter_async` |
| `annotation_pipeline_skill/runtime/local_scheduler.py` | Add routing branch for `arbiter_uncertain_needs_second` |
| `tests/test_subagent_cycle.py` | Tests for P3 slim-input, P2 stale-feedback filter, flag-setting, second-arbiter resolves |
| `tests/test_local_runtime_scheduler.py` | Test scheduler routes uncertain task to second-arbiter path |

---

## Task 1: P3 — Include full row text in arbiter slim prompt

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py:2763-2775`
- Test: `tests/test_subagent_cycle.py`

### Background

`_run_arbiter_llm` builds a "slim prompt" that only includes rows whose `row_index` appears in `qc.target.row_index` of any feedback item. When a feedback item has no `target.row_index` (e.g. schema-validation complaints, empty-annotation complaints), the row containing the disputed span is silently omitted. The arbiter then says "I cannot safely apply a correction without the row text" and marks its verdict `tentative`. Fix: always send all source rows in `input`; keep `current_annotation` filtered to disputed rows only (the arbiter only needs to produce corrections for those).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_subagent_cycle.py`:

```python
def test_arbiter_slim_prompt_includes_all_input_rows_regardless_of_target(tmp_path):
    """Rows whose row_index is NOT referenced in any qc.target must still
    appear in the arbiter prompt's input.rows so the arbiter has full context.
    Previously, a feedback item with no target.row_index caused its row to be
    silently omitted, making the arbiter mark verdicts tentative."""
    import json
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task, FeedbackRecord, ArtifactRef
    from annotation_pipeline_skill.core.states import (
        FeedbackSeverity, FeedbackSource, TaskStatus,
    )
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from tests.test_subagent_cycle import StubLLMClient  # reuse existing stub

    store = SqliteStore.open(tmp_path)
    # Source has 3 rows; feedback only references row_index=1 in its target.
    source_payload = {
        "rows": [
            {"row_id": "r0", "row_index": 0, "input": {"text": "row zero text"}},
            {"row_id": "r1", "row_index": 1, "input": {"text": "row one text"}},
            {"row_id": "r2", "row_index": 2, "input": {"text": "row two text"}},
        ]
    }
    task = Task.new(
        task_id="t-slim",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": source_payload},
    )
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)

    # Write an annotation_result artifact so _latest_annotation_artifact returns something.
    ann_payload = {"rows": [
        {"row_id": "r0", "row_index": 0, "output": {"entities": {}}},
        {"row_id": "r1", "row_index": 1, "output": {"entities": {"org": ["Acme"]}}},
        {"row_id": "r2", "row_index": 2, "output": {"entities": {}}},
    ]}
    rel = f"artifact_payloads/t-slim/ann.json"
    (store.root / "artifact_payloads" / "t-slim").mkdir(parents=True, exist_ok=True)
    (store.root / rel).write_text(json.dumps({"text": json.dumps(ann_payload)}), encoding="utf-8")
    store.append_artifact(ArtifactRef.new(
        task_id="t-slim", kind="annotation_result", path=rel, content_type="application/json",
    ))

    # Feedback with NO target.row_index → this row was previously omitted.
    fb_no_row = FeedbackRecord.new(
        task_id="t-slim", attempt_id="t-slim-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="schema_invalid", message="annotation is empty",
        target={},  # no row_index key
        suggested_action="annotator_rerun", created_by="qc",
    )
    # Feedback WITH target.row_index=1.
    fb_with_row = FeedbackRecord.new(
        task_id="t-slim", attempt_id="t-slim-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="Acme not labelled",
        target={"row_index": 1},
        suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb_no_row)
    store.append_feedback(fb_with_row)

    # Add annotator rebuttal for both so require_rebuttal passes.
    from annotation_pipeline_skill.core.models import FeedbackDiscussionEntry
    for fid in [fb_no_row.feedback_id, fb_with_row.feedback_id]:
        store.append_feedback_discussion(FeedbackDiscussionEntry.new(
            task_id="t-slim", feedback_id=fid, role="annotator",
            message="I disagree.", consensus=False,
        ))

    captured_prompts: list[str] = []

    class _CapturingClient:
        async def generate(self, request):
            from annotation_pipeline_skill.core.runtime import LLMGenerateResult
            captured_prompts.append(request.prompt)
            return LLMGenerateResult(
                final_text=json.dumps({
                    "verdicts": [
                        {"feedback_id": fb_no_row.feedback_id, "verdict": "annotator",
                         "confidence": "confident", "reasoning": "ok"},
                        {"feedback_id": fb_with_row.feedback_id, "verdict": "annotator",
                         "confidence": "confident", "reasoning": "ok"},
                    ]
                }),
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="arbiter", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _CapturingClient())
    import asyncio
    asyncio.run(runtime._arbitrate_and_apply(task, "t-slim-attempt-0", stage="qc"))

    assert captured_prompts, "arbiter must have been called"
    prompt_data = json.loads(captured_prompts[0])
    row_indices_in_prompt = {r["row_index"] for r in prompt_data["input"]["rows"]}
    assert row_indices_in_prompt == {0, 1, 2}, (
        f"all 3 rows must appear in input.rows; got {row_indices_in_prompt}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_subagent_cycle.py::test_arbiter_slim_prompt_includes_all_input_rows_regardless_of_target -xvs 2>&1 | tail -20
```

Expected: FAIL — `row_indices_in_prompt == {1}` (only the row with `target.row_index=1` is present).

- [ ] **Step 3: Change `slim_input_rows` to use all rows**

In `annotation_pipeline_skill/runtime/subagent_cycle.py`, replace lines 2771–2775:

```python
        slim_input_rows = [r for r in full_rows
                           if isinstance(r, dict) and r.get("row_index") in ref_rows]
        slim_input = {**{k: v for k, v in full_payload.items() if k != "rows"},
                      "rows": slim_input_rows,
                      "_omitted_unchanged_rows": max(0, len(full_rows) - len(slim_input_rows))}
```

with:

```python
        # Include ALL source rows in input so the arbiter has full text context
        # even for feedback items that lack a target.row_index (schema errors,
        # empty-annotation complaints, etc.). Previously omitting those rows
        # caused the arbiter to mark verdicts tentative due to missing context.
        # current_annotation is still filtered to disputed rows only.
        slim_input = {**{k: v for k, v in full_payload.items() if k != "rows"},
                      "rows": full_rows}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_subagent_cycle.py::test_arbiter_slim_prompt_includes_all_input_rows_regardless_of_target -xvs 2>&1 | tail -5
```

Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
python -m pytest tests/test_subagent_cycle.py tests/test_local_runtime_scheduler.py tests/test_prior_verifier_integration.py -x --no-header -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_subagent_cycle.py
git commit -m "fix(arbiter): include all source rows in slim prompt input

Arbiter marked verdicts tentative when feedback items lacked
target.row_index, because the row containing the disputed span
was silently omitted from slim_input. Sending all rows removes
that context gap. current_annotation filtering is unchanged."
```

---

## Task 2: P2 — Filter disputed items to latest QC round only

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py:2444-2449`
- Test: `tests/test_subagent_cycle.py`

### Background

`_arbitrate_and_apply` builds `open_feedbacks` from ALL unclosed feedback records for the task, across every QC round. If QC round 1 filed feedback about an empty annotation, and round 2's annotation fixed it but didn't close the round-1 feedback via consensus, the arbiter sees the stale complaint alongside a `current_annotation` that contradicts it. The arbiter correctly notices the mismatch but marks its verdict `tentative` instead of dismissing confidently.

Fix: after filtering for unclosed feedbacks, keep only those whose `attempt_id` matches the latest QC round's `attempt_id`. `list_feedback` returns records ordered by `seq` (insertion order); feedbacks from the same QC run are inserted together, so `open_feedbacks[-1].attempt_id` is the latest round's ID.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_subagent_cycle.py`:

```python
def test_arbiter_receives_only_latest_qc_round_feedback(tmp_path):
    """Stale feedback from an earlier QC round must NOT be sent to the arbiter.
    Only the most recent round's feedback (by attempt_id) should appear in the
    disputed_items the arbiter sees."""
    import json
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import (
        Task, FeedbackRecord, ArtifactRef, FeedbackDiscussionEntry,
    )
    from annotation_pipeline_skill.core.states import (
        FeedbackSeverity, FeedbackSource, TaskStatus,
    )
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.core.runtime import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    source_payload = {"rows": [{"row_id": "r0", "row_index": 0, "input": {"text": "hello"}}]}
    task = Task.new(
        task_id="t-stale", pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": source_payload},
    )
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)

    rel = "artifact_payloads/t-stale/ann.json"
    (store.root / "artifact_payloads" / "t-stale").mkdir(parents=True, exist_ok=True)
    ann = {"rows": [{"row_id": "r0", "row_index": 0, "output": {"entities": {"org": ["Acme"]}}}]}
    (store.root / rel).write_text(json.dumps({"text": json.dumps(ann)}), encoding="utf-8")
    store.append_artifact(ArtifactRef.new(
        task_id="t-stale", kind="annotation_result", path=rel, content_type="application/json",
    ))

    # Round 1: stale feedback (old attempt_id).
    fb_stale = FeedbackRecord.new(
        task_id="t-stale", attempt_id="t-stale-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="schema_invalid", message="annotation was empty (stale)",
        target={}, suggested_action="annotator_rerun", created_by="qc",
    )
    # Round 2: current feedback (newer attempt_id).
    fb_current = FeedbackRecord.new(
        task_id="t-stale", attempt_id="t-stale-attempt-1",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="Acme should be labelled org",
        target={"row_index": 0}, suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb_stale)
    store.append_feedback(fb_current)

    # Annotator rebuttals for both.
    for fid in [fb_stale.feedback_id, fb_current.feedback_id]:
        store.append_feedback_discussion(FeedbackDiscussionEntry.new(
            task_id="t-stale", feedback_id=fid, role="annotator",
            message="addressed", consensus=False,
        ))

    seen_feedback_ids: list[list[str]] = []

    class _CapturingClient:
        async def generate(self, request):
            data = json.loads(request.prompt)
            seen_feedback_ids.append([it["feedback_id"] for it in data["disputed_items"]])
            return LLMGenerateResult(
                final_text=json.dumps({
                    "verdicts": [{"feedback_id": fb_current.feedback_id,
                                  "verdict": "annotator", "confidence": "confident",
                                  "reasoning": "correct"}]
                }),
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="arbiter", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _CapturingClient())
    import asyncio
    asyncio.run(runtime._arbitrate_and_apply(task, "t-stale-attempt-1", stage="qc"))

    assert seen_feedback_ids, "arbiter must have been called"
    ids_sent = seen_feedback_ids[0]
    assert fb_stale.feedback_id not in ids_sent, (
        "stale round-1 feedback must NOT be sent to arbiter"
    )
    assert fb_current.feedback_id in ids_sent, (
        "current round-2 feedback must be sent to arbiter"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_subagent_cycle.py::test_arbiter_receives_only_latest_qc_round_feedback -xvs 2>&1 | tail -15
```

Expected: FAIL — both feedback IDs appear in `ids_sent`.

- [ ] **Step 3: Add latest-round filter to `_arbitrate_and_apply`**

In `annotation_pipeline_skill/runtime/subagent_cycle.py`, replace the `open_feedbacks` block (lines 2444–2449):

```python
        open_feedbacks = [
            f for f in self.store.list_feedback(task.task_id)
            if (include_closed_feedbacks or f.feedback_id not in consensus_ids)
            and (not require_rebuttal or f.feedback_id in replies_by_feedback)
            and (f.source_stage is FeedbackSource.QC or f.source_stage is FeedbackSource.VALIDATION)
        ]
```

with:

```python
        open_feedbacks = [
            f for f in self.store.list_feedback(task.task_id)
            if (include_closed_feedbacks or f.feedback_id not in consensus_ids)
            and (not require_rebuttal or f.feedback_id in replies_by_feedback)
            and (f.source_stage is FeedbackSource.QC or f.source_stage is FeedbackSource.VALIDATION)
        ]
        # Send only the latest QC round's feedback. list_feedback returns records
        # ordered by seq (insertion order); feedbacks from the same QC run share an
        # attempt_id and are inserted together, so the last record's attempt_id is
        # the most recent round. Stale feedback from prior rounds references
        # annotation states that no longer exist, causing the arbiter to mark
        # verdicts tentative when it sees the mismatch with current_annotation.
        if open_feedbacks:
            latest_attempt_id = open_feedbacks[-1].attempt_id
            open_feedbacks = [f for f in open_feedbacks if f.attempt_id == latest_attempt_id]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_subagent_cycle.py::test_arbiter_receives_only_latest_qc_round_feedback -xvs 2>&1 | tail -5
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests/test_subagent_cycle.py tests/test_local_runtime_scheduler.py tests/test_prior_verifier_integration.py -x --no-header -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_subagent_cycle.py
git commit -m "fix(arbiter): send only latest QC round feedback to arbiter

Stale feedback from earlier rounds referenced annotation states
that no longer existed, causing the arbiter to flag its own
verdicts as tentative. Filter open_feedbacks to the latest
attempt_id before building disputed_items."
```

---

## Task 3: Second arbiter for uncertain — `_arbitrate_and_apply` target parameter

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py:2404-2498`

This is a prerequisite for Task 4. `_arbitrate_and_apply` currently hardcodes `target_name="arbiter"`. The second-arbiter resolver needs to call the same method with `target_name="arbiter_secondary"`.

- [ ] **Step 1: Add `target_name` parameter**

In `annotation_pipeline_skill/runtime/subagent_cycle.py`, replace the `_arbitrate_and_apply` signature (line 2404):

```python
    async def _arbitrate_and_apply(
        self,
        task: Task,
        attempt_id: str,
        stage: str,
        *,
        include_closed_feedbacks: bool = False,
        require_rebuttal: bool = True,
    ) -> dict[str, Any]:
```

with:

```python
    async def _arbitrate_and_apply(
        self,
        task: Task,
        attempt_id: str,
        stage: str,
        *,
        include_closed_feedbacks: bool = False,
        require_rebuttal: bool = True,
        target_name: str = "arbiter",
    ) -> dict[str, Any]:
```

And replace line 2494–2498 (the `_run_arbiter_llm` call inside `_arbitrate_and_apply`):

```python
        try:
            payload = await self._run_arbiter_llm(
                task=task,
                items=items,
                target_name="arbiter",
            )
```

with:

```python
        try:
            payload = await self._run_arbiter_llm(
                task=task,
                items=items,
                target_name=target_name,
            )
```

- [ ] **Step 2: Run the full suite to verify no regressions**

```bash
python -m pytest tests/test_subagent_cycle.py tests/test_local_runtime_scheduler.py tests/test_prior_verifier_integration.py -x --no-header -q 2>&1 | tail -5
```

Expected: all pass (pure signature addition, no behaviour change).

- [ ] **Step 3: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py
git commit -m "refactor(arbiter): add target_name param to _arbitrate_and_apply

Allows the second-arbiter resolver to invoke the same method
with arbiter_secondary without duplicating the call logic."
```

---

## Task 4: Second arbiter for uncertain — flag setting and resolver method

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py` (two HR-transition sites + new methods)
- Test: `tests/test_subagent_cycle.py`

### Background

There are two sites where `arb["unresolved"] > 0` causes an immediate HUMAN_REVIEW transition:
- **Validation path** (`_run_validation_and_qc`, line ~617)
- **QC path** (`_run_validation_and_qc`, line ~821)

Both must be changed to: set `task.metadata["arbiter_uncertain_needs_second"] = True`, leave the task in ARBITRATING, and save. The scheduler (Task 5) will detect the flag and call `_resolve_uncertain_arbiter_async`.

The resolver method mirrors `_resolve_first_arbiter_divergence_async`:
- Calls `_arbitrate_and_apply` with `target_name="arbiter_secondary"`
- If second arbiter has no unresolved → apply via `_terminal_from_arbiter` → ACCEPTED
- If second arbiter also has `unresolved > 0` → HUMAN_REVIEW with reason "both arbiters uncertain"
- If second arbiter fails (exception) → HUMAN_REVIEW with reason "second arbiter unavailable"

- [ ] **Step 1: Write the failing test for flag-setting**

Add to `tests/test_subagent_cycle.py`:

```python
def test_arbiter_uncertain_sets_flag_instead_of_immediate_hr(tmp_path):
    """When the first arbiter is uncertain (tentative/unsure verdict), the task
    must stay in ARBITRATING with arbiter_uncertain_needs_second=True, NOT go
    straight to HUMAN_REVIEW. The second-arbiter resolver handles the HR decision."""
    import json
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task, FeedbackRecord
    from annotation_pipeline_skill.core.states import (
        FeedbackSeverity, FeedbackSource, TaskStatus,
    )
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.core.runtime import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-unc",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "alpha beta"}},
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    for i in range(3):
        fb = FeedbackRecord.new(
            task_id="t-unc", attempt_id=f"t-unc-attempt-{i}",
            source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
            category="missing_phrase", message=f"Complaint {i}",
            target={}, suggested_action="annotator_rerun", created_by="qc",
        )
        store.append_feedback(fb)

    feedback_ids = [f.feedback_id for f in store.list_feedback("t-unc")]
    # Arbiter responds with tentative confidence → unresolved > 0.
    arbiter_resp = json.dumps({
        "verdicts": [
            {"feedback_id": fid, "verdict": "annotator",
             "confidence": "tentative", "reasoning": "not sure"}
            for fid in feedback_ids
        ]
    })
    annotation_payload = {
        "entities": [],
        "discussion_replies": [
            {"feedback_id": fid, "confidence": 0.7, "message": "I disagree."}
            for fid in feedback_ids
        ],
    }
    qc_resp = '{"passed": false, "failures": [{"category": "missing_phrase", "confidence": 0.92, "message": "still missing"}]}'

    def factory(target: str):
        if target == "arbiter":
            return StubLLMClient(final_text=arbiter_resp, provider="arbiter")
        if target == "qc":
            return StubLLMClient(final_text=qc_resp, provider="qc")
        return StubLLMClient(final_text=json.dumps({"entities": []}), provider=target)

    runtime = SubagentRuntime(store=store, client_factory=factory, max_qc_rounds=3)
    runtime.run_once(stage_target="annotation")

    after = store.load_task("t-unc")
    assert after.status is TaskStatus.ARBITRATING, (
        f"expected ARBITRATING (waiting for second arbiter), got {after.status}"
    )
    assert after.metadata.get("arbiter_uncertain_needs_second") is True, (
        "arbiter_uncertain_needs_second flag must be set"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_subagent_cycle.py::test_arbiter_uncertain_sets_flag_instead_of_immediate_hr -xvs 2>&1 | tail -10
```

Expected: FAIL — task status is `HUMAN_REVIEW`, not `ARBITRATING`.

- [ ] **Step 3: Change the validation-path HR site (line ~617)**

In `annotation_pipeline_skill/runtime/subagent_cycle.py`, replace lines ~617–631:

```python
                if arb["unresolved"] > 0:
                    self._transition(
                        task,
                        TaskStatus.HUMAN_REVIEW,
                        reason="Arbiter flagged its own answer as uncertain (tentative/unsure verdict); needs human review",
                        stage="validation",
                        attempt_id=annotation_attempt_id,
                        metadata={
                            "auto_escalated": True,
                            "round_count": round_count,
                            "max_qc_rounds": self.max_qc_rounds,
                            "arbiter_ran": arb["ran"],
                            "arbiter_unresolved": arb["unresolved"],
                            "arbiter_mechanical_fail": arb["mechanical_fail"],
                        },
                    )
```

with:

```python
                if arb["unresolved"] > 0:
                    # First arbiter uncertain — defer to a second arbiter rather
                    # than escalating immediately. Scheduler detects the flag and
                    # calls _resolve_uncertain_arbiter_async.
                    task.metadata["arbiter_uncertain_needs_second"] = True
                    self.store.save_task(task)
```

- [ ] **Step 4: Change the QC-path HR site (line ~821)**

In `annotation_pipeline_skill/runtime/subagent_cycle.py`, replace lines ~821–838:

```python
                if arb["unresolved"] > 0:
                    self._transition(
                        task,
                        TaskStatus.HUMAN_REVIEW,
                        reason="Arbiter flagged its own answer as uncertain (tentative/unsure verdict); needs human review",
                        stage="qc",
                        attempt_id=qc_attempt_id,
                        metadata={
                            "auto_escalated": True,
                            "round_count": round_count,
                            "max_qc_rounds": self.max_qc_rounds,
                            "feedback_id": feedbacks[0].feedback_id,
                            "qc_artifact_id": qc_artifact.artifact_id,
                            "arbiter_ran": arb["ran"],
                            "arbiter_unresolved": arb["unresolved"],
                            "arbiter_mechanical_fail": arb["mechanical_fail"],
                        },
                    )
```

with:

```python
                if arb["unresolved"] > 0:
                    # First arbiter uncertain — defer to second arbiter.
                    task.metadata["arbiter_uncertain_needs_second"] = True
                    self.store.save_task(task)
```

- [ ] **Step 5: Run flag test to verify it passes**

```bash
python -m pytest tests/test_subagent_cycle.py::test_arbiter_uncertain_sets_flag_instead_of_immediate_hr -xvs 2>&1 | tail -5
```

Expected: PASS

- [ ] **Step 6: Write the resolver tests**

Add to `tests/test_subagent_cycle.py`:

```python
def _make_uncertain_task(store, task_id: str, feedback_ids: list[str]):
    """Helper: task in ARBITRATING with arbiter_uncertain_needs_second flag and
    an annotation_result artifact."""
    import json
    from annotation_pipeline_skill.core.models import ArtifactRef
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.core.models import Task

    task = Task.new(
        task_id=task_id, pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_id": "r0", "row_index": 0, "input": {"text": "hello"}}]
        }},
    )
    task.status = TaskStatus.ARBITRATING
    task.metadata["arbiter_uncertain_needs_second"] = True
    store.save_task(task)

    rel = f"artifact_payloads/{task_id}/ann.json"
    (store.root / "artifact_payloads" / task_id).mkdir(parents=True, exist_ok=True)
    ann = {"rows": [{"row_id": "r0", "row_index": 0, "output": {"entities": {}}}]}
    (store.root / rel).write_text(json.dumps({"text": json.dumps(ann)}), encoding="utf-8")
    store.append_artifact(ArtifactRef.new(
        task_id=task_id, kind="annotation_result", path=rel, content_type="application/json",
    ))
    return store.load_task(task_id)


def test_resolve_uncertain_arbiter_second_confident_accepts(tmp_path):
    """When the second arbiter responds with confident verdicts (annotator wins),
    the task is ACCEPTED and the flag is cleared."""
    import json
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import FeedbackRecord, FeedbackDiscussionEntry
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource, TaskStatus
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.core.runtime import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    fb = FeedbackRecord.new(
        task_id="t-unc2", attempt_id="t-unc2-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="missing span",
        target={}, suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb)
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id="t-unc2", feedback_id=fb.feedback_id,
        role="annotator", message="ok", consensus=False,
    ))
    task = _make_uncertain_task(store, "t-unc2", [fb.feedback_id])

    second_resp = json.dumps({
        "verdicts": [{"feedback_id": fb.feedback_id, "verdict": "annotator",
                      "confidence": "confident", "reasoning": "clear"}]
    })

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: StubLLMClient(final_text=second_resp, provider="arbiter_secondary"),
    )
    runtime._resolve_uncertain_arbiter(task)

    after = store.load_task("t-unc2")
    assert after.status is TaskStatus.ACCEPTED, f"expected ACCEPTED, got {after.status}"
    assert not after.metadata.get("arbiter_uncertain_needs_second"), "flag must be cleared"


def test_resolve_uncertain_arbiter_second_also_uncertain_goes_to_hr(tmp_path):
    """When the second arbiter is ALSO uncertain (tentative/unsure), the task
    escalates to HUMAN_REVIEW with a clear reason."""
    import json
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import FeedbackRecord, FeedbackDiscussionEntry
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource, TaskStatus
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.core.runtime import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    fb = FeedbackRecord.new(
        task_id="t-unc3", attempt_id="t-unc3-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="missing span",
        target={}, suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb)
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id="t-unc3", feedback_id=fb.feedback_id,
        role="annotator", message="ok", consensus=False,
    ))
    task = _make_uncertain_task(store, "t-unc3", [fb.feedback_id])

    second_resp = json.dumps({
        "verdicts": [{"feedback_id": fb.feedback_id, "verdict": "annotator",
                      "confidence": "tentative", "reasoning": "still unsure"}]
    })

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: StubLLMClient(final_text=second_resp, provider="arbiter_secondary"),
    )
    runtime._resolve_uncertain_arbiter(task)

    after = store.load_task("t-unc3")
    assert after.status is TaskStatus.HUMAN_REVIEW, f"expected HUMAN_REVIEW, got {after.status}"
    assert not after.metadata.get("arbiter_uncertain_needs_second"), "flag must be cleared"
    events = store.list_audit_events("t-unc3")
    hr_event = next((e for e in events if e.next_status.value == "human_review"), None)
    assert hr_event is not None
    assert "both arbiters" in hr_event.reason.lower() or "second arbiter" in hr_event.reason.lower()


def test_resolve_uncertain_arbiter_unavailable_goes_to_hr(tmp_path):
    """When the second arbiter client raises (unavailable), the task goes to
    HUMAN_REVIEW rather than staying stuck in ARBITRATING."""
    import json
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import FeedbackRecord, FeedbackDiscussionEntry
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource, TaskStatus
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime

    store = SqliteStore.open(tmp_path)
    fb = FeedbackRecord.new(
        task_id="t-unc4", attempt_id="t-unc4-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="missing",
        target={}, suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb)
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id="t-unc4", feedback_id=fb.feedback_id,
        role="annotator", message="ok", consensus=False,
    ))
    task = _make_uncertain_task(store, "t-unc4", [fb.feedback_id])

    def bad_factory(target):
        raise RuntimeError("provider not configured")

    runtime = SubagentRuntime(store=store, client_factory=bad_factory)
    runtime._resolve_uncertain_arbiter(task)

    after = store.load_task("t-unc4")
    assert after.status is TaskStatus.HUMAN_REVIEW, f"expected HUMAN_REVIEW, got {after.status}"
    assert not after.metadata.get("arbiter_uncertain_needs_second")
```

- [ ] **Step 7: Run the new tests to verify they fail**

```bash
python -m pytest tests/test_subagent_cycle.py::test_resolve_uncertain_arbiter_second_confident_accepts tests/test_subagent_cycle.py::test_resolve_uncertain_arbiter_second_also_uncertain_goes_to_hr tests/test_subagent_cycle.py::test_resolve_uncertain_arbiter_unavailable_goes_to_hr -xvs 2>&1 | tail -10
```

Expected: FAIL with `AttributeError: 'SubagentRuntime' object has no attribute '_resolve_uncertain_arbiter'`

- [ ] **Step 8: Add `_resolve_uncertain_arbiter` and `_resolve_uncertain_arbiter_async`**

Add the following two methods to `SubagentRuntime` in `annotation_pipeline_skill/runtime/subagent_cycle.py`, immediately after the `_clear_divergence_flag` method (after line 2335):

```python
    def _resolve_uncertain_arbiter(self, task: Task) -> None:
        """Sync entry called by the scheduler when it sees a task with the
        ``arbiter_uncertain_needs_second`` flag set. Runs the second arbiter
        via arbiter_secondary and applies the resolution:

        - second arbiter resolves (unresolved == 0) → ACCEPTED or arbiter fix
        - second arbiter also uncertain (unresolved > 0) → HUMAN_REVIEW
        - second arbiter unavailable (exception) → HUMAN_REVIEW
        """
        import asyncio
        asyncio.run(self._resolve_uncertain_arbiter_async(task))

    async def _resolve_uncertain_arbiter_async(self, task: Task) -> None:
        """Async implementation of the second-arbiter-for-uncertain path."""
        attempt_id = self._next_attempt_id(task)
        task.metadata.pop("arbiter_uncertain_needs_second", None)

        try:
            arb = await self._arbitrate_and_apply(
                task,
                attempt_id,
                stage="arbitration",
                require_rebuttal=False,
                target_name="arbiter_secondary",
            )
        except Exception:  # noqa: BLE001
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason="Arbiter flagged its own answer as uncertain; second arbiter unavailable — needs human review",
                stage="arbitration",
                attempt_id=attempt_id,
            )
            self.store.save_task(task)
            return

        terminal = self._terminal_from_arbiter(task, attempt_id, "arbitration", arb)
        if terminal is not None:
            self.store.save_task(task)
            return

        if arb["unresolved"] > 0:
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason=(
                    "Both arbiters flagged their answers as uncertain "
                    "(tentative/unsure verdict); needs human review"
                ),
                stage="arbitration",
                attempt_id=attempt_id,
                metadata={
                    "auto_escalated": True,
                    "arbiter_ran": arb["ran"],
                    "arbiter_unresolved": arb["unresolved"],
                },
            )
        else:
            self._handle_arbiter_mechanical_fail(
                task, attempt_id, arb, stage="arbitration",
            )
        self.store.save_task(task)
```

- [ ] **Step 9: Run resolver tests**

```bash
python -m pytest tests/test_subagent_cycle.py::test_resolve_uncertain_arbiter_second_confident_accepts tests/test_subagent_cycle.py::test_resolve_uncertain_arbiter_second_also_uncertain_goes_to_hr tests/test_subagent_cycle.py::test_resolve_uncertain_arbiter_unavailable_goes_to_hr -xvs 2>&1 | tail -10
```

Expected: all PASS

- [ ] **Step 10: Run the full suite**

```bash
python -m pytest tests/test_subagent_cycle.py tests/test_local_runtime_scheduler.py tests/test_prior_verifier_integration.py -x --no-header -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_subagent_cycle.py
git commit -m "feat(arbiter): add second-arbiter path for uncertain first-arbiter verdicts

Instead of immediately escalating to HR when the first arbiter is
uncertain (tentative/unsure), set arbiter_uncertain_needs_second=True
and leave the task in ARBITRATING. The scheduler routes it to
_resolve_uncertain_arbiter_async which invokes arbiter_secondary.
Only goes to HR if the second arbiter is also uncertain or unavailable."
```

---

## Task 5: Second arbiter for uncertain — scheduler routing

**Files:**
- Modify: `annotation_pipeline_skill/runtime/local_scheduler.py:458-469`
- Test: `tests/test_local_runtime_scheduler.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_local_runtime_scheduler.py`:

```python
def test_scheduler_routes_uncertain_flag_to_second_arbiter(tmp_path):
    """An ARBITRATING task with arbiter_uncertain_needs_second=True must be
    routed to _resolve_uncertain_arbiter, not the normal run_task_async path."""
    import asyncio
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.runtime.local_scheduler import LocalScheduler
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-sched-unc",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "hello"}},
    )
    task.status = TaskStatus.ARBITRATING
    task.metadata["arbiter_uncertain_needs_second"] = True
    store.save_task(task)

    resolver_called: list[str] = []

    class _PatchedRuntime(SubagentRuntime):
        def _resolve_uncertain_arbiter(self, t):
            resolver_called.append(t.task_id)
            # Move task out of ARBITRATING so scheduler doesn't loop.
            t.metadata.pop("arbiter_uncertain_needs_second", None)
            from annotation_pipeline_skill.core.states import TaskStatus
            t.status = TaskStatus.ACCEPTED
            self.store.save_task(t)

        def _resolve_first_arbiter_divergence(self, t):
            raise AssertionError("wrong resolver called")

    runtime = _PatchedRuntime(store=store, client_factory=lambda _t: None)
    scheduler = LocalScheduler(store=store, runtime=runtime, worker_count=1)

    async def _run_one():
        import asyncio
        stop = asyncio.Event()
        stop.set()
        await scheduler._worker(worker_id=0, stop=stop)

    asyncio.run(_run_one())

    assert resolver_called == ["t-sched-unc"], (
        f"_resolve_uncertain_arbiter must be called for the flagged task; got {resolver_called}"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_local_runtime_scheduler.py::test_scheduler_routes_uncertain_flag_to_second_arbiter -xvs 2>&1 | tail -10
```

Expected: FAIL — `AssertionError: _resolve_first_arbiter_divergence must not be called` or resolver_called is empty (wrong path taken).

- [ ] **Step 3: Add routing branch in the scheduler worker**

In `annotation_pipeline_skill/runtime/local_scheduler.py`, replace lines 458–469:

```python
                    if (
                        task.status is TaskStatus.ARBITRATING
                        and task.metadata.get("prior_verifier_first_arbiter_divergent")
                    ):
                        # Divergent-flag path: the first arbiter accepted an
                        # annotation that still diverges from project prior.
                        # Route to the dedicated resolver (which invokes a
                        # second arbiter) instead of the manual re-arbitrate
                        # flow that run_task_async would dispatch to.
                        work_coro = runtime._resolve_first_arbiter_divergence_async(task)
                    else:
                        work_coro = runtime.run_task_async(task, stage_target=stage_target)
```

with:

```python
                    if (
                        task.status is TaskStatus.ARBITRATING
                        and task.metadata.get("prior_verifier_first_arbiter_divergent")
                    ):
                        # Divergent-flag path: first arbiter accepted an annotation
                        # that diverges from project prior; second arbiter adjudicates.
                        work_coro = runtime._resolve_first_arbiter_divergence_async(task)
                    elif (
                        task.status is TaskStatus.ARBITRATING
                        and task.metadata.get("arbiter_uncertain_needs_second")
                    ):
                        # Uncertain-flag path: first arbiter was tentative/unsure;
                        # second arbiter gets a fresh attempt before escalating to HR.
                        work_coro = runtime._resolve_uncertain_arbiter_async(task)
                    else:
                        work_coro = runtime.run_task_async(task, stage_target=stage_target)
```

- [ ] **Step 4: Run the scheduler test**

```bash
python -m pytest tests/test_local_runtime_scheduler.py::test_scheduler_routes_uncertain_flag_to_second_arbiter -xvs 2>&1 | tail -5
```

Expected: PASS

- [ ] **Step 5: Run the full suite**

```bash
python -m pytest tests/test_subagent_cycle.py tests/test_local_runtime_scheduler.py tests/test_prior_verifier_integration.py -x --no-header -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add annotation_pipeline_skill/runtime/local_scheduler.py tests/test_local_runtime_scheduler.py
git commit -m "feat(scheduler): route arbiter_uncertain_needs_second tasks to second arbiter

Mirrors the prior_verifier_first_arbiter_divergent routing pattern.
ARBITRATING tasks with the uncertain flag bypass run_task_async and
go directly to _resolve_uncertain_arbiter_async."
```

---

## Self-Review

### 1. Spec coverage

| Requirement | Task |
|-------------|------|
| P3: full row text in slim prompt | Task 1 |
| P2: latest QC round feedback only | Task 2 |
| second arbiter: `_arbitrate_and_apply` accepts target | Task 3 |
| second arbiter: flag set instead of immediate HR | Task 4 (flag sites) |
| second arbiter: resolver method (confident → accept) | Task 4 (resolver) |
| second arbiter: both uncertain → HR | Task 4 (resolver test) |
| second arbiter: unavailable → HR | Task 4 (resolver test) |
| second arbiter: scheduler dispatch | Task 5 |

All requirements covered. ✓

### 2. Placeholder scan

No TBD, TODO, "similar to", or "add appropriate" patterns. All code blocks are complete. ✓

### 3. Type consistency

- `_arbitrate_and_apply` gains `target_name: str = "arbiter"` in Task 3; used as `target_name="arbiter_secondary"` in Task 4. ✓
- `_resolve_uncertain_arbiter` / `_resolve_uncertain_arbiter_async` defined in Task 4; called from scheduler in Task 5. ✓
- `task.metadata["arbiter_uncertain_needs_second"]` set in Task 4, popped in `_resolve_uncertain_arbiter_async`. ✓
- `StubLLMClient` referenced in tests — exists in `tests/test_subagent_cycle.py` at the module level; all test files import it from there. ✓

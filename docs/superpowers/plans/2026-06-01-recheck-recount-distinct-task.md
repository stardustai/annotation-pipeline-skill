# Re-check = Recount + Stop Weighted Accumulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `entity_statistics` an honest distinct-task projection of current accepted state — rebuilt on operator "Re-check" — and stop the lifetime weighted vote-accumulation that inflated it 10–30×.

**Architecture:** Today `entity_statistics` is a lifetime accumulator: every ACCEPTED decision adds +1 and every human-review (HR) commit adds +5 (`HR_WEIGHT`), so historical counts drift far above current task reality and spuriously flag spans as "divergent". This plan (1) adds `EntityStatisticsService.recount_project()` that rebuilds the whole project's stats from the current annotation of every accepted task using **distinct-task** semantics (each task contributes +1 per distinct `(span, type)` it tags, no HR weighting); (2) makes `build_posterior_audit` (the POST /api/posterior-audit "Re-check"/rebuild handler) call `recount_project` first, so the audit reflects freshly-recounted stats; (3) removes the live weighted accumulation in `subagent_cycle.py` and `human_review_service.py` so `entity_statistics` is only ever written by recount. **Behavioral consequence (intended):** between recounts the verifier reads the last recount snapshot; a project with no recount yet has empty stats → the verifier returns `cold_start` (no opinion), which is safe. The prompt-injection gate (`check_past_experience`) reads `entity_conventions`, NOT `entity_statistics`, so injection is unaffected.

**Tech Stack:** Python 3, sqlite (custom `SqliteStore`), pytest. Frontend: React/TypeScript (Vite), one wording tweak only.

---

### Task 1: `recount_project()` distinct-task rebuild on the service

Add a whole-project recount that mirrors the existing per-span `recount_span` but writes distinct-task counts for every span in one DELETE-all + INSERT. Extract the artifact-loading helper currently nested inside `recount_span` so both methods share it (DRY).

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_statistics_service.py`
- Test: `tests/test_entity_statistics_service.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_entity_statistics_service.py`:

```python
def test_recount_project_distinct_task_counts(tmp_path):
    """recount_project rebuilds the whole project from accepted tasks using
    distinct-task semantics: each accepted task contributes +1 per distinct
    (span, type), regardless of how many rows repeat it, and pre-existing
    inflated rows are wiped."""
    import json
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.core.models import ArtifactRef
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)

    # Pre-existing INFLATED stats that recount must overwrite.
    svc.increment(project_id="p", span="Apple", entity_type="organization", weight=99)
    svc.increment(project_id="p", span="Stale", entity_type="product", weight=40)

    def _add_accepted(task_id, annotation):
        task = Task.new(
            task_id=task_id, pipeline_id="p",
            source_ref={"kind": "jsonl", "payload": {
                "text": "x", "rows": [{"row_index": 0, "input": "x"}],
            }},
        )
        task.status = TaskStatus.ACCEPTED
        store.save_task(task)
        rel = f"artifact_payloads/{task_id}/final.json"
        abs_path = store.root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(json.dumps({"text": json.dumps(annotation)}), encoding="utf-8")
        store.append_artifact(ArtifactRef.new(
            task_id=task_id, kind="annotation_result", path=rel,
            content_type="application/json",
        ))

    # Task A tags Apple as organization in TWO rows -> counts once.
    _add_accepted("a", {"rows": [
        {"row_index": 0, "output": {"entities": {"organization": ["Apple"]}}},
        {"row_index": 1, "output": {"entities": {"organization": ["Apple"]}}},
    ]})
    # Task B tags Apple as organization (+1) AND product (+1).
    _add_accepted("b", {"rows": [
        {"row_index": 0, "output": {"entities": {"organization": ["Apple"], "product": ["Apple"]}}},
    ]})

    result = svc.recount_project(project_id="p")

    assert svc.distribution(project_id="p", span="Apple") == {
        "organization": 2, "product": 1,
    }
    # Stale span had no accepted task -> wiped entirely.
    assert svc.distribution(project_id="p", span="Stale") == {}
    assert result["apple"] == {"organization": 2, "product": 1}


def test_recount_project_only_counts_accepted_tasks(tmp_path):
    import json
    from annotation_pipeline_skill.core.models import Task, ArtifactRef
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)

    def _add(task_id, status, annotation):
        task = Task.new(
            task_id=task_id, pipeline_id="p",
            source_ref={"kind": "jsonl", "payload": {
                "text": "x", "rows": [{"row_index": 0, "input": "x"}]}},
        )
        task.status = status
        store.save_task(task)
        rel = f"artifact_payloads/{task_id}/final.json"
        abs_path = store.root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(json.dumps({"text": json.dumps(annotation)}), encoding="utf-8")
        store.append_artifact(ArtifactRef.new(
            task_id=task_id, kind="annotation_result", path=rel,
            content_type="application/json"))

    _add("acc", TaskStatus.ACCEPTED,
         {"rows": [{"row_index": 0, "output": {"entities": {"organization": ["Apple"]}}}]})
    _add("pending", TaskStatus.ANNOTATING,
         {"rows": [{"row_index": 0, "output": {"entities": {"product": ["Apple"]}}}]})

    svc.recount_project(project_id="p")
    assert svc.distribution(project_id="p", span="Apple") == {"organization": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_entity_statistics_service.py::test_recount_project_distinct_task_counts tests/test_entity_statistics_service.py::test_recount_project_only_counts_accepted_tasks -v`
Expected: FAIL with `AttributeError: 'EntityStatisticsService' object has no attribute 'recount_project'`

- [ ] **Step 3: Extract the shared loader, then implement `recount_project`**

In `annotation_pipeline_skill/services/entity_statistics_service.py`, add a module-level loader near the top of the file (after the imports, before the `VerifierResult` dataclass). This is the SAME logic currently nested as `_load_latest` inside `recount_span` (prefer `human_review_answer`, fall back to `annotation_result`, strip `<think>`):

```python
def _load_latest_annotation(store, task_id: str) -> dict | None:
    """Load a task's effective annotation payload: prefer the human-review
    answer, else the latest annotation_result (with <think> stripped).
    Shared by recount_span and recount_project."""
    import json as _json
    import re as _re

    arts = store.list_artifacts(task_id)
    hr = [a for a in arts if a.kind == "human_review_answer"]
    if hr:
        try:
            outer = _json.loads((store.root / hr[-1].path).read_text(encoding="utf-8"))
            return outer.get("answer") if isinstance(outer, dict) else None
        except (OSError, _json.JSONDecodeError):
            return None
    anns = [a for a in arts if a.kind == "annotation_result"]
    if not anns:
        return None
    try:
        outer = _json.loads((store.root / anns[-1].path).read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return None
    text = outer.get("text") if isinstance(outer, dict) else None
    if not isinstance(text, str):
        return None
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
    try:
        return _json.loads(text)
    except (ValueError, _json.JSONDecodeError):
        return None
```

Then refactor `recount_span` to delegate to it — replace the nested `def _load_latest(task_id: str)` (lines ~112-135) and its single call `payload = _load_latest(row["task_id"])` (line ~139) with:

```python
            payload = _load_latest_annotation(self.store, row["task_id"])
```

(Delete the now-redundant nested `_load_latest` definition entirely.)

Now add the new method directly after `recount_span` (after its `return new_counts`, around line 169):

```python
    def recount_project(self, *, project_id: str) -> dict[str, dict[str, int]]:
        """Rebuild entity_statistics for the WHOLE project from ground truth.

        Distinct-task semantics: each ACCEPTED task contributes +1 to
        (span_lower, entity_type) for every DISTINCT type it tags that span
        as (deduped within the task). No HR weighting, no lifetime
        accumulation — the table becomes an honest projection of current
        accepted state. Replaces ALL rows for the project atomically.

        Returns {span_lower: {entity_type: count}}.
        """
        from annotation_pipeline_skill.core.states import TaskStatus

        counts: dict[str, dict[str, int]] = {}
        for task in self.store.list_tasks_by_pipeline(project_id):
            if task.status is not TaskStatus.ACCEPTED:
                continue
            payload = _load_latest_annotation(self.store, task.task_id)
            if not isinstance(payload, dict):
                continue
            seen: set[tuple[str, str]] = set()
            for span, entity_type in iter_span_decisions(payload):
                span_lower = span.strip().lower()
                if not span_lower or not entity_type:
                    continue
                seen.add((span_lower, entity_type))
            for span_lower, entity_type in seen:
                counts.setdefault(span_lower, {})
                counts[span_lower][entity_type] = counts[span_lower].get(entity_type, 0) + 1

        now = datetime.now(timezone.utc).isoformat()
        with self.store._conn:
            self.store._conn.execute(
                "DELETE FROM entity_statistics WHERE project_id=?", (project_id,)
            )
            for span_lower, dist in counts.items():
                for entity_type, count in dist.items():
                    self.store._conn.execute(
                        "INSERT INTO entity_statistics "
                        "(project_id, span_lower, entity_type, count, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (project_id, span_lower, entity_type, count, now),
                    )
        return counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_entity_statistics_service.py -v`
Expected: PASS (new tests + all pre-existing recount_span / increment / divergent tests still green — the loader refactor is behavior-preserving)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/services/entity_statistics_service.py tests/test_entity_statistics_service.py
git commit -m "feat: add recount_project distinct-task rebuild to EntityStatisticsService"
```

---

### Task 2: `build_posterior_audit` recounts before auditing

Make the Re-check / rebuild handler recount the whole project first, so divergent/low-info/deviation outputs reflect fresh distinct-task stats instead of stale accumulated ones.

**Files:**
- Modify: `annotation_pipeline_skill/interfaces/api.py:128-261` (`build_posterior_audit`)
- Test: `tests/test_posterior_audit_api.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_posterior_audit_api.py`:

```python
def test_build_posterior_audit_recounts_before_auditing(tmp_path):
    """build_posterior_audit rebuilds entity_statistics from accepted tasks
    first, so a span inflated by stale historical votes collapses to its
    true distinct-task distribution and drops out of divergent."""
    import json
    from annotation_pipeline_skill.core.models import Task, ArtifactRef
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.interfaces.api import build_posterior_audit
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)

    # Stale inflated distribution that WOULD be "divergent" (org 30 / product 30,
    # total 60, no dominant >= 0.80, both >= 0.20).
    svc.increment(project_id="p", span="Apple", entity_type="organization", weight=30)
    svc.increment(project_id="p", span="Apple", entity_type="product", weight=30)

    # But the current accepted reality: 10 tasks ALL tag Apple as organization.
    for i in range(10):
        tid = f"t{i}"
        task = Task.new(task_id=tid, pipeline_id="p",
                        source_ref={"kind": "jsonl", "payload": {
                            "text": "Apple", "rows": [{"row_index": 0, "input": "Apple"}]}})
        task.status = TaskStatus.ACCEPTED
        store.save_task(task)
        rel = f"artifact_payloads/{tid}/final.json"
        ap = store.root / rel
        ap.parent.mkdir(parents=True, exist_ok=True)
        ap.write_text(json.dumps({"text": json.dumps(
            {"rows": [{"row_index": 0, "output": {"entities": {"organization": ["Apple"]}}}]})}),
            encoding="utf-8")
        store.append_artifact(ArtifactRef.new(
            task_id=tid, kind="annotation_result", path=rel,
            content_type="application/json"))

    result = build_posterior_audit(store, project_id="p")

    # After the in-handler recount, Apple is org:10 -> NOT divergent.
    spans = {e["span"] for e in result["divergent_entries"]}
    assert "apple" not in spans
    assert svc.distribution(project_id="p", span="Apple") == {"organization": 10}
    # task_deviations is ALSO computed against the recounted stats (svc.check
    # reads the fresh distribution): with all 10 tasks tagging Apple as
    # organization, none deviates from the now-honest consensus.
    assert all(d["span"].lower() != "apple" for d in result["task_deviations"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_posterior_audit_api.py::test_build_posterior_audit_recounts_before_auditing -v`
Expected: FAIL — `"apple"` is still in divergent_entries (stale stats not recounted), and distribution still `{"organization": 30, "product": 30}`.

- [ ] **Step 3: Add the recount call at the top of `build_posterior_audit`**

In `annotation_pipeline_skill/interfaces/api.py`, the function already imports `EntityStatisticsService` and constructs `svc = EntityStatisticsService(store)` at line ~161. Insert the recount immediately after that construction so every downstream read (`svc.check`, `svc.divergent_entries`, and the raw `entity_statistics` SELECT for low_info) sees fresh counts. Change:

```python
    svc = EntityStatisticsService(store)
```

to:

```python
    svc = EntityStatisticsService(store)
    # Re-check == re-count: rebuild entity_statistics from the current
    # annotation of every accepted task (distinct-task semantics) BEFORE
    # auditing. This replaces the old lifetime vote-accumulator reading,
    # so divergent/low-info/deviation flags reflect current reality rather
    # than inflated historical counts.
    svc.recount_project(project_id=project_id)
```

Also update the function docstring (lines 129-132) to reflect that it now recounts:

```python
    """Recount entity_statistics for the project from the current annotation
    of every ACCEPTED task (distinct-task semantics), then compare each
    task's (span, type) decisions to the freshly-recounted stats. Return
    task-level deviations and project-level contested spans.
    """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_posterior_audit_api.py -v`
Expected: PASS (new test green; pre-existing posterior-audit tests still green — they seed stats via `increment` then have NO accepted tasks, so `recount_project` wipes the seeded rows. **If a pre-existing test now fails because its seeded span was wiped, that test must be updated in Task 5** to back its stats with accepted tasks rather than bare `increment` calls. Note which tests fail here and carry them into Task 5.)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/interfaces/api.py tests/test_posterior_audit_api.py
git commit -m "feat: recount entity_statistics inside build_posterior_audit (Re-check = re-count)"
```

---

### Task 3: Stop live weighted accumulation in the subagent cycle

Remove the `+1`-per-accept live increment so `entity_statistics` is only written by recount.

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py` (method at 1200-1227; call sites 935, 1904-ish, 2098, 2449, 2469)
- Test: `tests/test_prior_verifier_integration.py`

- [ ] **Step 1: Write the failing test (new semantics: accept does NOT mutate stats)**

Replace the body assertion of `test_qc_pass_with_prior_agree_accepts_and_increments_stats` (and rename it). Change lines ~61-89 of `tests/test_prior_verifier_integration.py` from the increment assertion to a no-mutation assertion. New version:

```python
def test_qc_pass_with_prior_agree_accepts_without_mutating_stats(tmp_path):
    store = SqliteStore.open(tmp_path)
    project = "p"
    _seed_prior(store, project_id=project, span="Apple",
                type_to_count={"organization": 9, "project": 1})

    annotation = {
        "rows": [{
            "row_index": 0,
            "output": {"entities": {"organization": ["Apple"]}},
        }]
    }
    task = _make_task("t-agree", input_text="Apple is a company")
    task.status = TaskStatus.PENDING
    store.save_task(task)

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: _RecorderClient(qc_passed=True, annotation=annotation),
    )
    asyncio.run(runtime.run_task_async(store.load_task("t-agree")))

    after = store.load_task("t-agree")
    assert after.status is TaskStatus.ACCEPTED
    svc = EntityStatisticsService(store)
    # Acceptance no longer mutates entity_statistics — stats are a recount
    # projection now, refreshed only via build_posterior_audit / Re-check.
    assert svc.distribution(project_id=project, span="Apple") == {
        "organization": 9, "project": 1,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prior_verifier_integration.py::test_qc_pass_with_prior_agree_accepts_without_mutating_stats -v`
Expected: FAIL — distribution is `{"organization": 10, "project": 1}` because the runtime still live-increments.

- [ ] **Step 3: Remove the live increment method and all call sites**

In `annotation_pipeline_skill/runtime/subagent_cycle.py`:

1. Delete the entire method `_increment_entity_statistics_for_task` (lines ~1200-1227).
2. Delete each of the 5 call sites. The single-line sites (935, 2098, 2449, 2469) are of the form:

```python
            self._increment_entity_statistics_for_task(task, annotation_artifact, weight=1)
```

Delete each such line outright. For the multi-line site near 1904, it reads (verify exact text with Read before editing):

```python
            self._increment_entity_statistics_for_task(
                task, <artifact-var>, weight=1
            )
```

Delete the full call (all its lines). Do not leave a dangling blank `if:` body — if any call was the sole statement in a block, replace it with `pass` only if Python would otherwise have an empty block (it won't here; each site sits among other statements — verify with Read).

**Verification grep after editing:**

```bash
grep -n "_increment_entity_statistics_for_task" annotation_pipeline_skill/runtime/subagent_cycle.py
```

Expected: no matches.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prior_verifier_integration.py -v`
Expected: the renamed agree-test PASSES. **`test_qc_pass_with_cold_start_accepts` and `test_arbiter_acceptance_increments_stats` will now FAIL** (they assert `+1` from acceptance) — those are fixed in Step 5.

- [ ] **Step 5: Update the remaining two increment-asserting integration tests**

In `tests/test_prior_verifier_integration.py`:

For `test_qc_pass_with_cold_start_accepts` (the assertion at lines ~149-151), change the expected distribution to the seeded value only (no live increment):

```python
    assert svc.distribution(project_id=project, span="Apple") == {
        "organization": 5,
    }
```

For `test_arbiter_acceptance_increments_stats`: rename to `test_arbiter_acceptance_does_not_mutate_stats`, update its docstring, and change the assertion at lines ~201-203 to:

```python
    svc = EntityStatisticsService(store)
    # Arbiter-driven acceptance no longer mutates stats (recount-only model).
    assert svc.distribution(project_id=project, span="Acme") == {"organization": 12}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_prior_verifier_integration.py -v`
Expected: PASS (all)

- [ ] **Step 7: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_prior_verifier_integration.py
git commit -m "refactor: stop live entity_statistics accumulation in subagent cycle"
```

---

### Task 4: Stop HR-weighted accumulation in human_review_service

Remove the `HR_WEIGHT` (×5) bumps and the approximate decrement; `entity_statistics` is recount-only.

**Files:**
- Modify: `annotation_pipeline_skill/services/human_review_service.py` (call sites 216, 368-370; methods `_increment_stats_from_hr` 373-387, `_apply_scoped_stat_bumps` 389-end)
- Modify: `annotation_pipeline_skill/services/entity_statistics_service.py` (drop now-unused `HR_WEIGHT` if no remaining references)
- Test: `tests/test_human_review_service.py`

- [ ] **Step 1: Confirm no test asserts post-HR stat mutation**

Run:

```bash
grep -n "distribution\|HR_WEIGHT\|_increment_stats_from_hr\|_apply_scoped_stat_bumps" tests/test_human_review_service.py
```

Expected: only `increment`-based seeding for the prior-disagreement GATE tests (`test_submit_correction_rejects_when_against_prior`, `test_submit_correction_with_force_bypasses_verifier`), which assert status/error kind, NOT post-correction distribution. If this grep reveals a test asserting a distribution AFTER `submit_correction`/`decide`, update that test in this task to expect no HR mutation.

- [ ] **Step 2: Write a guard test pinning the new no-HR-mutation behavior**

Add to `tests/test_human_review_service.py`:

```python
def test_submit_correction_force_does_not_mutate_stats(tmp_path):
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.entity_statistics_service import (
        EntityStatisticsService,
    )
    from annotation_pipeline_skill.services.human_review_service import (
        HumanReviewService,
    )
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    svc = EntityStatisticsService(store)
    for _ in range(12):
        svc.increment(project_id="p", span="Apple", entity_type="organization")

    schema = {"type": "object", "additionalProperties": False, "required": ["rows"],
              "properties": {"rows": {"type": "array"}}}
    task = Task.new(
        task_id="hr-3", pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "Apple is mentioned"}],
            "annotation_guidance": {"output_schema": schema}}},
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    hr = HumanReviewService(store)
    answer = {"rows": [{"row_index": 0, "output": {"entities": {"technology": ["Apple"]}}}]}
    hr.submit_correction(task_id="hr-3", answer=answer, actor="op", note=None, force=True)

    # HR no longer bumps stats by HR_WEIGHT; distribution is unchanged until
    # an operator runs Re-check (recount).
    assert svc.distribution(project_id="p", span="Apple") == {"organization": 12}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_human_review_service.py::test_submit_correction_force_does_not_mutate_stats -v`
Expected: FAIL — distribution is `{"organization": 12, "technology": 5}` (HR_WEIGHT bump still applied).

- [ ] **Step 4: Remove the HR accumulation**

In `annotation_pipeline_skill/services/human_review_service.py`:

1. Delete the call at line ~216 (inside the `if next_status is TaskStatus.ACCEPTED:` block in `decide`). Verify with Read; that block is:

```python
        # Update stats with HR weight (5x) when accepting via decide.
        if next_status is TaskStatus.ACCEPTED:
            _ann = self._latest_annotation_payload(task_id)
            if _ann is not None:
                self._increment_stats_from_hr(task, _ann)
```

Delete the whole block (comment + `if` + body).

2. Delete the `stat_bumps`-dispatch block at lines ~357-370. It is:

```python
        # Update stats with HR weight (5x)
        # ... (multi-line comment) ...
        if stat_bumps is None:
            self._increment_stats_from_hr(task, answer)
        else:
            self._apply_scoped_stat_bumps(task, stat_bumps)
```

Delete the comment + the `if/else`. **Check whether `stat_bumps` is still used afterwards** (it was a parameter to this method, likely `submit_correction`/`apply_correction`). If `stat_bumps` becomes unused, leave the parameter in the signature for API stability but it is now ignored — add a one-line comment `# stat_bumps retained for signature compatibility; stats are recount-only now`. If it is referenced elsewhere in the method, leave those references intact.

3. Delete the methods `_increment_stats_from_hr` (373-387) and `_apply_scoped_stat_bumps` (389 to its end — read to find the closing line).

**Verification grep:**

```bash
grep -n "_increment_stats_from_hr\|_apply_scoped_stat_bumps" annotation_pipeline_skill/services/human_review_service.py
```

Expected: no matches.

4. In `annotation_pipeline_skill/services/entity_statistics_service.py`, check if `HR_WEIGHT` is still referenced anywhere:

```bash
grep -rn "HR_WEIGHT" annotation_pipeline_skill/ tests/
```

If the only remaining references are the definition (line 22) and `tests/` that you can update, remove the `HR_WEIGHT = 5` constant and the docstring sentence at lines 6-7 ("HR decisions count with extra weight..."). If any non-test production code still imports it, leave it. Update/remove any test import of `HR_WEIGHT` that becomes dead.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_human_review_service.py -v`
Expected: PASS (new guard test green; the prior-disagreement gate tests still pass because they only read stats)

- [ ] **Step 6: Commit**

```bash
git add annotation_pipeline_skill/services/human_review_service.py annotation_pipeline_skill/services/entity_statistics_service.py tests/test_human_review_service.py
git commit -m "refactor: stop HR-weighted entity_statistics accumulation (recount-only model)"
```

---

### Task 5: Repair any stat-seeded tests broken by recount-wipe

`build_posterior_audit` now wipes `entity_statistics` rows that aren't backed by accepted tasks. Any test that seeded stats via bare `increment` and then called `build_posterior_audit` will see those rows disappear. Fix them to back stats with accepted tasks (the realistic path) OR assert against the recounted reality.

**Files:**
- Modify: `tests/test_posterior_audit_api.py` (and any other test flagged in Task 2 Step 4)

- [ ] **Step 1: Identify breakages**

Run: `pytest tests/test_posterior_audit_api.py -v`
Expected: note every test that fails because a seeded span vanished after the in-handler recount.

- [ ] **Step 2: Fix each flagged test**

For each failing test, add a small helper that creates accepted tasks tagging the span the intended number of times, replacing the bare `svc.increment(...)` seed. Pattern (reuse the `_add_accepted` shape from Task 1's test):

```python
def _accepted_task_tagging(store, project_id, task_id, span, entity_type):
    import json
    from annotation_pipeline_skill.core.models import Task, ArtifactRef
    from annotation_pipeline_skill.core.states import TaskStatus
    task = Task.new(task_id=task_id, pipeline_id=project_id,
                    source_ref={"kind": "jsonl", "payload": {
                        "text": span, "rows": [{"row_index": 0, "input": span}]}})
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    rel = f"artifact_payloads/{task_id}/final.json"
    ap = store.root / rel
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text(json.dumps({"text": json.dumps(
        {"rows": [{"row_index": 0, "output": {"entities": {entity_type: [span]}}}]})}),
        encoding="utf-8")
    store.append_artifact(ArtifactRef.new(
        task_id=task_id, kind="annotation_result", path=rel,
        content_type="application/json"))
```

Then replace each `for _ in range(N): svc.increment(... span=S, entity_type=T)` that feeds a `build_posterior_audit` assertion with `N` calls to `_accepted_task_tagging(store, "p", f"{S}-{i}", S, T)`. Recompute expected distributions on a distinct-task basis (each task = +1). Tests that call `divergent_entries`/`check` directly (NOT through `build_posterior_audit`) need no change — those don't recount.

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_posterior_audit_api.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_posterior_audit_api.py
git commit -m "test: back posterior-audit stat fixtures with accepted tasks for recount model"
```

---

### Task 6: Update the Re-check button wording

Reflect that the action now recounts, not just re-reads.

**Files:**
- Modify: `web/src/components/PosteriorAuditPanel.tsx:263`

- [ ] **Step 1: Read the current label line**

Run: `grep -n "Re-check\|Checking\|cachedExists" web/src/components/PosteriorAuditPanel.tsx`
Expected: the ternary at line ~263: `{loading ? "Checking…" : stale ? "Re-check (stale)" : cachedExists ? "Re-check" : "Check"}`

- [ ] **Step 2: Update the label**

Replace that ternary with recount-oriented wording:

```tsx
{loading ? "Recounting…" : stale ? "Re-count (stale)" : cachedExists ? "Re-count" : "Count"}
```

If there is nearby helper/tooltip text describing the button as "re-reads stats", update it to "rebuilds stats from current accepted tasks (distinct-task recount)". Search:

```bash
grep -n "re-read\|re-check\|recompute\|stats" web/src/components/PosteriorAuditPanel.tsx
```

- [ ] **Step 3: Typecheck the frontend**

Run: `cd web && npx tsc --noEmit` (or the project's configured typecheck script — check `web/package.json` scripts first)
Expected: no type errors introduced by the wording change.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/PosteriorAuditPanel.tsx
git commit -m "ui: relabel posterior-audit action Re-check -> Re-count"
```

(Note: `web/dist` is NOT git-tracked — do not stage build output.)

---

### Task 7: Full regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the full backend suite**

Run: `pytest tests/ -q`
Expected: all green. Pay special attention to:
- `tests/test_entity_statistics_service.py`
- `tests/test_prior_verifier_integration.py`
- `tests/test_human_review_service.py`
- `tests/test_posterior_audit_api.py`
- `tests/test_bootstrap_entity_statistics.py`, `tests/test_entity_statistics_pagination.py`, `tests/test_entity_statistics_schema.py`, `tests/test_knowledge_summary_endpoint.py`

- [ ] **Step 2: Grep for orphaned references**

Run:

```bash
grep -rn "_increment_entity_statistics_for_task\|_increment_stats_from_hr\|_apply_scoped_stat_bumps" annotation_pipeline_skill/
```

Expected: no matches (all live-accumulation paths removed).

- [ ] **Step 3: Fix any failures, re-run until green, then final commit if anything changed**

```bash
git add -p   # stage reviewed changes only; never git add -A
git commit -m "test: regression fixes for recount-only entity_statistics model"
```

---

## Notes / Out-of-scope

- **Performance:** `build_posterior_audit` now scans accepted tasks twice (once in `recount_project`, once in its own deviation loop). This is acceptable for a manual/background "Re-check"; the prior per-span HTTP recount took ~700s for 2,415 spans, whereas a single in-process full scan is far cheaper even done twice. A future optimization could fold both into one scan, but that couples recount with audit concerns and is deliberately deferred (YAGNI).
- **`increment()` is intentionally kept** as a low-level primitive (used by tests, seeding, and the per-span recount endpoint). Only the *automatic* pipeline-flow accumulation is removed.
- **`recount_span` stays** — it backs the per-span recount endpoint used in residual review. It now shares `_load_latest_annotation` with `recount_project`.
- **Plan file is NOT committed** (per repo policy: `docs/superpowers/plans/*.md` are excluded).

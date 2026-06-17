# Convention Recount-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `entity_conventions` a recount-only projection — `EntityConventionService.recount_project` recomputes each convention's empirical fields from the current ACCEPTED annotations, auto-fixing frozen-wrong values (e.g. `vue`→`project` becomes `vue`→`technology`), while preserving operator/HR locks; wired into the Posterior-Audit "Re-count" flow.

**Architecture:** A new `recount_project(project_id)` method scans the project's ACCEPTED tasks once (same latest-annotation loader as `EntityStatisticsService.recount_project`, so the two tables agree by construction), tallies **one vote per task per span** (all-accepted, arbiter included), and UPDATEs only the materialized columns (`entity_type`, `dominant_type`, `distinct_task_count`, `dispute_count`, `dispute_pct`) that actually drive injection. Operator/HR-locked conventions (`created_by` carries an operator prefix, or `status='disputed'`) keep their `entity_type` and injection bypass — only their descriptive stats are refreshed. `proposals_json` is NOT rebuilt (no behavioral path reads its recomputed aggregates; rebuilding it is cosmetic and would be O(votes²)). No rows are created or deleted — saturation cleanup is explicitly out of scope. Finally `build_posterior_audit` calls it right after the stats recount and before building `convention_index`.

**Tech Stack:** Python 3.13, sqlite3 (WAL), pytest. No new dependencies.

---

## Background the implementer needs

- **What drives injection** (verified): `EntityConventionService.find_matches_in_text` → `_iter_injection_candidates` reads ONLY the materialized columns (`distinct_task_count`, `dispute_pct`, `status`, `created_by`) and `entity_type`; it never parses `proposals_json`. The injection gate is: `status='active' AND ((distinct_task_count >= INJECT_MIN_DISTINCT_TASKS(=5) AND dispute_pct < INJECT_MAX_DISPUTE_PCT(=0.20)) OR created_by LIKE '<operator-prefix>%')`. So to change what gets injected you write columns; to preserve an operator bypass you must NOT change that row's `created_by`.
- **No behavioral consumer reads `EntityConvention.dominant_type`/`.dispute_pct`** off a `list_for_project` result (the only `.dominant_type` references in the codebase are on `VerifierResult`, a different type). `build_posterior_audit`'s `convention_index` reads `c.entity_type` (a column). Therefore writing columns and leaving `proposals_json` stale changes no behavior; the proposals audit trail simply remains a historical log.
- **Operator/HR prefixes** are the module-level tuple `_OPERATOR_DECLARATION_SOURCE_PREFIXES` in `annotation_pipeline_skill/services/entity_convention_service.py:62` (`"declared:"`, `"hr_correction:"`, `"posterior_audit_operator"`, `"batch_operator_resolve"`, `"batch_operator_correction"`, `"dispute_resolved_by:"`). The class re-exposes it as `EntityConventionService.OPERATOR_DECLARATION_SOURCE_PREFIXES`. Every operator action stamps `created_by` with its source, so a `created_by`-prefix test is sufficient to detect a lock (no `proposals_json` scan needed).
- **Vote model** (decided): all ACCEPTED tasks, arbiter included, **one vote per task** — a task that tags a span under multiple types contributes a single deterministic vote (`max(types)`), so intra-task multi-typing does not inflate `dispute_pct` (which gates injection at 0.20). This mirrors the spirit of `_distinct_task_tally` (one vote per distinct task) rather than `entity_statistics`'s per-`(span,type)` counting.
- **Loader reuse:** import `_load_latest_annotation` and `iter_span_decisions` from `annotation_pipeline_skill.services.entity_statistics_service` — the exact functions the stats recount uses — so conventions and stats can't disagree on what a task's current annotation is.
- **Git branch:** `feat/convention-distinct-task-gate` (NOT main). Do NOT `git add -A`. Never commit `db.sqlite*`, `backups/`, `reports/`, `scratch/`, `docs/superpowers/plans/*.md`, `web/db.sqlite`. Subagents must NOT add `Co-Authored-By`.

## File Structure

- **Modify** `annotation_pipeline_skill/services/entity_convention_service.py` — add `recount_project` method (self-contained; uses the stats-service loader). ~45 lines.
- **Modify** `annotation_pipeline_skill/interfaces/api.py` — call `EntityConventionService(store).recount_project(project_id=project_id)` inside `build_posterior_audit`, after `svc.recount_project(...)` (stats) and before the `convention_index` loop (currently ~line 175→186).
- **Test** `tests/test_entity_convention_recount.py` — new file, unit tests for the method.
- **Test** `tests/test_posterior_audit_api.py` — add one integration test that a frozen-wrong convention is corrected by `build_posterior_audit` and an operator lock survives.

---

### Task 1: `EntityConventionService.recount_project`

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py` (add method after `rebuild_from_accepted_tasks`, ~line 514)
- Test: `tests/test_entity_convention_recount.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_entity_convention_recount.py`:

```python
import json

from annotation_pipeline_skill.core.models import ArtifactRef, Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _add_accepted_task(store, project_id, task_id, entities_by_type):
    """One ACCEPTED task whose final annotation tags {type: [spans]} in row 0."""
    task = Task.new(
        task_id=task_id, pipeline_id=project_id,
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_index": 0, "input": "x"}]}},
    )
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    rel = f"artifact_payloads/{task_id}/final.json"
    abs_path = store.root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(json.dumps({"text": json.dumps({
        "rows": [{"row_index": 0, "output": {"entities": entities_by_type}}]
    })}))
    store.append_artifact(ArtifactRef.new(
        task_id=task_id, kind="annotation_result", path=rel,
        content_type="application/json",
    ))


def test_recount_fixes_frozen_dominant(tmp_path):
    """A convention frozen at the wrong type is corrected to the current
    empirical dominant. Mirrors the real 'vue' case: convention says
    project, but every accepted task now tags it technology."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    # Seed a stale convention via the normal write path, frozen at 'project'.
    for i in range(3):
        svc.record_decision(project_id="p", span="vue", entity_type="project",
                            source="qc_consensus", task_id=f"old-{i}")
    assert svc.list_for_project("p")[0].entity_type == "project"
    # Current reality: 12 accepted tasks tag vue as technology.
    for i in range(12):
        _add_accepted_task(store, "p", f"vue-{i}", {"technology": ["vue"]})

    summary = svc.recount_project(project_id="p")

    conv = [c for c in svc.list_for_project("p") if c.span_lower == "vue"][0]
    assert conv.entity_type == "technology"
    assert conv.dominant_type == "technology"
    assert conv.distinct_task_count == 12
    assert conv.dispute_count == 0
    assert conv.dispute_pct == 0.0
    assert summary["conventions_seen"] >= 1


def test_recount_preserves_operator_lock(tmp_path):
    """An operator-declared convention keeps its entity_type and its
    injection bypass even when the empirical dominant disagrees."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    # Operator declares span 'edge' -> product (locks it; created_by stamped).
    svc.record_decision(project_id="p", span="edge", entity_type="product",
                        source="declared:operator-1", task_id=None)
    # Reality: 8 accepted tasks tag 'edge' as technology.
    for i in range(8):
        _add_accepted_task(store, "p", f"edge-{i}", {"technology": ["edge"]})

    svc.recount_project(project_id="p")

    conv = [c for c in svc.list_for_project("p") if c.span_lower == "edge"][0]
    # entity_type stays the operator's pick...
    assert conv.entity_type == "product"
    # ...but descriptive stats reflect current reality (dashboard honesty).
    assert conv.dominant_type == "technology"
    assert conv.distinct_task_count == 8
    # bypass intact: created_by still operator-stamped, so it still injects.
    assert conv.created_by.startswith("declared:")
    matches = svc.find_matches_in_text("p", "we use edge in prod")
    assert any(m.span_lower == "edge" and m.entity_type == "product" for m in matches)


def test_recount_zeroes_vanished_span_without_deleting(tmp_path):
    """A convention whose span no longer appears in any accepted task is
    zeroed (so it can't inject) but NOT deleted (saturation deferred)."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    for i in range(6):
        svc.record_decision(project_id="p", span="ghost", entity_type="organization",
                            source="qc_consensus", task_id=f"g-{i}")
    # No accepted tasks mention 'ghost' anymore.
    _add_accepted_task(store, "p", "other-0", {"technology": ["python"]})

    svc.recount_project(project_id="p")

    rows = {c.span_lower: c for c in svc.list_for_project("p")}
    assert "ghost" in rows  # NOT deleted
    assert rows["ghost"].distinct_task_count == 0
    assert rows["ghost"].dominant_type is None


def test_recount_one_vote_per_task_for_multi_type_span(tmp_path):
    """A task tagging the same span under two types counts as ONE vote
    (deterministic max(types)), so dispute_pct is not inflated."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p", span="alpha", entity_type="organization",
                        source="qc_consensus", task_id="seed")
    # One task tags alpha as BOTH technology and product.
    _add_accepted_task(store, "p", "multi-0",
                       {"technology": ["alpha"], "product": ["alpha"]})

    svc.recount_project(project_id="p")

    conv = [c for c in svc.list_for_project("p") if c.span_lower == "alpha"][0]
    assert conv.distinct_task_count == 1  # one task = one vote, not two
    assert conv.dispute_count == 0
    assert conv.dispute_pct == 0.0
    assert conv.dominant_type == max(["technology", "product"])  # 'technology'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_entity_convention_recount.py -q`
Expected: FAIL with `AttributeError: 'EntityConventionService' object has no attribute 'recount_project'`

- [ ] **Step 3: Implement `recount_project`**

In `annotation_pipeline_skill/services/entity_convention_service.py`, add this method to `EntityConventionService` immediately after `rebuild_from_accepted_tasks` (before the `MIN_INJECTION_SPAN_LEN` class constants, ~line 514):

```python
    def recount_project(self, *, project_id: str) -> dict[str, int]:
        """Recompute each convention's empirical fields from the CURRENT
        annotation of every ACCEPTED task — the convention analog of
        ``EntityStatisticsService.recount_project``.

        Vote model: all ACCEPTED tasks (arbiter included), ONE vote per task
        per span. A task tagging a span under multiple types contributes a
        single deterministic vote (``max(types)``) so intra-task multi-typing
        does not inflate ``dispute_pct`` (the 0.20 injection threshold).

        Operator/HR-locked conventions (``created_by`` carries an operator
        prefix, or ``status='disputed'``) keep their ``entity_type`` and their
        injection bypass — only their descriptive stats are refreshed so the
        dashboard can surface operator-vs-data conflicts. All other
        conventions get ``entity_type`` set to the empirical dominant.

        Writes ONLY the materialized columns that drive injection
        (``entity_type``, ``dominant_type``, ``distinct_task_count``,
        ``dispute_count``, ``dispute_pct``). ``proposals_json`` is left as a
        historical audit trail (no behavioral path reads its recomputed
        aggregates; rebuilding it would be O(votes^2)). No rows are created
        or deleted — saturation cleanup is out of scope.

        Returns a summary dict with ``conventions_seen``, ``recomputed``,
        ``operator_preserved`` and ``zeroed`` counts.
        """
        from annotation_pipeline_skill.core.states import TaskStatus
        from annotation_pipeline_skill.services.entity_statistics_service import (
            _load_latest_annotation,
            iter_span_decisions,
        )

        # Pass 1: one vote per task per span from current accepted annotations.
        votes: dict[str, dict[str, str]] = {}  # span_lower -> {task_id: type}
        for task in self.store.list_tasks_by_pipeline(project_id):
            if task.status is not TaskStatus.ACCEPTED:
                continue
            payload = _load_latest_annotation(self.store, task.task_id)
            if not isinstance(payload, dict):
                continue
            per_span: dict[str, set[str]] = {}
            for span, entity_type in iter_span_decisions(payload):
                span_lower = span.strip().lower()
                if not span_lower or not entity_type:
                    continue
                per_span.setdefault(span_lower, set()).add(entity_type)
            for span_lower, types in per_span.items():
                # One vote per task; deterministic pick for multi-type tasks.
                votes.setdefault(span_lower, {})[task.task_id] = max(types)

        prefixes = self.OPERATOR_DECLARATION_SOURCE_PREFIXES
        now = datetime.now(timezone.utc).isoformat()
        conventions_seen = recomputed = operator_preserved = zeroed = 0
        conn = self.store._conn
        with conn:
            rows = conn.execute(
                "SELECT convention_id, span_lower, entity_type, created_by, status "
                "FROM entity_conventions WHERE project_id=?",
                (project_id,),
            ).fetchall()
            for row in rows:
                conventions_seen += 1
                task_votes = votes.get(row["span_lower"], {})
                distinct = len(task_votes)
                if distinct:
                    tally = Counter(task_votes.values())
                    # Plurality with deterministic (count, type) tiebreak —
                    # matches _distinct_task_tally.
                    dominant = max(tally.items(), key=lambda kv: (kv[1], kv[0]))[0]
                    dispute_ct = distinct - tally[dominant]
                    dispute_pct = dispute_ct / distinct
                else:
                    dominant, dispute_ct, dispute_pct = None, 0, 0.0
                    zeroed += 1
                created_by = row["created_by"] or ""
                operator_locked = (
                    any(created_by.startswith(p) for p in prefixes)
                    or row["status"] == "disputed"
                )
                if operator_locked:
                    new_type = row["entity_type"]  # keep the lock
                    operator_preserved += 1
                else:
                    new_type = dominant if dominant is not None else row["entity_type"]
                    recomputed += 1
                conn.execute(
                    "UPDATE entity_conventions "
                    "SET entity_type=?, dominant_type=?, distinct_task_count=?, "
                    "    dispute_count=?, dispute_pct=?, updated_at=? "
                    "WHERE convention_id=?",
                    (new_type, dominant, distinct, dispute_ct, dispute_pct, now,
                     row["convention_id"]),
                )
        return {
            "conventions_seen": conventions_seen,
            "recomputed": recomputed,
            "operator_preserved": operator_preserved,
            "zeroed": zeroed,
        }
```

`Counter`, `datetime`, `timezone` are already imported at the top of this module (lines 22, 25). Do not re-import them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_entity_convention_recount.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py tests/test_entity_convention_recount.py
git commit -m "feat: add EntityConventionService.recount_project (recount-only conventions)"
```

---

### Task 2: Wire convention recount into `build_posterior_audit`

**Files:**
- Modify: `annotation_pipeline_skill/interfaces/api.py` (inside `build_posterior_audit`, between the stats recount and the `convention_index` loop)
- Test: `tests/test_posterior_audit_api.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_posterior_audit_api.py` (the `_add_accepted_task` helper already exists at the top of that file):

```python
def test_build_posterior_audit_recounts_conventions(tmp_path):
    """build_posterior_audit recounts conventions before auditing: a
    convention frozen at the wrong type is corrected from current accepted
    annotations, while an operator lock is preserved."""
    from annotation_pipeline_skill.interfaces.api import build_posterior_audit
    from annotation_pipeline_skill.services.entity_convention_service import (
        EntityConventionService,
    )

    store = SqliteStore.open(tmp_path)
    convs = EntityConventionService(store)
    # Frozen-wrong auto convention: says 'project'.
    for i in range(3):
        convs.record_decision(project_id="p", span="vue", entity_type="project",
                             source="qc_consensus", task_id=f"old-{i}")
    # Operator-locked convention: says 'product'.
    convs.record_decision(project_id="p", span="edge", entity_type="product",
                        source="declared:op-1", task_id=None)
    # Current reality.
    for i in range(11):
        _add_accepted_task(store, "p", f"vue-{i}", {"technology": ["vue"]})
    for i in range(8):
        _add_accepted_task(store, "p", f"edge-{i}", {"technology": ["edge"]})

    build_posterior_audit(store, project_id="p")

    by_span = {c.span_lower: c for c in convs.list_for_project("p")}
    assert by_span["vue"].entity_type == "technology"   # corrected
    assert by_span["edge"].entity_type == "product"     # lock preserved
    assert by_span["edge"].dominant_type == "technology"  # stats refreshed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_posterior_audit_api.py::test_build_posterior_audit_recounts_conventions -q`
Expected: FAIL — `vue` still `project` (conventions not yet recounted by the handler).

- [ ] **Step 3: Wire the call into `build_posterior_audit`**

In `annotation_pipeline_skill/interfaces/api.py`, find the stats recount line inside `build_posterior_audit`:

```python
    svc.recount_project(project_id=project_id)
```

Immediately after it (and before the `from ...entity_convention_service import EntityConventionService` / `convention_index` block that follows), add:

```python
    # Conventions are recount-only too: rebuild each convention's empirical
    # fields from the same current accepted annotations BEFORE we read the
    # convention_index below, so a frozen-wrong convention (e.g. vue->project
    # while every accepted task now tags it technology) is corrected and the
    # audit reflects current policy. Operator/HR locks are preserved inside
    # recount_project. SIDE EFFECT: persists column updates to entity_conventions.
    EntityConventionService(store).recount_project(project_id=project_id)
```

Note: `EntityConventionService` is already imported a few lines below in the same function; this earlier reference still resolves because the import statement executes before this line at runtime ONLY IF it precedes it. To be safe, add an explicit local import on the line above the call:

```python
    from annotation_pipeline_skill.services.entity_convention_service import (
        EntityConventionService,
    )
    EntityConventionService(store).recount_project(project_id=project_id)
```

(The existing later `from ...import EntityConventionService` is now redundant but harmless — leave it; removing it is optional cleanup, not required.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_posterior_audit_api.py -q`
Expected: PASS (all tests in the file, including the new one and the pre-existing recount tests).

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/interfaces/api.py tests/test_posterior_audit_api.py
git commit -m "feat: recount entity_conventions inside build_posterior_audit"
```

---

### Task 3: Regression sweep + real-store verification

**Files:**
- No production code changes (verification only). If a real bug surfaces, fix it under TDD and note it.

- [ ] **Step 1: Full unit suite**

Run: `python -m pytest -q`
Expected: the new convention-recount tests pass; the only failures are the 7 KNOWN pre-existing ones (`test_cli.py::test_cli_import_jsonl_prelabeled_creates_tasks_with_prelabel_metadata`, `test_cli.py::test_cli_import_omits_per_task_inline_schema`, `test_dashboard_api.py::test_dashboard_api_returns_config_files_and_can_update_allowed_yaml`, `test_dashboard_api_distribution.py::test_distribution_scan_populates_cache_and_get_returns_payload`, `test_local_cli_client.py::test_build_codex_command_includes_json_resume_and_model`, `test_local_cli_client.py::test_local_codex_client_propagates_continuity_handle`, `test_prior_verifier_integration.py::test_second_arbiter_verbatim_retry_exhausted_routes_to_hr_not_silent_drop`). No NEW failures.

- [ ] **Step 2: Real-store verification — capture BEFORE state**

The injection-facing change must be validated against the real store. Run this read-only probe (writes nothing) and record the output:

```bash
.venv/bin/python - <<'PY'
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.services.entity_convention_service import EntityConventionService
store = SqliteStore.open("projects/v4_ner_phrase/.annotation-pipeline")
svc = EntityConventionService(store)
sample = ("we deploy vue and react on aws; google and microsoft are vendors; "
          "the team uses experian data and huggingface transformers")
before = svc.find_matches_in_text("v4_ner_phrase", sample)
print("BEFORE inject candidates:", len(before))
print("BEFORE vue:", [(c.span_lower,c.entity_type) for c in svc.list_for_project("v4_ner_phrase") if c.span_lower=="vue"])
PY
```

Record `BEFORE inject candidates` count and the `vue` entity_type (expected `project`).

- [ ] **Step 3: Run the convention recount on the real store**

```bash
.venv/bin/python - <<'PY'
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.services.entity_convention_service import EntityConventionService
store = SqliteStore.open("projects/v4_ner_phrase/.annotation-pipeline")
svc = EntityConventionService(store)
print(svc.recount_project(project_id="v4_ner_phrase"))
PY
```

Expected: a summary dict like `{'conventions_seen': ~55750, 'recomputed': ~55730, 'operator_preserved': ~20, 'zeroed': <some>}`.

- [ ] **Step 4: Real-store verification — assert the three acceptance checks**

```bash
.venv/bin/python - <<'PY'
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.services.entity_convention_service import EntityConventionService
store = SqliteStore.open("projects/v4_ner_phrase/.annotation-pipeline")
svc = EntityConventionService(store)
# (a) vue corrected
vue = [c for c in svc.list_for_project("v4_ner_phrase") if c.span_lower=="vue"][0]
print("(a) vue ->", vue.entity_type, "dominant", vue.dominant_type, "dtc", vue.distinct_task_count)
assert vue.entity_type == "technology", "vue not corrected"
# (b) an operator-declared convention still injects with its locked type
ops = [c for c in svc.list_for_project("v4_ner_phrase")
       if (c.created_by or "").startswith(tuple(EntityConventionService.OPERATOR_DECLARATION_SOURCE_PREFIXES))]
print("(b) operator-locked conventions:", len(ops))
if ops:
    o = ops[0]
    m = svc.find_matches_in_text("v4_ner_phrase", f"context {o.span_original} here")
    print("    sample lock:", o.span_lower, o.entity_type,
          "injects:", any(x.span_lower==o.span_lower for x in m))
# (c) injection candidate count for the same sample text — no wild swing
sample = ("we deploy vue and react on aws; google and microsoft are vendors; "
          "the team uses experian data and huggingface transformers")
after = svc.find_matches_in_text("v4_ner_phrase", sample)
print("(c) AFTER inject candidates:", len(after),
      [(c.span_lower,c.entity_type) for c in after])
PY
```

Expected:
- (a) `vue -> technology` (assertion passes).
- (b) at least one operator-locked convention; the sampled one still injects with its locked `entity_type` (bypass intact).
- (c) AFTER candidate count is in the same ballpark as BEFORE (Step 2). A small change is expected (frozen-wrong spans flip type or fall out); a wild swing (e.g. 10× more or near-zero) means the gate broke — STOP and investigate. Note in the report the BEFORE→AFTER delta and that flips like `vue:project→technology`, `huggingface:?→organization` are the intended corrections.

- [ ] **Step 5: Dispatch the final code reviewer**

Dispatch a code-quality reviewer over the diff range for this plan (`git log --oneline` to find the first commit of Task 1; review `<that-commit>^..HEAD`). Confirm: operator-lock preservation correct, only columns written, `proposals_json` untouched, no row creation/deletion, loader matches the stats recount. Address any Critical/Important findings under TDD.

- [ ] **Step 6: Finish the branch**

Use superpowers:finishing-a-development-branch. Present merge/PR options to the human (do NOT auto-merge without consent).

---

## Self-Review

**1. Spec coverage:**
- "新增 EntityConventionService.recount_project" → Task 1. ✓
- "按当前 ACCEPTED 标注（全 accepted、含 arbiter、一任务一票）重算" → Task 1 Step 3 vote model (all ACCEPTED, `max(types)` one-vote-per-task). ✓
- "重算 entity_type/dominant_type/distinct_task_count/dispute_count/dispute_pct 列" → Task 1 UPDATE statement writes exactly these five columns + `updated_at`. ✓
- "保留 operator/HR 锁（created_by 带 operator 前缀或 status=disputed 不覆盖 entity_type、不动注入 bypass）" → Task 1 `operator_locked` branch keeps `entity_type`, never touches `created_by`; `test_recount_preserves_operator_lock` asserts the bypass still injects. ✓
- "不重建 proposals_json（cosmetic）" → method docstring + UPDATE omits `proposals_json`. ✓
- "不删任何行（饱和清理已 defer）" → no DELETE/INSERT; `test_recount_zeroes_vanished_span_without_deleting` asserts the row survives. ✓
- "接入 build_posterior_audit 在 stats recount 之后、convention_index 构建之前" → Task 2 Step 3 placement. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"; all code blocks are complete and runnable.

**3. Type consistency:** Method name `recount_project(*, project_id)` is consistent across Task 1 (definition), Task 2 (call site), Task 3 (verification). Return keys (`conventions_seen`, `recomputed`, `operator_preserved`, `zeroed`) match between the implementation and the Task 3 expected summary. Columns written match the `_INJECT_COLUMNS`/`_load_row_light` set. `OPERATOR_DECLARATION_SOURCE_PREFIXES` referenced via the class attribute (matches the source).

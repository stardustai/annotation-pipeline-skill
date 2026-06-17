# Fully Recount-Only Conventions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `entity_conventions` empirical columns (`distinct_task_count`, `dispute_count`, `dispute_pct`, `dominant_type`) and the auto-derived `entity_type` maintained SOLELY by `recount_project`, so a recount fix (e.g. `vue→technology`) is durable and never clobbered by the live `record_decision` path.

**Architecture (Option A — chosen by the user):** `record_decision` and `clear_dispute` stop computing/writing the four empirical columns and stop auto-deriving `entity_type` from the distinct-task plurality. They still append proposals (audit trail), bump `evidence_count`, set `updated_at`, and — for OPERATOR actions (`declared:` / dispute resolution) — set `entity_type` to the explicitly chosen type + stamp `created_by` + manage `status`. Operator-declared conventions keep injecting immediately via the `created_by` bypass. AUTO (`qc_consensus`) conventions only become injectable after a recount. `check_past_experience` is switched to read the four aggregates from the materialized columns (not recomputed from `proposals_json`) so the MCP tool agrees with the injection gate.

**Behavioral tradeoff (accepted):** live auto-learning of conventions during a run is replaced by batch updates at Re-count time. A span first seen mid-run won't inject until the operator runs Re-count. Operator declarations are unaffected (immediate via bypass). This mirrors what was already done for `entity_statistics` (recount-only).

**Tech Stack:** Python 3.13, sqlite3 (WAL), pytest. No new dependencies.

---

## Background the implementer needs

- **The columns** live on `entity_conventions`: `entity_type` (the injected type), `status` ('active'/'disputed'), `evidence_count` (display counter = number of proposals), `proposals_json` (audit trail), and the four empirical columns `distinct_task_count, dispute_count, dispute_pct, dominant_type`.
- **`recount_project(*, project_id)`** (already implemented in `annotation_pipeline_skill/services/entity_convention_service.py`) is the recount-only maintainer: it recomputes the four columns + auto `entity_type` from current ACCEPTED annotations, counting **agreement tasks only** — it EXCLUDES tasks whose effective annotation is an arbiter override (`corrected_annotation`; see `_effective_annotation_is_arbiter_correction`), while INCLUDING uncontested, annotator-wins, and HR-final tasks. (This is intentionally stricter than `EntityStatisticsService.recount_project`, which is all-accepted.) Operator-locked rows (`created_by` starts with an `OPERATOR_DECLARATION_SOURCE_PREFIXES` entry, or `status='disputed'`) keep their `entity_type`. THIS METHOD IS NOT CHANGED by this plan — it already has the correct population.
- **The injection gate** (`_iter_injection_candidates` → `find_matches_in_text`) reads ONLY columns: `status='active' AND ((distinct_task_count>=INJECT_MIN_DISTINCT_TASKS(5) AND dispute_pct<INJECT_MAX_DISPUTE_PCT(0.20)) OR created_by LIKE '<operator-prefix>%')`. So zeroed auto columns ⇒ no injection; operator `created_by` ⇒ injects regardless.
- **`_load_row`** already reads the four aggregates from columns (changed in the prior plan). No change needed.
- **`_distinct_task_tally(proposals)`** is the helper that derives `(dominant_type, distinct, dispute_count, dispute_pct)` from a proposals list. After this plan it is used ONLY by `recount_project`'s sibling code paths and `rebuild_from_accepted_tasks` (which is unchanged) — NOT by `record_decision`/`clear_dispute`/`check_past_experience`. Do not delete it.
- **Operator prefixes:** `EntityConventionService.OPERATOR_DECLARATION_SOURCE_PREFIXES` (the module tuple `_OPERATOR_DECLARATION_SOURCE_PREFIXES`). The helper `_is_operator_source(source)` tests a source string against them.
- **Git branch:** `feat/convention-distinct-task-gate`. Do NOT `git add -A`. Never commit `db.sqlite*`, `backups/`, `reports/`, `scratch/`, `docs/superpowers/plans/*.md`, `web/db.sqlite`. Subagents must NOT add a `Co-Authored-By` trailer.
- **Known pre-existing test failures (7, unrelated — do NOT try to fix):** `test_cli.py::test_cli_import_jsonl_prelabeled_creates_tasks_with_prelabel_metadata`, `test_cli.py::test_cli_import_omits_per_task_inline_schema`, `test_dashboard_api.py::test_dashboard_api_returns_config_files_and_can_update_allowed_yaml`, `test_dashboard_api_distribution.py::test_distribution_scan_populates_cache_and_get_returns_payload`, `test_local_cli_client.py::test_build_codex_command_includes_json_resume_and_model`, `test_local_cli_client.py::test_local_codex_client_propagates_continuity_handle`, `test_prior_verifier_integration.py::test_second_arbiter_verbatim_retry_exhausted_routes_to_hr_not_silent_drop`.

## File Structure

- **Modify** `annotation_pipeline_skill/services/entity_convention_service.py` — `record_decision` (INSERT + UPDATE) and `clear_dispute` stop maintaining the four columns + auto `entity_type`.
- **Modify** `annotation_pipeline_skill/llm/tools/check_past_experience.py` — read the four aggregates from columns.
- **Update tests** (uniform pattern, see Task 4) across: `tests/test_convention_aggregate_columns.py`, `tests/test_convention_distinct_task_gate.py`, `tests/test_convention_injection_prefilter.py`, `tests/test_convention_list_pagination.py`, `tests/test_convention_api_endpoint.py`, `tests/test_check_past_experience.py`.
- **Modify** comments/docstrings: module docstring + `sqlite_store.py` schema comment + service comments that claim record_decision maintains the columns.

---

### Task 1: `record_decision` stops maintaining empirical columns + auto entity_type

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py` (`record_decision`, the INSERT branch ~214-235 and the UPDATE branch ~254-310)
- Test: `tests/test_entity_convention_recount.py` (add a new-contract test)

- [ ] **Step 1: Write the failing new-contract test**

Append to `tests/test_entity_convention_recount.py` (the `_add_accepted_task` helper already exists there):

```python
def test_record_decision_does_not_maintain_empirical_columns(tmp_path):
    """Recount-only: an AUTO (qc_consensus) record_decision appends a proposal
    and bumps evidence_count, but does NOT write the empirical columns or
    auto-derive entity_type. Those stay zeroed until recount_project runs."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)

    # First auto proposal creates the row with zeroed aggregates.
    c = svc.record_decision(project_id="p", span="kafka", entity_type="technology",
                            source="qc_consensus", task_id="t1")
    assert c.evidence_count == 1
    assert c.distinct_task_count == 0
    assert c.dominant_type is None
    assert c.dispute_pct == 0.0

    # A conflicting auto proposal must NOT flip entity_type or move dispute
    # stats (no live soft-model recompute anymore).
    c = svc.record_decision(project_id="p", span="kafka", entity_type="project",
                            source="qc_consensus", task_id="t2")
    assert c.evidence_count == 2            # proposals still tracked
    assert c.entity_type == "technology"    # unchanged (NOT re-derived to plurality)
    assert c.distinct_task_count == 0       # still not maintained here
    assert c.dominant_type is None

    # Operator declaration still takes effect immediately (type + bypass).
    c = svc.record_decision(project_id="p", span="kafka", entity_type="product",
                            source="declared:op-9", task_id=None)
    assert c.entity_type == "product"
    assert c.created_by.startswith("declared:")


def test_record_decision_then_recount_populates_columns(tmp_path):
    """After auto proposals (which don't maintain columns), recount_project is
    what makes the convention injectable."""
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    for i in range(5):
        svc.record_decision(project_id="p", span="kafka", entity_type="technology",
                            source="qc_consensus", task_id=f"seed-{i}")
        _add_accepted_task(store, "p", f"kafka-{i}", {"technology": ["kafka"]})
    # Before recount: not injectable (columns zeroed).
    assert svc.find_matches_in_text("p", "we run kafka here") == []
    # After recount: injectable.
    svc.recount_project(project_id="p")
    matches = svc.find_matches_in_text("p", "we run kafka here")
    assert any(m.span_lower == "kafka" and m.entity_type == "technology" for m in matches)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_entity_convention_recount.py::test_record_decision_does_not_maintain_empirical_columns -v`
Expected: FAIL — current `record_decision` writes `distinct_task_count=1`/`dominant_type='technology'` on insert and flips `entity_type` to the plurality on the conflicting proposal.

- [ ] **Step 3: Change the INSERT branch**

In `record_decision`, the `if row is None:` branch currently computes `dom0, dist0, disp0, pct0 = _distinct_task_tally([proposal])` and inserts those. Replace it so the four empirical columns are seeded to `0/0/0.0/None` and the tally line is removed:

```python
        if row is None:
            conv_id = f"conv-{uuid4().hex[:16]}"
            # Recount-only: the empirical columns (distinct_task_count,
            # dispute_count, dispute_pct, dominant_type) and the auto entity_type
            # are maintained SOLELY by recount_project. A new convention starts
            # with zeroed aggregates, so an AUTO convention won't inject until
            # the next recount. Operator-declared conventions still inject
            # immediately via the created_by bypass (created_by=source below).
            conn.execute(
                """
                INSERT INTO entity_conventions
                (convention_id, project_id, span_lower, span_original, entity_type,
                 status, evidence_count, proposals_json, created_at, updated_at,
                 created_by, notes,
                 distinct_task_count, dispute_count, dispute_pct, dominant_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conv_id, project_id, span_lower, span.strip(), entity_type,
                    "active", 1, json.dumps([proposal]),
                    now.isoformat(), now.isoformat(), source, notes,
                    0, 0, 0.0, None,
                ),
            )
            return self._load_row(conn.execute(
                "SELECT * FROM entity_conventions WHERE convention_id=?", (conv_id,)
            ).fetchone())
```

- [ ] **Step 4: Change the UPDATE branch**

In `record_decision`, after `proposals.append(proposal)`, the code currently does `dominant_type, distinct_ct, dispute_ct, dispute_pct = _distinct_task_tally(proposals)` and the soft-model `else` branch sets `new_type = dominant_type`. Replace the block from `proposals.append(proposal)` through the `conn.execute(UPDATE ...)` with:

```python
        proposals.append(proposal)
        # evidence_count is a plain display counter (total proposals). The
        # empirical columns are NOT maintained here anymore — recount_project
        # owns them (recount-only model).
        new_count = len(proposals)
        if source.startswith("declared:"):
            # Explicit operator declaration wins: clears dispute, locks the type.
            new_status = "active"
            new_type = entity_type
        elif row["status"] == "disputed":
            # Operator-disputed stays disputed until the operator clears it;
            # automated proposals only append evidence.
            new_status = "disputed"
            new_type = row["entity_type"]
        else:
            # Recount-only: auto (qc_consensus) proposals and operator locks
            # already in the chain do NOT re-derive the type here. entity_type
            # for auto conventions is set by recount_project; operator locks
            # keep their type (created_by stays operator-stamped). No live
            # soft-model plurality recompute.
            new_status = "active"
            new_type = row["entity_type"]
        if _is_operator_source(source):
            new_created_by = source
        else:
            new_created_by = row["created_by"]
        conn.execute(
            """
            UPDATE entity_conventions
            SET entity_type=?, status=?, evidence_count=?, proposals_json=?,
                created_by=?, updated_at=?
            WHERE convention_id=?
            """,
            (new_type, new_status, new_count, json.dumps(proposals),
             new_created_by, now.isoformat(),
             row["convention_id"]),
        )
        return self._load_row(conn.execute(
            "SELECT * FROM entity_conventions WHERE convention_id=?",
            (row["convention_id"],),
        ).fetchone())
```

Note: the UPDATE no longer sets `distinct_task_count, dispute_count, dispute_pct, dominant_type` (they keep whatever recount last wrote, or the insert's zeros). The idempotent re-click guard above this block is unchanged.

- [ ] **Step 5: Run the new-contract tests and the recount tests**

Run: `python -m pytest tests/test_entity_convention_recount.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 6: Repair the directly-targeted unit tests**

These tests in `tests/test_convention_aggregate_columns.py` and `tests/test_convention_distinct_task_gate.py` assert the OLD contract (record_decision maintains columns / auto-derives entity_type). Insert `svc.recount_project(project_id=<pid>)` (using the same project id and store/service the test already builds; for tests needing accepted-task backing, add accepted tasks with the existing `_add_accepted_task`-style helper or `rebuild_from_accepted_tasks` where the test already seeds them) immediately before the broken assertions, OR convert the seed to go through `recount_project`. Specifically repair:

`tests/test_convention_aggregate_columns.py`:
- `test_insert_stores_aggregate_columns` — after the `record_decision` seed, the columns are now 0/None on insert. Reframe: assert the row exists with zeroed aggregates after insert, then (if the test's intent is to verify population) back the span with accepted tasks and `recount_project`, then assert `distinct_task_count==1`, `dominant_type=='technology'`.
- `test_update_path_keeps_columns_in_sync_with_tally` — rename intent: the update path NO LONGER syncs columns. Reframe to assert that after auto proposals the columns are unchanged (zeroed), and only after `recount_project` (with accepted-task backing) do they match the expected tally.
- `test_injection_path_does_not_read_proposals_json` — back the span with ≥6 accepted tasks and call `recount_project` before asserting `c.distinct_task_count == 6` and injection.
- `test_migration_backfills_columns_via_replay_not_silently_zero` — this asserts a count of 7 after a write; under recount-only that population happens at recount, so back with accepted tasks + `recount_project`, then assert 7. (`test_migration_is_idempotent` is unaffected — keep.)

`tests/test_convention_distinct_task_gate.py`:
- `test_load_row_attaches_derived_fields` — back the 3 votes with accepted tasks and `recount_project`, then assert `distinct_task_count==3`, `dominant_type=='organization'`, `dispute_count==1`, `dispute_pct==1/3`.
- `test_conflict_does_not_flip_to_disputed_soft_model` — after recording conflicting auto votes, call `recount_project` (with accepted-task backing reflecting the plurality), then assert `status=='active'` and `entity_type=='organization'` (recount derives plurality). The soft-model invariant (never hard-flips to 'disputed' on auto conflict) now holds via recount.
- `test_injection_requires_five_distinct_tasks` — back each step with accepted tasks and call `recount_project` after seeding before asserting injection on/off at the 5-task boundary.
- `test_injection_blocked_when_dispute_pct_too_high` and `test_injection_allowed_when_dispute_pct_under_threshold` — back with accepted tasks + `recount_project`, then assert the gate.
- KEEP unchanged (still pass): `test_evidence_count_tracks_total_proposals`, `test_operator_declaration_still_wins`, `test_operator_declared_bypasses_distinct_task_gate`, `test_operator_declaration_is_sticky_against_later_consensus`.

Run after edits: `python -m pytest tests/test_convention_aggregate_columns.py tests/test_convention_distinct_task_gate.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py tests/test_entity_convention_recount.py tests/test_convention_aggregate_columns.py tests/test_convention_distinct_task_gate.py
git commit -m "feat: record_decision stops maintaining convention empirical columns (recount-only)"
```

---

### Task 2: `clear_dispute` stops maintaining empirical columns

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py` (`clear_dispute`, ~328-372)
- Test: `tests/test_convention_aggregate_columns.py::test_clear_dispute_refreshes_columns`

- [ ] **Step 1: Update the test to the new contract**

`test_clear_dispute_refreshes_columns` currently asserts `clear_dispute` refreshes `distinct_task_count`/`dispute_count`/`dominant_type` from the tally. Reframe it to `test_clear_dispute_sets_type_without_maintaining_columns`: assert that after `clear_dispute(resolved_type=...)` the convention is `status=='active'`, `entity_type==resolved_type`, `created_by` starts with `dispute_resolved_by:`, and that the empirical columns are NOT recomputed by clear_dispute (they keep their prior value — e.g. still 0/None if no recount has run). Add a follow-up: after `recount_project` (with accepted-task backing) the columns reflect reality.

```python
def test_clear_dispute_sets_type_without_maintaining_columns(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    # Create + operator-dispute a convention.
    conv = svc.record_decision(project_id="p", span="spark", entity_type="technology",
                               source="qc_consensus", task_id="t1")
    # Force it disputed via an operator (simulate the operator dispute path).
    svc.store._conn.execute(
        "UPDATE entity_conventions SET status='disputed' WHERE convention_id=?",
        (conv.convention_id,))
    resolved = svc.clear_dispute(convention_id=conv.convention_id,
                                 resolved_type="organization", actor="op-1")
    assert resolved.status == "active"
    assert resolved.entity_type == "organization"
    assert resolved.created_by.startswith("dispute_resolved_by:")
    # clear_dispute does NOT recompute empirical columns (recount-only).
    assert resolved.distinct_task_count == 0
    assert resolved.dominant_type is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_convention_aggregate_columns.py::test_clear_dispute_sets_type_without_maintaining_columns -v`
Expected: FAIL — current `clear_dispute` writes the recomputed columns (so `distinct_task_count`/`dominant_type` would be non-zero from the single proposal tally).

- [ ] **Step 3: Change `clear_dispute`**

Remove the `_distinct_task_tally` call and drop the four columns from its UPDATE. The method should append the resolution proposal, set `entity_type=resolved_type`, `status='active'`, stamp `created_by=f"dispute_resolved_by:{actor}"`, and `updated_at` — nothing else:

```python
        proposals = json.loads(row["proposals_json"] or "[]")
        proposals.append({
            "type": resolved_type,
            "source": f"dispute_resolved_by:{actor}",
            "notes": notes,
            "at": now.isoformat(),
        })
        # Recount-only: clear_dispute resolves the operator dispute (type +
        # status + created_by stamp) but does NOT recompute the empirical
        # columns — recount_project owns those.
        conn.execute(
            """
            UPDATE entity_conventions
            SET entity_type=?, status='active', proposals_json=?,
                created_by=?, updated_at=?
            WHERE convention_id=?
            """,
            (resolved_type, json.dumps(proposals), f"dispute_resolved_by:{actor}",
             now.isoformat(), convention_id),
        )
        return self._load_row(conn.execute(
            "SELECT * FROM entity_conventions WHERE convention_id=?",
            (convention_id,),
        ).fetchone())
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/test_convention_aggregate_columns.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py tests/test_convention_aggregate_columns.py
git commit -m "feat: clear_dispute stops maintaining convention empirical columns (recount-only)"
```

---

### Task 3: `check_past_experience` reads aggregates from columns

**Files:**
- Modify: `annotation_pipeline_skill/llm/tools/check_past_experience.py`
- Test: `tests/test_check_past_experience.py::test_conflicting_proposals_track_dispute_soft_model`

- [ ] **Step 1: Update the test**

`test_conflicting_proposals_track_dispute_soft_model` asserts the response's `dominant_type`/`dispute_count`/`dispute_pct` reflect conflicting auto proposals. Under recount-only these come from columns, so the test must back the span with accepted tasks and call `recount_project` before calling `check_past_experience`. Reframe it to seed the conflicting decisions via accepted tasks + `recount_project`, then assert the response's `convention.dominant_type`/`dispute_count`/`dispute_pct` match the recounted values. (`test_active_convention_returns_examples` keeps passing — `distribution`/`evidence_count` are still proposals/counter-based.)

```python
def test_conflicting_proposals_track_dispute_via_recount(tmp_path):
    import json
    from annotation_pipeline_skill.core.models import ArtifactRef, Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.services.entity_convention_service import (
        EntityConventionService,
    )
    from annotation_pipeline_skill.llm.tools.check_past_experience import (
        check_past_experience,
    )

    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)

    def _accept(tid, etype):
        t = Task.new(task_id=tid, pipeline_id="p",
                     source_ref={"kind": "jsonl", "payload": {
                         "rows": [{"row_index": 0, "input": "x"}]}})
        t.status = TaskStatus.ACCEPTED
        store.save_task(t)
        rel = f"artifact_payloads/{tid}/final.json"
        ap = store.root / rel
        ap.parent.mkdir(parents=True, exist_ok=True)
        ap.write_text(json.dumps({"text": json.dumps({
            "rows": [{"row_index": 0, "output": {"entities": {etype: ["Acme"]}}}]})}))
        store.append_artifact(ArtifactRef.new(
            task_id=tid, kind="annotation_result", path=rel,
            content_type="application/json"))

    # 2 organization, 1 product -> dominant organization, dispute 1/3.
    _accept("a", "organization"); _accept("b", "organization"); _accept("c", "product")
    svc.recount_project(project_id="p")

    result = check_past_experience(store, project_id="p", entry="Acme")
    conv = result["convention"]
    assert conv["dominant_type"] == "organization"
    assert conv["distinct_task_count"] == 3
    assert conv["dispute_count"] == 1
    assert abs(conv["dispute_pct"] - 1/3) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_check_past_experience.py::test_conflicting_proposals_track_dispute_via_recount -v`
Expected: FAIL — `check_past_experience` currently recomputes from `proposals_json` (which here is empty because the convention was created by `recount_project`, not `record_decision`), so `dominant_type`/`distinct_task_count` come back `None`/`0`.

- [ ] **Step 3: Change `check_past_experience` to read columns**

In `annotation_pipeline_skill/llm/tools/check_past_experience.py`:

(a) Extend the SELECT to include the four columns:
```python
    row = store._conn.execute(
        "SELECT convention_id, entity_type, status, evidence_count, proposals_json, "
        "distinct_task_count, dispute_count, dispute_pct, dominant_type "
        "FROM entity_conventions WHERE project_id=? AND span_lower=?",
        (project_id, span_lower),
    ).fetchone()
```

(b) Replace the tally recompute (the `dominant_type, distinct_tasks, dispute_count, dispute_pct = _distinct_task_tally(proposals)` line) with column reads:
```python
    proposals = json.loads(row["proposals_json"] or "[]")
    # Recount-only: headline aggregates come from the materialized columns
    # (maintained by recount_project), so this tool agrees with the injection
    # gate. proposals_json is still used below for distribution + examples.
    dominant_type = row["dominant_type"]
    distinct_tasks = row["distinct_task_count"]
    dispute_count = row["dispute_count"]
    dispute_pct = row["dispute_pct"]
```

(c) Remove the now-unused import `from annotation_pipeline_skill.services.entity_convention_service import (_distinct_task_tally,)` at the top of the file (lines 14-16). Verify nothing else in the file references `_distinct_task_tally`.

The `distribution` (Counter over proposals) and `examples_by_type` (from proposal context snippets) stay proposals-based — unchanged.

- [ ] **Step 4: Run the test file**

Run: `python -m pytest tests/test_check_past_experience.py -q`
Expected: PASS (the new test plus the unaffected ones).

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/llm/tools/check_past_experience.py tests/test_check_past_experience.py
git commit -m "feat: check_past_experience reads convention aggregates from columns (recount-only)"
```

---

### Task 4: Repair seed-without-recount tests (injection prefilter, pagination, API endpoint)

**Files:**
- Test: `tests/test_convention_injection_prefilter.py`, `tests/test_convention_list_pagination.py`, `tests/test_convention_api_endpoint.py`

These tests seed auto conventions via `record_decision` and then assert on the columns / injection / SQL filters that read the columns. Under recount-only the columns are zeroed until a recount. The uniform fix: **back the seeded spans with ACCEPTED tasks and call `svc.recount_project(project_id=<pid>)` after seeding and before the assertions.** Use the project id each test already uses.

- [ ] **Step 1: Add a shared helper if absent**

If a test file lacks an accepted-task helper, add this near the top (adapt the entities mapping per test):
```python
import json
from annotation_pipeline_skill.core.models import ArtifactRef, Task
from annotation_pipeline_skill.core.states import TaskStatus

def _accept(store, pid, tid, entities_by_type):
    t = Task.new(task_id=tid, pipeline_id=pid,
                 source_ref={"kind": "jsonl", "payload": {
                     "rows": [{"row_index": 0, "input": "x"}]}})
    t.status = TaskStatus.ACCEPTED
    store.save_task(t)
    rel = f"artifact_payloads/{tid}/final.json"
    ap = store.root / rel
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text(json.dumps({"text": json.dumps({
        "rows": [{"row_index": 0, "output": {"entities": entities_by_type}}]})}))
    store.append_artifact(ArtifactRef.new(
        task_id=tid, kind="annotation_result", path=rel,
        content_type="application/json"))
```

- [ ] **Step 2: Repair each test**

For each test below, replace the "seed N qc_consensus votes for span X" loop with "seed N accepted tasks tagging X (via `_accept`)" and then call `EntityConventionService(store).recount_project(project_id=<pid>)` before the assertion. Where the test ALSO needs a convention row to exist with operator metadata, keep the `record_decision` call too (recount updates the existing row's columns).

- `tests/test_convention_injection_prefilter.py`: `test_prefilter_drops_singletons_but_keeps_eligible`, `test_prefilter_matches_full_scan_injection_result`.
- `tests/test_convention_list_pagination.py`: `test_min_count_filter_pushed_to_sql`, `test_ordered_by_distinct_task_count_desc`.
- `tests/test_convention_api_endpoint.py`: `test_full_mode_returns_all_rows_with_proposals` (the `distinct_task_count == 6` assertion). `test_default_mode_is_paginated_and_proposals_free` is unaffected — keep.

- [ ] **Step 3: Run the three files**

Run: `python -m pytest tests/test_convention_injection_prefilter.py tests/test_convention_list_pagination.py tests/test_convention_api_endpoint.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_convention_injection_prefilter.py tests/test_convention_list_pagination.py tests/test_convention_api_endpoint.py
git commit -m "test: back convention column assertions with recount (recount-only)"
```

---

### Task 5: Update docstrings/comments to the recount-only contract

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py` (module docstring lines 1-16; the `_iter_injection_candidates` comment ~670 and `_load_row_light` comment ~783 that say "maintained on write by record_decision/clear_dispute"); `annotation_pipeline_skill/store/sqlite_store.py` (schema comment ~line 92 "record_decision / clear_dispute" maintenance claim).

- [ ] **Step 1: Update the module docstring**

In `entity_convention_service.py`, the module docstring describes the "soft dispute model" as enforced live by record_decision. Add/adjust a sentence to state that the empirical columns (`distinct_task_count`, `dispute_count`, `dispute_pct`, `dominant_type`) and the auto-derived `entity_type` are maintained SOLELY by `recount_project` (recount-only); `record_decision`/`clear_dispute` only append proposals, bump `evidence_count`, and apply operator declarations.

- [ ] **Step 2: Fix the two service comments**

Change the comments in `_iter_injection_candidates` (the parenthetical "maintained on write by `record_decision` / `clear_dispute`") and in `_load_row_light` (the "`record_decision`/`clear_dispute` keep the columns in sync with the tally" comment) to say the columns are maintained by `recount_project` (recount-only); `record_decision`/`clear_dispute` no longer maintain them.

- [ ] **Step 3: Fix the sqlite_store schema comment**

In `annotation_pipeline_skill/store/sqlite_store.py` ~line 92, the comment says these columns are maintained "(record_decision / clear_dispute)". Change it to "(maintained by EntityConventionService.recount_project — recount-only)".

- [ ] **Step 4: Verify nothing references removed behavior**

Run: `grep -rn "record_decision / clear_dispute\|keep the columns in sync\|maintained on write by" annotation_pipeline_skill/`
Expected: no stale claims that record_decision/clear_dispute maintain the empirical columns.

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py annotation_pipeline_skill/store/sqlite_store.py
git commit -m "docs: conventions are recount-only; columns maintained by recount_project"
```

---

### Task 6: Regression sweep + real-store sanity + final review

- [ ] **Step 1: Full unit suite**

Run: `python -m pytest -q`
Expected: only the 7 KNOWN pre-existing failures (listed in Background); NO new failures. All convention/recount/check_past_experience tests pass.

- [ ] **Step 2: Grep for orphaned tally usage**

Run: `grep -rn "_distinct_task_tally" annotation_pipeline_skill/`
Expected: referenced only inside `entity_convention_service.py` (its definition + any recount-adjacent use) and NOT in `check_past_experience.py`. Confirm `record_decision`/`clear_dispute` no longer call it.

- [ ] **Step 3: Real-store sanity (read-only first)**

Confirm the durability fix: a recount result is no longer clobbered by a subsequent auto `record_decision`. On a scratch copy is unnecessary — use the real store read-only probe:
```bash
.venv/bin/python - <<'PY'
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.services.entity_convention_service import EntityConventionService
store = SqliteStore.open("projects/v4_ner_phrase/.annotation-pipeline")
svc = EntityConventionService(store)
vue = [c for c in svc.list_for_project("v4_ner_phrase") if c.span_lower=="vue"][0]
print("vue before:", vue.entity_type, vue.dominant_type, vue.distinct_task_count)
# Simulate a stray auto proposal on vue (the kind that used to clobber):
svc.record_decision(project_id="v4_ner_phrase", span="vue", entity_type="project",
                    source="qc_consensus", task_id="probe-does-not-exist")
vue2 = [c for c in svc.list_for_project("v4_ner_phrase") if c.span_lower=="vue"][0]
print("vue after stray auto proposal:", vue2.entity_type, vue2.dominant_type, vue2.distinct_task_count)
assert vue2.entity_type == "technology", "recount fix was clobbered!"
assert vue2.dominant_type == "technology", "dominant clobbered!"
print("DURABLE: stray auto proposal did NOT revert the recount fix")
PY
```
Expected: `vue` stays `technology` after the stray auto proposal (the whole point of this plan). NOTE: this writes one proposal to the real store's `vue` row (evidence_count +1, proposals_json appended) but must NOT change its columns. If you prefer zero writes, run it against a copied store.

- [ ] **Step 4: Dispatch the final code reviewer**

Review the full diff for this plan (`git log --oneline` to find Task 1's first commit; review `<that-commit>^..HEAD`). Confirm: record_decision/clear_dispute write no empirical columns; operator declarations still set type + bypass; check_past_experience reads columns; `_distinct_task_tally` not used by the live write/read paths; recount_project unchanged; no new full-suite failures. Address any Critical/Important findings under TDD.

- [ ] **Step 5: Finish the branch**

Use superpowers:finishing-a-development-branch. Present merge/PR options to the human (do NOT auto-merge without consent).

---

## Self-Review

**1. Spec coverage:**
- "record_decision stops maintaining empirical columns + auto entity_type" → Task 1 (INSERT zeroes columns; UPDATE drops the four columns + the soft-model plurality `new_type`). ✓
- "clear_dispute stops maintaining empirical columns" → Task 2. ✓
- "operator declarations stay live (type + bypass)" → Task 1 `declared:` branch keeps `entity_type=entity_type` + `created_by` stamp; KEEP tests `test_operator_declared_bypasses_distinct_task_gate` etc. ✓
- "recount becomes sole maintainer / fixes durable" → Task 6 Step 3 durability assertion. ✓
- "MCP tool agrees with gate" → Task 3 (`check_past_experience` reads columns). ✓
- "tests updated to new contract" → Tasks 1 Step 6, 2, 3, 4 (the full BREAKS list from the blast-radius map). ✓
- "docs reflect recount-only" → Task 5. ✓

**2. Placeholder scan:** Production-code steps show exact before/after. Test-repair Tasks 1.6 and 4 use one explicit uniform pattern (back with accepted tasks + `recount_project` before asserting) applied to a precisely enumerated list of test functions — no "TBD"/"handle edge cases".

**3. Type consistency:** `recount_project(*, project_id)` signature matches the existing implementation. The four column names (`distinct_task_count, dispute_count, dispute_pct, dominant_type`) are used identically across Tasks 1-3 and match the schema. `_is_operator_source` / `OPERATOR_DECLARATION_SOURCE_PREFIXES` match the source. `check_past_experience` return shape (`convention.dominant_type` etc.) is preserved — only the data source changes.

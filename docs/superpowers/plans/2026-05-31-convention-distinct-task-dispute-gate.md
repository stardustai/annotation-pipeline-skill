# Convention Distinct-Task + Soft Dispute Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the entity-convention injection gate from `evidence_count >= 5` (inflated by repeated same-type proposals) to a distinct-task vote model: inject only when `distinct_task_count >= 5 AND dispute_pct < 0.20`, and stop auto-flipping conventions to `disputed` on conflict (soft model — dominant type wins, dispute is tracked numerically).

**Architecture:** All vote aggregates (`dominant_type`, `distinct_task_count`, `dispute_count`, `dispute_pct`) are **derived from `proposals_json` at read time** — `proposals_json` already records `task_id` per proposal, making it the single source of truth. **No schema migration is added** (the codebase's additive-migration mechanism only supports `CREATE TABLE IF NOT EXISTS`, not column adds, and a cached aggregate would risk drifting from the proposal list). A new module-level helper `_distinct_task_tally()` computes the aggregates; `EntityConvention._load_row` attaches them as dataclass fields. `record_decision` keeps `entity_type` aligned to the plurality winner and never auto-sets `status='disputed'`. The dedup/idempotency key gains `task_id` so a new task with the same (source, type) is counted as a fresh vote rather than suppressed.

**Definition of "distinct task" (precise):** a distinct task is a unique `task_id` whose proposal came from the **three-party consensus path** (annotator + QC + prior verifier agree). In the pipeline this is exactly the `source="qc_consensus"` writer (`_record_conventions_from_qc_consensus`, gated on QC-pass + verifier-confirmed in `subagent_cycle.py`). Operator/HR proposals (`declared:`, `hr_correction:`, `dispute_resolved_by:`, `batch_operator_*`, `posterior_audit_operator`) are human overrides, **not** three-party consensus, so they are EXCLUDED from the distinct-task tally and from `dispute_pct` — they instead take effect through the operator-declared injection bypass. Therefore `_distinct_task_tally` counts only proposals that (a) have a `task_id` and (b) whose `source` is not an operator-declaration source.

**Tech Stack:** Python 3.11, sqlite3, pytest. Files are all under `annotation_pipeline_skill/` with tests under `tests/`.

**Scope note:** This change only affects how a *fresh* re-annotation project accumulates and injects conventions. The existing 10521-convention project will NOT be migrated — its data is intentionally abandoned. No backward-data-compat constraints apply, which is why `evidence_count`'s display meaning can be simplified.

---

## File Structure

- **Modify:** `annotation_pipeline_skill/services/entity_convention_service.py`
  - Add `_distinct_task_tally()` module helper (single source of vote math).
  - Add derived fields to `EntityConvention` dataclass + `to_dict()`.
  - Compute + attach derived fields in `_load_row()`.
  - Change `record_decision()` dedup key to `(source, type, task_id)` and switch to the soft model.
  - Replace the injection condition in `find_matches_in_text()` and add the two gate constants.
- **Modify:** `annotation_pipeline_skill/llm/tools/check_past_experience.py`
  - Surface `distinct_task_count`, `dispute_count`, `dispute_pct` in the `convention` block.
- **Modify (test updates):** `tests/test_check_past_experience.py`
  - `test_disputed_returns_examples_per_type` must reflect the soft model (no hard-flip).
- **Create (new tests):** `tests/test_convention_distinct_task_gate.py`
  - Tally helper, dedup-by-task, soft model, injection gate.

No other consumers change: `api.py` and `prompt_builder.py` read conventions through `list_for_project` / `find_matches_in_text` / `to_dict()`, all of which keep their existing keys and gain new ones additively.

---

### Task 1: Distinct-task tally helper

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py` (add helper after `_build_context_snippet`, ~line 50)
- Test: `tests/test_convention_distinct_task_gate.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_convention_distinct_task_gate.py`:

```python
import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
    _distinct_task_tally,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path):
    yield SqliteStore.open(tmp_path)


def _prop(task_id, ptype, source="qc_consensus"):
    return {"type": ptype, "source": source, "task_id": task_id}


def test_tally_counts_one_vote_per_distinct_task():
    # Same task proposes "technology" three times → one vote.
    proposals = [
        _prop("t1", "technology"),
        _prop("t1", "technology"),
        _prop("t1", "technology"),
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "technology"
    assert distinct == 1
    assert dispute == 0
    assert pct == 0.0


def test_tally_dominant_is_plurality_across_tasks():
    proposals = [
        _prop("t1", "organization"),
        _prop("t2", "organization"),
        _prop("t3", "product"),
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "organization"
    assert distinct == 3
    assert dispute == 1
    assert pct == pytest.approx(1 / 3)


def test_tally_uses_most_recent_type_per_task():
    # A task that changed its mind (later proposal wins for that task).
    proposals = [
        _prop("t1", "product"),
        _prop("t1", "technology"),  # later → t1 votes technology
        _prop("t2", "technology"),
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "technology"
    assert distinct == 2
    assert dispute == 0
    assert pct == 0.0


def test_tally_ignores_proposals_without_task_id():
    # Operator declarations (task_id=None) don't count as task votes.
    proposals = [
        {"type": "project", "source": "declared:operator", "task_id": None},
        _prop("t1", "project"),
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "project"
    assert distinct == 1
    assert dispute == 0


def test_tally_excludes_operator_source_even_with_task_id():
    # An HR correction carries a task_id but is a human override, NOT a
    # three-party consensus vote → excluded from the distinct-task tally.
    proposals = [
        _prop("t1", "organization"),
        _prop("t2", "organization"),
        {"type": "product", "source": "hr_correction:alice", "task_id": "t3"},
    ]
    dominant, distinct, dispute, pct = _distinct_task_tally(proposals)
    assert dominant == "organization"
    assert distinct == 2          # only the two qc_consensus tasks count
    assert dispute == 0
    assert pct == 0.0


def test_tally_empty_is_neutral():
    dominant, distinct, dispute, pct = _distinct_task_tally([])
    assert dominant is None
    assert distinct == 0
    assert dispute == 0
    assert pct == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convention_distinct_task_gate.py -k tally -v`
Expected: FAIL with `ImportError: cannot import name '_distinct_task_tally'`

- [ ] **Step 3: Write minimal implementation**

In `annotation_pipeline_skill/services/entity_convention_service.py`, add after the `_build_context_snippet` function (before the `EntityConvention` dataclass), and ensure `Counter` is imported at the top of the file (add `from collections import Counter` to the imports block near line 16).

First add a module-level constant for the operator/non-consensus source prefixes (it is the single source of truth; the `EntityConventionService` class attribute will reference it in Task 5):

```python
# Source prefixes that indicate a human override (operator / HR), NOT a
# three-party LLM consensus. These are EXCLUDED from the distinct-task tally
# (they don't count as "三方一致" consensus votes) and instead take effect via
# the operator-declared injection bypass.
_OPERATOR_DECLARATION_SOURCE_PREFIXES: tuple[str, ...] = (
    "declared:", "hr_correction:", "posterior_audit_operator",
    "batch_operator_resolve", "batch_operator_correction",
    "dispute_resolved_by:",
)


def _is_operator_source(source: Any) -> bool:
    return isinstance(source, str) and any(
        source.startswith(p) for p in _OPERATOR_DECLARATION_SOURCE_PREFIXES
    )


def _distinct_task_tally(
    proposals: list[dict[str, Any]],
) -> tuple[str | None, int, int, float]:
    """Aggregate three-party-consensus proposals into a one-vote-per-task tally.

    A "distinct task" is a unique ``task_id`` whose proposal came from the
    three-party consensus path (``source="qc_consensus"``: annotator + QC +
    prior verifier agree). Each such task contributes a SINGLE vote; that
    task's vote is the type of its MOST RECENT consensus proposal (later
    proposals overwrite earlier ones, so a task that changed its mind votes
    for its final answer).

    EXCLUDED from the tally:
      - proposals with no ``task_id`` (operator declarations, dispute
        resolutions), and
      - proposals from operator/HR sources (see
        ``_OPERATOR_DECLARATION_SOURCE_PREFIXES``) — those are human
        overrides, not three-party consensus, and take effect via the
        operator-declared injection bypass instead.

    Returns ``(dominant_type, distinct_task_count, dispute_count,
    dispute_pct)`` where:
      - ``dominant_type`` is the plurality winner across task votes (ties
        broken deterministically by the larger type string), or ``None``
        when there are no consensus task votes.
      - ``dispute_count`` is the number of consensus tasks whose vote !=
        dominant.
      - ``dispute_pct`` is ``dispute_count / distinct_task_count`` (0.0 when
        there are no consensus task votes).
    """
    votes: dict[str, str] = {}  # task_id -> most-recent consensus type
    for p in proposals:
        if not isinstance(p, dict):
            continue
        task_id = p.get("task_id")
        ptype = p.get("type")
        if not task_id or not isinstance(ptype, str):
            continue
        if _is_operator_source(p.get("source")):
            continue  # human override, not a three-party consensus vote
        votes[task_id] = ptype  # later proposals overwrite → most-recent wins
    distinct = len(votes)
    if distinct == 0:
        return (None, 0, 0, 0.0)
    tally = Counter(votes.values())
    # Plurality; deterministic tiebreak on (count, type) so equal counts
    # resolve consistently regardless of insertion order.
    dominant_type = max(tally.items(), key=lambda kv: (kv[1], kv[0]))[0]
    dispute_count = distinct - tally[dominant_type]
    return (dominant_type, distinct, dispute_count, dispute_count / distinct)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convention_distinct_task_gate.py -k tally -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py tests/test_convention_distinct_task_gate.py
git commit -m "feat(conventions): add distinct-task vote tally helper"
```

---

### Task 2: Expose derived fields on EntityConvention

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py` (dataclass ~52-80, `_load_row` ~339-353)
- Test: `tests/test_convention_distinct_task_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_convention_distinct_task_gate.py`:

```python
def test_load_row_attaches_derived_fields(store):
    svc = EntityConventionService(store)
    svc.record_decision(
        project_id="p1", span="Apple", entity_type="organization",
        source="qc_consensus:a", task_id="t1",
    )
    svc.record_decision(
        project_id="p1", span="Apple", entity_type="organization",
        source="qc_consensus:b", task_id="t2",
    )
    svc.record_decision(
        project_id="p1", span="Apple", entity_type="product",
        source="qc_consensus:c", task_id="t3",
    )
    conv = svc.list_for_project("p1")[0]
    assert conv.distinct_task_count == 3
    assert conv.dominant_type == "organization"
    assert conv.dispute_count == 1
    assert conv.dispute_pct == pytest.approx(1 / 3)
    d = conv.to_dict()
    assert d["distinct_task_count"] == 3
    assert d["dominant_type"] == "organization"
    assert d["dispute_count"] == 1
    assert d["dispute_pct"] == pytest.approx(1 / 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convention_distinct_task_gate.py::test_load_row_attaches_derived_fields -v`
Expected: FAIL with `AttributeError: 'EntityConvention' object has no attribute 'distinct_task_count'`

- [ ] **Step 3: Write minimal implementation**

In `entity_convention_service.py`, extend the `EntityConvention` dataclass (add fields after `notes`, all with defaults since `notes` has a default):

```python
@dataclass(frozen=True)
class EntityConvention:
    convention_id: str
    project_id: str
    span_lower: str
    span_original: str
    entity_type: str | None
    status: str   # 'active' or 'disputed'
    evidence_count: int
    proposals: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    created_by: str
    notes: str | None = None
    # Derived from ``proposals`` (one vote per distinct task). Not stored as
    # columns — computed in ``_load_row`` so ``proposals_json`` stays the
    # single source of truth.
    distinct_task_count: int = 0
    dispute_count: int = 0
    dispute_pct: float = 0.0
    dominant_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "convention_id": self.convention_id,
            "project_id": self.project_id,
            "span": self.span_original,
            "entity_type": self.entity_type,
            "status": self.status,
            "evidence_count": self.evidence_count,
            "proposals": self.proposals,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "notes": self.notes,
            "distinct_task_count": self.distinct_task_count,
            "dispute_count": self.dispute_count,
            "dispute_pct": self.dispute_pct,
            "dominant_type": self.dominant_type,
        }
```

Then update `_load_row` to compute and pass them:

```python
    def _load_row(self, row: sqlite3.Row) -> EntityConvention:
        proposals = json.loads(row["proposals_json"] or "[]")
        dominant_type, distinct_tasks, dispute_count, dispute_pct = (
            _distinct_task_tally(proposals)
        )
        return EntityConvention(
            convention_id=row["convention_id"],
            project_id=row["project_id"],
            span_lower=row["span_lower"],
            span_original=row["span_original"],
            entity_type=row["entity_type"],
            status=row["status"],
            evidence_count=row["evidence_count"],
            proposals=proposals,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            created_by=row["created_by"],
            notes=row["notes"],
            distinct_task_count=distinct_tasks,
            dispute_count=dispute_count,
            dispute_pct=dispute_pct,
            dominant_type=dominant_type,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convention_distinct_task_gate.py::test_load_row_attaches_derived_fields -v`
Expected: PASS

Note: this test currently relies on the soft model (Task 4) to keep `status='active'` and `entity_type='organization'` after the conflicting `product` proposal. With the OLD code still in place at this point, `_load_row`'s derived fields are computed from `proposals` directly and are independent of the stored `entity_type`/`status`, so the assertions on `dominant_type`/`dispute_*` pass regardless. The test does not assert `conv.entity_type` here.

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py tests/test_convention_distinct_task_gate.py
git commit -m "feat(conventions): expose distinct-task derived fields on EntityConvention"
```

---

### Task 3: Add task_id to the dedup/idempotency key

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py` (`record_decision` no-op check ~152-158)
- Test: `tests/test_convention_distinct_task_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_convention_distinct_task_gate.py`:

```python
def _proposals(store):
    import json
    rows = list(store._conn.execute("SELECT proposals_json FROM entity_conventions"))
    return json.loads(rows[0][0])


def test_same_task_same_source_type_is_noop(store):
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")  # exact repeat → no-op
    assert len(_proposals(store)) == 1


def test_different_task_same_source_type_is_recorded(store):
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t2")  # new task → new vote
    proposals = _proposals(store)
    assert len(proposals) == 2
    conv = svc.list_for_project("p1")[0]
    assert conv.distinct_task_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convention_distinct_task_gate.py -k "noop or different_task" -v`
Expected: `test_different_task_same_source_type_is_recorded` FAILS — under the current `(type, source)` dedup key the second call is suppressed as a no-op, so `len(proposals) == 1` (not 2).

- [ ] **Step 3: Write minimal implementation**

In `record_decision`, change the no-op check (currently ~152-158) to include `task_id`:

```python
        last = proposals[-1] if proposals else None
        if (
            isinstance(last, dict)
            and last.get("type") == entity_type
            and last.get("source") == source
            and last.get("task_id") == task_id
        ):
            return self._load_row(row)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convention_distinct_task_gate.py -k "noop or different_task" -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py tests/test_convention_distinct_task_gate.py
git commit -m "feat(conventions): include task_id in record_decision dedup key"
```

---

### Task 4: Soft dispute model in record_decision

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py` (`record_decision` state machine ~159-189, docstring ~99-104)
- Test: `tests/test_convention_distinct_task_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_convention_distinct_task_gate.py`:

```python
def test_conflict_does_not_flip_to_disputed_soft_model(store):
    svc = EntityConventionService(store)
    # Two tasks say organization, one says product → dominant=organization.
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t2")
    conv = svc.record_decision(project_id="p1", span="Apple", entity_type="product",
                              source="qc_consensus", task_id="t3")
    # Soft model: stays active, entity_type tracks the plurality winner.
    assert conv.status == "active"
    assert conv.entity_type == "organization"
    assert conv.dominant_type == "organization"
    assert conv.dispute_count == 1


def test_evidence_count_tracks_total_proposals(store):
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    conv = svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                              source="qc_consensus", task_id="t2")
    assert conv.evidence_count == 2  # two recorded proposals


def test_operator_declaration_still_wins(store):
    svc = EntityConventionService(store)
    svc.record_decision(project_id="p1", span="Apple", entity_type="organization",
                        source="qc_consensus", task_id="t1")
    conv = svc.record_decision(project_id="p1", span="Apple", entity_type="product",
                              source="declared:operator")
    # Operator declaration is final authority: type locks to declared value.
    assert conv.status == "active"
    assert conv.entity_type == "product"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convention_distinct_task_gate.py -k "soft_model or evidence_count_tracks or operator_declaration" -v`
Expected: `test_conflict_does_not_flip_to_disputed_soft_model` FAILS (old code sets `status='disputed'`, `entity_type=None`). `test_evidence_count_tracks_total_proposals` likely FAILS (old code only bumps `evidence_count` on same-type match — here both are organization so it would be 2, may pass; the soft model makes this deterministic regardless of type).

- [ ] **Step 3: Write minimal implementation**

Replace the `record_decision` state machine block (currently ~159-189, from `proposals.append(proposal)` through the `UPDATE`) with the soft model:

```python
        proposals.append(proposal)
        dominant_type, _distinct, _dispute, _pct = _distinct_task_tally(proposals)
        # evidence_count is now a plain display counter: the total number of
        # recorded proposals. The injection gate uses distinct_task_count /
        # dispute_pct (derived in _load_row), NOT this field.
        new_count = len(proposals)
        if source.startswith("declared:"):
            # Explicit operator declaration is the final authority: it always
            # wins, clears any prior dispute, and locks in the chosen type.
            new_status = "active"
            new_type = entity_type
        elif row["status"] == "disputed":
            # A convention disputed by an operator stays disputed until the
            # operator clears it; automated proposals only append evidence.
            new_status = "disputed"
            new_type = row["entity_type"]
        else:
            # Soft model: automated conflicts NEVER hard-flip to 'disputed'.
            # The plurality winner (one vote per distinct task) becomes the
            # convention's current type; disagreement is tracked numerically
            # via dispute_pct and enforced softly at injection time.
            new_status = "active"
            new_type = dominant_type if dominant_type is not None else row["entity_type"]
        conn.execute(
            """
            UPDATE entity_conventions
            SET entity_type=?, status=?, evidence_count=?, proposals_json=?, updated_at=?
            WHERE convention_id=?
            """,
            (new_type, new_status, new_count, json.dumps(proposals),
             now.isoformat(), row["convention_id"]),
        )
        return self._load_row(conn.execute(
            "SELECT * FROM entity_conventions WHERE convention_id=?",
            (row["convention_id"],),
        ).fetchone())
```

Also update the `record_decision` docstring (~99-104) to describe the soft model:

```python
        """Upsert a convention. Rules:
        - first time → insert as 'active'
        - automated proposals (qc_consensus etc.) → append a proposal and set
          entity_type to the plurality winner across distinct tasks. Conflicts
          do NOT flip the convention to 'disputed' (soft model); disagreement
          is tracked numerically (dispute_pct) and enforced at injection time.
        - explicit operator declaration ("declared:...") → wins, locks the
          chosen type, clears any prior dispute.
        - already operator-'disputed' → append the proposal but leave the
          status alone until an operator clears it.
        """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convention_distinct_task_gate.py -k "soft_model or evidence_count_tracks or operator_declaration" -v`
Expected: PASS (3)

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py tests/test_convention_distinct_task_gate.py
git commit -m "feat(conventions): soft dispute model — plurality wins, no auto-flip to disputed"
```

---

### Task 5: Distinct-task + dispute injection gate

**Files:**
- Modify: `annotation_pipeline_skill/services/entity_convention_service.py` (constants ~259-266, `find_matches_in_text` ~299-337)
- Test: `tests/test_convention_distinct_task_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_convention_distinct_task_gate.py`:

```python
def _seed_votes(svc, span, etype, n, project="p1", start=0):
    for i in range(start, start + n):
        svc.record_decision(project_id=project, span=span, entity_type=etype,
                            source="qc_consensus", task_id=f"t{i}",
                            row_content=f"{span} appears here {i}")


def test_injection_requires_five_distinct_tasks(store):
    svc = EntityConventionService(store)
    _seed_votes(svc, "Salesforce", "organization", 4)  # 4 < 5 distinct tasks
    assert svc.find_matches_in_text("p1", "We use Salesforce daily") == []
    _seed_votes(svc, "Salesforce", "organization", 1, start=4)  # now 5
    matches = svc.find_matches_in_text("p1", "We use Salesforce daily")
    assert [c.span_original for c in matches] == ["Salesforce"]


def test_injection_blocked_when_dispute_pct_too_high(store):
    svc = EntityConventionService(store)
    # 6 distinct tasks, 2 dissent → dispute_pct = 2/6 = 0.333 >= 0.20 → blocked.
    _seed_votes(svc, "Mercury", "organization", 4)
    _seed_votes(svc, "Mercury", "product", 2, start=4)
    conv = svc.list_for_project("p1")[0]
    assert conv.distinct_task_count == 6
    assert conv.dispute_pct >= 0.20
    assert svc.find_matches_in_text("p1", "Mercury launched a probe") == []


def test_injection_allowed_when_dispute_pct_under_threshold(store):
    svc = EntityConventionService(store)
    # 10 distinct tasks, 1 dissent → dispute_pct = 0.10 < 0.20 → injected.
    _seed_votes(svc, "Mercury", "organization", 9)
    _seed_votes(svc, "Mercury", "product", 1, start=9)
    conv = svc.list_for_project("p1")[0]
    assert conv.distinct_task_count == 10
    assert conv.dispute_pct < 0.20
    matches = svc.find_matches_in_text("p1", "Mercury launched a probe")
    assert [c.span_original for c in matches] == ["Mercury"]


def test_operator_declared_bypasses_distinct_task_gate(store):
    svc = EntityConventionService(store)
    # One operator declaration, zero distinct task votes → still injected.
    svc.record_decision(project_id="p1", span="Gmail", entity_type="project",
                        source="declared:operator", row_content="Gmail filters")
    matches = svc.find_matches_in_text("p1", "I set up a Gmail filter")
    assert [c.span_original for c in matches] == ["Gmail"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convention_distinct_task_gate.py -k "injection or bypasses" -v`
Expected: `test_injection_requires_five_distinct_tasks` and `test_injection_blocked_when_dispute_pct_too_high` FAIL — the old gate keys off `evidence_count >= 5` (which now equals total proposals), so it injects after 4 tasks (no), and does not consider `dispute_pct` at all.

- [ ] **Step 3: Write minimal implementation**

In `entity_convention_service.py`, add the two new constants alongside the existing ones (after `MIN_INJECTION_EVIDENCE`, ~266) and remove the now-unused evidence threshold from the gate. Add:

```python
    # Injection gate (replaces the old evidence_count >= MIN_INJECTION_EVIDENCE
    # rule). An auto-accumulated convention is injected only once enough
    # DISTINCT tasks have voted for it AND the cross-task disagreement is low.
    # One task = one vote (see _distinct_task_tally). Operator-declared
    # conventions bypass both thresholds.
    INJECT_MIN_DISTINCT_TASKS = 5
    INJECT_MAX_DISPUTE_PCT = 0.20
```

Also de-duplicate the operator-source prefixes: Task 1 added a module-level `_OPERATOR_DECLARATION_SOURCE_PREFIXES`. Change the existing class attribute (~269-273) to reference it so there is one source of truth:

```python
    OPERATOR_DECLARATION_SOURCE_PREFIXES: tuple[str, ...] = (
        _OPERATOR_DECLARATION_SOURCE_PREFIXES
    )
```

(`_is_operator_declared` keeps working unchanged — it reads the class attribute.)

Remove `MIN_INJECTION_EVIDENCE` and its comment block entirely, since nothing else references it (verified by grep: only `find_matches_in_text` used it; `check_past_experience` has its own separate `_GENERIC_WORD_MIN_EVIDENCE`).

Replace the gate inside `find_matches_in_text` (currently the `if conv.evidence_count < self.MIN_INJECTION_EVIDENCE and not self._is_operator_declared(conv): continue` block, ~327-331) with:

```python
            if not self._is_operator_declared(conv):
                if conv.distinct_task_count < self.INJECT_MIN_DISTINCT_TASKS:
                    continue
                if conv.dispute_pct >= self.INJECT_MAX_DISPUTE_PCT:
                    continue
```

Also update the `find_matches_in_text` docstring's third bullet (~313-316) to describe the new gate:

```python
          - The convention must have at least
            ``INJECT_MIN_DISTINCT_TASKS`` distinct tasks voting for it AND a
            cross-task ``dispute_pct < INJECT_MAX_DISPUTE_PCT`` — UNLESS it was
            operator-declared, in which case one declaration counts as policy.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convention_distinct_task_gate.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Verify no other reference to the removed constant**

Run: `grep -rn "MIN_INJECTION_EVIDENCE" annotation_pipeline_skill/ tests/`
Expected: no output (constant fully removed). If any reference remains, update it to the new gate before committing.

- [ ] **Step 6: Commit**

```bash
git add annotation_pipeline_skill/services/entity_convention_service.py tests/test_convention_distinct_task_gate.py
git commit -m "feat(conventions): gate injection on distinct-task count + dispute_pct"
```

---

### Task 6: Surface dispute metrics in check_past_experience + fix soft-model test

**Files:**
- Modify: `annotation_pipeline_skill/llm/tools/check_past_experience.py` (~40-103)
- Modify: `tests/test_check_past_experience.py` (`test_disputed_returns_examples_per_type` ~67-79)
- Test: `tests/test_convention_distinct_task_gate.py`

- [ ] **Step 1: Update the now-stale soft-model test**

`test_disputed_returns_examples_per_type` asserts the OLD hard-flip behavior (`status == "disputed"`, `type is None`). Under the soft model the convention stays active with the plurality type. Rewrite it in `tests/test_check_past_experience.py`:

```python
def test_conflicting_proposals_track_dispute_soft_model(store):
    svc = EntityConventionService(store)
    _seed(svc, "p1", "Apple", "organization", "qc_consensus",
          "task_a", "row_1", "Apple's customer support helped me yesterday")
    _seed(svc, "p1", "Apple", "organization", "qc_consensus",
          "task_b", "row_2", "Apple announced a new privacy policy")
    _seed(svc, "p1", "Apple", "product", "qc_consensus",
          "task_c", "row_3", "My Apple iPad keeps crashing on updates")
    result = check_past_experience(store, project_id="p1", entry="Apple")
    # Soft model: stays active, plurality (organization) wins.
    assert result["convention"]["status"] == "active"
    assert result["convention"]["type"] == "organization"
    assert result["convention"]["dominant_type"] == "organization"
    assert result["convention"]["distinct_task_count"] == 3
    assert result["convention"]["dispute_count"] == 1
    assert result["convention"]["dispute_pct"] == pytest.approx(1 / 3)
    # Distribution still counts every proposal by its declared type.
    assert result["distribution"] == {"organization": 2, "product": 1}
    assert set(result["examples_by_type"].keys()) == {"organization", "product"}
```

Ensure `import pytest` is present at the top of `tests/test_check_past_experience.py` (it is — used via fixtures; if `pytest.approx` errors on import, add `import pytest`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_check_past_experience.py::test_conflicting_proposals_track_dispute_soft_model -v`
Expected: FAIL with `KeyError: 'dominant_type'` (check_past_experience doesn't emit the new keys yet).

- [ ] **Step 3: Write minimal implementation**

In `check_past_experience.py`, the convention is currently read via a raw SQL query (lines 40-44) that selects only `convention_id, entity_type, status, evidence_count, proposals_json`. Compute the tally from `proposals` and add the three fields. Add the import at the top:

```python
from annotation_pipeline_skill.services.entity_convention_service import (
    _distinct_task_tally,
)
```

Then, after `proposals = json.loads(row["proposals_json"] or "[]")` (~line 60), compute the tally and include it in the returned `convention` block (~88-94):

```python
    proposals = json.loads(row["proposals_json"] or "[]")
    dominant_type, distinct_tasks, dispute_count, dispute_pct = (
        _distinct_task_tally(proposals)
    )
```

and change the returned dict's `convention` block to:

```python
        "convention": {
            "status": row["status"],
            "type": row["entity_type"],
            "evidence_count": evidence_count,
            "dominant_type": dominant_type,
            "distinct_task_count": distinct_tasks,
            "dispute_count": dispute_count,
            "dispute_pct": dispute_pct,
        },
```

Also add the same four keys (with neutral values) to the `row is None` early-return branch (~49-58) so the output shape is stable:

```python
            "convention": {
                "status": "none", "type": None, "evidence_count": 0,
                "dominant_type": None, "distinct_task_count": 0,
                "dispute_count": 0, "dispute_pct": 0.0,
            },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_check_past_experience.py -v`
Expected: PASS (all, including the rewritten soft-model test and the unchanged `evidence_count`/`generic_word` tests — those use unique sources and distinct/None task_ids so their counts are unaffected).

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/llm/tools/check_past_experience.py tests/test_check_past_experience.py
git commit -m "feat(conventions): surface dispute metrics in check_past_experience; update soft-model test"
```

---

### Task 7: Full regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the convention + KB + prompt-builder + api test suites**

Run:
```bash
pytest tests/test_convention_distinct_task_gate.py \
       tests/test_entity_convention_proposals_schema.py \
       tests/test_check_past_experience.py -v
```
Expected: all PASS.

- [ ] **Step 2: Run any tests that touch find_matches_in_text / prompt injection**

Run:
```bash
grep -rl "find_matches_in_text\|EntityConventionService\|build_conventions_block\|check_past_experience" tests/ | xargs pytest -v
```
Expected: all PASS. If a previously-green test asserted the old `evidence_count >= 5` injection behavior or the hard-flip-to-disputed behavior, update it to the new model (distinct-task gate / soft model) — those are intentional behavior changes, not regressions. Do NOT weaken a test to pass; align it with the new spec.

- [ ] **Step 3: Commit any test alignments**

```bash
git add -A
git commit -m "test(conventions): align existing tests with distinct-task gate + soft model"
```

---

## Self-Review

**1. Spec coverage:**
- "去重键加 task_id" → Task 3 (no-op key now `(source, type, task_id)`).
- "记录dispute的值" → dispute value derived from per-task votes recorded in `proposals_json`; surfaced via `EntityConvention.dispute_count/dispute_pct/distinct_task_count/dominant_type` (Task 2) and `check_past_experience` (Task 6).
- "dispute% < 20 AND distinct-task ≥ 5即可注入" → Task 5 (`INJECT_MIN_DISTINCT_TASKS = 5`, `INJECT_MAX_DISPUTE_PCT = 0.20`, gate condition `distinct >= 5 AND dispute_pct < 0.20`).
- "按 distinct task（每任务一票）" + "distinct-task 是指三方意见一致的独特task数量" → Task 1 (`_distinct_task_tally`: one vote per distinct task, counting ONLY three-party-consensus proposals — operator/HR sources excluded; most-recent vote per task).
- "改为软模型（auto 不再硬翻）" → Task 4 (auto path never sets `status='disputed'`; dominant type wins; operator declaration still authoritative; operator-set dispute still honored until cleared).

**2. Placeholder scan:** No TBD/TODO/"add validation"-style placeholders. Every code step has complete code.

**3. Type consistency:** `_distinct_task_tally` returns `(str|None, int, int, float)` and is unpacked identically in `_load_row`, `check_past_experience`, and tests. New dataclass fields `distinct_task_count: int`, `dispute_count: int`, `dispute_pct: float`, `dominant_type: str | None` match their `to_dict()` keys and the `check_past_experience` output keys. Constants `INJECT_MIN_DISTINCT_TASKS` / `INJECT_MAX_DISPUTE_PCT` are referenced with the same names in `find_matches_in_text` and tests.

**Threshold boundary check:** gate is `distinct < 5 → skip` (so 5 passes) and `dispute_pct >= 0.20 → skip` (so exactly 0.20 is blocked; "< 20%" honored). dispute_pct = 2/6 ≈ 0.333 blocked; 1/10 = 0.10 allowed — both asserted in Task 5 tests.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-31-convention-distinct-task-dispute-gate.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session with checkpoints for review.

Which approach?

# QC Accuracy Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three concrete defects that cause ~29% of accepted annotations to contain errors the pipeline never detects.

**Architecture:** Two prompt-level patches to `subagent_cycle.py` close the two architectural gaps found by tracing real tasks through the pipeline (task 000238 had 8 empty rows that QC never checked; task 000616 had 18 misclassified numbers that survived all QC+arbiter passes). A third change updates the annotation guideline document with rules the models are consistently missing (CVE→document, Chinese person names, numeric values).

**Tech Stack:** Python 3.13, pytest, SQLite (annotation store), Markdown (guideline doc)

---

## Background: What went wrong and why

Investigation traced three accepted tasks end-to-end through conversations, artifacts, and feedback records. Findings:

**Defect 1 — ROUND DISCIPLINE fires too early (task 000238):**
`_build_qc_instructions` says "if feedback_bundle.items is NON-EMPTY, this is a retry round — only check rows referenced by prior feedback." For v3-migrated (prelabeled) tasks, a validation-stage feedback record is created *before* QC ever runs (the prelabeled annotation failed a mechanical rule check). So QC's very first real scan enters retry mode immediately, sees one prior feedback item about row 1, and never looks at rows 2–9 — which Qwen left entirely empty. MiniMax QC correctly resolved the one feedback and passed the task.

**Defect 2 — Annotator drops unchanged rows (task 000238):**
When Qwen received the task with one feedback item about row 1, it re-annotated rows 0–1 and output `{"entities": {}, "json_structures": {}}` for rows 2–9. The system prompt says "include an entry for EVERY row" but doesn't say "copy unchanged rows from prior_artifacts." Qwen treated the prior annotation as context rather than a baseline to delta-edit, producing empty rows instead of carrying the rich v3 content forward.

**Defect 3 — Missing guideline rules (tasks 000089, 000616, 000459, etc.):**
The guideline `v1.md` has no rule for CVE identifiers (consistently tagged as `generic_entity` or `technology` instead of `document`), no rule for Chinese person names (tagged as `generic_entity` instead of `person`), and no rule for numeric/percentage values (tagged as `generic_entity` instead of `number`).

---

## File map

| File | Change |
|------|--------|
| `annotation_pipeline_skill/runtime/subagent_cycle.py` | Patch ROUND DISCIPLINE text (Task 1) + add BASELINE PRESERVATION instruction (Task 2) |
| `tests/test_subagent_cycle.py` | Add two regression tests (Tasks 1 and 2) |
| `projects/v4_ner_phrase/.annotation-pipeline/document_versions/doc-e14e8757b02543ceaa238fa65bfa2ed0/v1.md` | Add three new guideline rules (Task 3) |

---

## Task 1: Fix ROUND DISCIPLINE — validation-only feedback must not trigger retry mode

**Context:** `_build_qc_instructions` is the module-level function at line 3791 of `subagent_cycle.py`. It builds the system prompt for every QC call. The current ROUND DISCIPLINE text tells QC "if feedback_bundle.items is NON-EMPTY, this is a retry round." The fix adds a carve-out: if every item has `source_stage=="validation"` (none have `source_stage=="qc"`), QC must still scan exhaustively.

The feedback bundle already serializes `source_stage` for every item (see `feedback_service.py` line 49), so the LLM has the information it needs to apply this rule.

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py` (~line 3809)
- Test: `tests/test_subagent_cycle.py`

---

- [ ] **Step 1: Write the failing test**

Add this test at the end of `tests/test_subagent_cycle.py`:

```python
def test_qc_instructions_scan_exhaustively_when_feedback_is_validation_only():
    """Regression for task-000238: when all prior feedback has source_stage=='validation'
    (none from 'qc'), the QC instructions must tell the model to scan ALL rows
    exhaustively, not restrict to rows referenced by prior feedback."""
    from annotation_pipeline_skill.runtime.subagent_cycle import _build_qc_instructions
    from annotation_pipeline_skill.core.models import Task

    task = Task.new(
        task_id="t-rd-1",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"text": "hello"}},
        modality="text",
        annotation_requirements={"annotation_types": ["entity_span"]},
    )
    instructions = _build_qc_instructions(
        task,
        resolved_policy={"mode": "all", "sample_ratio": 1.0},
    )
    # Must include the validation-only exception so QC knows to scan exhaustively
    assert 'source_stage=="validation"' in instructions, (
        "QC instructions must distinguish validation-only feedback from QC feedback "
        "so prelabeled tasks get a real full-scan QC pass"
    )
    # Must also state the condition under which retry mode applies
    assert 'source_stage=="qc"' in instructions, (
        "QC instructions must explicitly state that retry mode requires at least one "
        "source_stage==\"qc\" item in the feedback bundle"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/derek/Projects/annotation-pipeline-skill
python -m pytest tests/test_subagent_cycle.py::test_qc_instructions_scan_exhaustively_when_feedback_is_validation_only -v
```

Expected: `FAILED` — `AssertionError: QC instructions must distinguish validation-only feedback...`

- [ ] **Step 3: Apply the fix to `_build_qc_instructions`**

In `annotation_pipeline_skill/runtime/subagent_cycle.py`, find this exact block (around line 3809):

```python
        "ROUND DISCIPLINE: The feedback_bundle in this prompt contains all prior feedback items for this task. "
        "If feedback_bundle.items is EMPTY, this is round 1 — scan every row exhaustively and report EVERY "
        "defect you detect. This is your only opportunity to flag issues anywhere in the annotation. "
        "If feedback_bundle.items is NON-EMPTY, this is a retry round. In retry rounds your failures list "
        "is STRICTLY RESTRICTED to: "
```

Replace it with:

```python
        "ROUND DISCIPLINE: The feedback_bundle in this prompt contains all prior feedback items for this task. "
        "If feedback_bundle.items is EMPTY, this is round 1 — scan every row exhaustively and report EVERY "
        "defect you detect. This is your only opportunity to flag issues anywhere in the annotation. "
        "ROUND-1 EXCEPTION: if feedback_bundle.items is NON-EMPTY but every item has "
        "source_stage==\"validation\" (none have source_stage==\"qc\"), treat this as your round 1 — "
        "scan every row exhaustively. Validation flags mechanical rule violations before QC runs; "
        "a validation-only bundle does NOT mean QC has already scanned this annotation. "
        "If feedback_bundle.items is NON-EMPTY and at least one item has source_stage==\"qc\", "
        "this is a retry round. In retry rounds your failures list "
        "is STRICTLY RESTRICTED to: "
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_subagent_cycle.py::test_qc_instructions_scan_exhaustively_when_feedback_is_validation_only -v
```

Expected: `PASSED`

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
python -m pytest tests/test_subagent_cycle.py tests/test_prompt_builder.py -v
```

Expected: all existing tests still pass (the change is additive — same retry-mode logic applies, just with a new carve-out for validation-only bundles).

- [ ] **Step 6: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_subagent_cycle.py
git commit -m "fix(qc): treat validation-only feedback bundle as round 1, not retry

Prelabeled (v3-migrated) tasks receive validation-stage feedback before
QC ever runs. The old ROUND DISCIPLINE rule entered retry mode as soon
as feedback_bundle.items was non-empty, causing MiniMax QC to only check
the one row referenced by the validation item — leaving all other rows
unscanned even on the first real QC pass (observed: task-000238 had 8
empty rows that QC never flagged).

Add ROUND-1 EXCEPTION: if all items have source_stage=='validation', QC
still scans exhaustively. Retry mode now requires at least one item with
source_stage=='qc'."
```

---

## Task 2: Add BASELINE PRESERVATION — annotator must carry forward unchanged rows

**Context:** `_annotation_instructions` builds the system prompt for annotation calls (line 3667 of `subagent_cycle.py`). The base string currently tells the model "include an entry for EVERY row," but gives no instruction about what to do with rows that aren't referenced by any feedback. Qwen re-annotated from scratch, zeroing out rows 2–9 of task 000238 instead of copying the rich v3 prior annotation. The fix adds a BASELINE PRESERVATION rule between the MANDATORY ROW COVERAGE paragraph and the HANDLING QC FEEDBACK paragraph.

**Files:**
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py` (~line 3712)
- Test: `tests/test_subagent_cycle.py`

---

- [ ] **Step 1: Write the failing test**

Add this test at the end of `tests/test_subagent_cycle.py` (after the test from Task 1):

```python
def test_annotation_instructions_include_baseline_preservation_rule():
    """Regression for task-000238: annotator must copy unchanged rows from
    prior_artifacts when feedback_bundle only references specific rows.
    Without this, Qwen drops rows it wasn't asked about."""
    from annotation_pipeline_skill.runtime.subagent_cycle import _annotation_instructions
    from annotation_pipeline_skill.core.models import Task

    task = Task.new(
        task_id="t-bp-1",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"text": "hello"}},
        modality="text",
        annotation_requirements={"annotation_types": ["entity_span"]},
    )
    instructions = _annotation_instructions(task)
    assert "BASELINE PRESERVATION" in instructions, (
        "Annotation instructions must include a BASELINE PRESERVATION rule so "
        "the model knows to copy unchanged rows from prior_artifacts"
    )
    assert "prior_artifacts" in instructions, (
        "BASELINE PRESERVATION rule must explicitly reference prior_artifacts "
        "so the model knows where to look for the baseline"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_subagent_cycle.py::test_annotation_instructions_include_baseline_preservation_rule -v
```

Expected: `FAILED` — `AssertionError: Annotation instructions must include a BASELINE PRESERVATION rule...`

- [ ] **Step 3: Apply the fix to `_annotation_instructions`**

In `annotation_pipeline_skill/runtime/subagent_cycle.py`, find this exact block (around line 3710):

```python
        "genuinely contains no instance of any declared type. "
        "\n\n"
        "HANDLING QC FEEDBACK: for each item in feedback_bundle, choose either to fix or to rebut:\n"
```

Replace it with:

```python
        "genuinely contains no instance of any declared type. "
        "\n\n"
        "BASELINE PRESERVATION: when prior_artifacts contains an annotation_result AND "
        "feedback_bundle.items is non-empty, treat the prior annotation as your baseline. "
        "For each row_index in the prior annotation that is NOT referenced by any feedback item "
        "(check feedback_bundle.items[*].target.row_index), copy that row's output exactly as-is "
        "into your response. Only re-evaluate rows whose row_index appears in at least one "
        "feedback item's target. This prevents silently dropping correct annotations on rows "
        "QC has already implicitly accepted."
        "\n\n"
        "HANDLING QC FEEDBACK: for each item in feedback_bundle, choose either to fix or to rebut:\n"
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_subagent_cycle.py::test_annotation_instructions_include_baseline_preservation_rule -v
```

Expected: `PASSED`

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests/test_subagent_cycle.py tests/test_prompt_builder.py -v
```

Expected: all tests pass. The BASELINE PRESERVATION text is inserted between existing paragraphs and doesn't break byte-stability of the system prompt (it's a constant string addition with no per-task content).

- [ ] **Step 6: Verify the byte-stability requirement is met**

The system prompt must be byte-stable across tasks for vLLM prefix caching. The new text is a constant string (no f-strings, no per-task variables). Verify:

```bash
python3 -c "
from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.runtime.subagent_cycle import _annotation_instructions

task_a = Task.new(task_id='x-1', pipeline_id='pipe',
    source_ref={'kind': 'jsonl', 'payload': {'text': 'hello'}},
    modality='text', annotation_requirements={'annotation_types': ['entity_span']})
task_b = Task.new(task_id='x-2', pipeline_id='pipe',
    source_ref={'kind': 'jsonl', 'payload': {'text': 'world', 'rows': [{'input': 'y'}]}},
    modality='text', annotation_requirements={'annotation_types': ['entity_span']})

inst_a = _annotation_instructions(task_a)
inst_b = _annotation_instructions(task_b)
assert inst_a == inst_b, 'Instructions differ between tasks — prefix cache will miss'
print('OK: instructions are byte-identical across tasks')
"
```

Expected output: `OK: instructions are byte-identical across tasks`

- [ ] **Step 7: Commit**

```bash
git add annotation_pipeline_skill/runtime/subagent_cycle.py tests/test_subagent_cycle.py
git commit -m "fix(annotator): add BASELINE PRESERVATION rule to annotation system prompt

Without this, Qwen re-annotates from scratch on every re-run and silently
zeros out rows not referenced by feedback — observed in task-000238 where
rows 2-9 had rich v3 annotations that Qwen replaced with empty {}.

BASELINE PRESERVATION instructs the model to copy unchanged rows from
prior_artifacts verbatim, only re-evaluating rows referenced in
feedback_bundle.items[*].target.row_index."
```

---

## Task 3: Update annotation guideline with missing type rules

**Context:** The annotation guideline at `projects/v4_ner_phrase/.annotation-pipeline/document_versions/doc-e14e8757b02543ceaa238fa65bfa2ed0/v1.md` is loaded by `_load_guideline` and appended to both the annotation and QC system prompts for every task in this project. Three entity types are consistently misclassified across the 311-row sample: CVE identifiers (tagged as `technology`/`generic_entity` instead of `document`), Chinese person names (tagged as `generic_entity` instead of `person`), and numeric/percentage values (tagged as `generic_entity` instead of `number`).

There is no automated test for this task — it is a configuration content change. Verify by reading the file after editing.

**Files:**
- Modify: `projects/v4_ner_phrase/.annotation-pipeline/document_versions/doc-e14e8757b02543ceaa238fa65bfa2ed0/v1.md`

---

- [ ] **Step 1: Add three new rules at the end of v1.md**

The file currently ends after the `json_structures` rule (line 217). Append the following three rules:

```markdown
  - id: cve_identifiers
    applies_to: [entity_span]
    instruction: |
      CVE identifiers are `document`, not `technology` or `generic_entity`.
      A CVE is a stable named identifier for a published security vulnerability report —
      it is a document reference, not a tool or product.
      Examples: "CVE-2022-22965", "CVE-2021-26855", "CVE-2022-22954" → all `document`.
      Tag the full identifier verbatim as it appears in the source text.
      Do NOT split the year from the sequence number.

  - id: person_names_all_scripts
    applies_to: [entity_span]
    instruction: |
      Person names in any script are `person` — not `generic_entity`.
      This applies equally to Chinese, Arabic, Cyrillic, and other non-Latin scripts.
      Examples: "李明", "王芳", "张伟", "Joachim Murat", "石田三成", "Sarah Chen" → all `person`.
      Historical figures, fictional characters, and pseudonymous handles that refer to
      a specific individual are also `person`.
      Do not downgrade a person name to `generic_entity` merely because the script or
      language is unfamiliar — if the surrounding context identifies it as a person, tag it.

  - id: numeric_quantities
    applies_to: [entity_span]
    instruction: |
      Numeric values with referential meaning are `number`, not `generic_entity`.
      This includes:
      - Percentages: "12%", "68%", "156%", "18%"
      - Raw counts and metrics: "2.3M", "2.3 million", "42", "38", "10MB"
      - Latency / size / performance values: "340ms", "200ms"
      - Multipliers: "1.2x", "3x"
      - Monetary amounts (currency symbol + digits): "$4.8M", "$5.2M", "$315,000"

      Use `generic_entity` only when a value's referential meaning cannot be determined
      from context. When the surrounding text makes clear this is a measurable quantity,
      use `number`.

      Exceptions that remain outside `number` (per conservative_entity_scope):
      - Structured metadata IDs such as "Intervention: 9442" or "Question: 107170"
        have no referential NER value — omit them.
      - Bare pedagogical operands in math word problems ("how many 3s are in 4?")
        have no stable entity referent — omit them.
```

- [ ] **Step 2: Verify the file was updated correctly**

```bash
tail -60 "/home/derek/Projects/annotation-pipeline-skill/projects/v4_ner_phrase/.annotation-pipeline/document_versions/doc-e14e8757b02543ceaa238fa65bfa2ed0/v1.md"
```

Expected: output shows all three new rules (`cve_identifiers`, `person_names_all_scripts`, `numeric_quantities`) at the end of the file, properly indented as YAML list items under `rules:`.

- [ ] **Step 3: Verify the guideline is still valid YAML**

```bash
python3 -c "
import yaml
with open('/home/derek/Projects/annotation-pipeline-skill/projects/v4_ner_phrase/.annotation-pipeline/document_versions/doc-e14e8757b02543ceaa238fa65bfa2ed0/v1.md') as f:
    content = f.read()
# Strip the leading '# ...' comment line before parsing
lines = content.split('\n')
yaml_start = next(i for i, l in enumerate(lines) if l.startswith('rules:'))
yaml_content = '\n'.join(lines[yaml_start:])
parsed = yaml.safe_load(yaml_content)
rule_ids = [r['id'] for r in parsed['rules']]
assert 'cve_identifiers' in rule_ids
assert 'person_names_all_scripts' in rule_ids
assert 'numeric_quantities' in rule_ids
print('OK: guideline parses correctly, new rules present:', rule_ids[-3:])
"
```

Expected: `OK: guideline parses correctly, new rules present: ['cve_identifiers', 'person_names_all_scripts', 'numeric_quantities']`

- [ ] **Step 4: Run the full test suite one final time**

```bash
python -m pytest tests/test_subagent_cycle.py tests/test_prompt_builder.py -v
```

Expected: all tests pass (guideline file change has no code-level test surface).

- [ ] **Step 5: Commit**

```bash
git add "projects/v4_ner_phrase/.annotation-pipeline/document_versions/doc-e14e8757b02543ceaa238fa65bfa2ed0/v1.md"
git commit -m "fix(guideline): add rules for CVE identifiers, CJK person names, numeric values

Three entity types are consistently misclassified across the 5% QC sample:
- CVE-XXXX → was tagged generic_entity/technology; correct type is document
- Chinese person names (李明, 王芳) → was generic_entity; correct type is person
- Percentages, counts, latency values → was generic_entity; correct type is number

Add three explicit rules to v1.md that the guideline was missing. These rules
are loaded by _load_guideline and appended to both annotation and QC system
prompts for every v4_ner_phrase task."
```

---

## Self-Review

**1. Spec coverage:**
- ROUND DISCIPLINE defect → Task 1 ✓
- Annotation drops unchanged rows → Task 2 ✓
- CVE misclassification → Task 3 ✓
- Chinese person names as generic_entity → Task 3 ✓
- Numbers/percentages as generic_entity → Task 3 ✓

**2. Placeholder scan:** None found. All code blocks are complete and runnable.

**3. Type consistency:**
- `_build_qc_instructions(task, resolved_policy=..., guideline=...)` — used consistently in Task 1
- `_annotation_instructions(task, ...)` — used consistently in Task 2
- Both are imported from `annotation_pipeline_skill.runtime.subagent_cycle`
- `Task.new(...)` constructor used identically in both tests (same pattern as existing tests)

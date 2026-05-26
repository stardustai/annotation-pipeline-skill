#!/usr/bin/env python3
"""Batch-fix v9 rule violations in ACCEPTED tasks for v3_initial_deployment.

Three fixes applied per task:
  1. Remove bare-name entries from json_structures.technology
     (entries with no predicate verb, or that appear verbatim in entities.technology,
     belong only in entities.technology, not json_structures.technology)
  2. Deduplicate + remove nested spans within each entity type
     (no two identical spans; shorter span that is a proper substring of
     a longer span in the same type is dropped)
  3. Remove redacted/masked placeholder spans (XXXX patterns) from all entity types

Tasks that change are transitioned ACCEPTED → HUMAN_REVIEW then corrected
via HumanReviewService.submit_correction(force=True) → ACCEPTED.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.core.transitions import transition_task, InvalidTransition
from annotation_pipeline_skill.services.human_review_service import (
    HumanReviewService,
    _autoclean_pre_existing_defects,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore

# ---------------------------------------------------------------------------
# Fix 1: bare names in json_structures.technology
# ---------------------------------------------------------------------------

# Common English verb/copula forms that indicate a predicated phrase.
# Use explicit alternation (not char-class hacks) for suffix groups.
_VERB_RE = re.compile(
    r"\b("
    # Copulas / auxiliaries
    r"is|are|was|were|be|been|being|has|have|had|do|does|did|"
    # Common tech verbs: stem + (s|ed|es|ing) and common irregulars
    r"uses?|used|using|"
    r"runs?|ran|running|"
    r"requires?|required|requiring|"
    r"supports?|supported|supporting|"
    r"drops?|dropped|dropping|"
    r"adds?|added|adding|"
    r"removes?|removed|removing|"
    r"provides?|provided|providing|"
    r"allows?|allowed|allowing|"
    r"enables?|enabled|enabling|"
    r"includes?|included|including|"
    r"works?|worked|working|"
    r"fix(es|ed)?|fixing|"
    r"handles?|handled|handling|"
    r"needs?|needed|needing|"
    r"gets?|got|gotten|getting|"
    r"sets?|setting|"
    r"makes?|made|making|"
    r"takes?|took|taken|taking|"
    r"builds?|built|building|"
    r"generates?|generated|generating|"
    r"returns?|returned|returning|"
    r"calls?|called|calling|"
    r"creates?|created|creating|"
    r"updates?|updated|updating|"
    r"deletes?|deleted|deleting|"
    r"reads?|reading|"
    r"writes?|wrote|written|writing|"
    r"sends?|sent|sending|"
    r"receives?|received|receiving|"
    r"connects?|connected|connecting|"
    r"depends?|depended|depending|"
    r"replaces?|replaced|replacing|"
    r"extends?|extended|extending|"
    r"imports?|imported|importing|"
    r"exports?|exported|exporting|"
    r"renders?|rendered|rendering|"
    r"loads?|loaded|loading|"
    r"fails?|failed|failing|"
    r"throws?|threw|thrown|throwing|"
    r"catches?|caught|catching|"
    r"validates?|validated|validating|"
    r"parses?|parsed|parsing|"
    r"formats?|formatted|formatting|"
    r"converts?|converted|converting|"
    r"compiles?|compiled|compiling|"
    r"executes?|executed|executing|"
    r"contains?|contained|containing|"
    r"implements?|implemented|implementing|"
    r"introduces?|introduced|introducing|"
    r"deprecates?|deprecated|deprecating|"
    r"launches?|launched|launching|"
    r"announces?|announced|announcing|"
    r"releases?|released|releasing|"
    r"integrates?|integrated|integrating|"
    r"migrates?|migrated|migrating|"
    r"breaks?|broke|broken|breaking|"
    r"changes?|changed|changing|"
    r"limits?|limited|limiting|"
    r"restricts?|restricted|restricting|"
    r"prevents?|prevented|preventing|"
    r"applies?|applied|applying|"
    r"resolves?|resolved|resolving"
    r")\b",
    re.IGNORECASE,
)


def _has_predicate(phrase: str) -> bool:
    """Return True if phrase contains a verb/predicate structure."""
    return bool(_VERB_RE.search(phrase))


def fix1_json_structures_technology(payload: dict) -> tuple[dict, int]:
    """Remove bare-name entries from json_structures.technology.

    A bare name is:
      - Appears verbatim in entities.technology (any row), OR
      - Contains no predicate verb

    Returns (updated_payload, n_removed).
    """
    rows = payload.get("rows", [])
    n_removed = 0
    for row in rows:
        output = row.get("output", {})
        js = output.get("json_structures", {})
        tech_phrases = js.get("technology", [])
        if not tech_phrases:
            continue

        # Collect entities.technology spans for this row (lowercase set)
        ent_tech = set(s.lower() for s in output.get("entities", {}).get("technology", []))

        kept = []
        for phrase in tech_phrases:
            # Criterion 1: verbatim match in entities.technology (case-insensitive)
            if phrase.lower() in ent_tech:
                n_removed += 1
                continue
            # Criterion 2: no predicate verb → bare name
            if not _has_predicate(phrase):
                n_removed += 1
                continue
            kept.append(phrase)

        if len(kept) != len(tech_phrases):
            if kept:
                js["technology"] = kept
            else:
                js.pop("technology", None)
            # IMPORTANT: do NOT remove json_structures even when empty —
            # the output schema requires "json_structures" to be present.
            # If json_structures key is missing from output, ensure it exists.
            if "json_structures" not in output:
                output["json_structures"] = {}

    return payload, n_removed


# ---------------------------------------------------------------------------
# Fix 2: dedup + remove nested spans within each entity type
# ---------------------------------------------------------------------------

def fix2_dedup_and_denest(payload: dict) -> tuple[dict, int]:
    """Deduplicate spans and remove spans nested inside longer spans (same type).

    Applied to both entities and json_structures.
    Returns (updated_payload, n_changed).
    """
    n_changed = 0
    rows = payload.get("rows", [])
    for row in rows:
        output = row.get("output", {})

        # Entities: dedup + denest
        entities = output.get("entities", {})
        for etype, spans in list(entities.items()):
            if not isinstance(spans, list):
                continue
            # Dedup preserving order
            seen: set[str] = set()
            unique: list[str] = []
            for s in spans:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)

            # Remove spans that are proper substrings of another span in the same type
            non_nested: list[str] = []
            for s in unique:
                if not any(s != other and s in other for other in unique):
                    non_nested.append(s)

            if non_nested != spans:
                n_changed += len(spans) - len(non_nested)
                if non_nested:
                    entities[etype] = non_nested
                else:
                    del entities[etype]

        # json_structures: dedup only (phrases are sentence fragments, nesting rare)
        js = output.get("json_structures", {})
        for jtype, phrases in list(js.items()):
            if not isinstance(phrases, list):
                continue
            seen_p: set[str] = set()
            unique_p: list[str] = []
            for p in phrases:
                if p not in seen_p:
                    seen_p.add(p)
                    unique_p.append(p)
            if unique_p != phrases:
                n_changed += len(phrases) - len(unique_p)
                if unique_p:
                    js[jtype] = unique_p
                else:
                    del js[jtype]

    return payload, n_changed


# ---------------------------------------------------------------------------
# Fix 3: remove redacted/masked placeholder spans
# ---------------------------------------------------------------------------

# Placeholder patterns:
#   - "XXXX", "XXXXXXXX" (3+ uppercase X's, possibly with non-alpha prefix/suffix)
#   - "$ XXXX", "#XXXX", "account #XXXX"
_XXXX_CORE = re.compile(r"X{3,}")  # 3+ consecutive uppercase X


def _is_placeholder(span: str) -> bool:
    """Return True if span looks like a redacted/masked placeholder.

    Does NOT flag date-format strings like "XX/XX/2021" (those are `time`
    entities and fix3 skips entity type 'time' entirely).
    """
    s = span.strip()
    if not s:
        return False
    # Pattern 1: only non-alpha chars and X's (covers "XXXX", "$ XXXX", "#XXXX",
    # "XXXX XXXX", "XXXXXXXX XXXX")
    # [^a-wyz] excludes a-w,y,z but allows X/x and non-letter chars
    if re.match(r"^[^a-wyz]*X{3,}[^a-wyz]*$", s, re.IGNORECASE):
        return bool(_XXXX_CORE.search(s))
    # Pattern 2: span starts with X{3+} (masked entity name prefix like "XXXX SCORE",
    # "XXXX site" — brand name replaced by masking token)
    if re.match(r"^X{3,}\s", s, re.IGNORECASE):
        return True
    # Pattern 3: after stripping common context words, only X's remain
    # Covers "account #XXXX", "Account Number XXXX"
    remainder = re.sub(
        r"\b(account|number|no|nr|#|account\s+no|account\s+number)\b",
        "",
        s,
        flags=re.IGNORECASE,
    )
    remainder = re.sub(r"[\s$#\d.,/()]+", "", remainder)
    if remainder and re.match(r"^X+$", remainder, re.IGNORECASE):
        return True
    return False


def fix3_remove_placeholders(payload: dict) -> tuple[dict, int]:
    """Remove placeholder spans from all entity types except 'time'.

    'time' is excluded because masked date tokens like "XX/XX/2021" or
    "XX/XX/XXXX" are valid per v9 entity_span_boundaries rule.

    Returns (updated_payload, n_removed).
    """
    n_removed = 0
    rows = payload.get("rows", [])
    for row in rows:
        output = row.get("output", {})
        entities = output.get("entities", {})
        for etype, spans in list(entities.items()):
            # Masked date tokens in time are valid per v9 — skip entirely
            if etype == "time":
                continue
            if not isinstance(spans, list):
                continue
            cleaned = [s for s in spans if not _is_placeholder(s)]
            if len(cleaned) != len(spans):
                n_removed += len(spans) - len(cleaned)
                if cleaned:
                    entities[etype] = cleaned
                else:
                    del entities[etype]

    return payload, n_removed


# ---------------------------------------------------------------------------
# Apply all fixes to one payload
# ---------------------------------------------------------------------------

def apply_all_fixes(payload: dict) -> tuple[dict, dict]:
    """Apply all three fixes and return (fixed_payload, stats).

    stats keys: f1_removed, f2_changed, f3_removed, total_changes, changed

    Post-condition: every row output contains both "entities" and
    "json_structures" keys (the schema requires both to be present).
    If either was absent or removed, it is replaced with {}.
    """
    p = copy.deepcopy(payload)

    p, f1 = fix1_json_structures_technology(p)
    p, f2 = fix2_dedup_and_denest(p)
    p, f3 = fix3_remove_placeholders(p)

    # Guarantee required output keys are always present
    for row in p.get("rows", []):
        out = row.get("output", {})
        if "entities" not in out:
            out["entities"] = {}
        if "json_structures" not in out:
            out["json_structures"] = {}

    total = f1 + f2 + f3
    return p, {
        "f1_js_tech_removed": f1,
        "f2_dedup_denest": f2,
        "f3_placeholders_removed": f3,
        "total_changes": total,
        "changed": total > 0,
    }


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

ACTOR = "batch-fix-v9"
PIPELINE_ID = "v3_initial_deployment"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Batch-fix v9 rule violations in v3_initial_deployment ACCEPTED tasks"
    )
    ap.add_argument(
        "--store-root",
        default="projects/v3_initial_deployment/.annotation-pipeline",
        help="Path to .annotation-pipeline directory",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be changed without writing anything",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N tasks (for testing)",
    )
    ap.add_argument(
        "--task-id",
        default=None,
        help="Process a single task ID only (for debugging)",
    )
    args = ap.parse_args(argv)

    store_root = Path(args.store_root)
    store = SqliteStore.open(store_root)
    hr_service = HumanReviewService(store)

    # Load tasks
    if args.task_id:
        tasks = [store.load_task(args.task_id)]
    else:
        all_accepted = store.list_tasks_by_status([TaskStatus.ACCEPTED])
        tasks = [t for t in all_accepted if t.pipeline_id == PIPELINE_ID]
        if args.limit:
            tasks = tasks[: args.limit]

    print(
        f"{'DRY RUN — ' if args.dry_run else ''}Processing {len(tasks)} ACCEPTED tasks "
        f"for pipeline={PIPELINE_ID}",
        file=sys.stderr,
    )

    stats_total = {
        "tasks_inspected": 0,
        "tasks_changed": 0,
        "tasks_skipped_no_payload": 0,
        "tasks_failed": 0,
        "f1_js_tech_removed": 0,
        "f2_dedup_denest": 0,
        "f3_placeholders_removed": 0,
        "total_span_changes": 0,
    }

    for i, task in enumerate(tasks):
        stats_total["tasks_inspected"] += 1

        # Load current payload
        payload = hr_service._latest_annotation_payload(task.task_id)
        if payload is None:
            print(
                f"  [{i+1}/{len(tasks)}] SKIP {task.task_id} — no annotation payload",
                file=sys.stderr,
            )
            stats_total["tasks_skipped_no_payload"] += 1
            continue

        # Apply fixes
        fixed_payload, fix_stats = apply_all_fixes(payload)
        if not fix_stats["changed"]:
            continue

        stats_total["tasks_changed"] += 1
        stats_total["f1_js_tech_removed"] += fix_stats["f1_js_tech_removed"]
        stats_total["f2_dedup_denest"] += fix_stats["f2_dedup_denest"]
        stats_total["f3_placeholders_removed"] += fix_stats["f3_placeholders_removed"]
        stats_total["total_span_changes"] += fix_stats["total_changes"]

        note = (
            f"batch-fix-v9: "
            f"f1={fix_stats['f1_js_tech_removed']} bare-name js.tech removed, "
            f"f2={fix_stats['f2_dedup_denest']} dedup/denest, "
            f"f3={fix_stats['f3_placeholders_removed']} placeholders removed"
        )

        if args.dry_run:
            print(
                f"  [{i+1}/{len(tasks)}] WOULD-FIX {task.task_id}: {note}",
                file=sys.stderr,
            )
            continue

        # Live: transition ACCEPTED → HUMAN_REVIEW, then submit correction
        try:
            # Reload fresh task object to avoid stale state
            task = store.load_task(task.task_id)

            event_hr = transition_task(
                task,
                TaskStatus.HUMAN_REVIEW,
                actor=ACTOR,
                reason="batch-fix-v9: correcting annotation quality issues per v9 rules",
                stage="human_review",
                metadata={"batch_fix": "v9"},
            )
            store.save_task(task)
            store.append_event(event_hr)

            # Pre-clean pre-existing defects (trailing punctuation, non-verbatim
            # spans, cross-type collisions, schema-disallowed old keys) so
            # submit_correction doesn't reject the payload for issues we didn't
            # introduce.
            _autoclean_pre_existing_defects(task, fixed_payload, store=store)

            # submit_correction transitions back to ACCEPTED internally
            hr_service.submit_correction(
                task_id=task.task_id,
                answer=fixed_payload,
                actor=ACTOR,
                note=note,
                force=True,
                record_conventions=False,  # mechanical fix — don't pollute conventions
                stat_bumps=[],  # don't bulk-bump stats for every span
            )

            print(
                f"  [{i+1}/{len(tasks)}] FIXED {task.task_id}: {note}",
            )
        except Exception as exc:  # noqa: BLE001
            stats_total["tasks_failed"] += 1
            print(
                f"  [{i+1}/{len(tasks)}] FAIL {task.task_id}: {exc}",
                file=sys.stderr,
            )
            # Attempt to put task back to ACCEPTED if it got stuck in HUMAN_REVIEW
            try:
                t2 = store.load_task(task.task_id)
                if t2.status is TaskStatus.HUMAN_REVIEW:
                    # Try to transition back
                    ev = transition_task(
                        t2,
                        TaskStatus.ACCEPTED,
                        actor=ACTOR,
                        reason="batch-fix-v9 rollback: submit_correction failed",
                        stage="human_review",
                    )
                    store.save_task(t2)
                    store.append_event(ev)
            except Exception:  # noqa: BLE001
                pass

    # Print summary
    print("\n=== Batch Fix v9 Summary ===")
    print(json.dumps(stats_total, indent=2))
    return 0 if stats_total["tasks_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

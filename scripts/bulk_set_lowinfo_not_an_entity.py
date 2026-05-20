"""Bulk mark all high-wordfreq spans as not_an_entity.

For each span in entity_statistics whose average Zipf score (via wordfreq) is
>= WORDFREQ_THRESHOLD, remove that span from every ACCEPTED task's annotation
in one efficient pass — one artifact write per task instead of one per span.

Steps
-----
1. Build target span set: scan entity_statistics for all distinct spans whose
   wordfreq score >= WORDFREQ_THRESHOLD.
2. Scan every ACCEPTED task once.  For each task whose annotation contains any
   target span, strip those spans from all entity-type buckets (entities +
   json_structures) and write a single human_review_answer artifact.
3. DELETE entity_statistics rows for every target span (they're all
   not_an_entity from now on — there's nothing to count).
4. Clear posterior_audit_cache so the next /posterior-audit call rebuilds.

Usage
-----
    # Preview only (no writes)
    python -m scripts.bulk_set_lowinfo_not_an_entity \\
        --project v3_initial_deployment

    # Apply changes
    python -m scripts.bulk_set_lowinfo_not_an_entity \\
        --project v3_initial_deployment --apply

    # Different threshold (default 5.0)
    python -m scripts.bulk_set_lowinfo_not_an_entity \\
        --project v3_initial_deployment --threshold 4.0 --apply
"""
from __future__ import annotations

import argparse
import copy
import json
import re as _re
import sys
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

# ── project root on path ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from annotation_pipeline_skill.core.models import ArtifactRef
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.store.sqlite_store import SqliteStore

DEFAULT_THRESHOLD = 5.0


# ── wordfreq helper ───────────────────────────────────────────────────────────

def _wordfreq_score(span: str) -> float:
    """Average Zipf frequency of the span's tokens.  0.0 for empty spans."""
    from wordfreq import zipf_frequency, tokenize
    lang = "zh" if any("一" <= ch <= "鿿" for ch in span) else "en"
    tokens = tokenize(span, lang)
    if not tokens:
        return 0.0
    return sum(zipf_frequency(t, lang) for t in tokens) / len(tokens)


# ── annotation loading ────────────────────────────────────────────────────────

def _load_annotation(store: SqliteStore, task) -> dict | None:
    """Load the best current annotation for a task.

    Prefers human_review_answer (latest seq) over annotation_result,
    mirroring build_posterior_audit._load_annotation exactly.
    """
    arts = store.list_artifacts(task.task_id)
    hr = [a for a in arts if a.kind == "human_review_answer"]
    if hr:
        try:
            outer = json.loads((store.root / hr[-1].path).read_text(encoding="utf-8"))
            return outer.get("answer") if isinstance(outer, dict) else None
        except (OSError, json.JSONDecodeError):
            return None
    anns = [a for a in arts if a.kind == "annotation_result"]
    if not anns:
        return None
    try:
        outer = json.loads((store.root / anns[-1].path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    text = outer.get("text") if isinstance(outer, dict) else None
    if not isinstance(text, str):
        return None
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return None


# ── span removal ──────────────────────────────────────────────────────────────

def _remove_spans_from_payload(payload: dict, spans_lower: set[str]) -> tuple[dict, int]:
    """Deep-copy payload with all target spans stripped from every row.

    Strips from both ``entities`` and ``json_structures`` buckets.
    Comparison is case-insensitive (spans_lower contains lowercased keys).

    Returns (new_payload, n_removed) where n_removed is the total number of
    individual span occurrences removed across all rows and type buckets.
    """
    out = copy.deepcopy(payload)
    # Drop runtime-only keys that don't belong in saved corrections.
    if isinstance(out, dict):
        for _key in ("discussion_replies", "feedback_resolution", "task_id"):
            out.pop(_key, None)
    rows = out.get("rows") if isinstance(out, dict) else None
    if not isinstance(rows, list):
        return out, 0

    n_removed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        for field_key in ("entities", "json_structures"):
            container = output.get(field_key)
            if not isinstance(container, dict):
                continue
            empty_keys = []
            for typ, items in container.items():
                if not isinstance(items, list):
                    continue
                before = len(items)
                container[typ] = [s for s in items if s.strip().lower() not in spans_lower]
                n_removed += before - len(container[typ])
                if not container[typ]:
                    empty_keys.append(typ)
            for k in empty_keys:
                container.pop(k, None)
    return out, n_removed


# ── artifact writing ──────────────────────────────────────────────────────────

def _write_hr_answer(store: SqliteStore, task_id: str, answer: dict, *, note: str) -> None:
    """Write a human_review_answer artifact file and register it in SQLite."""
    actor = "bulk_set_lowinfo_not_an_entity"
    relative_path = Path("artifact_payloads") / task_id / f"human_review_answer-{uuid4().hex}.json"
    absolute_path = store.root / relative_path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(
        json.dumps({"answer": answer, "actor": actor, "note": note}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    ref = ArtifactRef.new(
        task_id=task_id,
        kind="human_review_answer",
        path=relative_path.as_posix(),
        content_type="application/json",
        metadata={"actor": actor, "note": note},
    )
    store.append_artifact(ref)


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="pipeline_id / project_id")
    parser.add_argument(
        "--workspace",
        default="projects/v3_initial_deployment/.annotation-pipeline",
        help="Path to the project's .annotation-pipeline dir",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Zipf wordfreq floor (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--apply", action="store_true", help="actually write changes (default: dry-run)")
    args = parser.parse_args(argv)

    store = SqliteStore.open(Path(args.workspace))
    project_id: str = args.project
    threshold: float = args.threshold
    dry_run: bool = not args.apply

    mode_label = "DRY RUN" if dry_run else "APPLY"
    print(f"[{mode_label}] project={project_id!r}  wordfreq>={threshold}")

    # ── 1. Build target span set ───────────────────────────────────────────────
    print("\n── Step 1: scoring spans from entity_statistics ──")
    stat_rows = store._conn.execute(
        "SELECT DISTINCT span_lower FROM entity_statistics WHERE project_id = ?",
        (project_id,),
    ).fetchall()
    all_spans = [r["span_lower"] for r in stat_rows]
    print(f"  Distinct spans in entity_statistics: {len(all_spans):,}")

    target_spans: set[str] = set()
    for span in all_spans:
        if _wordfreq_score(span) >= threshold:
            target_spans.add(span)
    print(f"  Target spans (wordfreq>={threshold}): {len(target_spans):,}")

    if not target_spans:
        print("No target spans found — nothing to do.")
        return

    # ── 2. Scan ACCEPTED tasks ─────────────────────────────────────────────────
    print("\n── Step 2: scanning ACCEPTED tasks ──")
    all_tasks = [t for t in store.list_tasks_by_pipeline(project_id) if t.status is TaskStatus.ACCEPTED]
    print(f"  ACCEPTED tasks: {len(all_tasks):,}")

    affected_tasks: list[tuple] = []   # (task_id, payload_with_removals, n_removed)
    affected_spans: set[str] = set()   # spans that actually appeared in at least one task

    for i, task in enumerate(all_tasks):
        task_id = task.task_id
        payload = _load_annotation(store, task)
        if not isinstance(payload, dict):
            continue

        new_payload, n_removed = _remove_spans_from_payload(payload, target_spans)
        if n_removed == 0:
            continue

        # Track which target spans actually appeared in this task.
        rows_list = payload.get("rows") or []
        for trow in rows_list:
            if not isinstance(trow, dict):
                continue
            output = trow.get("output") or {}
            for fk in ("entities", "json_structures"):
                container = output.get(fk) or {}
                for items in container.values():
                    if not isinstance(items, list):
                        continue
                    for s in items:
                        if isinstance(s, str) and s.strip().lower() in target_spans:
                            affected_spans.add(s.strip().lower())

        affected_tasks.append((task_id, new_payload, n_removed))

        if (i + 1) % 500 == 0:
            print(f"  Scanned {i + 1:,}/{len(all_tasks):,} tasks, {len(affected_tasks):,} affected so far…")

    print(f"  Affected tasks: {len(affected_tasks):,}")
    print(f"  Target spans with task hits: {len(affected_spans):,}")
    total_span_removals = sum(n for _, _, n in affected_tasks)
    print(f"  Total span occurrences to remove: {total_span_removals:,}")

    if dry_run:
        print("\n[DRY RUN] No changes written.  Pass --apply to execute.")
        return

    # ── 3. Write corrected annotations ────────────────────────────────────────
    print(f"\n── Step 3: writing human_review_answer artifacts ({len(affected_tasks):,} tasks) ──")
    note = f"bulk_set_lowinfo_not_an_entity: wordfreq>={threshold} → not_an_entity"
    for i, (task_id, new_payload, n_removed) in enumerate(affected_tasks):
        _write_hr_answer(store, task_id, new_payload, note=note)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1:,}/{len(affected_tasks):,} done…")
    print(f"  {len(affected_tasks):,} artifacts written.")

    # ── 4. Purge entity_statistics for all target spans ────────────────────────
    # These are all not_an_entity now — there is no entity type to count.
    # Using affected_spans (those that actually had task hits) to avoid
    # deleting rows for spans that never appeared in any task (they'd have
    # no effect anyway, but let's be precise).
    print(f"\n── Step 4: purging entity_statistics for {len(affected_spans):,} affected spans ──")
    now_iso = datetime.now(timezone.utc).isoformat()
    deleted = 0
    with store._conn:
        for span_lower in affected_spans:
            cur = store._conn.execute(
                "DELETE FROM entity_statistics WHERE project_id = ? AND span_lower = ?",
                (project_id, span_lower),
            )
            deleted += cur.rowcount
    print(f"  Deleted {deleted:,} entity_statistics rows.")

    # ── 5. Clear posterior_audit_cache ────────────────────────────────────────
    print("\n── Step 5: clearing posterior_audit_cache ──")
    with store._conn:
        store._conn.execute(
            "DELETE FROM posterior_audit_cache WHERE project_id = ?",
            (project_id,),
        )
    print("  Cache cleared.")

    print(f"\n✓ Done.  {len(affected_tasks):,} tasks patched, {total_span_removals:,} span occurrences removed.")
    print("  Next /posterior-audit call will rebuild from scratch.")


if __name__ == "__main__":
    main()

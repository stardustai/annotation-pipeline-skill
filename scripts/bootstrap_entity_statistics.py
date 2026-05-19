"""One-time backfill of entity_statistics from existing ACCEPTED tasks.

Scans every ACCEPTED task's final artifact (human_review_answer first,
otherwise the latest annotation_result), iterates its (span, type)
pairs, and increments entity_statistics. HR-authored answers receive
HR_WEIGHT (5x); all other paths receive +1.

The historical sample is naturally "clean" because all current ACCEPTED
tasks predate the dictionary-injection feature — their decisions weren't
conditioned on any convention block in the prompt.

Usage:
  python scripts/bootstrap_entity_statistics.py <workspace_root>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.services.entity_statistics_service import (
    HR_WEIGHT,
    EntityStatisticsService,
    iter_span_decisions,
)
from annotation_pipeline_skill.services.row_mask_service import RowMaskService
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _load_payload(store: SqliteStore, artifact) -> dict | None:
    path = store.root / artifact.path
    if not path.exists():
        return None
    try:
        outer = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(outer, dict):
        return None
    if artifact.kind == "human_review_answer":
        ans = outer.get("answer")
        return ans if isinstance(ans, dict) else None
    text = outer.get("text")
    if not isinstance(text, str):
        return None
    try:
        return json.loads(_strip_think(text))
    except (json.JSONDecodeError, ValueError):
        return None


def _pick_final_artifact(store: SqliteStore, task_id: str):
    arts = store.list_artifacts(task_id)
    hr = [a for a in arts if a.kind == "human_review_answer"]
    if hr:
        return hr[-1], True
    anns = [a for a in arts if a.kind == "annotation_result"]
    return (anns[-1], False) if anns else (None, False)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace_root", type=Path)
    args = parser.parse_args(argv)

    store = SqliteStore.open(args.workspace_root)
    svc = EntityStatisticsService(store)
    mask_svc = RowMaskService(store)
    tasks = list(store.list_tasks_by_status({TaskStatus.ACCEPTED}))
    print(f"scanning {len(tasks)} ACCEPTED tasks...", file=sys.stderr)

    # Bulk-load all row masks upfront to avoid N+1 queries.
    task_ids = [t.task_id for t in tasks]
    masked_by_task = mask_svc.masked_indices_by_task(task_ids)

    incremented = 0
    skipped_no_artifact = 0
    skipped_parse_fail = 0
    for task in tasks:
        artifact, is_hr = _pick_final_artifact(store, task.task_id)
        if artifact is None:
            skipped_no_artifact += 1
            continue
        payload = _load_payload(store, artifact)
        if payload is None:
            skipped_parse_fail += 1
            continue
        weight = HR_WEIGHT if is_hr else 1
        masked_indices = masked_by_task.get(task.task_id) or set()
        for span, entity_type in iter_span_decisions(payload, masked_indices=masked_indices):
            svc.increment(
                project_id=task.pipeline_id,
                span=span,
                entity_type=entity_type,
                weight=weight,
            )
            incremented += 1

    print(json.dumps({
        "tasks_scanned": len(tasks),
        "increments_recorded": incremented,
        "skipped_no_artifact": skipped_no_artifact,
        "skipped_parse_fail": skipped_parse_fail,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

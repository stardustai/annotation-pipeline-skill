#!/usr/bin/env python3
"""
Migrate v4_ner_phrase ACCEPTED tasks → v5_ner_phrase project (B+prelabel).

Same-schema re-run: v5 keeps v4's 18-type output_schema, so the annotation
transform is IDENTITY (no entity-type rename). Each v4 ACCEPTED task's CURRENT
final annotation (post-cleanup) is carried into v5 as a pre-label
annotation_result, the v5 task is created at QC status, and bound to the v5
annotation-rules guideline (latest version) so the refined v5 rules drive
QC + any re-annotation.

QC re-validates the prelabel against the v5 rules: pass → ACCEPTED (fast path);
fail → ANNOTATING (re-annotate under v5 rules).

Usage:
    python scripts/migrate_v4_to_v5.py --pilot 10
    python scripts/migrate_v4_to_v5.py
    python scripts/migrate_v4_to_v5.py --force-rewrite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from annotation_pipeline_skill.core.models import ArtifactRef, Attempt, Task
from annotation_pipeline_skill.core.states import AttemptStatus, TaskStatus
from annotation_pipeline_skill.core.transitions import transition_task
from annotation_pipeline_skill.services.entity_statistics_service import (
    _load_latest_annotation,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore

V4_ANNOTATION_DIR = REPO_ROOT / "projects" / "v4_ner_phrase" / ".annotation-pipeline"
V5_ANNOTATION_DIR = REPO_ROOT / "projects" / "v5_ner_phrase" / ".annotation-pipeline"
V4_PIPELINE_ID = "v4_ner_phrase"
V5_PIPELINE_ID = "v5_ner_phrase"
V5_TASK_PREFIX = "v5_ner_phrase"


def _resolve_guideline_version_id(store: SqliteStore) -> str:
    """Return the latest annotation_rules document version_id in the store."""
    for doc in store.list_documents():
        if isinstance(doc.metadata, dict) and doc.metadata.get("role") == "annotation_rules":
            versions = store.list_document_versions(doc.document_id)
            if not versions:
                raise SystemExit("error: annotation_rules doc has no versions")
            latest = max(versions, key=lambda v: v.created_at)
            return latest.version_id
    raise SystemExit("error: no annotation_rules document in v5 — create the guideline first")


def _source_rows(task: Task) -> list[dict]:
    """Extract the input rows from a v4 task's source_ref payload."""
    sr = task.source_ref if isinstance(task.source_ref, dict) else {}
    payload = sr.get("payload", {}) if isinstance(sr.get("payload"), dict) else {}
    rows = payload.get("rows")
    return rows if isinstance(rows, list) and rows else []


def _make_prelabel_artifact(
    *, v5_store: SqliteStore, task_id: str, annotation_payload: dict
) -> ArtifactRef:
    """Write the prelabel annotation_result file and return its ArtifactRef.
    metadata.runtime='import' marks it as a pre-label for the runtime."""
    relative_path = f"artifact_payloads/{task_id}/prelabeled-annotation.json"
    payload_path = v5_store.root / relative_path
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    file_content = {
        "task_id": task_id,
        "text": json.dumps(annotation_payload, ensure_ascii=False, sort_keys=True),
        "raw_response": {"source": "v4_migration"},
        "usage": {},
        "diagnostics": {"source": "v4_migration"},
    }
    payload_path.write_text(
        json.dumps(file_content, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return ArtifactRef.new(
        task_id=task_id,
        kind="annotation_result",
        path=relative_path,
        content_type="application/json",
        metadata={
            "runtime": "import",
            "provider": "v4_migration",
            "model": "v4_accepted",
            "diagnostics": {"source": "v4_migration"},
        },
    )


def migrate(*, pilot: int | None = None, force_rewrite: bool = False, verbose: bool = False) -> None:
    v4_store = SqliteStore.open(V4_ANNOTATION_DIR)
    v5_store = SqliteStore.open(V5_ANNOTATION_DIR)

    rules_version_id = _resolve_guideline_version_id(v5_store)
    print(f"v5 guideline version_id: {rules_version_id}")

    v4_tasks = [
        t for t in v4_store.list_tasks_by_pipeline(V4_PIPELINE_ID)
        if t.status is TaskStatus.ACCEPTED
    ]
    v4_tasks.sort(key=lambda t: t.task_id)  # stable idx ordering
    total = len(v4_tasks)
    if pilot is not None:
        v4_tasks = v4_tasks[:pilot]
        print(f"PILOT mode: processing {len(v4_tasks)} of {total} accepted tasks")
    else:
        print(f"Processing all {total} accepted tasks")

    existing_ids = {t.task_id for t in v5_store.list_tasks()}
    imported = skipped = 0
    errors: list[tuple[str, str, str]] = []

    for idx, src in enumerate(v4_tasks):
        v5_task_id = f"{V5_TASK_PREFIX}-{idx:06d}"
        if v5_task_id in existing_ids:
            if force_rewrite:
                v5_store.delete_task(v5_task_id)
            else:
                skipped += 1
                continue

        rows = _source_rows(src)
        if not rows:
            errors.append((v5_task_id, src.task_id, "no source rows"))
            continue
        annotation_payload = _load_latest_annotation(v4_store, src.task_id)
        if not isinstance(annotation_payload, dict):
            errors.append((v5_task_id, src.task_id, "no loadable annotation"))
            continue

        task = Task.new(
            task_id=v5_task_id,
            pipeline_id=V5_PIPELINE_ID,
            source_ref={
                "kind": "v4_migration",
                "v4_task_id": src.task_id,
                "batch_index": idx,
                "row_count": len(rows),
                "payload": {"rows": rows},
            },
            modality="text",
            annotation_requirements={"annotation_types": ["entity_span", "json_structure"]},
            metadata={
                "prelabeled": True,
                "prelabel_source": "v4_migration",
                "v4_task_id": src.task_id,
                "batch_size": len(rows),
                "row_ids": [r.get("row_id", f"row-{i}") for i, r in enumerate(rows)],
            },
            document_version_id=rules_version_id,
        )

        ev_pending = transition_task(
            task, TaskStatus.PENDING,
            actor="migrate_v4_to_v5", reason="migrated from v4 accepted", stage="prepare",
            metadata={"v4_task_id": src.task_id},
        )
        ev_annotating = transition_task(
            task, TaskStatus.ANNOTATING,
            actor="migrate_v4_to_v5", reason="pre-label injected", stage="annotation",
        )
        ev_qc = transition_task(
            task, TaskStatus.QC,
            actor="migrate_v4_to_v5", reason="pre-label ready for QC", stage="qc",
        )

        artifact = _make_prelabel_artifact(
            v5_store=v5_store, task_id=v5_task_id, annotation_payload=annotation_payload,
        )

        v5_store.save_task(task)
        for ev in (ev_pending, ev_annotating, ev_qc):
            v5_store.append_event(ev)
        v5_store.append_artifact(artifact)
        v5_store.append_attempt(
            Attempt(
                attempt_id=f"{v5_task_id}-attempt-0-prelabel",
                task_id=v5_task_id,
                index=0,
                stage="annotation",
                status=AttemptStatus.SUCCEEDED,
                started_at=task.created_at,
                finished_at=task.created_at,
                provider_id="v4_migration",
                model="v4_accepted",
                route_role="import",
                summary=f"migrated from v4 task {src.task_id}",
                artifacts=[artifact],
            )
        )

        imported += 1
        if verbose or (imported % 250 == 0):
            print(f"  [{idx:6d}] {v5_task_id} ← {src.task_id}  (imported: {imported})")

    print(f"\nDone.\n  Imported: {imported}\n  Skipped : {skipped}\n  Errors  : {len(errors)}")
    for v5_id, v4_id, reason in errors[:20]:
        print(f"  ERROR {v5_id} ← {v4_id}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate v4 accepted tasks → v5_ner_phrase (prelabel)")
    parser.add_argument("--pilot", type=int, default=None)
    parser.add_argument("--force-rewrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    migrate(pilot=args.pilot, force_rewrite=args.force_rewrite, verbose=args.verbose)


if __name__ == "__main__":
    main()

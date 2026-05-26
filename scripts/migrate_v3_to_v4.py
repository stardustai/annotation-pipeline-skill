#!/usr/bin/env python3
"""
Migrate v3_initial_deployment ACCEPTED tasks → v4_ner_phrase project.

Strategy: B+prelabel
  - Read each v3 ACCEPTED task from its SQLite store
  - Migrate entity → generic_entity in the annotation payload
  - Create a v4 task at QC status with the migrated annotation as a
    pre-label annotation_result artifact
  - QC failure path (handled by runtime): task goes back to ANNOTATING

Usage:
    python scripts/migrate_v3_to_v4.py --pilot 10
    python scripts/migrate_v3_to_v4.py
    python scripts/migrate_v3_to_v4.py --force-rewrite
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ── project root on sys.path so we can import annotation_pipeline_skill ──────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from annotation_pipeline_skill.core.models import (
    AnnotationDocument,
    AnnotationDocumentVersion,
    ArtifactRef,
    Attempt,
    AuditEvent,
    Task,
)
from annotation_pipeline_skill.core.states import AttemptStatus, TaskStatus
from annotation_pipeline_skill.core.transitions import transition_task
from annotation_pipeline_skill.store.sqlite_store import SqliteStore

# ─── constants ───────────────────────────────────────────────────────────────
V3_PROJECT_ROOT = REPO_ROOT / "projects" / "v3_initial_deployment"
V4_PROJECT_ROOT = REPO_ROOT / "projects" / "v4_ner_phrase"
V3_ANNOTATION_DIR = V3_PROJECT_ROOT / ".annotation-pipeline"
V4_ANNOTATION_DIR = V4_PROJECT_ROOT / ".annotation-pipeline"

V4_PIPELINE_ID = "v4_ner_phrase"
V4_TASK_PREFIX = "v4_ner_phrase"


# ─── helpers ─────────────────────────────────────────────────────────────────

def _migrate_entities(entities: dict) -> dict:
    """Rename 'entity' key → 'generic_entity' in an entities dict."""
    if not isinstance(entities, dict):
        return entities
    if "entity" not in entities:
        return entities
    out = dict(entities)
    out["generic_entity"] = out.pop("entity")
    return out


def _migrate_annotation_payload(payload: dict) -> dict:
    """Walk rows and rename entity→generic_entity in each row's output."""
    if not isinstance(payload, dict):
        return payload
    rows = payload.get("rows", [])
    new_rows = []
    for row in rows:
        new_row = dict(row)
        output = row.get("output", {})
        if isinstance(output, dict) and "entities" in output:
            new_output = dict(output)
            new_output["entities"] = _migrate_entities(output["entities"])
            new_row["output"] = new_output
        new_rows.append(new_row)
    return {**payload, "rows": new_rows}


def _get_latest_annotation(
    v3_cur: sqlite3.Cursor,
    task_id: str,
    v3_base: Path,
) -> dict | None:
    """
    Return the parsed annotation payload for an accepted v3 task.
    Prefers human_review_answer over annotation_result (highest seq wins).
    The payload is the inner 'answer' dict from human_review_answer,
    or the inner parsed 'text' from annotation_result.
    """
    # Try human_review_answer first (highest seq)
    v3_cur.execute(
        "SELECT path FROM artifact_refs WHERE task_id=? AND kind='human_review_answer' ORDER BY seq DESC LIMIT 1",
        (task_id,),
    )
    row = v3_cur.fetchone()
    if row:
        path = v3_base / row["path"]
        raw = json.loads(path.read_text(encoding="utf-8"))
        # human_review_answer format: {"actor": "...", "answer": {"rows": [...]}}
        answer = raw.get("answer", raw)
        return answer

    # Fall back to latest annotation_result
    v3_cur.execute(
        "SELECT path FROM artifact_refs WHERE task_id=? AND kind='annotation_result' ORDER BY seq DESC LIMIT 1",
        (task_id,),
    )
    row = v3_cur.fetchone()
    if row:
        path = v3_base / row["path"]
        raw = json.loads(path.read_text(encoding="utf-8"))
        # annotation_result format: {"task_id": ..., "text": "<json>", ...}
        text = raw.get("text", "")
        if isinstance(text, str) and text.strip():
            return json.loads(text)
        # Some formats store the payload directly
        return raw.get("rows") and raw or None

    return None


def _make_artifact_file(
    *,
    v4_store: SqliteStore,
    task_id: str,
    annotation_payload: dict,
) -> tuple[str, ArtifactRef]:
    """Write the annotation_result file and return (rel_path, ArtifactRef)."""
    relative_path = f"artifact_payloads/{task_id}/prelabeled-annotation.json"
    payload_path = v4_store.root / relative_path
    payload_path.parent.mkdir(parents=True, exist_ok=True)

    payload_text = json.dumps(annotation_payload, ensure_ascii=False, sort_keys=True)
    file_content = {
        "task_id": task_id,
        "text": payload_text,
        "raw_response": {"source": "v3_migration"},
        "usage": {},
        "diagnostics": {"source": "v3_migration"},
    }
    payload_path.write_text(
        json.dumps(file_content, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    artifact = ArtifactRef.new(
        task_id=task_id,
        kind="annotation_result",
        path=relative_path,
        content_type="application/json",
        metadata={
            "runtime": "import",
            "provider": "v3_migration",
            "model": "v3_accepted",
            "diagnostics": {"source": "v3_migration"},
        },
    )
    return relative_path, artifact


def _init_v4_project(v4_store: SqliteStore, v3_annotation_dir: Path) -> str:
    """
    Create the annotation rules document in the v4 store.
    Reads v9 content from v3, replaces 'entity' references with 'generic_entity'.
    Returns the version_id of the created v1 rule.
    """
    # Read v9 from v3
    v9_content = (
        v3_annotation_dir
        / "document_versions"
        / "doc-a6a843cf20d548eb9978568b49d2965f"
        / "v9.md"
    ).read_text(encoding="utf-8")

    # Patch entity references → generic_entity in the rules text
    v1_content = (
        v9_content
        .replace('"entity"', '"generic_entity"')
        .replace("Use \"entity\"", 'Use "generic_entity"')
        .replace("use \"entity\"", 'use "generic_entity"')
        .replace("entities.entity", "entities.generic_entity")
        .replace("type: entity", "type: generic_entity")
        # Description line in rules: "entity" only for genuinely ambiguous...
        .replace(
            '      Use "generic_entity" only for genuinely ambiguous',
            '      Use "generic_entity" only for genuinely ambiguous',
        )
    )

    # Check if a document already exists
    docs = v4_store.list_documents()
    annotation_rules_doc = next(
        (d for d in docs if d.metadata.get("role") == "annotation_rules"), None
    )

    if annotation_rules_doc is None:
        doc = AnnotationDocument.new(
            title="Annotation Rules",
            description="NER + phrase annotation rules for v4_ner_phrase",
            created_by="migrate_v3_to_v4",
            metadata={"role": "annotation_rules"},
        )
        v4_store.save_document(doc)
        annotation_rules_doc = doc

    # Check if v1 already exists
    existing_versions = v4_store.list_document_versions(annotation_rules_doc.document_id)
    if any(v.version == "v1" for v in existing_versions):
        ver = next(v for v in existing_versions if v.version == "v1")
        print(f"  annotation rules v1 already exists: {ver.version_id}")
        return ver.version_id

    ver = AnnotationDocumentVersion.new(
        document_id=annotation_rules_doc.document_id,
        version="v1",
        content=v1_content,
        changelog="Copied from v3_initial_deployment v9; entity → generic_entity",
        created_by="migrate_v3_to_v4",
    )
    v4_store.save_document_version(ver)
    print(f"  Created annotation rules v1: {ver.version_id}")
    return ver.version_id


def migrate(
    *,
    pilot: int | None = None,
    force_rewrite: bool = False,
    verbose: bool = False,
) -> None:
    # ── Open v3 (read-only via plain sqlite3) ──────────────────────────────
    v3_db = sqlite3.connect(f"file:{V3_ANNOTATION_DIR / 'db.sqlite'}?mode=ro", uri=True)
    v3_db.row_factory = sqlite3.Row
    v3_cur = v3_db.cursor()

    # ── Open / create v4 store ─────────────────────────────────────────────
    V4_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure subdirectories
    for subdir in (
        "tasks", "events", "feedback", "feedback_discussions",
        "attempts", "artifacts", "outbox", "exports", "runtime",
        "snapshots", "coordination", "documents", "document_versions",
    ):
        (V4_ANNOTATION_DIR / subdir).mkdir(parents=True, exist_ok=True)

    v4_store = SqliteStore.open(V4_ANNOTATION_DIR)

    # ── Set up annotation rules document ───────────────────────────────────
    print("Setting up v4 annotation rules document...")
    rules_version_id = _init_v4_project(v4_store, V3_ANNOTATION_DIR)

    # ── Fetch v3 accepted tasks ────────────────────────────────────────────
    v3_cur.execute(
        "SELECT task_id, source_ref_json, metadata_json FROM tasks WHERE status='accepted' ORDER BY task_id"
    )
    v3_tasks = v3_cur.fetchall()

    total = len(v3_tasks)
    if pilot is not None:
        v3_tasks = v3_tasks[:pilot]
        print(f"PILOT mode: processing {len(v3_tasks)} of {total} accepted tasks")
    else:
        print(f"Processing all {total} accepted tasks")

    # ── Existing v4 task_ids (for idempotency) ────────────────────────────
    existing_ids = {t.task_id for t in v4_store.list_tasks()}

    imported = 0
    skipped = 0
    errors = []

    for idx, v3_row in enumerate(v3_tasks):
        v3_task_id = v3_row["task_id"]
        v4_task_id = f"{V4_TASK_PREFIX}-{idx:06d}"

        if v4_task_id in existing_ids and not force_rewrite:
            if verbose:
                print(f"  [{idx:5d}] SKIP {v4_task_id} (already exists)")
            skipped += 1
            continue

        if v4_task_id in existing_ids and force_rewrite:
            v4_store.delete_task(v4_task_id)

        # ── Get source rows from v3 ────────────────────────────────────
        v3_source = json.loads(v3_row["source_ref_json"])
        payload = v3_source.get("payload", {})
        rows_data = payload.get("rows", [])
        if not rows_data and "input" in payload:
            # Single-row jsonl format
            rows_data = [{"row_index": 0, "row_id": "row-0", "input": payload["input"]}]

        if not rows_data:
            errors.append((v4_task_id, v3_task_id, "no rows in source_ref"))
            continue

        # ── Get accepted annotation from v3 ───────────────────────────
        annotation_payload = _get_latest_annotation(v3_cur, v3_task_id, V3_ANNOTATION_DIR)
        if annotation_payload is None:
            errors.append((v4_task_id, v3_task_id, "no annotation found"))
            continue

        # ── Migrate entity → generic_entity ───────────────────────────
        migrated_payload = _migrate_annotation_payload(annotation_payload)

        # ── Create v4 task ─────────────────────────────────────────────
        v3_meta = json.loads(v3_row["metadata_json"] or "{}")
        task = Task.new(
            task_id=v4_task_id,
            pipeline_id=V4_PIPELINE_ID,
            source_ref={
                "kind": "v3_migration",
                "v3_task_id": v3_task_id,
                "batch_index": idx,
                "row_count": len(rows_data),
                "payload": {"rows": rows_data},
            },
            modality="text",
            annotation_requirements={"annotation_types": ["entity_span", "json_structure"]},
            metadata={
                "prelabeled": True,
                "prelabel_source": "v3_migration",
                "v3_task_id": v3_task_id,
                "batch_size": len(rows_data),
                "row_ids": [r.get("row_id", f"row-{i}") for i, r in enumerate(rows_data)],
            },
            document_version_id=rules_version_id,
        )

        # ── Transition DRAFT → PENDING → ANNOTATING → QC ─────────────
        ev_pending = transition_task(
            task, TaskStatus.PENDING,
            actor="migrate_v3_to_v4", reason="migrated from v3 accepted", stage="prepare",
            metadata={"v3_task_id": v3_task_id},
        )
        ev_annotating = transition_task(
            task, TaskStatus.ANNOTATING,
            actor="migrate_v3_to_v4", reason="pre-label injected", stage="annotation",
        )
        ev_qc = transition_task(
            task, TaskStatus.QC,
            actor="migrate_v3_to_v4", reason="pre-label ready for QC", stage="qc",
        )

        # ── Write annotation artifact file ────────────────────────────
        _, artifact = _make_artifact_file(
            v4_store=v4_store,
            task_id=v4_task_id,
            annotation_payload=migrated_payload,
        )

        # ── Persist ───────────────────────────────────────────────────
        v4_store.save_task(task)
        for ev in (ev_pending, ev_annotating, ev_qc):
            v4_store.append_event(ev)
        v4_store.append_artifact(artifact)
        v4_store.append_attempt(
            Attempt(
                attempt_id=f"{v4_task_id}-attempt-0-prelabel",
                task_id=v4_task_id,
                index=0,
                stage="annotation",
                status=AttemptStatus.SUCCEEDED,
                started_at=task.created_at,
                finished_at=task.created_at,
                provider_id="v3_migration",
                model="v3_accepted",
                route_role="import",
                summary=f"migrated from v3 task {v3_task_id}",
                artifacts=[artifact],
            )
        )

        imported += 1
        if verbose or (imported % 100 == 0):
            print(f"  [{idx:5d}] {v4_task_id} ← {v3_task_id}  (total imported: {imported})")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\nDone.")
    print(f"  Imported : {imported}")
    print(f"  Skipped  : {skipped}")
    print(f"  Errors   : {len(errors)}")
    if errors:
        print("\nErrors:")
        for v4_id, v3_id, reason in errors:
            print(f"  {v4_id} ← {v3_id}: {reason}")

    v3_db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate v3 accepted tasks to v4_ner_phrase")
    parser.add_argument(
        "--pilot", type=int, default=None,
        help="Only migrate the first N tasks (for testing)",
    )
    parser.add_argument(
        "--force-rewrite", action="store_true",
        help="Delete and re-create v4 tasks that already exist",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every task",
    )
    args = parser.parse_args()
    migrate(pilot=args.pilot, force_rewrite=args.force_rewrite, verbose=args.verbose)


if __name__ == "__main__":
    main()

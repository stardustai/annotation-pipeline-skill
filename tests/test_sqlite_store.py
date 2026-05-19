import json
from datetime import datetime, timezone
from pathlib import Path

from annotation_pipeline_skill.core.models import AuditEvent, Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def test_open_creates_schema_and_sets_pragmas(tmp_path: Path):
    store = SqliteStore.open(tmp_path)

    assert (tmp_path / "db.sqlite").exists()
    # foreign_keys is a per-connection pragma — verify it on the store's own
    # connection. journal_mode and user_version are persisted in the database
    # file so they're visible from any connection.
    conn = store._conn
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tasks", "audit_events", "attempts", "feedback_records", "outbox_records",
            "runtime_leases", "documents", "document_versions", "export_manifests"} <= names
    store.close()


def test_open_is_idempotent_on_existing_db(tmp_path: Path):
    SqliteStore.open(tmp_path).close()
    store = SqliteStore.open(tmp_path)
    store.close()


def _make_task(task_id: str, pipeline_id: str = "pipe-1", status: TaskStatus = TaskStatus.DRAFT) -> Task:
    task = Task.new(task_id=task_id, pipeline_id=pipeline_id, source_ref={"kind": "jsonl"})
    task.status = status
    return task


def test_save_and_load_task(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = _make_task("task-1")

    store.save_task(task)
    loaded = store.load_task("task-1")

    assert loaded == task
    store.close()


def test_save_task_is_upsert(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = _make_task("task-1")
    store.save_task(task)

    task.status = TaskStatus.PENDING
    task.metadata = {"note": "updated"}
    store.save_task(task)

    loaded = store.load_task("task-1")
    assert loaded.status is TaskStatus.PENDING
    assert loaded.metadata == {"note": "updated"}
    store.close()


def test_save_task_auto_stamps_brand_new_with_active_rules_version(tmp_path):
    """When a brand-new task is saved and an active annotation-rules
    document exists, the task should be stamped with that version_id.
    """
    from annotation_pipeline_skill.core.models import (
        AnnotationDocument,
        AnnotationDocumentVersion,
    )

    store = SqliteStore.open(tmp_path)
    doc = AnnotationDocument.new(
        title="Annotation Rules",
        description="",
        created_by="system",
        metadata={"role": "annotation_rules"},
    )
    store.save_document(doc)
    ver = AnnotationDocumentVersion.new(
        document_id=doc.document_id,
        version="v1",
        content="rule body",
        changelog="initial",
        created_by="system",
    )
    store.save_document_version(ver)

    task = _make_task("task-new")
    assert task.document_version_id is None
    store.save_task(task)

    loaded = store.load_task("task-new")
    assert loaded.document_version_id == ver.version_id
    store.close()


def test_save_task_does_not_stamp_historical_null_version_tasks(tmp_path):
    """A historical task already in the DB with document_version_id = NULL
    must STAY NULL across subsequent save_task calls — otherwise we'd be
    fabricating provenance ("annotated against rules v3" when it was
    actually annotated under the pre-versioning era).
    """
    from annotation_pipeline_skill.core.models import (
        AnnotationDocument,
        AnnotationDocumentVersion,
    )

    store = SqliteStore.open(tmp_path)
    # Historical task lands first, BEFORE any rules document exists.
    task = _make_task("task-historical")
    store.save_task(task)
    assert store.load_task("task-historical").document_version_id is None

    # Operator later creates the first versioned rules document.
    doc = AnnotationDocument.new(
        title="Annotation Rules",
        description="",
        created_by="system",
        metadata={"role": "annotation_rules"},
    )
    store.save_document(doc)
    ver = AnnotationDocumentVersion.new(
        document_id=doc.document_id,
        version="v1",
        content="rule body",
        changelog="initial",
        created_by="system",
    )
    store.save_document_version(ver)

    # Any subsequent save_task on the historical row (status flip, etc.)
    # MUST NOT silently backfill document_version_id — that'd be a lie.
    task.status = TaskStatus.PENDING
    store.save_task(task)
    assert store.load_task("task-historical").document_version_id is None

    # A brand-new task created after the version exists, however, gets stamped.
    new_task = _make_task("task-new")
    store.save_task(new_task)
    assert store.load_task("task-new").document_version_id == ver.version_id
    store.close()


def test_list_tasks_returns_all(tmp_path):
    store = SqliteStore.open(tmp_path)
    store.save_task(_make_task("task-1", pipeline_id="a"))
    store.save_task(_make_task("task-2", pipeline_id="b"))

    ids = sorted(t.task_id for t in store.list_tasks())
    assert ids == ["task-1", "task-2"]
    store.close()


def test_list_tasks_by_pipeline_filters(tmp_path):
    store = SqliteStore.open(tmp_path)
    store.save_task(_make_task("a-1", pipeline_id="a"))
    store.save_task(_make_task("a-2", pipeline_id="a"))
    store.save_task(_make_task("b-1", pipeline_id="b"))

    rows = store.list_tasks_by_pipeline("a")
    assert sorted(t.task_id for t in rows) == ["a-1", "a-2"]
    store.close()


def test_list_tasks_by_status_filters(tmp_path):
    store = SqliteStore.open(tmp_path)
    store.save_task(_make_task("draft-1", status=TaskStatus.DRAFT))
    store.save_task(_make_task("pend-1", status=TaskStatus.PENDING))
    store.save_task(_make_task("pend-2", status=TaskStatus.PENDING))

    rows = store.list_tasks_by_status({TaskStatus.PENDING})
    assert sorted(t.task_id for t in rows) == ["pend-1", "pend-2"]
    store.close()


def test_save_task_roundtrips_all_fields(tmp_path):
    from annotation_pipeline_skill.core.models import ExternalTaskRef
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="task-full",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "path": "x.jsonl"},
        external_ref=ExternalTaskRef(
            system_id="sys-1",
            external_task_id="ext-1",
            source_url="http://example.com/1",
            idempotency_key="key-1",
        ),
        modality="text",
        annotation_requirements={"schema": "ner"},
        selected_annotator_id="annot-A",
        metadata={"note": "fully populated"},
        document_version_id="docver-1",
    )
    store.save_task(task)
    loaded = store.load_task("task-full")
    assert loaded == task
    store.close()


def test_append_and_list_events_preserves_order(tmp_path):
    store = SqliteStore.open(tmp_path)
    e1 = AuditEvent.new("task-1", TaskStatus.DRAFT, TaskStatus.PENDING, actor="a", reason="r1", stage="ingest")
    e2 = AuditEvent.new("task-1", TaskStatus.PENDING, TaskStatus.ANNOTATING, actor="a", reason="r2", stage="annotate")

    store.append_event(e1)
    store.append_event(e2)

    rows = store.list_events("task-1")
    assert [e.event_id for e in rows] == [e1.event_id, e2.event_id]
    store.close()


def test_list_events_returns_empty_for_unknown_task(tmp_path):
    store = SqliteStore.open(tmp_path)
    assert store.list_events("nope") == []
    store.close()


from annotation_pipeline_skill.core.models import (
    ArtifactRef, Attempt, FeedbackDiscussionEntry, FeedbackRecord,
)
from annotation_pipeline_skill.core.states import (
    AttemptStatus, FeedbackSeverity, FeedbackSource,
)


def test_append_and_list_attempts(tmp_path):
    store = SqliteStore.open(tmp_path)
    a = Attempt(
        attempt_id="att-1", task_id="task-1", index=0, stage="annotate",
        status=AttemptStatus.SUCCEEDED,
    )
    store.append_attempt(a)
    assert store.list_attempts("task-1") == [a]
    store.close()


def test_append_and_list_feedback(tmp_path):
    store = SqliteStore.open(tmp_path)
    f = FeedbackRecord.new(
        task_id="task-1", attempt_id="att-1",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.ERROR,
        category="missing_entity", message="m", target={"f": "x"},
        suggested_action="rerun", created_by="qc",
    )
    store.append_feedback(f)
    assert store.list_feedback("task-1") == [f]
    store.close()


def test_append_and_list_feedback_discussion(tmp_path):
    store = SqliteStore.open(tmp_path)
    d = FeedbackDiscussionEntry.new(
        task_id="task-1", feedback_id="fb-1",
        role="annotator", stance="agree", message="ok", created_by="annotator-1",
    )
    store.append_feedback_discussion(d)
    assert store.list_feedback_discussions("task-1") == [d]
    store.close()


def test_append_and_list_artifact(tmp_path):
    store = SqliteStore.open(tmp_path)
    a = ArtifactRef.new(
        task_id="task-1", kind="annotation_result",
        path="artifacts/task-1.json", content_type="application/json",
    )
    store.append_artifact(a)
    assert store.list_artifacts("task-1") == [a]
    store.close()


from annotation_pipeline_skill.core.models import OutboxRecord
from annotation_pipeline_skill.core.states import OutboxKind, OutboxStatus


def test_save_and_list_outbox(tmp_path):
    store = SqliteStore.open(tmp_path)
    rec = OutboxRecord.new("task-1", OutboxKind.STATUS, {"foo": "bar"})
    store.save_outbox(rec)

    listed = store.list_outbox()
    assert len(listed) == 1 and listed[0] == rec
    store.close()


def test_save_outbox_is_upsert(tmp_path):
    store = SqliteStore.open(tmp_path)
    rec = OutboxRecord.new("task-1", OutboxKind.STATUS, {"foo": "bar"})
    store.save_outbox(rec)
    rec.status = OutboxStatus.SENT
    store.save_outbox(rec)

    listed = store.list_outbox()
    assert listed[0].status is OutboxStatus.SENT
    store.close()


def test_list_pending_outbox_filters_by_status_and_retry(tmp_path):
    from datetime import datetime, timedelta, timezone
    store = SqliteStore.open(tmp_path)

    a = OutboxRecord.new("t-1", OutboxKind.STATUS, {})
    b = OutboxRecord.new("t-2", OutboxKind.STATUS, {})
    b.status = OutboxStatus.SENT
    c = OutboxRecord.new("t-3", OutboxKind.STATUS, {})
    c.next_retry_at = datetime.now(timezone.utc) + timedelta(hours=1)
    for r in (a, b, c):
        store.save_outbox(r)

    pending = store.list_pending_outbox(now=datetime.now(timezone.utc))
    assert [r.record_id for r in pending] == [a.record_id]
    store.close()


from datetime import datetime, timedelta, timezone

from annotation_pipeline_skill.core.runtime import ActiveRun, RuntimeLease


def _now():
    return datetime.now(timezone.utc)


def test_active_run_save_list_delete(tmp_path):
    store = SqliteStore.open(tmp_path)
    run = ActiveRun(
        run_id="run-1", task_id="t-1", stage="annotate", attempt_id="a-1",
        provider_target="local", started_at=_now(), heartbeat_at=_now(),
    )
    store.save_active_run(run)
    assert store.list_active_runs() == [run]

    store.delete_active_run("run-1")
    assert store.list_active_runs() == []
    store.close()


def test_save_runtime_lease_returns_true_on_first_acquire(tmp_path):
    store = SqliteStore.open(tmp_path)
    lease = RuntimeLease(
        lease_id="L1", task_id="t-1", stage="annotate",
        acquired_at=_now(), heartbeat_at=_now(), expires_at=_now() + timedelta(minutes=10),
        owner="worker-A",
    )
    assert store.save_runtime_lease(lease) is True
    assert len(store.list_runtime_leases()) == 1
    store.close()


def test_save_runtime_lease_returns_false_when_task_stage_locked(tmp_path):
    store = SqliteStore.open(tmp_path)
    a = RuntimeLease(
        lease_id="L1", task_id="t-1", stage="annotate",
        acquired_at=_now(), heartbeat_at=_now(), expires_at=_now() + timedelta(minutes=10),
        owner="worker-A",
    )
    b = RuntimeLease(
        lease_id="L2", task_id="t-1", stage="annotate",
        acquired_at=_now(), heartbeat_at=_now(), expires_at=_now() + timedelta(minutes=10),
        owner="worker-B",
    )
    assert store.save_runtime_lease(a) is True
    assert store.save_runtime_lease(b) is False
    leases = store.list_runtime_leases()
    assert [l.owner for l in leases] == ["worker-A"]
    store.close()


def test_delete_runtime_lease_releases_slot(tmp_path):
    store = SqliteStore.open(tmp_path)
    a = RuntimeLease(
        lease_id="L1", task_id="t-1", stage="annotate",
        acquired_at=_now(), heartbeat_at=_now(), expires_at=_now() + timedelta(minutes=10),
        owner="worker-A",
    )
    store.save_runtime_lease(a)
    store.delete_runtime_lease("L1")

    b = RuntimeLease(
        lease_id="L2", task_id="t-1", stage="annotate",
        acquired_at=_now(), heartbeat_at=_now(), expires_at=_now() + timedelta(minutes=10),
        owner="worker-B",
    )
    assert store.save_runtime_lease(b) is True
    store.close()


def test_coordination_record_append_and_list(tmp_path):
    store = SqliteStore.open(tmp_path)
    store.append_coordination_record("rule_updates", {"id": 1, "project_id": "p"})
    store.append_coordination_record("rule_updates", {"id": 2, "project_id": "p"})
    store.append_coordination_record("long_tail_issues", {"id": 3, "project_id": "p"})

    rules = store.list_coordination_records("rule_updates")
    assert [r["id"] for r in rules] == [1, 2]
    long_tail = store.list_coordination_records("long_tail_issues")
    assert [r["id"] for r in long_tail] == [3]
    store.close()


def test_runtime_heartbeat_roundtrip(tmp_path):
    from datetime import datetime, timezone
    store = SqliteStore.open(tmp_path)
    assert store.load_runtime_heartbeat() is None

    now = datetime.now(timezone.utc)
    store.save_runtime_heartbeat(now)
    loaded = store.load_runtime_heartbeat()
    assert loaded.isoformat() == now.isoformat()
    store.close()


def test_runtime_snapshot_save_and_load(tmp_path):
    from datetime import datetime, timezone
    from annotation_pipeline_skill.core.runtime import (
        CapacitySnapshot, QueueCounts, RuntimeSnapshot, RuntimeStatus,
    )
    store = SqliteStore.open(tmp_path)
    snap = RuntimeSnapshot(
        generated_at=datetime.now(timezone.utc),
        runtime_status=RuntimeStatus(healthy=True, heartbeat_at=None, heartbeat_age_seconds=None, active=False),
        queue_counts=QueueCounts(
            pending=0, annotating=0, qc=0, human_review=0, accepted=0, rejected=0,
        ),
        active_runs=[], capacity=CapacitySnapshot(
            max_concurrent_tasks=4, active_count=0, available_slots=4,
        ),
        stale_tasks=[], due_retries=[], project_summaries=[],
    )
    store.save_runtime_snapshot(snap)
    assert store.load_runtime_snapshot() == snap
    store.close()


def test_save_and_load_document(tmp_path):
    from annotation_pipeline_skill.core.models import AnnotationDocument
    store = SqliteStore.open(tmp_path)
    doc = AnnotationDocument.new(title="t", description="d", created_by="u")
    store.save_document(doc)
    assert store.load_document(doc.document_id) == doc
    assert store.list_documents() == [doc]
    store.close()


def test_save_document_version_writes_content_to_file(tmp_path):
    import hashlib
    from annotation_pipeline_skill.core.models import AnnotationDocument, AnnotationDocumentVersion
    store = SqliteStore.open(tmp_path)
    doc = AnnotationDocument.new(title="t", description="d", created_by="u")
    store.save_document(doc)
    ver = AnnotationDocumentVersion.new(
        document_id=doc.document_id, version="v1", content="# Title\n\nbody",
        changelog="initial", created_by="u",
    )

    store.save_document_version(ver)

    content_path = tmp_path / "document_versions" / doc.document_id / "v1.md"
    assert content_path.exists()
    assert content_path.read_text(encoding="utf-8") == "# Title\n\nbody"

    loaded = store.load_document_version(ver.version_id)
    assert loaded == ver

    versions = store.list_document_versions(doc.document_id)
    assert versions == [ver]
    store.close()


def test_document_version_sha256_is_stored(tmp_path):
    import hashlib
    import sqlite3
    from annotation_pipeline_skill.core.models import AnnotationDocument, AnnotationDocumentVersion
    store = SqliteStore.open(tmp_path)
    doc = AnnotationDocument.new(title="t", description="d", created_by="u")
    store.save_document(doc)
    ver = AnnotationDocumentVersion.new(
        document_id=doc.document_id, version="v1", content="abc",
        changelog="x", created_by="u",
    )
    store.save_document_version(ver)

    expected = hashlib.sha256(b"abc").hexdigest()
    with sqlite3.connect(tmp_path / "db.sqlite") as conn:
        sha = conn.execute(
            "SELECT content_sha256 FROM document_versions WHERE version_id = ?",
            (ver.version_id,),
        ).fetchone()[0]
    assert sha == expected
    store.close()


def test_save_and_list_export_manifest(tmp_path):
    from annotation_pipeline_skill.core.models import ExportManifest
    store = SqliteStore.open(tmp_path)
    m = ExportManifest.new(
        project_id="p", output_paths=["exports/e/training.jsonl"],
        task_ids_included=["t-1"], task_ids_excluded=[],
        artifact_ids=["a-1"], source_files=["in.jsonl"],
        annotation_rules_hash=None, schema_version="v1",
        validator_version="vv1", validation_summary={"ok": 1},
    )
    store.save_export_manifest(m)
    assert store.list_export_manifests() == [m]
    store.close()


def test_delete_pipeline_removes_tasks_and_children(tmp_path):
    store = SqliteStore.open(tmp_path)

    # Two tasks in pipeline_a and one in pipeline_b.
    task_a1 = _make_task("a-1", pipeline_id="pipeline_a")
    task_a2 = _make_task("a-2", pipeline_id="pipeline_a")
    task_b1 = _make_task("b-1", pipeline_id="pipeline_b")
    for t in (task_a1, task_a2, task_b1):
        store.save_task(t)

    # Children for each task — events, attempts, feedback, discussions, artifacts.
    for tid in ("a-1", "a-2", "b-1"):
        store.append_event(
            AuditEvent.new(tid, TaskStatus.DRAFT, TaskStatus.PENDING, actor="x", reason="r", stage="ingest")
        )
        store.append_attempt(
            Attempt(attempt_id=f"{tid}-att-1", task_id=tid, index=0, stage="annotate", status=AttemptStatus.SUCCEEDED)
        )
        store.append_feedback(
            FeedbackRecord.new(
                task_id=tid, attempt_id=f"{tid}-att-1",
                source_stage=FeedbackSource.QC, severity=FeedbackSeverity.ERROR,
                category="cat", message="m", target={"x": "y"},
                suggested_action="rerun", created_by="qc",
            )
        )
        store.append_feedback_discussion(
            FeedbackDiscussionEntry.new(
                task_id=tid, feedback_id="fb-1",
                role="annotator", stance="agree", message="ok", created_by="annotator-1",
            )
        )
        store.append_artifact(
            ArtifactRef.new(
                task_id=tid, kind="annotation_result",
                path=f"artifact_payloads/{tid}/annotation_result.json",
                content_type="application/json",
            )
        )
        store.save_outbox(OutboxRecord.new(tid, OutboxKind.STATUS, {"v": 1}))

    # On-disk artifact files for each task.
    artifact_root = store.root / "artifact_payloads"
    for tid in ("a-1", "a-2", "b-1"):
        d = artifact_root / tid
        d.mkdir(parents=True, exist_ok=True)
        (d / "annotation_result.json").write_text("{}", encoding="utf-8")

    report = store.delete_pipeline("pipeline_a")

    # Tasks: only pipeline_b remains.
    remaining = [t.task_id for t in store.list_tasks()]
    assert remaining == ["b-1"]

    # pipeline_a children gone, pipeline_b children intact.
    assert store.list_events("a-1") == [] and store.list_events("a-2") == []
    assert len(store.list_events("b-1")) == 1
    assert store.list_attempts("a-1") == [] and store.list_attempts("a-2") == []
    assert len(store.list_attempts("b-1")) == 1
    assert store.list_feedback("a-1") == [] and store.list_feedback("a-2") == []
    assert len(store.list_feedback("b-1")) == 1
    assert store.list_feedback_discussions("a-1") == [] and store.list_feedback_discussions("a-2") == []
    assert len(store.list_feedback_discussions("b-1")) == 1
    assert store.list_artifacts("a-1") == [] and store.list_artifacts("a-2") == []
    assert len(store.list_artifacts("b-1")) == 1
    outbox_task_ids = {r.task_id for r in store.list_outbox()}
    assert outbox_task_ids == {"b-1"}

    # On-disk dirs gone for pipeline_a, intact for pipeline_b.
    assert not (artifact_root / "a-1").exists()
    assert not (artifact_root / "a-2").exists()
    assert (artifact_root / "b-1" / "annotation_result.json").exists()

    # Report has correct counts.
    assert report["tasks"] == 2
    assert report["audit_events"] == 2
    assert report["attempts"] == 2
    assert report["feedback_records"] == 2
    assert report["feedback_discussions"] == 2
    assert report["artifact_refs"] == 2
    assert report["outbox_records"] == 2
    assert report["active_runs"] == 0
    assert report["runtime_leases"] == 0
    assert report["artifact_files"] == 2

    store.close()


def test_delete_pipeline_returns_zero_counts_when_pipeline_missing(tmp_path):
    store = SqliteStore.open(tmp_path)
    report = store.delete_pipeline("nope")
    assert report["tasks"] == 0
    assert report["artifact_files"] == 0
    store.close()


def test_store_delete_task_removes_all_children_and_files(tmp_path):
    store = SqliteStore.open(tmp_path)

    # Two tasks: one we'll delete, one we'll keep, both in the same pipeline.
    target = _make_task("target", pipeline_id="pipe-x")
    keep = _make_task("keep", pipeline_id="pipe-x")
    store.save_task(target)
    store.save_task(keep)

    for tid in ("target", "keep"):
        store.append_event(
            AuditEvent.new(tid, TaskStatus.DRAFT, TaskStatus.PENDING, actor="x", reason="r", stage="ingest")
        )
        store.append_attempt(
            Attempt(attempt_id=f"{tid}-att-1", task_id=tid, index=1, stage="annotate", status=AttemptStatus.SUCCEEDED)
        )
        store.append_feedback(
            FeedbackRecord.new(
                task_id=tid, attempt_id=f"{tid}-att-1",
                source_stage=FeedbackSource.QC, severity=FeedbackSeverity.ERROR,
                category="cat", message="m", target={"x": "y"},
                suggested_action="rerun", created_by="qc",
            )
        )
        store.append_feedback_discussion(
            FeedbackDiscussionEntry.new(
                task_id=tid, feedback_id="fb-1",
                role="annotator", stance="agree", message="ok", created_by="annotator-1",
            )
        )
        store.append_artifact(
            ArtifactRef.new(
                task_id=tid, kind="annotation_result",
                path=f"artifact_payloads/{tid}/annotation_result.json",
                content_type="application/json",
            )
        )
        store.save_outbox(OutboxRecord.new(tid, OutboxKind.STATUS, {"v": 1}))

    artifact_root = store.root / "artifact_payloads"
    for tid in ("target", "keep"):
        d = artifact_root / tid
        d.mkdir(parents=True, exist_ok=True)
        (d / "annotation_result.json").write_text("{}", encoding="utf-8")

    report = store.delete_task("target")

    assert report["tasks"] == 1
    assert report["audit_events"] == 1
    assert report["attempts"] == 1
    assert report["feedback_records"] == 1
    assert report["feedback_discussions"] == 1
    assert report["artifact_refs"] == 1
    assert report["outbox_records"] == 1
    assert report["artifact_files"] == 1

    # Target is gone, including all children.
    import pytest as _pytest
    with _pytest.raises(KeyError):
        store.load_task("target")
    assert store.list_events("target") == []
    assert store.list_attempts("target") == []
    assert store.list_feedback("target") == []
    assert store.list_feedback_discussions("target") == []
    assert store.list_artifacts("target") == []
    assert not (artifact_root / "target").exists()

    # Sibling task in same pipeline is untouched.
    assert store.load_task("keep") is not None
    assert len(store.list_events("keep")) == 1
    assert len(store.list_attempts("keep")) == 1
    assert len(store.list_feedback("keep")) == 1
    assert len(store.list_feedback_discussions("keep")) == 1
    assert len(store.list_artifacts("keep")) == 1
    assert (artifact_root / "keep" / "annotation_result.json").exists()

    # Outbox: only "keep" remains.
    outbox_task_ids = {r.task_id for r in store.list_outbox()}
    assert outbox_task_ids == {"keep"}

    # Deleting a non-existent task is a no-op with zero counts.
    empty_report = store.delete_task("does-not-exist")
    assert empty_report["tasks"] == 0
    assert empty_report["artifact_files"] == 0

    store.close()

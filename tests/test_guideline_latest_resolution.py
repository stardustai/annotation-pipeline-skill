"""The runtime must annotate/QC against the LATEST published annotation_rules
version, not the version a task was originally bound to. Republishing the
guideline should immediately govern every in-flight task with no rebind."""
from datetime import datetime, timedelta, timezone

from annotation_pipeline_skill.core.models import (
    AnnotationDocument,
    AnnotationDocumentVersion,
    Task,
)
from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _runtime(store):
    return SubagentRuntime(store, client_factory=lambda target: None)


def _publish(store, doc_id, version, content, *, when):
    ver = AnnotationDocumentVersion(
        version_id=f"docver-{version}",
        document_id=doc_id,
        version=version,
        content=content,
        changelog=f"{version} changelog",
        created_at=when,
        created_by="op",
    )
    store.save_document_version(ver)
    return ver


def test_latest_version_wins_over_bound(tmp_path):
    store = SqliteStore.open(tmp_path / "A")
    doc = AnnotationDocument.new(
        title="Annotation Rules", description="", created_by="op",
        metadata={"role": "annotation_rules"},
    )
    store.save_document(doc)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    v2 = _publish(store, doc.document_id, "v2", "OLD v2 rules", when=t0)
    _publish(store, doc.document_id, "v4", "NEW v4 rules", when=t0 + timedelta(days=10))

    # Task pinned to v2 (as migrate_v4_to_v5 binds them).
    task = Task.new(task_id="t-1", pipeline_id="p",
                    source_ref={"kind": "jsonl", "payload": {"rows": []}},
                    document_version_id=v2.version_id)

    guideline = _runtime(store)._load_guideline(task)
    assert guideline is not None
    assert "NEW v4 rules" in guideline  # latest wins
    assert "OLD v2 rules" not in guideline
    assert "(v4)" in guideline


def test_falls_back_to_bound_when_no_rules_doc(tmp_path):
    """A store with no annotation_rules document still resolves the task's
    bound version (legacy / non-rules documents)."""
    store = SqliteStore.open(tmp_path / "B")
    doc = AnnotationDocument.new(
        title="Some Other Doc", description="", created_by="op",
        metadata={"role": "scratch"},
    )
    store.save_document(doc)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bound = _publish(store, doc.document_id, "x1", "bound content", when=t0)

    task = Task.new(task_id="t-2", pipeline_id="p",
                    source_ref={"kind": "jsonl", "payload": {"rows": []}},
                    document_version_id=bound.version_id)

    guideline = _runtime(store)._load_guideline(task)
    assert guideline is not None
    assert "bound content" in guideline

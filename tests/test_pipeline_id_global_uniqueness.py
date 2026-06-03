"""project_id is the dashboard's sole addressing key (?project= with no
store=), so creating a pipeline_id that already exists in a DIFFERENT store
must be rejected. Adding tasks to the SAME store's existing pipeline is fine."""
import pytest

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.interfaces.cli import (
    assert_pipeline_id_globally_unique,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _seed(project_root, pipeline_id, task_id):
    store = SqliteStore.open(project_root / ".annotation-pipeline")
    t = Task.new(task_id=task_id, pipeline_id=pipeline_id,
                 source_ref={"kind": "jsonl", "payload": {"rows": []}})
    t.status = TaskStatus.PENDING
    store.save_task(t)


def test_rejects_duplicate_pipeline_id_in_other_store(tmp_path):
    ws = tmp_path / "projects"
    (ws / "A").mkdir(parents=True)
    (ws / "B").mkdir(parents=True)
    _seed(ws / "A", "shared", "shared-000000")  # 'shared' lives in store A

    # Creating 'shared' in store B must be rejected (global collision).
    with pytest.raises(SystemExit):
        assert_pipeline_id_globally_unique(ws / "B", "shared", workspace=ws)


def test_allows_unique_pipeline_id(tmp_path):
    ws = tmp_path / "projects"
    (ws / "A").mkdir(parents=True)
    (ws / "B").mkdir(parents=True)
    _seed(ws / "A", "alpha", "alpha-000000")

    # A different name in store B is fine.
    assert_pipeline_id_globally_unique(ws / "B", "beta", workspace=ws)  # no raise


def test_allows_same_store_existing_pipeline(tmp_path):
    """Re-running create/import on the SAME store's pipeline (appending tasks)
    is allowed — the current project root is excluded from the check."""
    ws = tmp_path / "projects"
    (ws / "A").mkdir(parents=True)
    _seed(ws / "A", "alpha", "alpha-000000")

    assert_pipeline_id_globally_unique(ws / "A", "alpha", workspace=ws)  # no raise


def test_allows_when_no_sibling_stores(tmp_path):
    ws = tmp_path / "projects"
    (ws / "A").mkdir(parents=True)
    assert_pipeline_id_globally_unique(ws / "A", "anything", workspace=ws)  # no raise

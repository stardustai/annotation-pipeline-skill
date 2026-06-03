"""With project-only addressing (?project= and no store=), the dashboard
resolves which store/db holds that pipeline_id. An explicit store= still wins."""
from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _store_with_pipeline(root, pipeline_id):
    store = SqliteStore.open(root)
    t = Task.new(task_id=f"{pipeline_id}-000000", pipeline_id=pipeline_id,
                 source_ref={"kind": "jsonl", "payload": {"rows": []}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    return store


def test_resolves_store_from_project_alone(tmp_path):
    a = _store_with_pipeline(tmp_path / "A", "proj_a")
    b = _store_with_pipeline(tmp_path / "B", "proj_b")
    api = DashboardApi(a, stores={"ka": a, "kb": b}, default_store_key="ka")

    assert api._resolve_store({"project": ["proj_b"]}) is b
    assert api._resolve_store({"project": ["proj_a"]}) is a


def test_explicit_store_key_wins_over_project(tmp_path):
    a = _store_with_pipeline(tmp_path / "A", "proj_a")
    b = _store_with_pipeline(tmp_path / "B", "proj_b")
    api = DashboardApi(a, stores={"ka": a, "kb": b}, default_store_key="ka")

    # store= takes precedence even if project points elsewhere.
    assert api._resolve_store({"store": ["ka"], "project": ["proj_b"]}) is a


def test_unknown_project_falls_back_to_default(tmp_path):
    a = _store_with_pipeline(tmp_path / "A", "proj_a")
    b = _store_with_pipeline(tmp_path / "B", "proj_b")
    api = DashboardApi(a, stores={"ka": a, "kb": b}, default_store_key="ka")

    assert api._resolve_store({"project": ["nonexistent"]}) is a  # default

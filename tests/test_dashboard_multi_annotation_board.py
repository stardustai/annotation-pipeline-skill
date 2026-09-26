"""Kanban columns for multi-annotation projects (replicas > 1) and the live sub-stage they use."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.services import dashboard_service
from annotation_pipeline_skill.services.dashboard_service import build_kanban_snapshot
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _project(tmp_path: Path, annotation: dict | None) -> SqliteStore:
    root = tmp_path / "proj" / ".annotation-pipeline"
    root.mkdir(parents=True)
    if annotation is not None:
        (root / "workflow.yaml").write_text(yaml.safe_dump({"stages": {"annotation": annotation}}), encoding="utf-8")
    return SqliteStore.open(root)


def _task(store: SqliteStore, task_id: str, status: TaskStatus, substage: str | None = None) -> None:
    task = Task.new(task_id=task_id, pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = status
    if substage:
        task.metadata["runtime_substage"] = substage
    store.save_task(task)


def _cards(snapshot: dict, column_id: str) -> list[str]:
    column = next(col for col in snapshot["columns"] if col["id"] == column_id)
    return sorted(card["task_id"] for card in column["cards"])


def test_single_annotation_keeps_the_classic_columns(tmp_path):
    dashboard_service._COLUMN_CONFIG_CACHE.clear()
    store = _project(tmp_path, None)
    _task(store, "t1", TaskStatus.QC)
    ids = [col["id"] for col in build_kanban_snapshot(store, project_id="pipe")["columns"]]
    assert "qc" in ids and not any(col.startswith("annotator_") for col in ids)


def test_multi_annotation_has_one_column_per_annotator_and_an_arbiter(tmp_path):
    dashboard_service._COLUMN_CONFIG_CACHE.clear()
    store = _project(tmp_path, {"replicas": 2, "targets": ["annotation", "annotation_2"], "arbiter_target": "arbiter"})
    _task(store, "running", TaskStatus.ANNOTATING, "annotating")
    _task(store, "arbitrating", TaskStatus.ANNOTATING, "arbitrating")
    _task(store, "queued", TaskStatus.PENDING)
    snapshot = build_kanban_snapshot(store, project_id="pipe")
    ids = [col["id"] for col in snapshot["columns"]]
    assert ids == ["pending", "annotator_1", "annotator_2", "arbiter", "human_review", "accepted", "rejected"]
    # the annotators run in parallel: an annotating task is in EVERY annotator column, not the arbiter's
    assert _cards(snapshot, "annotator_1") == ["running"] == _cards(snapshot, "annotator_2")
    assert _cards(snapshot, "arbiter") == ["arbitrating"]
    assert _cards(snapshot, "pending") == ["queued"]


def test_multi_annotation_never_hides_a_task_that_is_in_qc(tmp_path):
    dashboard_service._COLUMN_CONFIG_CACHE.clear()
    store = _project(tmp_path, {"replicas": 2, "targets": ["annotation", "annotation_2"]})
    _task(store, "in-qc", TaskStatus.QC)
    snapshot = build_kanban_snapshot(store, project_id="pipe")
    assert _cards(snapshot, "qc") == ["in-qc"]


def test_column_config_is_cached_until_workflow_yaml_changes(tmp_path):
    dashboard_service._COLUMN_CONFIG_CACHE.clear()
    store = _project(tmp_path, {"replicas": 2, "targets": ["annotation", "annotation_2"]})
    assert dashboard_service._annotation_column_config(store)[0] == 2
    workflow = store.root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump({"stages": {"annotation": {"replicas": 1}}}), encoding="utf-8")
    import os
    os.utime(workflow, (workflow.stat().st_atime, workflow.stat().st_mtime + 5))
    assert dashboard_service._annotation_column_config(store)[0] == 1


def test_stores_listing_reports_the_project_ids_each_store_holds(tmp_path):
    store = _project(tmp_path, None)
    _task(store, "a", TaskStatus.PENDING)
    status, _headers, body = DashboardApi(store, stores={"proj": store}, workspace_root=tmp_path).handle_get("/api/stores")
    assert status == 200
    stores = json.loads(body.decode("utf-8"))["stores"]
    assert [entry["project_ids"] for entry in stores] == [["pipe"]]
    assert stores[0]["pipeline_count"] == 1


def test_board_does_not_need_llm_profiles_beside_the_project(tmp_path):
    """The board reads only workflow.yaml: a workspace whose llm_profiles.yaml is elsewhere
    (or absent) must still get the multi-annotation columns, not a silent single-annotation board."""
    dashboard_service._COLUMN_CONFIG_CACHE.clear()
    store = _project(tmp_path, {"replicas": 2, "targets": ["annotation", "annotation_2"]})
    assert not (store.root / "llm_profiles.yaml").exists()
    assert [col["id"] for col in build_kanban_snapshot(store, project_id="pipe")["columns"]][:4] == [
        "pending", "annotator_1", "annotator_2", "arbiter"]

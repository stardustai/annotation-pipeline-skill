"""Tests for AnnotationPromptBuilder extracted from SubagentRuntime."""
import json
import pytest


def _make_task(task_id="pb-001", rows=None):
    from annotation_pipeline_skill.core.models import Task
    return Task.new(
        task_id=task_id,
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"rows": rows or [{"input": "hello"}]}},
        modality="text",
        annotation_requirements={"annotation_types": ["extraction"]},
        metadata={},
    )


def test_annotation_prompt_is_non_empty_string(tmp_path):
    from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    builder = AnnotationPromptBuilder(store=store, project_id="pipe", config={})
    task = _make_task()
    prompt = builder.build_annotation_prompt(task)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    store.close()


def test_build_conventions_block_returns_none_when_no_conventions(tmp_path):
    from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    builder = AnnotationPromptBuilder(store=store, project_id="pipe", config={})
    task = _make_task()
    result = builder.build_conventions_block(task)
    # No conventions registered → should return None or empty string
    assert result is None or result == ""
    store.close()


def test_annotation_prompt_starts_with_stable_task_key(tmp_path):
    """Prefix-cache locality: the prompt JSON must begin with the stable
    `task` key (source rows, schema) so cross-turn calls of the same task
    share a byte-identical head. sort_keys=True (old behaviour) made
    `feedback_bundle` the first key, busting cache from byte one."""
    from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    builder = AnnotationPromptBuilder(store=store, project_id="pipe", config={})
    task = _make_task()
    prompt = builder.build_annotation_prompt(task)
    assert prompt.startswith('{"task":'), f"prompt head: {prompt[:80]!r}"
    # Volatile sections must sit at the tail so the head stays stable
    # across turns of the same task.
    feedback_pos = prompt.index('"feedback_bundle"')
    task_pos = prompt.index('"task"')
    schema_pos = prompt.index('"output_schema"')
    assert task_pos < schema_pos < feedback_pos
    store.close()


def test_annotation_prompt_filters_unstable_task_metadata(tmp_path):
    """task.metadata must be filtered to a stable allowlist so scheduler
    accounting (bail counts, retry counters, continuity handles) doesn't
    leak into the prompt and bust the prefix-cache prefix from byte one."""
    from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    builder = AnnotationPromptBuilder(store=store, project_id="pipe", config={})
    task = _make_task()
    # Mix mutating (must drop) and stable (must keep) metadata.
    task.metadata.update({
        "worker_bail_count": 7,
        "arbiter_mechanical_retries": 2,
        "continuity_handle": "secret-handle",
        "_ann_sent_feedback_ids": ["x"],
        "qc_policy": {"mode": "all"},      # whitelisted
        "prelabeled": True,                # whitelisted
    })
    prompt = builder.build_annotation_prompt(task)
    assert "worker_bail_count" not in prompt
    assert "arbiter_mechanical_retries" not in prompt
    assert "continuity_handle" not in prompt
    assert "_ann_sent_feedback_ids" not in prompt
    # Allowlisted keys must survive.
    assert "qc_policy" in prompt
    assert "prelabeled" in prompt
    store.close()


def test_qc_prompt_returns_non_empty_string(tmp_path):
    import datetime
    from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import ArtifactRef

    store_root = tmp_path / ".annotation-pipeline"
    store = SqliteStore.open(store_root)
    builder = AnnotationPromptBuilder(store=store, project_id="pipe", config={})
    task = _make_task()

    # Write a minimal annotation artifact the builder can read
    artifact_dir = store_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = artifact_dir / "ann.json"
    artifact_file.write_text(json.dumps({
        "text": json.dumps({"rows": [{"input": "hello", "entities": []}]}),
        "provider": "test",
        "model": "test-model",
    }))

    artifact = ArtifactRef(
        artifact_id="a1",
        task_id="pb-001",
        kind="annotation_result",
        path="artifacts/ann.json",
        content_type="application/json",
        created_at=datetime.datetime.now(datetime.timezone.utc),
        metadata={},
    )

    prompt = builder.build_qc_prompt(task, artifact)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    store.close()

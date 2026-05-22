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

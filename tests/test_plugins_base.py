"""Verify Plugin Protocol contracts are runtime-checkable."""
from __future__ import annotations

import json
import pytest
from annotation_pipeline_skill.plugins.base import DatasetAdapter, Validator, MergeSink


def test_dataset_adapter_protocol_is_checkable():
    class GoodAdapter:
        def load_rows(self, source_path):
            return []

        def make_batches(self, rows, batch_size, *, group_by=None):
            return []

    assert isinstance(GoodAdapter(), DatasetAdapter)


def test_dataset_adapter_missing_method_fails_check():
    class BadAdapter:
        def load_rows(self, source_path):
            return []
        # make_batches missing

    assert not isinstance(BadAdapter(), DatasetAdapter)


def test_validator_protocol_is_checkable():
    class GoodValidator:
        def validate(self, task, payload):
            return None

    assert isinstance(GoodValidator(), Validator)


def test_merge_sink_protocol_is_checkable():
    class GoodSink:
        def write(self, tasks, output_path):
            return 0

    assert isinstance(GoodSink(), MergeSink)


def test_register_and_get_adapter():
    from annotation_pipeline_skill.plugins.registry import register_adapter, get_adapter, _adapters

    class MyAdapter:
        def load_rows(self, source_path): return []
        def make_batches(self, rows, batch_size, *, group_by=None): return []

    _adapters.clear()
    register_adapter("my_format", MyAdapter())
    assert isinstance(get_adapter("my_format"), MyAdapter)


def test_get_unknown_adapter_raises():
    from annotation_pipeline_skill.plugins.registry import get_adapter, _adapters
    import pytest
    _adapters.clear()
    with pytest.raises(KeyError):
        get_adapter("nonexistent")


def test_jsonl_adapter_load_rows(tmp_path):
    from annotation_pipeline_skill.plugins.jsonl_adapter import JsonlDatasetAdapter

    source = tmp_path / "data.jsonl"
    source.write_text(
        json.dumps({"input": "hello"}) + "\n" +
        json.dumps({"input": "world"}) + "\n"
    )
    adapter = JsonlDatasetAdapter()
    rows = adapter.load_rows(source)
    assert len(rows) == 2
    assert rows[0]["input"] == "hello"


def test_jsonl_adapter_make_batches():
    from annotation_pipeline_skill.plugins.jsonl_adapter import JsonlDatasetAdapter

    adapter = JsonlDatasetAdapter()
    rows = [{"input": str(i)} for i in range(7)]
    batches = adapter.make_batches(rows, batch_size=3)
    assert len(batches) == 3
    assert len(batches[0]) == 3
    assert len(batches[2]) == 1


def test_jsonl_adapter_implements_protocol():
    from annotation_pipeline_skill.plugins.jsonl_adapter import JsonlDatasetAdapter
    from annotation_pipeline_skill.plugins.base import DatasetAdapter

    assert isinstance(JsonlDatasetAdapter(), DatasetAdapter)

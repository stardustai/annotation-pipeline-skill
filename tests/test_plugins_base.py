"""Verify Plugin Protocol contracts are runtime-checkable."""
from __future__ import annotations

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

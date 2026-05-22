# annotation_pipeline_skill/plugins/registry.py
from __future__ import annotations

from annotation_pipeline_skill.plugins.base import DatasetAdapter, MergeSink, Validator

_adapters: dict[str, DatasetAdapter] = {}
_validators: dict[str, Validator] = {}
_merge_sinks: dict[str, MergeSink] = {}


def register_adapter(name: str, adapter: DatasetAdapter) -> None:
    _adapters[name] = adapter


def get_adapter(name: str) -> DatasetAdapter:
    if name not in _adapters:
        raise KeyError(f"No DatasetAdapter registered under {name!r}")
    return _adapters[name]


def register_validator(name: str, validator: Validator) -> None:
    _validators[name] = validator


def get_validator(name: str) -> Validator:
    if name not in _validators:
        raise KeyError(f"No Validator registered under {name!r}")
    return _validators[name]


def register_merge_sink(name: str, sink: MergeSink) -> None:
    _merge_sinks[name] = sink


def get_merge_sink(name: str) -> MergeSink:
    if name not in _merge_sinks:
        raise KeyError(f"No MergeSink registered under {name!r}")
    return _merge_sinks[name]

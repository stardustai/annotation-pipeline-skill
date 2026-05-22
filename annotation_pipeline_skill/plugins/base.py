# annotation_pipeline_skill/plugins/base.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class DatasetAdapter(Protocol):
    """Reads a data source and produces batches of raw rows."""

    def load_rows(self, source_path: Path) -> list[dict]:
        """Read all rows from the source file."""
        ...

    def make_batches(
        self,
        rows: list[dict],
        batch_size: int,
        *,
        group_by: list[str] | None = None,
    ) -> list[list[dict]]:
        """Split rows into batches of at most batch_size."""
        ...


@runtime_checkable
class Validator(Protocol):
    """Validates an annotation payload against task constraints."""

    def validate(self, task: object, payload: dict) -> dict | None:
        """Return None if valid; return a dict with 'errors' key if invalid."""
        ...


@runtime_checkable
class MergeSink(Protocol):
    """Writes accepted tasks to an output destination."""

    def write(self, tasks: Iterable[object], output_path: Path) -> int:
        """Write tasks to output_path. Return number of tasks written."""
        ...

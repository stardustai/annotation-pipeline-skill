# annotation_pipeline_skill/plugins/jsonl_adapter.py
from __future__ import annotations

import json
from pathlib import Path


class JsonlDatasetAdapter:
    """Reads JSONL files and splits rows into batches.

    Implements the DatasetAdapter protocol from plugins.base.
    """

    def load_rows(self, source_path: Path) -> list[dict]:
        """Read every non-empty line in a JSONL file as a dict."""
        rows = []
        with open(source_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def make_batches(
        self,
        rows: list[dict],
        batch_size: int,
        *,
        group_by: list[str] | None = None,
    ) -> list[list[dict]]:
        """Split rows into batches of at most batch_size.

        If group_by keys are given, rows with the same values for those keys
        are kept together in the same batch when possible.
        """
        if not group_by:
            return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
        # Group consecutive rows with same group_by values
        batches: list[list[dict]] = []
        current: list[dict] = []
        current_key: tuple | None = None
        for row in rows:
            key = tuple(row.get(k) for k in group_by)
            if current_key is not None and key != current_key and len(current) >= batch_size:
                batches.append(current)
                current = []
            if len(current) >= batch_size:
                batches.append(current)
                current = []
            current.append(row)
            current_key = key
        if current:
            batches.append(current)
        return batches

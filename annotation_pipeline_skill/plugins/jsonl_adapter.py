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
        # Global bucketing: collect all rows per key, then chunk each group.
        # Non-consecutive rows with the same key values are merged into one
        # group, matching the semantics of cli.py:build_batches.
        buckets: dict[tuple, list[dict]] = {}
        order: list[tuple] = []
        for row in rows:
            key = tuple(str(row.get(k) or "") for k in group_by)
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(row)
        batches: list[list[dict]] = []
        for key in order:
            group = buckets[key]
            batches.extend(group[i : i + batch_size] for i in range(0, len(group), batch_size))
        return batches

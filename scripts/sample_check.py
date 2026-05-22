"""Sample-check a task's latest annotation/arbiter-correction artifact.

Usage: python scripts/sample_check.py <task_id>

Prints a one-liner: rows count + first entity sample + filename.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: sample_check.py <task_id>")
        return 1
    task = sys.argv[1]
    base = PROJECT_ROOT / "projects/v3_initial_deployment/.annotation-pipeline/artifact_payloads" / task
    files = glob.glob(str(base / f"{task}-attempt-*.json"))

    def numkey(path: str) -> tuple[int, str]:
        m = re.search(r"attempt-(\d+)_", path)
        return (int(m.group(1)) if m else 0, path)

    files.sort(key=numkey)
    ann = [f for f in files if "annotation_result" in f]
    corr = [f for f in files if "arbiter_correction" in f]
    src = corr[-1] if corr else (ann[-1] if ann else None)
    if not src:
        print("no_artifact")
        return 0
    try:
        with open(src) as fh:
            d = json.load(fh)
        text = d.get("text", "")
        obj = json.loads(text) if text and text.lstrip().startswith("{") else None
        rows = obj.get("rows", []) if isinstance(obj, dict) else []
        # Pick first non-empty entity span across the first few rows for sanity
        first_ent = None
        for r in rows[:3]:
            ents = (r.get("output") or {}).get("entities", {}) or {}
            for _, vs in ents.items():
                if vs:
                    first_ent = vs[0][:50]
                    break
            if first_ent:
                break
        print(f"rows={len(rows)} sample_entity={first_ent!r} src={Path(src).name}")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

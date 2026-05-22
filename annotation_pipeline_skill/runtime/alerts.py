"""Operator alerts sink.

Single source of truth for writing to ``<store_root>/alerts.jsonl``,
which the dashboard polls and renders. Three callers today:
  - ``SubagentRuntime._emit_provider_alert``  (4xx provider error)
  - ``SubagentRuntime._emit_enum_coerce_alert`` (arbiter invented type)
  - ``LocalRuntimeScheduler._write_health_alert`` (5-min probe)

Centralizing also gives us bounded growth: the file caps at
``MAX_LINES`` (default 10,000) and is trimmed in place when it grows
past ``MAX_LINES + ROTATION_HEADROOM``. The check runs probabilistically
(1-in-K writes) to keep the common path O(1).

"In place" trim writes a temp file and atomic-renames, so a crash
mid-rotation leaves either the old or the new file, never a half-
written one. Concurrent writers across processes use ``flock``.
"""
from __future__ import annotations

import fcntl
import json
import os
import random
from pathlib import Path
from typing import Any

MAX_LINES: int = 10_000
ROTATION_HEADROOM: int = 500  # trim only when over MAX_LINES + HEADROOM
ROTATION_CHECK_PROBABILITY: float = 0.02  # ~1 in 50 writes does the count


def append_alert(store_root: Path, entry: dict[str, Any]) -> None:
    """Append ``entry`` as a JSON line to ``<store_root>/alerts.jsonl``.
    Best-effort: never raises. Probabilistically trims to keep the
    last ``MAX_LINES`` lines once the file grows past the cap.
    """
    try:
        alerts_path = store_root / "alerts.jsonl"
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(alerts_path, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
            except OSError:
                pass  # filesystem doesn't support flock — fall through
            f.write(line)
        # Probabilistically check size; rotate if over cap.
        if random.random() < ROTATION_CHECK_PROBABILITY:
            _trim_if_oversize(alerts_path)
    except Exception:  # noqa: BLE001 — alerts are best-effort
        pass


def _trim_if_oversize(path: Path) -> None:
    """In-place trim to keep last MAX_LINES lines. Atomic rename so a
    crash mid-rotation never leaves a half-written file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
            except OSError:
                pass
            lines = f.readlines()
    except OSError:
        return
    if len(lines) <= MAX_LINES + ROTATION_HEADROOM:
        return
    kept = lines[-MAX_LINES:]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
            except OSError:
                pass
            f.writelines(kept)
        os.replace(tmp_path, path)
    except OSError:
        # Cleanup partial tmp file; original is intact.
        try:
            tmp_path.unlink()
        except OSError:
            pass

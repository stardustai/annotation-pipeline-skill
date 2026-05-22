# Maintainability Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three biggest maintainability gaps: missing CLI operator commands, absent plugin extension interfaces, and two architecture documents that no longer match the code.

**Architecture:** Three independent tracks (CLI, Plugin Protocol, Docs) that produce separate PRs. Track 1 adds CLI commands as thin wrappers over existing store/service calls. Track 2 declares `Protocol` interfaces in a new `plugins/` package and extracts two focused modules from `subagent_cycle.py`. Track 3 is pure documentation.

**Tech Stack:** Python 3.11+, `typing.Protocol`, `argparse`, SQLite via `SqliteStore`, existing `transition_task` + `TrainingDataExportService`.

---

## File Map

**Track 1 — CLI**
- Modify: `annotation_pipeline_skill/interfaces/cli.py` — add 4 commands + verify `serve` registration

**Track 2 — Plugin Protocol**
- Create: `annotation_pipeline_skill/plugins/__init__.py`
- Create: `annotation_pipeline_skill/plugins/base.py` — Protocol declarations
- Create: `annotation_pipeline_skill/plugins/registry.py` — dict-based registry
- Create: `annotation_pipeline_skill/plugins/jsonl_adapter.py` — `JsonlDatasetAdapter` reference impl
- Create: `annotation_pipeline_skill/runtime/annotation_validator.py` — extracted from subagent_cycle
- Create: `annotation_pipeline_skill/runtime/prompt_builder.py` — extracted from subagent_cycle
- Modify: `annotation_pipeline_skill/interfaces/cli.py` — use `JsonlDatasetAdapter` in `handle_create_tasks`
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py` — delegate to extracted classes

**Track 3 — Docs**
- Modify: `TECHNICAL_ARCHITECTURE.md` — §6.1, §9, §14.2
- Modify: `PRODUCT_DESIGN.md` — §10.1

**Tests**
- Modify: `tests/test_cli.py`
- Create: `tests/test_plugins_base.py`
- Create: `tests/test_annotation_validator.py`
- Create: `tests/test_prompt_builder.py`

---

## Track 1 — CLI Commands

### Task 1: Verify `serve` registration

**Files:**
- Read: `annotation_pipeline_skill/interfaces/cli.py:390-410`

- [ ] **Step 1: Confirm `serve` is registered**

  Run:
  ```bash
  grep -n "serve_parser\|add_parser.*serve" annotation_pipeline_skill/interfaces/cli.py
  ```

  Expected output should include a line with `add_parser("serve")`. If present, this task is done — mark complete and move to Task 2.

- [ ] **Step 2 (only if Step 1 found nothing): Add `serve` registration**

  In `build_parser()` just before `_register_db_commands(subparsers)`, add:
  ```python
  serve_parser = subparsers.add_parser("serve", help="Start the dashboard API server")
  serve_parser.add_argument("--workspace", type=Path, default=Path.cwd() / "projects")
  serve_parser.add_argument("--host", default="127.0.0.1")
  serve_parser.add_argument("--port", type=int, default=8509)
  serve_parser.set_defaults(handler=handle_serve)
  ```

- [ ] **Step 3: Smoke test**

  ```bash
  python -m annotation_pipeline_skill.interfaces.cli serve --help
  ```

  Expected: help text with `--workspace`, `--host`, `--port` options.

---

### Task 2: `inspect` command

**Files:**
- Modify: `annotation_pipeline_skill/interfaces/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

  In `tests/test_cli.py`, add:
  ```python
  def test_inspect_prints_task_summary(tmp_path):
      """inspect prints status, attempts, and feedback for a task."""
      from annotation_pipeline_skill.interfaces.cli import main
      from annotation_pipeline_skill.store.sqlite_store import SqliteStore
      from annotation_pipeline_skill.core.models import Task
      from annotation_pipeline_skill.core.states import TaskStatus
      from annotation_pipeline_skill.core.transitions import transition_task

      config_root = tmp_path / ".annotation-pipeline"
      config_root.mkdir()
      store = SqliteStore.open(config_root)
      task = Task.new(
          task_id="test-000001",
          pipeline_id="test-pipe",
          source_ref={"kind": "jsonl", "payload": {"rows": []}},
          modality="text",
          annotation_requirements={"annotation_types": ["extraction"]},
          metadata={},
      )
      event = transition_task(task, TaskStatus.PENDING, actor="test", reason="init", stage="prepare")
      store.save_task(task)
      store.append_event(event)
      store.close()

      result = main(["inspect", "test-000001", "--project-root", str(tmp_path)])
      assert result == 0
  ```

- [ ] **Step 2: Run to confirm it fails**

  ```bash
  python -m pytest tests/test_cli.py::test_inspect_prints_task_summary -v
  ```

  Expected: FAIL — `unrecognized arguments: inspect` or similar.

- [ ] **Step 3: Add `inspect` parser registration**

  In `build_parser()`, before `_register_db_commands(subparsers)`:
  ```python
  inspect_parser = subparsers.add_parser("inspect", help="Print task state, attempts, and feedback")
  inspect_parser.add_argument("task_id")
  inspect_parser.add_argument("--project-root", type=Path, default=Path.cwd())
  inspect_parser.set_defaults(handler=handle_inspect)
  ```

- [ ] **Step 4: Add `handle_inspect` function**

  Add after `handle_task_unblock` in `cli.py`:
  ```python
  def handle_inspect(args: argparse.Namespace) -> int:
      store = SqliteStore.open(args.project_root / ".annotation-pipeline")
      try:
          task = store.load_task(args.task_id)
      except KeyError:
          print(json.dumps({"error": "task_not_found", "task_id": args.task_id}))
          store.close()
          return 1

      attempts = store.list_attempts(args.task_id)
      feedbacks = store.list_feedback(args.task_id)
      open_feedbacks = [f for f in feedbacks if f.metadata.get("status") != "closed"]

      result = {
          "task_id": task.task_id,
          "pipeline_id": task.pipeline_id,
          "status": task.status,
          "current_attempt": task.current_attempt,
          "created_at": task.created_at,
          "updated_at": task.updated_at,
          "next_retry_at": task.metadata.get("next_retry_at"),
          "recent_attempts": [
              {
                  "index": a.index,
                  "stage": a.stage,
                  "status": a.status,
                  "provider_id": a.provider_id,
                  "model": a.model,
                  "error": a.error,
                  "summary": (a.summary or "")[:200],
              }
              for a in attempts[-3:]
          ],
          "open_feedback": [
              {
                  "feedback_id": f.feedback_id,
                  "severity": f.severity,
                  "category": f.category,
                  "message": f.message,
                  "source_stage": f.source_stage,
              }
              for f in open_feedbacks
          ],
      }
      print(json.dumps(result, sort_keys=True, indent=2))
      store.close()
      return 0
  ```

  Also add to imports at top of `cli.py` if not already present:
  ```python
  # (SqliteStore, json, Path, TaskStatus, transition_task are already imported)
  ```

- [ ] **Step 5: Run test to confirm it passes**

  ```bash
  python -m pytest tests/test_cli.py::test_inspect_prints_task_summary -v
  ```

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add annotation_pipeline_skill/interfaces/cli.py tests/test_cli.py
  git commit -m "feat(cli): add inspect command"
  ```

---

### Task 3: `approve` and `reject` commands

**Files:**
- Modify: `annotation_pipeline_skill/interfaces/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

  In `tests/test_cli.py`:
  ```python
  def _make_task_at_status(tmp_path, task_id, status):
      """Helper: create a task and advance it to the given status."""
      from annotation_pipeline_skill.store.sqlite_store import SqliteStore
      from annotation_pipeline_skill.core.models import Task
      from annotation_pipeline_skill.core.states import TaskStatus
      from annotation_pipeline_skill.core.transitions import transition_task

      config_root = tmp_path / ".annotation-pipeline"
      config_root.mkdir(exist_ok=True)
      store = SqliteStore.open(config_root)
      task = Task.new(
          task_id=task_id,
          pipeline_id="pipe",
          source_ref={"kind": "jsonl", "payload": {"rows": []}},
          modality="text",
          annotation_requirements={"annotation_types": ["extraction"]},
          metadata={},
      )
      e0 = transition_task(task, TaskStatus.PENDING, actor="test", reason="init", stage="prepare")
      store.save_task(task)
      store.append_event(e0)
      if status != TaskStatus.PENDING:
          e1 = transition_task(task, status, actor="test", reason="setup", stage="test")
          store.save_task(task)
          store.append_event(e1)
      store.close()


  def test_approve_transitions_task_to_accepted(tmp_path):
      from annotation_pipeline_skill.interfaces.cli import main
      from annotation_pipeline_skill.store.sqlite_store import SqliteStore
      from annotation_pipeline_skill.core.states import TaskStatus

      _make_task_at_status(tmp_path, "t-001", TaskStatus.QC)
      result = main(["approve", "t-001", "--project-root", str(tmp_path)])
      assert result == 0
      store = SqliteStore.open(tmp_path / ".annotation-pipeline")
      task = store.load_task("t-001")
      store.close()
      assert task.status == TaskStatus.ACCEPTED


  def test_reject_transitions_task_to_rejected(tmp_path):
      from annotation_pipeline_skill.interfaces.cli import main
      from annotation_pipeline_skill.store.sqlite_store import SqliteStore
      from annotation_pipeline_skill.core.states import TaskStatus

      _make_task_at_status(tmp_path, "t-002", TaskStatus.HUMAN_REVIEW)
      result = main(["reject", "t-002", "--project-root", str(tmp_path)])
      assert result == 0
      store = SqliteStore.open(tmp_path / ".annotation-pipeline")
      task = store.load_task("t-002")
      store.close()
      assert task.status == TaskStatus.REJECTED


  def test_approve_prints_event_id(tmp_path, capsys):
      from annotation_pipeline_skill.interfaces.cli import main
      from annotation_pipeline_skill.core.states import TaskStatus

      _make_task_at_status(tmp_path, "t-003", TaskStatus.QC)
      main(["approve", "t-003", "--project-root", str(tmp_path)])
      out = capsys.readouterr().out
      data = json.loads(out)
      assert "event_id" in data
  ```

- [ ] **Step 2: Run to confirm they fail**

  ```bash
  python -m pytest tests/test_cli.py::test_approve_transitions_task_to_accepted tests/test_cli.py::test_reject_transitions_task_to_rejected tests/test_cli.py::test_approve_prints_event_id -v
  ```

  Expected: all FAIL.

- [ ] **Step 3: Add `approve`/`reject` parsers**

  In `build_parser()`, before `_register_db_commands(subparsers)`:
  ```python
  approve_parser = subparsers.add_parser("approve", help="Accept a task")
  approve_parser.add_argument("task_id")
  approve_parser.add_argument("--reason", default="approved via CLI")
  approve_parser.add_argument("--project-root", type=Path, default=Path.cwd())
  approve_parser.set_defaults(handler=handle_approve)

  reject_parser = subparsers.add_parser("reject", help="Reject a task")
  reject_parser.add_argument("task_id")
  reject_parser.add_argument("--reason", default="rejected via CLI")
  reject_parser.add_argument("--project-root", type=Path, default=Path.cwd())
  reject_parser.set_defaults(handler=handle_reject)
  ```

- [ ] **Step 4: Add `handle_approve` and `handle_reject`**

  Add after `handle_inspect` in `cli.py`:
  ```python
  def handle_approve(args: argparse.Namespace) -> int:
      from annotation_pipeline_skill.core.transitions import InvalidTransition

      store = SqliteStore.open(args.project_root / ".annotation-pipeline")
      try:
          task = store.load_task(args.task_id)
      except KeyError:
          print(json.dumps({"error": "task_not_found", "task_id": args.task_id}))
          store.close()
          return 1
      try:
          event = transition_task(
              task,
              TaskStatus.ACCEPTED,
              actor="cli",
              reason=args.reason,
              stage="approve",
          )
      except InvalidTransition as exc:
          print(json.dumps({"error": str(exc), "task_id": args.task_id, "status": task.status}))
          store.close()
          return 2
      store.save_task(task)
      store.append_event(event)
      store.close()
      print(json.dumps({"task_id": task.task_id, "status": task.status, "event_id": event.event_id}))
      return 0


  def handle_reject(args: argparse.Namespace) -> int:
      from annotation_pipeline_skill.core.transitions import InvalidTransition

      store = SqliteStore.open(args.project_root / ".annotation-pipeline")
      try:
          task = store.load_task(args.task_id)
      except KeyError:
          print(json.dumps({"error": "task_not_found", "task_id": args.task_id}))
          store.close()
          return 1
      try:
          event = transition_task(
              task,
              TaskStatus.REJECTED,
              actor="cli",
              reason=args.reason,
              stage="reject",
          )
      except InvalidTransition as exc:
          print(json.dumps({"error": str(exc), "task_id": args.task_id, "status": task.status}))
          store.close()
          return 2
      store.save_task(task)
      store.append_event(event)
      store.close()
      print(json.dumps({"task_id": task.task_id, "status": task.status, "event_id": event.event_id}))
      return 0
  ```

- [ ] **Step 5: Run tests**

  ```bash
  python -m pytest tests/test_cli.py::test_approve_transitions_task_to_accepted tests/test_cli.py::test_reject_transitions_task_to_rejected tests/test_cli.py::test_approve_prints_event_id -v
  ```

  Expected: all PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add annotation_pipeline_skill/interfaces/cli.py tests/test_cli.py
  git commit -m "feat(cli): add approve and reject commands"
  ```

---

### Task 4: `merge` command

**Files:**
- Modify: `annotation_pipeline_skill/interfaces/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

  ```python
  def test_merge_creates_output_dir_and_returns_zero(tmp_path):
      """merge runs without error and creates the output directory."""
      from annotation_pipeline_skill.interfaces.cli import main
      from annotation_pipeline_skill.store.sqlite_store import SqliteStore
      from annotation_pipeline_skill.core.models import Task
      from annotation_pipeline_skill.core.states import TaskStatus
      from annotation_pipeline_skill.core.transitions import transition_task

      config_root = tmp_path / ".annotation-pipeline"
      config_root.mkdir()
      store = SqliteStore.open(config_root)
      task = Task.new(
          task_id="merge-t-001",
          pipeline_id="merge-pipe",
          source_ref={"kind": "jsonl", "payload": {"rows": [{"input": "hello"}]}},
          modality="text",
          annotation_requirements={"annotation_types": ["extraction"]},
          metadata={},
      )
      e0 = transition_task(task, TaskStatus.PENDING, actor="test", reason="init", stage="prepare")
      store.save_task(task)
      store.append_event(e0)
      e1 = transition_task(task, TaskStatus.ACCEPTED, actor="test", reason="done", stage="approve")
      store.save_task(task)
      store.append_event(e1)
      store.close()

      output_dir = tmp_path / "out"
      result = main([
          "merge",
          "--pipeline-id", "merge-pipe",
          "--output", str(output_dir),
          "--project-root", str(tmp_path),
      ])
      assert result == 0
      assert output_dir.exists()
  ```

- [ ] **Step 2: Run to confirm it fails**

  ```bash
  python -m pytest tests/test_cli.py::test_merge_writes_jsonl_for_accepted_tasks -v
  ```

  Expected: FAIL.

- [ ] **Step 3: Add `merge` parser**

  In `build_parser()`, before `_register_db_commands(subparsers)`:
  ```python
  merge_parser = subparsers.add_parser("merge", help="Export accepted tasks to JSONL")
  merge_parser.add_argument("--pipeline-id", required=True)
  merge_parser.add_argument("--output", type=Path, default=None, help="Output directory (default: exports/<timestamp>)")
  merge_parser.add_argument("--project-root", type=Path, default=Path.cwd())
  merge_parser.set_defaults(handler=handle_merge)
  ```

- [ ] **Step 4: Add `handle_merge`**

  Add after `handle_reject` in `cli.py`:
  ```python
  def handle_merge(args: argparse.Namespace) -> int:
      import datetime

      store = SqliteStore.open(args.project_root / ".annotation-pipeline")
      timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
      output_dir = args.output or (args.project_root / "exports" / f"merged-{timestamp}")
      manifest = TrainingDataExportService(store).export_jsonl(
          project_id=args.pipeline_id,
          output_dir=output_dir,
      )
      store.close()
      print(json.dumps({
          "output_dir": str(output_dir),
          "accepted_tasks": manifest.metadata.get("accepted_tasks", "?"),
          "export_id": manifest.export_id,
      }, indent=2))
      return 0
  ```

  Verify `TrainingDataExportService` is already imported at the top of `cli.py`:
  ```python
  from annotation_pipeline_skill.services.export_service import TrainingDataExportService
  ```
  (It is — at line 39.)

- [ ] **Step 5: Run test**

  ```bash
  python -m pytest tests/test_cli.py::test_merge_writes_jsonl_for_accepted_tasks -v
  ```

  Expected: PASS. (If `save_artifact` doesn't exist on `SqliteStore`, check the store API and use the correct method name.)

- [ ] **Step 6: Run full CLI test suite**

  ```bash
  python -m pytest tests/test_cli.py -v
  ```

  Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

  ```bash
  git add annotation_pipeline_skill/interfaces/cli.py tests/test_cli.py
  git commit -m "feat(cli): add merge command"
  ```

---

## Track 2 — Plugin Protocol

### Task 5: `plugins/base.py` — Protocol declarations

**Files:**
- Create: `annotation_pipeline_skill/plugins/__init__.py`
- Create: `annotation_pipeline_skill/plugins/base.py`
- Create: `tests/test_plugins_base.py`

- [ ] **Step 1: Write failing tests**

  Create `tests/test_plugins_base.py`:
  ```python
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
  ```

- [ ] **Step 2: Run to confirm they fail**

  ```bash
  python -m pytest tests/test_plugins_base.py -v
  ```

  Expected: FAIL — `ModuleNotFoundError: annotation_pipeline_skill.plugins`.

- [ ] **Step 3: Create `plugins/__init__.py`**

  ```python
  # annotation_pipeline_skill/plugins/__init__.py
  from annotation_pipeline_skill.plugins.base import DatasetAdapter, MergeSink, Validator

  __all__ = ["DatasetAdapter", "MergeSink", "Validator"]
  ```

- [ ] **Step 4: Create `plugins/base.py`**

  ```python
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
  ```

- [ ] **Step 5: Run tests**

  ```bash
  python -m pytest tests/test_plugins_base.py -v
  ```

  Expected: all PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add annotation_pipeline_skill/plugins/ tests/test_plugins_base.py
  git commit -m "feat(plugins): add Protocol declarations for DatasetAdapter, Validator, MergeSink"
  ```

---

### Task 6: `plugins/registry.py` + `plugins/jsonl_adapter.py`

**Files:**
- Create: `annotation_pipeline_skill/plugins/registry.py`
- Create: `annotation_pipeline_skill/plugins/jsonl_adapter.py`
- Modify: `annotation_pipeline_skill/interfaces/cli.py` — use adapter in `handle_create_tasks`
- Modify: `tests/test_plugins_base.py`

- [ ] **Step 1: Write failing tests for registry + adapter**

  Add to `tests/test_plugins_base.py`:
  ```python
  from pathlib import Path
  import json
  import tempfile


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
  ```

- [ ] **Step 2: Run to confirm they fail**

  ```bash
  python -m pytest tests/test_plugins_base.py -v -k "registry or jsonl"
  ```

  Expected: FAIL.

- [ ] **Step 3: Create `plugins/registry.py`**

  ```python
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
  ```

- [ ] **Step 4: Create `plugins/jsonl_adapter.py`**

  Extract the JSONL reading and batching logic from `cli.py:read_jsonl` and `cli.py:build_batches`:
  ```python
  # annotation_pipeline_skill/plugins/jsonl_adapter.py
  from __future__ import annotations

  import json
  from pathlib import Path


  class JsonlDatasetAdapter:
      """Reads JSONL files and splits rows into batches."""

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
  ```

- [ ] **Step 5: Update `plugins/__init__.py` to expose the adapter**

  ```python
  from annotation_pipeline_skill.plugins.base import DatasetAdapter, MergeSink, Validator
  from annotation_pipeline_skill.plugins.jsonl_adapter import JsonlDatasetAdapter

  __all__ = ["DatasetAdapter", "MergeSink", "Validator", "JsonlDatasetAdapter"]
  ```

- [ ] **Step 6: Run tests**

  ```bash
  python -m pytest tests/test_plugins_base.py -v
  ```

  Expected: all PASS.

- [ ] **Step 7: Commit**

  ```bash
  git add annotation_pipeline_skill/plugins/ tests/test_plugins_base.py
  git commit -m "feat(plugins): add registry and JsonlDatasetAdapter reference implementation"
  ```

---

### Task 7: Extract `runtime/annotation_validator.py`

**Files:**
- Create: `annotation_pipeline_skill/runtime/annotation_validator.py`
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py`
- Create: `tests/test_annotation_validator.py`

The methods to extract from `SubagentRuntime` (keep their logic, change `self` → constructor-injected deps):
- `_check_annotation_validation` (~line 1428) — main entry point
- `_check_verbatim_spans` (~line 1562) — verbatim violation checker
- `_verbatim_candidate_spans` (~line 1590) — helper for verbatim extraction
- `_auto_align_corrected_annotation` (~line 1582) — post-process helper

- [ ] **Step 1: Write failing tests**

  Create `tests/test_annotation_validator.py`:
  ```python
  """Tests for AnnotationValidator extracted from SubagentRuntime."""
  import json
  import pytest
  from annotation_pipeline_skill.runtime.annotation_validator import AnnotationValidator
  from annotation_pipeline_skill.plugins.base import Validator


  def _make_task(task_id="test-001", rows=None):
      """Return a minimal Task-like object for testing."""
      from annotation_pipeline_skill.core.models import Task
      return Task.new(
          task_id=task_id,
          pipeline_id="pipe",
          source_ref={
              "kind": "jsonl",
              "payload": {"rows": rows or [{"input": "Apple is a company"}]},
          },
          modality="text",
          annotation_requirements={"annotation_types": ["extraction"]},
          metadata={},
      )


  def test_validate_returns_none_for_valid_payload():
      task = _make_task()
      validator = AnnotationValidator(output_schema=None)
      payload = {"rows": [{"input": "Apple is a company", "entities": [{"span": "Apple", "type": "organization"}]}]}
      result = validator.validate(task, payload)
      assert result is None


  def test_validate_catches_verbatim_violation():
      task = _make_task(rows=[{"input": "Apple is a company"}])
      validator = AnnotationValidator(output_schema=None)
      # "Appl" is not a verbatim substring of "Apple is a company"
      payload = {"rows": [{"input": "Apple is a company", "entities": [{"span": "Appl", "type": "organization"}]}]}
      result = validator.validate(task, payload)
      assert result is not None
      assert "errors" in result or "violations" in result or "feedback_data" in result


  def test_validator_implements_protocol():
      from annotation_pipeline_skill.plugins.base import Validator
      v = AnnotationValidator(output_schema=None)
      assert isinstance(v, Validator)
  ```

- [ ] **Step 2: Run to confirm they fail**

  ```bash
  python -m pytest tests/test_annotation_validator.py -v
  ```

  Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `runtime/annotation_validator.py`**

  Open `subagent_cycle.py` and copy the bodies of `_check_annotation_validation`, `_check_verbatim_spans`, `_verbatim_candidate_spans`, and `_auto_align_corrected_annotation` into the new class. Replace `self.store` with `self._store` (passed in constructor), and `self._project_schema` with constructor-injected `output_schema`.

  ```python
  # annotation_pipeline_skill/runtime/annotation_validator.py
  """Annotation output validation: schema + verbatim span checking.

  Extracted from SubagentRuntime to allow independent testing and reuse.
  """
  from __future__ import annotations

  import json
  import re
  from typing import Any

  from annotation_pipeline_skill.core.models import Task
  from annotation_pipeline_skill.store.sqlite_store import SqliteStore


  class AnnotationValidator:
      """Validates annotation payloads against schema and verbatim constraints.

      Implements the Validator protocol from plugins.base.
      """

      def __init__(
          self,
          output_schema: dict | None,
          store: SqliteStore | None = None,
      ):
          self._output_schema = output_schema
          self._store = store

      def validate(self, task: Task, payload: dict) -> dict | None:
          """Return None if payload is valid; dict with 'feedback_data' key if not.

          Runs two checks in sequence:
          1. JSON schema validation against output_schema (if set)
          2. Verbatim span check — entity spans must be exact substrings of input
          """
          # Schema check
          if self._output_schema:
              errors = self._check_schema(payload)
              if errors:
                  return {"feedback_data": {"source": "schema_validation", "errors": errors}}

          # Verbatim check
          violations = self.check_verbatim_spans(task, payload)
          if violations:
              return {"feedback_data": {"source": "verbatim_validation", "violations": violations}}

          return None

      def check_verbatim_spans(self, task: Task, payload: Any) -> list[dict]:
          """Return list of non-verbatim span violations found in payload."""
          source_ref = task.source_ref if isinstance(task.source_ref, dict) else {}
          source_payload = source_ref.get("payload", {})
          rows = source_payload.get("rows") if isinstance(source_payload, dict) else None
          if not isinstance(rows, list):
              return []

          violations = []
          ann_rows = payload.get("rows") if isinstance(payload, dict) else None
          if not isinstance(ann_rows, list):
              return violations

          for i, (src_row, ann_row) in enumerate(zip(rows, ann_rows)):
              input_text = src_row.get("input", "") if isinstance(src_row, dict) else ""
              for span in self._extract_candidate_spans(ann_row):
                  if span and span not in input_text:
                      violations.append({"row_index": i, "span": span, "input": input_text[:200]})
          return violations

      def _extract_candidate_spans(self, ann_row: Any) -> list[str]:
          """Return all entity/structure spans declared in an annotation row."""
          if not isinstance(ann_row, dict):
              return []
          spans = []
          for entity in ann_row.get("entities", []):
              if isinstance(entity, dict) and isinstance(entity.get("span"), str):
                  spans.append(entity["span"])
          for struct in ann_row.get("json_structures", []):
              if isinstance(struct, dict) and isinstance(struct.get("phrase"), str):
                  spans.append(struct["phrase"])
          return spans

      def _check_schema(self, payload: dict) -> list[str]:
          """Validate payload against self._output_schema. Return list of error messages."""
          try:
              import jsonschema
              validator = jsonschema.Draft202012Validator(self._output_schema)
              return [e.message for e in validator.iter_errors(payload)]
          except Exception as exc:  # noqa: BLE001
              return [str(exc)]
  ```

  **Note:** The bodies of the existing methods in `subagent_cycle.py` may be more complex (they call internal helpers, use `self.store` for mask lookups, etc.). Copy the actual implementation from `subagent_cycle.py` rather than the stub above — the stub illustrates the structure and interface. Look at lines 1428–1652 for the real implementation.

- [ ] **Step 4: Wire into `SubagentRuntime`**

  In `subagent_cycle.py`, add to `SubagentRuntime.__init__`:
  ```python
  from annotation_pipeline_skill.runtime.annotation_validator import AnnotationValidator
  # inside __init__, after self.store is set:
  self._validator = AnnotationValidator(
      output_schema=self._project_schema,
      store=self.store,
  )
  ```

  Replace the body of `SubagentRuntime._check_annotation_validation` with a delegation call:
  ```python
  def _check_annotation_validation(self, task: Task, final_text: str) -> dict | None:
      try:
          payload = json.loads(final_text) if isinstance(final_text, str) else final_text
      except json.JSONDecodeError:
          return {"feedback_data": {"source": "parse_error", "errors": ["invalid JSON"]}}
      return self._validator.validate(task, payload)
  ```

  Keep the old method bodies in `subagent_cycle.py` commented out (not deleted) until all tests pass — makes rollback easy.

- [ ] **Step 5: Run new tests**

  ```bash
  python -m pytest tests/test_annotation_validator.py -v
  ```

  Expected: all PASS.

- [ ] **Step 6: Run existing subagent_cycle tests**

  ```bash
  python -m pytest tests/test_subagent_cycle.py tests/test_schema_validation.py -v
  ```

  Expected: all still PASS.

- [ ] **Step 7: Remove commented-out old bodies from subagent_cycle.py**

- [ ] **Step 8: Commit**

  ```bash
  git add annotation_pipeline_skill/runtime/annotation_validator.py \
          annotation_pipeline_skill/runtime/subagent_cycle.py \
          tests/test_annotation_validator.py
  git commit -m "refactor(runtime): extract AnnotationValidator from subagent_cycle"
  ```

---

### Task 8: Extract `runtime/prompt_builder.py`

**Files:**
- Create: `annotation_pipeline_skill/runtime/prompt_builder.py`
- Modify: `annotation_pipeline_skill/runtime/subagent_cycle.py`
- Create: `tests/test_prompt_builder.py`

Methods to extract (from `SubagentRuntime`):
- `_build_conventions_block` (~line 3172) — entity conventions prompt block
- `_annotation_prompt` (~line 3224)
- `_delta_feedback_items` (~line 3246)
- `_snapshot_sent_feedback` (~line 3251)
- `_qc_prompt` (~line 3257)
- `_slim_annotation_payload` (~line 3271)

- [ ] **Step 1: Write failing tests**

  Create `tests/test_prompt_builder.py`:
  ```python
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


  def test_annotation_prompt_is_valid_json(tmp_path):
      from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
      from annotation_pipeline_skill.store.sqlite_store import SqliteStore

      store = SqliteStore.open(tmp_path / ".annotation-pipeline")
      builder = AnnotationPromptBuilder(store=store, project_id="pipe", config={})
      task = _make_task()
      prompt = builder.build_annotation_prompt(task)
      # Should produce valid JSON
      parsed = json.loads(prompt)
      assert "task" in parsed
      store.close()


  def test_qc_prompt_is_valid_json(tmp_path):
      from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
      from annotation_pipeline_skill.store.sqlite_store import SqliteStore
      from annotation_pipeline_skill.core.models import ArtifactRef
      import datetime

      store = SqliteStore.open(tmp_path / ".annotation-pipeline")
      (tmp_path / ".annotation-pipeline" / "artifacts").mkdir(exist_ok=True)

      artifact_path = tmp_path / ".annotation-pipeline" / "artifacts" / "ann.json"
      artifact_path.write_text(json.dumps({"text": json.dumps({"rows": []}), "provider": "test", "model": "m"}))
      artifact = ArtifactRef(
          artifact_id="a1", task_id="pb-001", kind="annotation_result",
          path="artifacts/ann.json", content_type="application/json",
          created_at=datetime.datetime.now().isoformat(), metadata={},
      )
      builder = AnnotationPromptBuilder(store=store, project_id="pipe", config={})
      task = _make_task()
      prompt = builder.build_qc_prompt(task, artifact)
      parsed = json.loads(prompt)
      assert "task" in parsed
      assert "annotation_artifact" in parsed
      store.close()


  def test_build_conventions_block_returns_none_when_no_matches(tmp_path):
      from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
      from annotation_pipeline_skill.store.sqlite_store import SqliteStore

      store = SqliteStore.open(tmp_path / ".annotation-pipeline")
      builder = AnnotationPromptBuilder(store=store, project_id="pipe", config={})
      task = _make_task()
      result = builder.build_conventions_block(task)
      assert result is None
      store.close()
  ```

- [ ] **Step 2: Run to confirm they fail**

  ```bash
  python -m pytest tests/test_prompt_builder.py -v
  ```

  Expected: FAIL.

- [ ] **Step 3: Create `runtime/prompt_builder.py`**

  Copy the bodies of the six methods from `subagent_cycle.py` into `AnnotationPromptBuilder`. The key dependencies are `store`, `project_id`, and `config` (for QC policy resolution). Replace `self.store` with `self._store`, `self._project_id` with `self._project_id`.

  ```python
  # annotation_pipeline_skill/runtime/prompt_builder.py
  """Prompt construction for annotator and QC subagents.

  Extracted from SubagentRuntime to allow independent testing.
  """
  from __future__ import annotations

  from typing import Any

  from annotation_pipeline_skill.core.models import ArtifactRef, Task
  from annotation_pipeline_skill.store.sqlite_store import SqliteStore


  class AnnotationPromptBuilder:
      """Builds annotation and QC prompts, including conventions blocks."""

      def __init__(
          self,
          store: SqliteStore,
          project_id: str,
          config: dict,
      ):
          self._store = store
          self._project_id = project_id
          self._config = config

      def build_annotation_prompt(
          self,
          task: Task,
          *,
          continuation_handle: str | None = None,
      ) -> str:
          # Copy body of SubagentRuntime._annotation_prompt here,
          # replacing self.store → self._store, self._project_id stays same.
          raise NotImplementedError("copy from subagent_cycle._annotation_prompt")

      def build_qc_prompt(self, task: Task, annotation_artifact: ArtifactRef) -> str:
          # Copy body of SubagentRuntime._qc_prompt here.
          raise NotImplementedError("copy from subagent_cycle._qc_prompt")

      def build_conventions_block(self, task: Task) -> str | None:
          # Copy body of SubagentRuntime._build_conventions_block here.
          raise NotImplementedError("copy from subagent_cycle._build_conventions_block")

      def slim_annotation_payload(self, artifact: ArtifactRef) -> Any:
          # Copy body of SubagentRuntime._slim_annotation_payload here.
          raise NotImplementedError("copy from subagent_cycle._slim_annotation_payload")

      def delta_feedback_items(self, task: Task) -> list[dict]:
          # Copy body of SubagentRuntime._delta_feedback_items here.
          raise NotImplementedError("copy from subagent_cycle._delta_feedback_items")

      def snapshot_sent_feedback(self, task: Task) -> None:
          # Copy body of SubagentRuntime._snapshot_sent_feedback here.
          raise NotImplementedError("copy from subagent_cycle._snapshot_sent_feedback")

      def _read_artifact_payload(self, artifact: ArtifactRef) -> Any:
          # Copy body of SubagentRuntime._read_artifact_payload here.
          raise NotImplementedError("copy from subagent_cycle._read_artifact_payload")
  ```

  **Important:** Replace the `raise NotImplementedError` stubs with the actual method bodies copied from `subagent_cycle.py`. The stubs show the interface; the real code is in lines 3172–3370 of `subagent_cycle.py`.

- [ ] **Step 4: Wire into `SubagentRuntime`**

  In `subagent_cycle.py`, add to `SubagentRuntime.__init__`:
  ```python
  from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
  # inside __init__:
  self._prompt_builder = AnnotationPromptBuilder(
      store=self.store,
      project_id=self._project_id,
      config=self.config,
  )
  ```

  Replace each method's body in `SubagentRuntime` with a one-line delegation:
  ```python
  def _annotation_prompt(self, task, *, continuation_handle=None):
      return self._prompt_builder.build_annotation_prompt(task, continuation_handle=continuation_handle)

  def _qc_prompt(self, task, annotation_artifact):
      return self._prompt_builder.build_qc_prompt(task, annotation_artifact)

  def _build_conventions_block(self, task):
      return self._prompt_builder.build_conventions_block(task)

  def _slim_annotation_payload(self, artifact):
      return self._prompt_builder.slim_annotation_payload(artifact)

  def _delta_feedback_items(self, task):
      return self._prompt_builder.delta_feedback_items(task)

  def _snapshot_sent_feedback(self, task):
      return self._prompt_builder.snapshot_sent_feedback(task)
  ```

- [ ] **Step 5: Run prompt builder tests**

  ```bash
  python -m pytest tests/test_prompt_builder.py -v
  ```

  Expected: all PASS.

- [ ] **Step 6: Run full test suite to confirm nothing regressed**

  ```bash
  python -m pytest tests/ -v --tb=short 2>&1 | tail -30
  ```

  Expected: all existing tests pass.

- [ ] **Step 7: Verify line count reduction**

  ```bash
  wc -l annotation_pipeline_skill/runtime/subagent_cycle.py
  ```

  Expected: ≤ 3700 (from 3917; prompt methods ~120 lines removed, validator methods already handled in Task 7).

- [ ] **Step 8: Commit**

  ```bash
  git add annotation_pipeline_skill/runtime/prompt_builder.py \
          annotation_pipeline_skill/runtime/subagent_cycle.py \
          tests/test_prompt_builder.py
  git commit -m "refactor(runtime): extract AnnotationPromptBuilder from subagent_cycle"
  ```

---

## Track 3 — Documentation

### Task 9: Update `TECHNICAL_ARCHITECTURE.md`

**Files:**
- Modify: `TECHNICAL_ARCHITECTURE.md`

Three sections to update. Open the file and make each change in sequence.

- [ ] **Step 1: Fix §6.1 state machine**

  Find and replace the paragraph starting "当前实现使用 7 个 task status" and update the table to:

  | Status | 含义 |
  |---|---|
  | `draft` | task 已创建，等待 manifest 生成 |
  | `pending` | 等待 worker claim |
  | `annotating` | 标注 LLM 调用进行中 |
  | `qc` | validation 通过，QC 进行中 |
  | `arbitrating` | 仲裁 LLM 调用进行中，或 mechanical retry 等待 pickup |
  | `human_review` | 需要人工判断或 arbiter 不确定 |
  | `accepted` | 终态 — 通过所有检查 |
  | `rejected` | 终态 — 人工拒绝 |
  | `blocked` | 需要人工介入才能继续 |
  | `cancelled` | 已取消 |

  Add note after the table:
  > `validating` 是 inline 步骤（annotation 写完后立刻执行），不是独立 task status；失败后 task 回 `pending` 重试，write a BLOCKING FeedbackRecord。

- [ ] **Step 2: Update §9 应用服务 — add undocumented services**

  After the existing listed services (`TaskFactoryService`, `PipelineService`, etc.), add a subsection "当前实现中存在的服务":

  ```markdown
  ### 当前代码中存在但文档未覆盖的服务

  - `coordinator_service` — 跨 pipeline 协调调度，生成统一 report
  - `distribution_service` — span/entity 类型分布统计与缓存，供看板的 DistributionPanel 使用
  - `entity_convention_service` — 高确定性约定读写，用于向 annotator/QC prompt 注入 KNOWN ENTITY CONVENTIONS
  - `entity_statistics_service` — 所有 ACCEPTED 决策的统计分布，供 prior verifier（§11.9）查询
  - `export_service` (`TrainingDataExportService`) — 将 ACCEPTED task 的 annotation_result 导出为 JSONL
  - `human_review_service` — HR 阶段决策写入（accept/reject）和 operator 修正提交
  - `outbox_dispatch_service` — 外部状态回传和结果提交的 outbox drain，可靠重试
  - `provider_config_service` — provider profile 读取、校验和 stage target 更新
  - `readiness_service` — 项目就绪度检查（环境、配置、schema）
  - `row_dedup_service` — 输入行去重检测，防止重复标注
  - `row_mask_service` — 屏蔽低质量或重复行，在 annotation 和 export 前过滤
  ```

- [ ] **Step 3: Update §14.2 API 端点**

  Replace the existing endpoint list with a two-column table mapping doc-described paths to actual paths, and add a section of undocumented endpoints. Follow the mapping from the gap analysis:

  ```markdown
  ### 文档路径 → 实际路径

  | 文档描述 | 实际路径 |
  |---|---|
  | `GET /dashboard` | `GET /api/kanban` + `GET /api/dashboard-stats` |
  | `GET /settings` | `GET /api/config` + `GET /api/providers` + `GET /api/annotators` |
  | `POST /settings/validate` | 未实现 |
  | `POST /providers/test` | 未实现 |
  | `POST /tasks/<id>/retry` | `POST /api/tasks/<id>/move` (body: `{"to": "pending"}`) |
  | `POST /tasks/<id>/approve` | `POST /api/tasks/<id>/move` (body: `{"to": "accepted"}`) |
  | `POST /tasks/<id>/reject` | `POST /api/tasks/<id>/move` (body: `{"to": "rejected"}`) |
  | `GET /tasks/<id>/feedback` | 包含在 `GET /api/tasks/<id>` 响应体内 |
  | `POST /external/tasks/pull` | 仅 CLI `apl external pull` |

  ### 代码中存在但文档未提及的端点

  `GET /api/conventions`, `POST /api/conventions`, `DELETE /api/conventions/clear`,
  `GET /api/posterior-audit`, `POST /api/posterior-audit`, `POST /api/posterior-audit/retroactive-fix`,
  `GET /api/distribution`, `POST /api/distribution/scan`, `POST /api/distribution/reject`,
  `GET /api/row-dedup`, `POST /api/row-dedup/scan`, `POST /api/row-dedup/mask`, `DELETE /api/row-dedup/mask`,
  `GET /api/type-statistics`, `POST /api/type-statistics`,
  `GET /api/entity-statistics`, `POST /api/entity-statistics/recount`,
  `GET /api/typical-text`, `GET /api/alerts`, `GET /api/coordinator`,
  `GET /api/readiness`, `GET /api/export-file`,
  `GET /api/annotation-rules-document`, `POST /api/annotation-rules-document/versions`,
  `GET /api/documents`, `POST /api/documents`,
  `GET /api/runtime/monitor`, `POST /api/runtime/run-once`,
  `GET /api/jobs/<id>`
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add TECHNICAL_ARCHITECTURE.md
  git commit -m "docs(arch): fix state count, add undocumented services and API paths"
  ```

---

### Task 10: Update `PRODUCT_DESIGN.md`

**Files:**
- Modify: `PRODUCT_DESIGN.md`

- [ ] **Step 1: Add implementation note to §10.1**

  In `PRODUCT_DESIGN.md`, find the heading `### 10.1 生命周期` and the state machine line:
  ```
  `draft -> ready -> annotating -> validating -> qc -> human_review -> accepted/rejected/repair_needed -> merged`
  ```

  Insert the following paragraph **before** that line:
  ```markdown
  > **当前实现说明**（与通用状态机的偏差）：
  > - `validating` 是 annotation worker 内部的 inline 步骤，不是独立 task status；失败后 task 回 `pending` 并写 BLOCKING FeedbackRecord。
  > - `repair_needed` 在当前实现中通过 `pending` 重试循环 + `arbitrating` 仲裁状态覆盖，没有独立状态。
  > - `ready` 等同于 `pending`（draft 写完即直接变 pending）。
  > - `merged` 通过 `ExportService` 实现为操作而非状态；accepted task 的 status 不改变。
  > - `retry_scheduled` 由 `next_retry_at` 字段在 task metadata 里表达，不是独立状态。
  > - `arbitrating` 是当前实现特有的仲裁状态，对应通用模型中"进入 repair/arbiter 判断"这一阶段。
  >
  > 上述偏差反映了当前 NER 项目驱动的实现选择；通用框架扩展时可以重新建模这些为独立状态。
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add PRODUCT_DESIGN.md
  git commit -m "docs(product): add implementation note to §10.1 state machine"
  ```

---

## Final Integration Check

After all three tracks are merged:

- [ ] Run full test suite:
  ```bash
  python -m pytest tests/ -v --tb=short 2>&1 | tail -20
  ```
  Expected: all pass.

- [ ] Verify new CLI commands are accessible:
  ```bash
  python -m annotation_pipeline_skill.interfaces.cli inspect --help
  python -m annotation_pipeline_skill.interfaces.cli approve --help
  python -m annotation_pipeline_skill.interfaces.cli reject --help
  python -m annotation_pipeline_skill.interfaces.cli merge --help
  ```

- [ ] Verify plugin protocol check:
  ```python
  from annotation_pipeline_skill.plugins import DatasetAdapter, JsonlDatasetAdapter
  assert isinstance(JsonlDatasetAdapter(), DatasetAdapter)
  ```

- [ ] Verify `subagent_cycle.py` line count:
  ```bash
  wc -l annotation_pipeline_skill/runtime/subagent_cycle.py
  ```
  Expected: ≤ 3700.

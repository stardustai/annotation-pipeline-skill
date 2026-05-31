from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from annotation_pipeline_skill.core.models import (
    AnnotationDocument,
    AnnotationDocumentVersion,
    ArtifactRef,
    Attempt,
    AuditEvent,
    ExportManifest,
    FeedbackDiscussionEntry,
    FeedbackRecord,
    OutboxRecord,
    Task,
)
from annotation_pipeline_skill.core.runtime import (
    ActiveRun,
    RuntimeLease,
    RuntimeSnapshot,
)
from annotation_pipeline_skill.core.states import TaskStatus

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _migrate_convention_aggregate_columns(conn: "sqlite3.Connection") -> None:
    """Idempotently add the materialized aggregate columns to an existing
    entity_conventions table.

    SQLite has no ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``, so we inspect
    ``PRAGMA table_info`` and only add columns that are missing. Fresh DBs get
    the columns straight from the CREATE TABLE above, so this is a no-op for
    them. Existing rows keep the column DEFAULTs (0 / NULL) until a write
    refreshes them or a rebuild replays their proposals.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(entity_conventions)")}
    if not existing:
        return  # table not created yet (shouldn't happen post-migrations)
    additions = (
        ("distinct_task_count", "INTEGER NOT NULL DEFAULT 0"),
        ("dispute_count", "INTEGER NOT NULL DEFAULT 0"),
        ("dispute_pct", "REAL NOT NULL DEFAULT 0.0"),
        ("dominant_type", "TEXT"),
    )
    for name, decl in additions:
        if name not in existing:
            conn.execute(f"ALTER TABLE entity_conventions ADD COLUMN {name} {decl}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_inject "
        "ON entity_conventions(project_id, status, distinct_task_count)"
    )

# Additive migrations: CREATE TABLE IF NOT EXISTS only, applied on every open
# of an existing DB. Keeps the door open for adding tables without a formal
# migration system. Drop additions here once a year by promoting them into
# schema.sql proper.
_ADDITIVE_MIGRATIONS_SQL = """
CREATE TABLE IF NOT EXISTS entity_conventions (
    convention_id  TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    span_lower     TEXT NOT NULL,
    span_original  TEXT NOT NULL,
    entity_type    TEXT,
    status         TEXT NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    proposals_json TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    created_by     TEXT NOT NULL,
    notes          TEXT,
    -- Materialized aggregates of proposals_json, maintained on every write
    -- (record_decision / clear_dispute) so the injection gate can be
    -- evaluated in SQL without parsing the JSON blob per row. proposals_json
    -- stays the source of truth; these are a write-maintained cache.
    distinct_task_count INTEGER NOT NULL DEFAULT 0,
    dispute_count       INTEGER NOT NULL DEFAULT 0,
    dispute_pct         REAL NOT NULL DEFAULT 0.0,
    dominant_type       TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_project_span ON entity_conventions(project_id, span_lower);
CREATE INDEX IF NOT EXISTS idx_conv_project_status ON entity_conventions(project_id, status);
-- NOTE: idx_conv_inject references distinct_task_count, which on a pre-existing
-- DB is added by _migrate_convention_aggregate_columns *after* this script
-- runs. Creating it here would fail (no such column) before the ALTER lands,
-- so that index is created inside the migration instead, once the column
-- exists. Fresh DBs get it from schema.sql.

CREATE TABLE IF NOT EXISTS entity_statistics (
    project_id   TEXT NOT NULL,
    span_lower   TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    count        INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (project_id, span_lower, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_entity_stats_span ON entity_statistics(project_id, span_lower);

-- Cache of the most recent Posterior Audit scan per project, so the dashboard
-- can auto-load the previous result without re-running the (expensive) full
-- scan, and so we can show whether the cache is stale relative to current
-- ACCEPTED tasks. ``accepted_hash`` is a sha256 over (task_id, updated_at) of
-- all ACCEPTED tasks at scan time; comparing the cached hash to a freshly
-- computed one tells us whether anything has changed since the cache was
-- written.
CREATE TABLE IF NOT EXISTS posterior_audit_cache (
    project_id    TEXT PRIMARY KEY,
    payload_json  TEXT NOT NULL,
    accepted_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

-- Cached per-project distribution of entity types and json_structure
-- phrase types across ACCEPTED tasks. Powers the Statistics dashboard
-- view. Recomputed on POST /api/type-statistics (GET serves cache).
CREATE TABLE IF NOT EXISTS type_statistics_cache (
    project_id   TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

-- Cached per-(project, profile) embedding-cluster scan + 2D scatter
-- coords. Payload shape: {"clusters": [...], "coords": [...], "params": {...}}.
-- One row per (project_id, embedding profile name) — distinct profiles
-- (e.g. jina_small vs random_baseline) keep separate caches. content_hash
-- is a fingerprint of the input set (task_ids + canonical_text hashes) so
-- the UI can mark the cache stale when tasks are added / rejected.
CREATE TABLE IF NOT EXISTS distribution_cache (
    project_id    TEXT NOT NULL,
    profile_name  TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (project_id, profile_name)
);

-- Per-task embedding cache keyed by (task_id, profile_name). vector is a
-- raw float32 BLOB (dim values, little-endian). content_hash is sha256 of
-- the canonical_task_text — when the task's input text changes the cache
-- row is invalidated by hash mismatch (NOT auto-deleted; lookup compares
-- and re-embeds on miss). One profile's embeddings don't poison another's
-- so jina_small and random_baseline coexist.
CREATE TABLE IF NOT EXISTS task_embeddings (
    task_id       TEXT NOT NULL,
    profile_name  TEXT NOT NULL,
    model         TEXT NOT NULL,
    dim           INTEGER NOT NULL,
    content_hash  TEXT NOT NULL,
    vector        BLOB NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (task_id, profile_name)
);

-- Row-level mask: per (task_id, row_index), a row that downstream consumers
-- (export, entity_statistics, posterior audit, scatter) treat AS IF it didn't
-- exist. The task itself stays ACCEPTED — only the marked rows disappear at
-- the read boundary. Mostly populated by RowDedupService when it finds rows
-- that duplicate other rows across tasks.
CREATE TABLE IF NOT EXISTS row_masks (
    task_id     TEXT NOT NULL,
    row_index   INTEGER NOT NULL,
    reason      TEXT NOT NULL,
    masked_by   TEXT NOT NULL,
    masked_at   TEXT NOT NULL,
    metadata_json TEXT,
    PRIMARY KEY (task_id, row_index)
);

-- Per-row embedding cache, parallel to task_embeddings but keyed by row.
-- Same float32 BLOB layout; content_hash is sha256 of the row's input
-- text salted with provider-specific params (model name, or shingle_size
-- / num_perm for MinHash).
CREATE TABLE IF NOT EXISTS row_embeddings (
    task_id       TEXT NOT NULL,
    row_index     INTEGER NOT NULL,
    profile_name  TEXT NOT NULL,
    model         TEXT NOT NULL,
    dim           INTEGER NOT NULL,
    content_hash  TEXT NOT NULL,
    vector        BLOB NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (task_id, row_index, profile_name)
);

-- Cached per-(project, profile) row-level dedup scan, mirrors
-- distribution_cache for task-level. Payload shape:
--   {"params": {...}, "clusters": [...], "row_count": int}
-- One row per (project_id, profile_name).
CREATE TABLE IF NOT EXISTS row_dedup_cache (
    project_id    TEXT NOT NULL,
    profile_name  TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (project_id, profile_name)
);
"""


def _row_to_task(row: sqlite3.Row) -> Task:
    return Task.from_dict({
        "task_id": row["task_id"],
        "pipeline_id": row["pipeline_id"],
        "source_ref": json.loads(row["source_ref_json"]),
        "external_ref": json.loads(row["external_ref_json"]) if row["external_ref_json"] else None,
        "modality": row["modality"],
        "annotation_requirements": json.loads(row["annotation_requirements_json"]),
        "selected_annotator_id": row["selected_annotator_id"],
        "status": row["status"],
        "current_attempt": row["current_attempt"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "active_run_id": row["active_run_id"],
        "next_retry_at": row["next_retry_at"],
        "metadata": json.loads(row["metadata_json"]),
        "document_version_id": row["document_version_id"],
    })


class SqliteStore:
    def __init__(self, root: Path | str, db_path: Path):
        self.root = Path(root)
        self._db_path = Path(db_path)
        self._local = threading.local()

    @classmethod
    def open(cls, root: Path | str) -> "SqliteStore":
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        for sub in ("artifacts", "exports", "runtime", "documents", "document_versions", "backups"):
            (root_path / sub).mkdir(parents=True, exist_ok=True)
        db_path = root_path / "db.sqlite"
        first_time = not db_path.exists()
        store = cls(root_path, db_path)
        # Open one connection now to apply schema if first_time.
        conn = store._conn
        if first_time:
            conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        # Ensure additive schema migrations (CREATE TABLE IF NOT EXISTS only)
        # land on both new and existing DBs. New DBs need them for tables not
        # yet promoted into schema.sql; existing DBs need them for the same
        # tables that were added after their initial creation.
        conn.executescript(_ADDITIVE_MIGRATIONS_SQL)
        _migrate_convention_aggregate_columns(conn)
        return store

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self._db_path,
                isolation_level=None,
                timeout=5.0,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            del self._local.conn

    def save_task(self, task: Task) -> None:
        # Auto-stamp brand-new tasks with the active annotation-rules
        # version so the runtime can reproduce later exactly which rule
        # set the task was annotated against. Only stamps when the row
        # does not yet exist — historical NULL-version tasks must stay
        # NULL (stamping them with the current active version would be
        # fake provenance: those tasks were annotated under a different
        # / unrecorded rule state). NULL means "pre-versioning era";
        # _load_guideline handles that by falling back to the latest
        # singleton version when the task is re-run.
        if not task.document_version_id:
            existing_row = self._conn.execute(
                "SELECT task_id FROM tasks WHERE task_id = ?",
                (task.task_id,),
            ).fetchone()
            if existing_row is None:
                version_id = self._active_annotation_rules_version_id()
                if version_id:
                    task.document_version_id = version_id
        d = task.to_dict()
        self._conn.execute(
            """
            INSERT INTO tasks (
                task_id, pipeline_id, status, current_attempt, modality,
                selected_annotator_id, active_run_id, next_retry_at,
                created_at, updated_at, document_version_id,
                source_ref_json, external_ref_json,
                annotation_requirements_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                pipeline_id=excluded.pipeline_id,
                status=excluded.status,
                current_attempt=excluded.current_attempt,
                modality=excluded.modality,
                selected_annotator_id=excluded.selected_annotator_id,
                active_run_id=excluded.active_run_id,
                next_retry_at=excluded.next_retry_at,
                updated_at=excluded.updated_at,
                document_version_id=excluded.document_version_id,
                source_ref_json=excluded.source_ref_json,
                external_ref_json=excluded.external_ref_json,
                annotation_requirements_json=excluded.annotation_requirements_json,
                metadata_json=excluded.metadata_json
            """,
            (
                d["task_id"], d["pipeline_id"], d["status"], d["current_attempt"], d["modality"],
                d["selected_annotator_id"], d["active_run_id"], d["next_retry_at"],
                d["created_at"], d["updated_at"], d["document_version_id"],
                json.dumps(d["source_ref"], sort_keys=True),
                json.dumps(d["external_ref"], sort_keys=True) if d["external_ref"] else None,
                json.dumps(d["annotation_requirements"], sort_keys=True),
                json.dumps(d["metadata"], sort_keys=True),
            ),
        )

    def load_task(self, task_id: str) -> Task:
        """Return the task with this id; raise KeyError if it does not exist."""
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return _row_to_task(row)

    def list_tasks(self) -> list[Task]:
        rows = self._conn.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()
        return [_row_to_task(r) for r in rows]

    def list_tasks_by_pipeline(self, pipeline_id: str) -> list[Task]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE pipeline_id = ? ORDER BY created_at",
            (pipeline_id,),
        ).fetchall()
        return [_row_to_task(r) for r in rows]

    def list_tasks_by_status(self, statuses: Iterable[TaskStatus]) -> list[Task]:
        values = [s.value for s in statuses]
        if not values:
            return []
        placeholders = ",".join("?" for _ in values)
        rows = self._conn.execute(
            f"SELECT * FROM tasks WHERE status IN ({placeholders}) ORDER BY created_at",
            values,
        ).fetchall()
        return [_row_to_task(r) for r in rows]

    def delete_task(self, task_id: str) -> dict[str, int]:
        """Cascade-delete a single task and all of its children, plus the on-disk
        artifact_payloads/<task_id>/ directory.

        Returns a dict of {table_name: rows_deleted} including an
        "artifact_files" entry counting on-disk files removed. The "tasks" entry
        is 1 if the task existed and 0 otherwise.
        """
        empty_report = {
            "tasks": 0,
            "audit_events": 0,
            "attempts": 0,
            "feedback_records": 0,
            "feedback_discussions": 0,
            "artifact_refs": 0,
            "outbox_records": 0,
            "active_runs": 0,
            "runtime_leases": 0,
            "artifact_files": 0,
        }
        row = self._conn.execute(
            "SELECT task_id FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return empty_report

        child_tables = (
            "audit_events",
            "attempts",
            "feedback_records",
            "feedback_discussions",
            "artifact_refs",
            "outbox_records",
            "active_runs",
            "runtime_leases",
        )

        report = dict(empty_report)
        report["tasks"] = 1
        for table in child_tables:
            count_row = self._conn.execute(
                f"SELECT COUNT(*) AS c FROM {table} WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            report[table] = count_row["c"]

        with self._conn:
            for table in child_tables:
                self._conn.execute(
                    f"DELETE FROM {table} WHERE task_id = ?",
                    (task_id,),
                )
            self._conn.execute(
                "DELETE FROM tasks WHERE task_id = ?",
                (task_id,),
            )

        files_removed = 0
        task_dir = self.root / "artifact_payloads" / task_id
        if task_dir.exists():
            files_removed = sum(1 for p in task_dir.rglob("*") if p.is_file())
            shutil.rmtree(task_dir, ignore_errors=False)
        report["artifact_files"] = files_removed
        return report

    def delete_pipeline(self, pipeline_id: str) -> dict[str, int]:
        """Cascade-delete every task with the given pipeline_id and all of its
        children, plus the on-disk artifact_payloads directory for each task.

        Returns a dict of {table_name: rows_deleted} including an
        "artifact_files" entry counting on-disk files removed.
        """
        task_rows = self._conn.execute(
            "SELECT task_id FROM tasks WHERE pipeline_id = ?",
            (pipeline_id,),
        ).fetchall()
        task_ids = [r["task_id"] for r in task_rows]

        empty_report = {
            "tasks": 0,
            "audit_events": 0,
            "attempts": 0,
            "feedback_records": 0,
            "feedback_discussions": 0,
            "artifact_refs": 0,
            "outbox_records": 0,
            "active_runs": 0,
            "runtime_leases": 0,
            "artifact_files": 0,
        }
        if not task_ids:
            return empty_report

        child_tables = (
            "audit_events",
            "attempts",
            "feedback_records",
            "feedback_discussions",
            "artifact_refs",
            "outbox_records",
            "active_runs",
            "runtime_leases",
        )

        # Chunk IN-clause args to stay well under SQLite's variable limit.
        chunk_size = 500
        chunks = [task_ids[i : i + chunk_size] for i in range(0, len(task_ids), chunk_size)]

        report = dict(empty_report)
        report["tasks"] = len(task_ids)

        # Pre-count child rows for the report.
        for table in child_tables:
            total = 0
            for chunk in chunks:
                placeholders = ",".join("?" for _ in chunk)
                row = self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM {table} WHERE task_id IN ({placeholders})",
                    chunk,
                ).fetchone()
                total += row["c"]
            report[table] = total

        with self._conn:
            for table in child_tables:
                for chunk in chunks:
                    placeholders = ",".join("?" for _ in chunk)
                    self._conn.execute(
                        f"DELETE FROM {table} WHERE task_id IN ({placeholders})",
                        chunk,
                    )
            self._conn.execute(
                "DELETE FROM tasks WHERE pipeline_id = ?",
                (pipeline_id,),
            )

        # Remove on-disk artifact_payloads/<task_id>/ directories.
        files_removed = 0
        artifact_root = self.root / "artifact_payloads"
        for tid in task_ids:
            task_dir = artifact_root / tid
            if task_dir.exists():
                # Count files before removal for the report.
                files_removed += sum(1 for _ in task_dir.rglob("*") if _.is_file())
                shutil.rmtree(task_dir, ignore_errors=False)
        report["artifact_files"] = files_removed
        return report

    def append_event(self, event: AuditEvent) -> None:
        d = event.to_dict()
        self._conn.execute(
            """
            INSERT INTO audit_events (
                event_id, task_id, previous_status, next_status, actor,
                reason, stage, attempt_id, created_at, metadata_json, seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT MAX(seq) + 1 FROM audit_events WHERE task_id = ?), 1)
            )
            """,
            (
                d["event_id"], d["task_id"], d["previous_status"], d["next_status"],
                d["actor"], d["reason"], d["stage"], d["attempt_id"], d["created_at"],
                json.dumps(d["metadata"], sort_keys=True),
                d["task_id"],
            ),
        )

    def list_events_paginated(
        self,
        *,
        pipeline_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditEvent], int]:
        """Return (events, total_count) ordered by created_at DESC.

        Optionally filtered to a single pipeline (joins audit_events ↔ tasks
        on task_id). Used by /api/events for paginated display — the dashboard
        was loading the full table every poll which got slow past ~30k rows.
        """
        if pipeline_id is None:
            count_sql = "SELECT COUNT(*) FROM audit_events"
            page_sql = (
                "SELECT * FROM audit_events ORDER BY created_at DESC LIMIT ? OFFSET ?"
            )
            count_args: tuple = ()
            page_args: tuple = (limit, offset)
        else:
            count_sql = (
                "SELECT COUNT(*) FROM audit_events e "
                "JOIN tasks t ON e.task_id = t.task_id WHERE t.pipeline_id = ?"
            )
            page_sql = (
                "SELECT e.* FROM audit_events e "
                "JOIN tasks t ON e.task_id = t.task_id "
                "WHERE t.pipeline_id = ? "
                "ORDER BY e.created_at DESC LIMIT ? OFFSET ?"
            )
            count_args = (pipeline_id,)
            page_args = (pipeline_id, limit, offset)

        total = int(self._conn.execute(count_sql, count_args).fetchone()[0])
        rows = self._conn.execute(page_sql, page_args).fetchall()
        events = [
            AuditEvent.from_dict({
                "event_id": r["event_id"],
                "task_id": r["task_id"],
                "previous_status": r["previous_status"],
                "next_status": r["next_status"],
                "actor": r["actor"],
                "reason": r["reason"],
                "stage": r["stage"],
                "attempt_id": r["attempt_id"],
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata_json"]),
            })
            for r in rows
        ]
        return events, total

    def list_events(self, task_id: str) -> list[AuditEvent]:
        rows = self._conn.execute(
            "SELECT * FROM audit_events WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ).fetchall()
        return [
            AuditEvent.from_dict({
                "event_id": r["event_id"],
                "task_id": r["task_id"],
                "previous_status": r["previous_status"],
                "next_status": r["next_status"],
                "actor": r["actor"],
                "reason": r["reason"],
                "stage": r["stage"],
                "attempt_id": r["attempt_id"],
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata_json"]),
            })
            for r in rows
        ]

    def append_attempt(self, attempt) -> None:
        d = attempt.to_dict()
        self._conn.execute(
            """
            INSERT INTO attempts (
                attempt_id, task_id, idx, stage, status,
                started_at, finished_at, provider_id, model, effort,
                route_role, summary, error_json, artifacts_json, seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT MAX(seq) + 1 FROM attempts WHERE task_id = ?), 1)
            )
            """,
            (
                d["attempt_id"], d["task_id"], d["index"], d["stage"], d["status"],
                d["started_at"], d["finished_at"], d["provider_id"], d["model"], d["effort"],
                d["route_role"], d["summary"],
                json.dumps(d["error"], sort_keys=True) if d["error"] else None,
                json.dumps(d["artifacts"], sort_keys=True),
                d["task_id"],
            ),
        )

    def count_succeeded_attempts_since(
        self,
        since_iso: str,
        *,
        pipeline_id: str | None = None,
    ) -> dict[str, int]:
        """Return {stage: count} for attempts with status='succeeded' and
        finished_at >= since_iso. Optionally filtered to one pipeline.

        Zero-duration attempts (started_at = finished_at) are excluded because
        they are synthetic records injected by migration/import scripts rather
        than real LLM calls, and would otherwise inflate the throughput metric.

        Used by the dashboard stats bar to compute per-stage throughput.
        """
        if pipeline_id is None:
            rows = self._conn.execute(
                "SELECT stage, COUNT(*) AS c FROM attempts "
                "WHERE status = 'succeeded' AND finished_at >= ? "
                "AND started_at != finished_at "
                "GROUP BY stage",
                (since_iso,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT a.stage, COUNT(*) AS c FROM attempts a "
                "JOIN tasks t ON a.task_id = t.task_id "
                "WHERE a.status = 'succeeded' AND a.finished_at >= ? "
                "AND a.started_at != a.finished_at "
                "AND t.pipeline_id = ? "
                "GROUP BY a.stage",
                (since_iso, pipeline_id),
            ).fetchall()
        return {r["stage"]: r["c"] for r in rows}

    def count_accepted_since(
        self,
        since_iso: str,
        *,
        pipeline_id: str | None = None,
    ) -> int:
        """Return the number of audit_events transitions to 'accepted' since
        *since_iso*.  Used by the dashboard stats bar to compute ETA.
        """
        if pipeline_id is None:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM audit_events "
                "WHERE next_status = 'accepted' AND created_at >= ?",
                (since_iso,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM audit_events e "
                "JOIN tasks t ON e.task_id = t.task_id "
                "WHERE e.next_status = 'accepted' AND e.created_at >= ? "
                "AND t.pipeline_id = ?",
                (since_iso, pipeline_id),
            ).fetchone()
        return int(row["c"]) if row else 0

    def fetch_pipeline_health_metrics(
        self,
        *,
        pipeline_id: str | None = None,
    ) -> dict:
        """Return aggregate quality metrics for the dashboard stats bar.

        Returns a dict with:
          - ``accepted_count``      total accepted tasks
          - ``terminal_count``      accepted + rejected tasks
          - ``first_pass_count``    accepted tasks with no arbitration attempt
          - ``arb_entered_count``   terminal tasks that had ≥1 arbitration attempt
          - ``avg_llm_calls``       avg annotation+qc+arb succeeded attempts per
                                    accepted task (excludes zero-duration synthetic)
        """
        pid_filter_task = "AND t.pipeline_id = ?" if pipeline_id else ""
        pid_args: tuple = (pipeline_id,) if pipeline_id else ()

        # accepted_count and terminal_count from tasks table
        rows = self._conn.execute(
            f"SELECT status, COUNT(*) AS c FROM tasks t "
            f"WHERE t.status IN ('accepted','rejected') {pid_filter_task} "
            f"GROUP BY t.status",
            pid_args,
        ).fetchall()
        counts = {r["status"]: r["c"] for r in rows}
        accepted_count = counts.get("accepted", 0)
        terminal_count = accepted_count + counts.get("rejected", 0)

        # accepted tasks that entered arbitration (at least one arb attempt)
        row = self._conn.execute(
            f"SELECT COUNT(DISTINCT t.task_id) AS c FROM tasks t "
            f"JOIN attempts a ON a.task_id = t.task_id "
            f"WHERE t.status = 'accepted' AND a.stage = 'arbitration' "
            f"{pid_filter_task}",
            pid_args,
        ).fetchone()
        accepted_with_arb = int(row["c"]) if row else 0
        first_pass_count = accepted_count - accepted_with_arb

        # terminal tasks (accepted or rejected) that entered arbitration
        row = self._conn.execute(
            f"SELECT COUNT(DISTINCT t.task_id) AS c FROM tasks t "
            f"JOIN attempts a ON a.task_id = t.task_id "
            f"WHERE t.status IN ('accepted','rejected') AND a.stage = 'arbitration' "
            f"{pid_filter_task}",
            pid_args,
        ).fetchone()
        arb_entered_count = int(row["c"]) if row else 0

        # avg succeeded annotation+qc+arb calls per accepted task
        row = self._conn.execute(
            f"SELECT AVG(call_count) AS avg FROM ("
            f"  SELECT COUNT(*) AS call_count FROM tasks t "
            f"  JOIN attempts a ON a.task_id = t.task_id "
            f"  WHERE t.status = 'accepted' "
            f"  AND a.stage IN ('annotation','qc','arbitration') "
            f"  AND a.status = 'succeeded' "
            f"  AND a.started_at != a.finished_at "
            f"  {pid_filter_task} "
            f"  GROUP BY t.task_id"
            f")",
            pid_args,
        ).fetchone()
        avg_llm_calls = round(float(row["avg"]), 2) if row and row["avg"] is not None else 0.0

        return {
            "accepted_count": accepted_count,
            "terminal_count": terminal_count,
            "first_pass_count": first_pass_count,
            "arb_entered_count": arb_entered_count,
            "avg_llm_calls": avg_llm_calls,
        }

    def list_attempts(self, task_id: str):
        rows = self._conn.execute(
            "SELECT * FROM attempts WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ).fetchall()
        return [
            Attempt.from_dict({
                "attempt_id": r["attempt_id"], "task_id": r["task_id"],
                "index": r["idx"], "stage": r["stage"], "status": r["status"],
                "started_at": r["started_at"], "finished_at": r["finished_at"],
                "provider_id": r["provider_id"], "model": r["model"], "effort": r["effort"],
                "route_role": r["route_role"], "summary": r["summary"],
                "error": json.loads(r["error_json"]) if r["error_json"] else None,
                "artifacts": json.loads(r["artifacts_json"]),
            })
            for r in rows
        ]

    def append_feedback(self, feedback) -> None:
        d = feedback.to_dict()
        self._conn.execute(
            """
            INSERT INTO feedback_records (
                feedback_id, task_id, attempt_id, source_stage, severity,
                category, message, target_json, suggested_action,
                created_at, created_by, metadata_json, seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT MAX(seq) + 1 FROM feedback_records WHERE task_id = ?), 1)
            )
            """,
            (
                d["feedback_id"], d["task_id"], d["attempt_id"], d["source_stage"], d["severity"],
                d["category"], d["message"],
                json.dumps(d["target"], sort_keys=True),
                d["suggested_action"], d["created_at"], d["created_by"],
                json.dumps(d["metadata"], sort_keys=True),
                d["task_id"],
            ),
        )

    def list_feedback(self, task_id: str):
        rows = self._conn.execute(
            "SELECT * FROM feedback_records WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ).fetchall()
        return [
            FeedbackRecord.from_dict({
                "feedback_id": r["feedback_id"], "task_id": r["task_id"],
                "attempt_id": r["attempt_id"], "source_stage": r["source_stage"],
                "severity": r["severity"], "category": r["category"], "message": r["message"],
                "target": json.loads(r["target_json"]),
                "suggested_action": r["suggested_action"],
                "created_at": r["created_at"], "created_by": r["created_by"],
                "metadata": json.loads(r["metadata_json"]),
            })
            for r in rows
        ]

    def append_feedback_many(self, feedbacks) -> None:
        self._conn.executemany(
            """
            INSERT INTO feedback_records (
                feedback_id, task_id, attempt_id, source_stage, severity,
                category, message, target_json, suggested_action,
                created_at, created_by, metadata_json, seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT MAX(seq) + 1 FROM feedback_records WHERE task_id = ?), 1)
            )
            """,
            [
                (
                    d["feedback_id"], d["task_id"], d["attempt_id"], d["source_stage"], d["severity"],
                    d["category"], d["message"],
                    json.dumps(d["target"], sort_keys=True),
                    d["suggested_action"], d["created_at"], d["created_by"],
                    json.dumps(d["metadata"], sort_keys=True),
                    d["task_id"],
                )
                for d in (fb.to_dict() for fb in feedbacks)
            ],
        )

    def append_feedback_discussion(self, entry) -> None:
        d = entry.to_dict()
        self._conn.execute(
            """
            INSERT INTO feedback_discussions (
                entry_id, task_id, feedback_id, role, stance, message,
                agreed_points_json, disputed_points_json, proposed_resolution,
                consensus, created_at, created_by, metadata_json, seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT MAX(seq) + 1 FROM feedback_discussions WHERE task_id = ?), 1)
            )
            """,
            (
                d["entry_id"], d["task_id"], d["feedback_id"], d["role"], d["stance"], d["message"],
                json.dumps(d["agreed_points"], sort_keys=True),
                json.dumps(d["disputed_points"], sort_keys=True),
                d["proposed_resolution"], 1 if d["consensus"] else 0,
                d["created_at"], d["created_by"],
                json.dumps(d["metadata"], sort_keys=True),
                d["task_id"],
            ),
        )

    def list_feedback_discussions(self, task_id: str):
        rows = self._conn.execute(
            "SELECT * FROM feedback_discussions WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ).fetchall()
        return [
            FeedbackDiscussionEntry.from_dict({
                "entry_id": r["entry_id"], "task_id": r["task_id"],
                "feedback_id": r["feedback_id"], "role": r["role"], "stance": r["stance"],
                "message": r["message"],
                "agreed_points": json.loads(r["agreed_points_json"]),
                "disputed_points": json.loads(r["disputed_points_json"]),
                "proposed_resolution": r["proposed_resolution"],
                "consensus": bool(r["consensus"]),
                "created_at": r["created_at"], "created_by": r["created_by"],
                "metadata": json.loads(r["metadata_json"]),
            })
            for r in rows
        ]

    def clear_feedback_for_attempt(self, task_id: str, attempt_id: str) -> int:
        """Delete all feedback records and their discussion entries for a
        specific attempt. Returns the number of feedback records deleted.

        Used by the high-hallucination reset path: when an annotation attempt
        produces so many non-verbatim spans that recording per-span feedback
        would blow the context window, we wipe the feedback for that attempt
        and reset the task to PENDING for a clean re-annotation.
        """
        # Collect the feedback_ids so we can cascade to discussions.
        rows = self._conn.execute(
            "SELECT feedback_id FROM feedback_records WHERE task_id = ? AND attempt_id = ?",
            (task_id, attempt_id),
        ).fetchall()
        feedback_ids = [r["feedback_id"] for r in rows]
        if not feedback_ids:
            return 0
        placeholders = ",".join("?" * len(feedback_ids))
        self._conn.execute(
            f"DELETE FROM feedback_discussions WHERE feedback_id IN ({placeholders})",
            feedback_ids,
        )
        self._conn.execute(
            "DELETE FROM feedback_records WHERE task_id = ? AND attempt_id = ?",
            (task_id, attempt_id),
        )
        return len(feedback_ids)

    def append_artifact(self, artifact) -> None:
        d = artifact.to_dict()
        self._conn.execute(
            """
            INSERT INTO artifact_refs (
                artifact_id, task_id, kind, path, content_type,
                created_at, metadata_json, seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT MAX(seq) + 1 FROM artifact_refs WHERE task_id = ?), 1)
            )
            """,
            (
                d["artifact_id"], d["task_id"], d["kind"], d["path"], d["content_type"],
                d["created_at"], json.dumps(d["metadata"], sort_keys=True),
                d["task_id"],
            ),
        )

    def list_artifacts(self, task_id: str):
        rows = self._conn.execute(
            "SELECT * FROM artifact_refs WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ).fetchall()
        return [
            ArtifactRef.from_dict({
                "artifact_id": r["artifact_id"], "task_id": r["task_id"],
                "kind": r["kind"], "path": r["path"], "content_type": r["content_type"],
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata_json"]),
            })
            for r in rows
        ]

    def save_outbox(self, record) -> None:
        d = record.to_dict()
        self._conn.execute(
            """
            INSERT INTO outbox_records (
                record_id, task_id, kind, payload_json, status,
                retry_count, next_retry_at, last_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                kind=excluded.kind,
                payload_json=excluded.payload_json,
                status=excluded.status,
                retry_count=excluded.retry_count,
                next_retry_at=excluded.next_retry_at,
                last_error=excluded.last_error
            """,
            (
                d["record_id"], d["task_id"], d["kind"],
                json.dumps(d["payload"], sort_keys=True),
                d["status"], d["retry_count"], d["next_retry_at"], d["last_error"], d["created_at"],
            ),
        )

    def _row_to_outbox(self, r):
        return OutboxRecord.from_dict({
            "record_id": r["record_id"], "task_id": r["task_id"], "kind": r["kind"],
            "payload": json.loads(r["payload_json"]), "status": r["status"],
            "retry_count": r["retry_count"], "next_retry_at": r["next_retry_at"],
            "last_error": r["last_error"], "created_at": r["created_at"],
        })

    def list_outbox(self):
        rows = self._conn.execute("SELECT * FROM outbox_records ORDER BY created_at").fetchall()
        return [self._row_to_outbox(r) for r in rows]

    def list_pending_outbox(self, *, now):
        rows = self._conn.execute(
            """
            SELECT * FROM outbox_records
            WHERE status = ?
              AND (next_retry_at IS NULL OR next_retry_at <= ?)
            ORDER BY created_at
            """,
            ("pending", now.isoformat()),
        ).fetchall()
        return [self._row_to_outbox(r) for r in rows]

    def save_active_run(self, run) -> None:
        d = run.to_dict()
        self._conn.execute(
            """
            INSERT INTO active_runs (
                run_id, task_id, stage, attempt_id, provider_target,
                started_at, heartbeat_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                stage=excluded.stage,
                attempt_id=excluded.attempt_id,
                provider_target=excluded.provider_target,
                heartbeat_at=excluded.heartbeat_at,
                metadata_json=excluded.metadata_json
            """,
            (
                d["run_id"], d["task_id"], d["stage"], d["attempt_id"], d["provider_target"],
                d["started_at"], d["heartbeat_at"],
                json.dumps(d["metadata"], sort_keys=True),
            ),
        )

    def list_active_runs(self):
        rows = self._conn.execute("SELECT * FROM active_runs ORDER BY started_at").fetchall()
        return [
            ActiveRun.from_dict({
                "run_id": r["run_id"], "task_id": r["task_id"], "stage": r["stage"],
                "attempt_id": r["attempt_id"], "provider_target": r["provider_target"],
                "started_at": r["started_at"], "heartbeat_at": r["heartbeat_at"],
                "metadata": json.loads(r["metadata_json"]),
            })
            for r in rows
        ]

    def delete_active_run(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM active_runs WHERE run_id = ?", (run_id,))

    def save_runtime_lease(self, lease) -> bool:
        d = lease.to_dict()
        try:
            self._conn.execute(
                """
                INSERT INTO runtime_leases (
                    lease_id, task_id, stage, acquired_at, heartbeat_at,
                    expires_at, owner, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["lease_id"], d["task_id"], d["stage"],
                    d["acquired_at"], d["heartbeat_at"], d["expires_at"], d["owner"],
                    json.dumps(d["metadata"], sort_keys=True),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def list_runtime_leases(self):
        rows = self._conn.execute("SELECT * FROM runtime_leases ORDER BY acquired_at").fetchall()
        return [
            RuntimeLease.from_dict({
                "lease_id": r["lease_id"], "task_id": r["task_id"], "stage": r["stage"],
                "acquired_at": r["acquired_at"], "heartbeat_at": r["heartbeat_at"],
                "expires_at": r["expires_at"], "owner": r["owner"],
                "metadata": json.loads(r["metadata_json"]),
            })
            for r in rows
        ]

    def delete_runtime_lease(self, lease_id: str) -> None:
        self._conn.execute("DELETE FROM runtime_leases WHERE lease_id = ?", (lease_id,))

    def append_coordination_record(self, kind: str, record: dict) -> None:
        self._conn.execute(
            "INSERT INTO coordination_records (kind, record_json, created_at) VALUES (?, ?, ?)",
            (kind, json.dumps(record, sort_keys=True), record.get("created_at") or ""),
        )

    def list_coordination_records(self, kind: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT record_json FROM coordination_records WHERE kind = ? ORDER BY rowid_pk",
            (kind,),
        ).fetchall()
        return [json.loads(r["record_json"]) for r in rows]

    @property
    def _runtime_dir(self) -> Path:
        return self.root / "runtime"

    @property
    def _runtime_heartbeat_path(self) -> Path:
        return self._runtime_dir / "heartbeat.json"

    @property
    def _runtime_snapshot_path(self) -> Path:
        return self._runtime_dir / "runtime_snapshot.json"

    def save_runtime_heartbeat(self, heartbeat_at) -> None:
        self._runtime_heartbeat_path.write_text(
            json.dumps({"heartbeat_at": heartbeat_at.isoformat()}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load_runtime_heartbeat(self):
        from datetime import datetime
        if not self._runtime_heartbeat_path.exists():
            return None
        payload = json.loads(self._runtime_heartbeat_path.read_text(encoding="utf-8"))
        return datetime.fromisoformat(payload["heartbeat_at"])

    def save_runtime_snapshot(self, snap) -> None:
        self._runtime_snapshot_path.write_text(
            json.dumps(snap.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def load_runtime_snapshot(self):
        if not self._runtime_snapshot_path.exists():
            return None
        return RuntimeSnapshot.from_dict(json.loads(self._runtime_snapshot_path.read_text(encoding="utf-8")))

    def save_document(self, doc) -> None:
        d = doc.to_dict()
        self._conn.execute(
            """
            INSERT INTO documents (document_id, title, description, created_at, created_by, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                title=excluded.title,
                description=excluded.description,
                metadata_json=excluded.metadata_json
            """,
            (
                d["document_id"], d["title"], d["description"], d["created_at"], d["created_by"],
                json.dumps(d["metadata"], sort_keys=True),
            ),
        )

    def load_document(self, document_id: str):
        row = self._conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        if row is None:
            raise KeyError(document_id)
        return AnnotationDocument.from_dict({
            "document_id": row["document_id"], "title": row["title"], "description": row["description"],
            "created_at": row["created_at"], "created_by": row["created_by"],
            "metadata": json.loads(row["metadata_json"]),
        })

    def list_documents(self):
        rows = self._conn.execute("SELECT * FROM documents ORDER BY created_at").fetchall()
        return [
            AnnotationDocument.from_dict({
                "document_id": r["document_id"], "title": r["title"], "description": r["description"],
                "created_at": r["created_at"], "created_by": r["created_by"],
                "metadata": json.loads(r["metadata_json"]),
            })
            for r in rows
        ]

    def _active_annotation_rules_version_id(self) -> str | None:
        """Return the version_id of the latest version of the singleton
        annotation-rules document, or None if no such document or version
        exists yet. Used to auto-stamp newly created tasks.
        """
        try:
            doc_row = self._conn.execute(
                "SELECT document_id, metadata_json FROM documents"
            ).fetchall()
        except Exception:
            return None
        target_doc_id: str | None = None
        for r in doc_row:
            try:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                meta = {}
            if isinstance(meta, dict) and meta.get("role") == "annotation_rules":
                target_doc_id = r["document_id"]
                break
        if not target_doc_id:
            return None
        ver_row = self._conn.execute(
            """
            SELECT version_id FROM document_versions
            WHERE document_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (target_doc_id,),
        ).fetchone()
        if ver_row is None:
            return None
        return ver_row["version_id"]

    def _content_path_for(self, document_id: str, version: str) -> Path:
        return self.root / "document_versions" / document_id / f"{version}.md"

    def save_document_version(self, ver) -> None:
        d = ver.to_dict()
        content = d["content"]
        path = self._content_path_for(d["document_id"], d["version"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        rel_path = path.relative_to(self.root).as_posix()
        self._conn.execute(
            """
            INSERT INTO document_versions (
                version_id, document_id, version, content_path, content_sha256,
                changelog, created_at, created_by, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version_id) DO UPDATE SET
                content_path=excluded.content_path,
                content_sha256=excluded.content_sha256,
                changelog=excluded.changelog,
                metadata_json=excluded.metadata_json
            """,
            (
                d["version_id"], d["document_id"], d["version"], rel_path, sha,
                d["changelog"], d["created_at"], d["created_by"],
                json.dumps(d["metadata"], sort_keys=True),
            ),
        )

    def _row_to_doc_version(self, row):
        path = self.root / row["content_path"]
        content = path.read_text(encoding="utf-8")
        return AnnotationDocumentVersion.from_dict({
            "version_id": row["version_id"], "document_id": row["document_id"],
            "version": row["version"], "content": content,
            "changelog": row["changelog"], "created_at": row["created_at"],
            "created_by": row["created_by"], "metadata": json.loads(row["metadata_json"]),
        })

    def load_document_version(self, version_id: str):
        row = self._conn.execute(
            "SELECT * FROM document_versions WHERE version_id = ?", (version_id,)
        ).fetchone()
        if row is None:
            raise KeyError(version_id)
        return self._row_to_doc_version(row)

    def list_document_versions(self, document_id: str):
        rows = self._conn.execute(
            "SELECT * FROM document_versions WHERE document_id = ? ORDER BY created_at",
            (document_id,),
        ).fetchall()
        return [self._row_to_doc_version(r) for r in rows]

    def save_export_manifest(self, manifest) -> None:
        d = manifest.to_dict()
        self._conn.execute(
            """
            INSERT INTO export_manifests (
                export_id, project_id, created_at,
                output_paths_json, task_ids_included_json, task_ids_excluded_json,
                artifact_ids_json, source_files_json,
                annotation_rules_hash, schema_version, validator_version,
                validation_summary_json, known_limitations_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(export_id) DO UPDATE SET
                project_id=excluded.project_id,
                output_paths_json=excluded.output_paths_json,
                task_ids_included_json=excluded.task_ids_included_json,
                task_ids_excluded_json=excluded.task_ids_excluded_json,
                artifact_ids_json=excluded.artifact_ids_json,
                source_files_json=excluded.source_files_json,
                annotation_rules_hash=excluded.annotation_rules_hash,
                schema_version=excluded.schema_version,
                validator_version=excluded.validator_version,
                validation_summary_json=excluded.validation_summary_json,
                known_limitations_json=excluded.known_limitations_json
            """,
            (
                d["export_id"], d["project_id"], d["created_at"],
                json.dumps(d["output_paths"], sort_keys=True),
                json.dumps(d["task_ids_included"], sort_keys=True),
                json.dumps(d["task_ids_excluded"], sort_keys=True),
                json.dumps(d["artifact_ids"], sort_keys=True),
                json.dumps(d["source_files"], sort_keys=True),
                d["annotation_rules_hash"], d["schema_version"], d["validator_version"],
                json.dumps(d["validation_summary"], sort_keys=True),
                json.dumps(d["known_limitations"], sort_keys=True),
            ),
        )

    def list_export_manifests(self):
        rows = self._conn.execute(
            "SELECT * FROM export_manifests ORDER BY project_id, created_at"
        ).fetchall()
        return [
            ExportManifest.from_dict({
                "export_id": r["export_id"], "project_id": r["project_id"],
                "created_at": r["created_at"],
                "output_paths": json.loads(r["output_paths_json"]),
                "task_ids_included": json.loads(r["task_ids_included_json"]),
                "task_ids_excluded": json.loads(r["task_ids_excluded_json"]),
                "artifact_ids": json.loads(r["artifact_ids_json"]),
                "source_files": json.loads(r["source_files_json"]),
                "annotation_rules_hash": r["annotation_rules_hash"],
                "schema_version": r["schema_version"], "validator_version": r["validator_version"],
                "validation_summary": json.loads(r["validation_summary_json"]),
                "known_limitations": json.loads(r["known_limitations_json"]),
            })
            for r in rows
        ]

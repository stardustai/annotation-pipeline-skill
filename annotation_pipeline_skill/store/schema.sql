PRAGMA user_version = 1;

CREATE TABLE tasks (
    task_id                 TEXT PRIMARY KEY,
    pipeline_id             TEXT NOT NULL,
    status                  TEXT NOT NULL,
    current_attempt         INTEGER NOT NULL DEFAULT 0,
    modality                TEXT NOT NULL DEFAULT 'text',
    selected_annotator_id   TEXT,
    active_run_id           TEXT,
    next_retry_at           TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    document_version_id     TEXT,
    source_ref_json         TEXT NOT NULL,
    external_ref_json       TEXT,
    annotation_requirements_json TEXT NOT NULL,
    metadata_json           TEXT NOT NULL
);
CREATE INDEX idx_tasks_pipeline_status ON tasks(pipeline_id, status);
CREATE INDEX idx_tasks_status_created ON tasks(status, created_at);
CREATE INDEX idx_tasks_next_retry ON tasks(next_retry_at) WHERE next_retry_at IS NOT NULL;

CREATE TABLE audit_events (
    event_id        TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    previous_status TEXT NOT NULL,
    next_status     TEXT NOT NULL,
    actor           TEXT NOT NULL,
    reason          TEXT NOT NULL,
    stage           TEXT NOT NULL,
    attempt_id      TEXT,
    created_at      TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,
    seq             INTEGER NOT NULL
);
CREATE INDEX idx_audit_task_seq ON audit_events(task_id, seq);

CREATE TABLE attempts (
    attempt_id   TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL,
    idx          INTEGER NOT NULL,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    provider_id  TEXT,
    model        TEXT,
    effort       TEXT,
    route_role   TEXT,
    summary      TEXT,
    error_json   TEXT,
    artifacts_json TEXT NOT NULL,
    seq          INTEGER NOT NULL
);
CREATE INDEX idx_attempts_task_seq ON attempts(task_id, seq);

CREATE TABLE feedback_records (
    feedback_id      TEXT PRIMARY KEY,
    task_id          TEXT NOT NULL,
    attempt_id       TEXT NOT NULL,
    source_stage     TEXT NOT NULL,
    severity         TEXT NOT NULL,
    category         TEXT NOT NULL,
    message          TEXT NOT NULL,
    target_json      TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    created_by       TEXT NOT NULL,
    metadata_json    TEXT NOT NULL,
    seq              INTEGER NOT NULL
);
CREATE INDEX idx_feedback_task_seq ON feedback_records(task_id, seq);

CREATE TABLE feedback_discussions (
    entry_id            TEXT PRIMARY KEY,
    task_id             TEXT NOT NULL,
    feedback_id         TEXT NOT NULL,
    role                TEXT NOT NULL,
    stance              TEXT NOT NULL,
    message             TEXT NOT NULL,
    agreed_points_json  TEXT NOT NULL,
    disputed_points_json TEXT NOT NULL,
    proposed_resolution TEXT,
    consensus           INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    created_by          TEXT NOT NULL,
    metadata_json       TEXT NOT NULL,
    seq                 INTEGER NOT NULL
);
CREATE INDEX idx_discussion_task_seq ON feedback_discussions(task_id, seq);

CREATE TABLE artifact_refs (
    artifact_id   TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    path          TEXT NOT NULL,
    content_type  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    seq           INTEGER NOT NULL
);
CREATE INDEX idx_artifact_task_seq ON artifact_refs(task_id, seq);

CREATE TABLE outbox_records (
    record_id      TEXT PRIMARY KEY,
    task_id        TEXT NOT NULL,
    kind           TEXT NOT NULL,
    payload_json   TEXT NOT NULL,
    status         TEXT NOT NULL,
    retry_count    INTEGER NOT NULL DEFAULT 0,
    next_retry_at  TEXT,
    last_error     TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX idx_outbox_status_retry ON outbox_records(status, next_retry_at);
CREATE INDEX idx_outbox_task ON outbox_records(task_id);

CREATE TABLE active_runs (
    run_id           TEXT PRIMARY KEY,
    task_id          TEXT NOT NULL,
    stage            TEXT NOT NULL,
    attempt_id       TEXT NOT NULL,
    provider_target  TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    heartbeat_at     TEXT NOT NULL,
    metadata_json    TEXT NOT NULL
);
CREATE INDEX idx_active_runs_task ON active_runs(task_id);

CREATE TABLE runtime_leases (
    lease_id      TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL,
    stage         TEXT NOT NULL,
    acquired_at   TEXT NOT NULL,
    heartbeat_at  TEXT NOT NULL,
    expires_at    TEXT NOT NULL,
    owner         TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE(task_id, stage)
);

CREATE TABLE coordination_records (
    rowid_pk    INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX idx_coord_kind_created ON coordination_records(kind, created_at);

CREATE TABLE documents (
    document_id   TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    created_by    TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE document_versions (
    version_id    TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    version       TEXT NOT NULL,
    content_path  TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    changelog     TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    created_by    TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX idx_docver_doc_version ON document_versions(document_id, version);

CREATE TABLE export_manifests (
    export_id              TEXT PRIMARY KEY,
    project_id             TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    output_paths_json      TEXT NOT NULL,
    task_ids_included_json TEXT NOT NULL,
    task_ids_excluded_json TEXT NOT NULL,
    artifact_ids_json      TEXT NOT NULL,
    source_files_json      TEXT NOT NULL,
    annotation_rules_hash  TEXT,
    schema_version         TEXT NOT NULL,
    validator_version      TEXT NOT NULL,
    validation_summary_json TEXT NOT NULL,
    known_limitations_json TEXT NOT NULL
);
CREATE INDEX idx_export_project_created ON export_manifests(project_id, created_at);

-- Entity convention store: case-by-case "lesson learned" knowledge accumulated
-- from QC discussions, arbiter rulings, and HR feedback. Used to inject
-- per-project entity-type guidance into annotator/QC/arbiter prompts so
-- ambiguous spans get consistent classification across runs.
CREATE TABLE IF NOT EXISTS entity_conventions (
    convention_id  TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    span_lower     TEXT NOT NULL,        -- lowercased for case-insensitive match
    span_original  TEXT NOT NULL,        -- as first seen
    entity_type    TEXT,                  -- canonical type; NULL when disputed
    status         TEXT NOT NULL,         -- 'active' | 'disputed'
    evidence_count INTEGER NOT NULL DEFAULT 1,
    proposals_json TEXT NOT NULL DEFAULT '[]',  -- audit trail of (type, source, task_id, created_at)
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    created_by     TEXT NOT NULL,        -- first source: 'hr_correction'/'declared'/'arbiter_consensus'
    notes          TEXT,
    -- Materialized aggregates of proposals_json, maintained on every write so
    -- the injection gate is a plain indexed SQL predicate (no JSON parse).
    distinct_task_count INTEGER NOT NULL DEFAULT 0,
    dispute_count       INTEGER NOT NULL DEFAULT 0,
    dispute_pct         REAL NOT NULL DEFAULT 0.0,
    dominant_type       TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_project_span ON entity_conventions(project_id, span_lower);
CREATE INDEX IF NOT EXISTS idx_conv_project_status ON entity_conventions(project_id, status);
CREATE INDEX IF NOT EXISTS idx_conv_inject ON entity_conventions(project_id, status, distinct_task_count);

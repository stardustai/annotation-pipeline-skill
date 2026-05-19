export interface TaskCard {
  task_id: string;
  status: string;
  operator_stage: string;
  pipeline_chain: string;
  modality: string;
  annotation_types: string[];
  selected_annotator_id: string | null;
  annotator_model: string | null;
  qc_model: string | null;
  status_age_seconds: number;
  latest_attempt_status: string | null;
  feedback_count: number;
  retry_pending: boolean;
  blocked: boolean;
  external_sync_pending: boolean;
  row_count: number | null;
  attempt_count: number;
}

export interface KanbanColumn {
  id: string;
  title: string;
  cards: TaskCard[];
}

export interface KanbanSnapshot {
  project_id: string | null;
  stage_view?: string;
  columns: KanbanColumn[];
}

export interface ProjectSummary {
  project_id: string;
  task_count: number;
  status_counts: Record<string, number>;
}

export interface ProjectSnapshot {
  projects: ProjectSummary[];
}

export interface TaskDetailArtifact {
  artifact_id: string;
  task_id: string;
  kind: string;
  path: string;
  content_type: string;
  created_at: string;
  metadata: Record<string, unknown>;
  payload: unknown;
}

export interface StoreInfo {
  key: string;
  name: string;
  path: string;
  pipeline_count: number;
  task_count: number;
}

export interface StoresSnapshot {
  workspace_path?: string;
  stores: StoreInfo[];
}

export interface AnnotationDocument {
  document_id: string;
  title: string;
  description: string;
  created_at: string;
  created_by: string;
  metadata: Record<string, unknown>;
}

export interface AnnotationDocumentVersion {
  version_id: string;
  document_id: string;
  version: string;
  content: string;
  changelog: string;
  created_at: string;
  created_by: string;
  metadata: Record<string, unknown>;
}

export interface DocumentDetail {
  document: AnnotationDocument;
  versions: AnnotationDocumentVersion[];
}

export interface DocumentsSnapshot {
  documents: AnnotationDocument[];
}

export interface TaskDetail {
  task: {
    task_id: string;
    pipeline_id: string;
    source_ref: Record<string, unknown>;
    modality: string;
    annotation_requirements: Record<string, unknown>;
    selected_annotator_id: string | null;
    status: string;
    current_attempt: number;
    metadata: Record<string, unknown>;
    document_version_id: string | null;
  };
  attempts: Array<Record<string, unknown>>;
  artifacts: TaskDetailArtifact[];
  events: Array<Record<string, unknown>>;
  feedback: Array<Record<string, unknown>>;
  feedback_discussions: Array<Record<string, unknown>>;
  feedback_consensus: {
    total_feedback: number;
    consensus_feedback: number;
    open_feedback: string[];
    can_accept_by_consensus: boolean;
  };
}

export interface ConfigFile {
  id: string;
  title: string;
  path: string;
  exists: boolean;
  content: string;
}

export interface ConfigSnapshot {
  files: ConfigFile[];
}

export interface EventLog {
  events: Array<Record<string, unknown>>;
  total?: number;
  limit?: number;
  offset?: number;
}

export interface RuntimeStatus {
  healthy: boolean;
  heartbeat_at: string | null;
  heartbeat_age_seconds: number | null;
  active: boolean;
  errors: string[];
}

export interface QueueCounts {
  draft: number;
  pending: number;
  annotating: number;
  qc: number;
  arbitrating: number;
  human_review: number;
  accepted: number;
  rejected: number;
  blocked: number;
  cancelled: number;
}

export interface ActiveRun {
  run_id: string;
  task_id: string;
  stage: string;
  attempt_id: string;
  provider_target: string;
  started_at: string;
  heartbeat_at: string;
  metadata: Record<string, unknown>;
}

export interface CapacitySnapshot {
  max_concurrent_tasks: number;
  active_count: number;
  available_slots: number;
}

export interface RuntimeSnapshot {
  generated_at: string;
  runtime_status: RuntimeStatus;
  queue_counts: QueueCounts;
  active_runs: ActiveRun[];
  capacity: CapacitySnapshot;
  stale_tasks: string[];
  due_retries: string[];
  project_summaries: ProjectSummary[];
}

export interface RuntimeMonitorReport {
  ok: boolean;
  failures: string[];
  details: Record<string, Record<string, unknown>>;
}

export interface RuntimeRunOnceResponse {
  ok: boolean;
  snapshot: RuntimeSnapshot;
}

export interface ReadinessReport {
  project_id: string;
  ready_for_training: boolean;
  accepted_count: number;
  exported_count: number;
  pending_export_count: number;
  open_feedback_count: number;
  resolved_feedback_count: number;
  closed_feedback_count: number;
  human_review_count: number;
  validation_blockers: Array<Record<string, unknown>>;
  pending_outbox_count: number;
  dead_letter_outbox_count: number;
  latest_export: {
    export_id: string;
    created_at: string;
    output_paths: string[];
    included: number;
    excluded: number;
  } | null;
  exports: Array<{
    export_id: string;
    created_at: string;
    output_paths: string[];
    included: number;
    excluded: number;
  }>;
  recommended_next_action: string;
  next_command: string | null;
  export_command: string;
}

export interface OutboxRecord {
  record_id: string;
  task_id: string;
  kind: string;
  payload: Record<string, unknown>;
  status: string;
  retry_count: number;
  created_at: string;
  next_retry_at: string | null;
  last_error: string | null;
}

export interface OutboxSummary {
  counts: {
    pending: number;
    sent: number;
    dead_letter: number;
  };
  records: OutboxRecord[];
}

export type ProviderName = "openai_responses" | "openai_compatible" | "local_cli";
export type ProviderFlavor = "deepseek" | "glm" | "minimax";
export type CliKind = "codex" | "claude";

export interface ProviderProfileConfig {
  name: string;
  provider: ProviderName;
  provider_flavor: ProviderFlavor | null;
  cli_kind: CliKind | null;
  cli_binary: string | null;
  model: string;
  api_key_env: string | null;
  // Write-only: when non-empty, the server stores it as inline api_key.
  // The GET response never includes the raw value (api_key_set echoes it).
  api_key?: string | null;
  // Read-only echo: true when the stored profile has an inline api_key.
  api_key_set?: boolean;
  base_url: string | null;
  reasoning_effort: string | null;
  permission_mode: string | null;
  timeout_seconds: number | null;
  max_retries: number | null;
  concurrency_limit: number | null;
  no_progress_timeout_seconds: number | null;
}

export interface ProviderCheck {
  id: string;
  status: "ok" | "warning" | "error";
  message: string;
}

export interface ProviderDiagnostic {
  status: "ok" | "warning" | "error";
  checks: ProviderCheck[];
}

export interface ProviderConfigSnapshot {
  config_valid: boolean;
  profiles: ProviderProfileConfig[];
  targets: Record<string, string>;
  limits: {
    local_cli_global_concurrency: number | null;
  };
  diagnostics: Record<string, ProviderDiagnostic>;
}

export type TaskDeviation = {
  task_id: string;
  row_index: number;
  span: string;
  current_type: string;
  prior_dominant_type: string;
  prior_distribution: Record<string, number>;
  prior_total: number;
};

export type DivergentEntry = {
  span: string;
  prior_total: number;
  prior_distribution: Record<string, number>;
  top_share: number;
  runner_up_share: number;
  type_entropy: number;
  resolved_convention_type?: string;
};

export type LowInfoEntry = {
  span: string;
  prior_total: number;
  prior_distribution: Record<string, number>;
  wordfreq: number;
};

export type PosteriorAudit = {
  task_deviations: TaskDeviation[];
  divergent_entries: DivergentEntry[];
  low_info_entries: LowInfoEntry[];
};

export type EntityConvention = {
  convention_id: string;
  project_id: string;
  // API field is `span` (originally `span_original` in the DB). May be
  // null in legacy rows where the original capitalization wasn't kept.
  span: string | null;
  entity_type: string | null;
  status: string;
  evidence_count: number;
  created_by: string;
  notes?: string | null;
  proposals?: { entity_type: string; evidence_count: number }[];
  created_at: string;
  updated_at: string;
};

export type EntityStatsItem = {
  span: string;
  distribution: Record<string, number>;
  total: number;
};


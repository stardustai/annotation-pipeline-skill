import type {
  AnnotationDocument,
  AnnotationDocumentVersion,
  ConfigSnapshot,
  DocumentDetail,
  DocumentsSnapshot,
  EventLog,
  KanbanSnapshot,
  ProjectSnapshot,
  ProviderConfigSnapshot,
  RuntimeMonitorReport,
  RuntimeRunOnceResponse,
  RuntimeSnapshot,
  StoresSnapshot,
  TaskDetail,
  ReadinessReport,
  OutboxSummary,
} from "./types";

function projectQuery(projectId: string | null): string {
  return projectId ? `?project=${encodeURIComponent(projectId)}` : "";
}

function storeParam(storeKey: string | null): string {
  return storeKey ? `store=${encodeURIComponent(storeKey)}` : "";
}

function withStore(base: string, storeKey: string | null): string {
  const sp = storeParam(storeKey);
  if (!sp) return base;
  return base.includes("?") ? `${base}&${sp}` : `${base}?${sp}`;
}

export async function fetchStores(): Promise<StoresSnapshot> {
  const response = await fetch("/api/stores");
  if (!response.ok) {
    throw new Error(`Stores API returned ${response.status}`);
  }
  return response.json() as Promise<StoresSnapshot>;
}

export async function fetchProjects(storeKey: string | null = null): Promise<ProjectSnapshot> {
  const response = await fetch(withStore("/api/projects", storeKey));
  if (!response.ok) {
    throw new Error(`Projects API returned ${response.status}`);
  }
  return response.json() as Promise<ProjectSnapshot>;
}

export interface DashboardStats {
  project_id: string | null;
  task_count: number;
  status_counts: Record<string, number>;
  open_feedback_count: number;
  outbox_pending_count: number;
  throughput_per_window: Record<string, number>;
  throughput_window_minutes: number;
}

export async function fetchDashboardStats(
  projectId: string | null = null,
  storeKey: string | null = null,
): Promise<DashboardStats> {
  const base = `/api/dashboard-stats${projectQuery(projectId)}`;
  const response = await fetch(withStore(base, storeKey));
  if (!response.ok) {
    throw new Error(`Dashboard stats API returned ${response.status}`);
  }
  return response.json() as Promise<DashboardStats>;
}

export async function fetchKanbanSnapshot(projectId: string | null = null, storeKey: string | null = null): Promise<KanbanSnapshot> {
  const base = `/api/kanban${projectQuery(projectId)}`;
  const response = await fetch(withStore(base, storeKey));
  if (!response.ok) {
    throw new Error(`Kanban API returned ${response.status}`);
  }
  return response.json() as Promise<KanbanSnapshot>;
}

export async function fetchTaskDetail(taskId: string, storeKey: string | null = null): Promise<TaskDetail> {
  const response = await fetch(withStore(`/api/tasks/${encodeURIComponent(taskId)}`, storeKey));
  if (!response.ok) {
    throw new Error(`Task detail API returned ${response.status}`);
  }
  return response.json() as Promise<TaskDetail>;
}

export async function postFeedbackDiscussion(
  taskId: string,
  payload: Record<string, unknown>,
  storeKey: string | null = null,
): Promise<TaskDetail> {
  const response = await fetch(withStore(`/api/tasks/${encodeURIComponent(taskId)}/feedback-discussions`, storeKey), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Feedback discussion API returned ${response.status}`);
  }
  await response.json();
  return fetchTaskDetail(taskId, storeKey);
}

export async function postTaskMove(
  taskId: string,
  targetStatus: string,
  reason: string,
  storeKey: string | null = null,
): Promise<{ ok: boolean; task: Record<string, unknown> }> {
  const response = await fetch(withStore(`/api/tasks/${encodeURIComponent(taskId)}/move`, storeKey), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ target_status: targetStatus, reason }),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Task move API returned ${response.status}`);
  }
  return response.json();
}

export async function clearConvention(
  projectId: string,
  span: string,
  storeKey: string | null = null,
): Promise<{ removed: boolean }> {
  const response = await fetch(withStore("/api/conventions/clear", storeKey), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ project_id: projectId, span }),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { error?: string } | null;
    throw new Error(err?.error ?? `Clear convention returned ${response.status}`);
  }
  return response.json() as Promise<{ removed: boolean }>;
}

export interface EntityConvention {
  convention_id: string;
  project_id: string;
  span: string;
  entity_type: string | null;
  status: "active" | "disputed";
  evidence_count: number;
  proposals: Array<Record<string, unknown>>;
  created_at: string;
  updated_at: string;
  created_by: string;
  notes: string | null;
}

export async function fetchConventions(
  projectId: string,
  storeKey: string | null = null,
): Promise<EntityConvention[]> {
  const url = withStore(`/api/conventions?project=${encodeURIComponent(projectId)}`, storeKey);
  const response = await fetch(url);
  if (!response.ok) return [];
  const data = (await response.json()) as { conventions?: EntityConvention[] };
  return data.conventions ?? [];
}

export async function declareConvention(
  payload: {
    project_id: string;
    span: string;
    entity_type: string;
    task_id?: string;
    notes?: string;
    actor?: string;
  },
  storeKey: string | null = null,
): Promise<EntityConvention> {
  const response = await fetch(withStore("/api/conventions", storeKey), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Convention API returned ${response.status}`);
  }
  return response.json();
}

export async function resolveConventionDispute(
  conventionId: string,
  entityType: string,
  storeKey: string | null = null,
  actor?: string,
  notes?: string,
): Promise<EntityConvention> {
  const response = await fetch(
    withStore(`/api/conventions/${encodeURIComponent(conventionId)}/resolve`, storeKey),
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ entity_type: entityType, actor, notes }),
    },
  );
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Resolve API returned ${response.status}`);
  }
  return response.json();
}

export async function postHumanReviewDecision(
  taskId: string,
  payload: Record<string, unknown>,
  storeKey: string | null = null,
): Promise<TaskDetail> {
  const response = await fetch(withStore(`/api/tasks/${encodeURIComponent(taskId)}/human-review`, storeKey), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Human Review API returned ${response.status}`);
  }
  await response.json();
  return fetchTaskDetail(taskId, storeKey);
}

export async function saveTaskQcPolicy(
  taskId: string,
  payload: Record<string, unknown>,
  storeKey: string | null = null,
): Promise<TaskDetail> {
  const response = await fetch(withStore(`/api/tasks/${encodeURIComponent(taskId)}/qc-policy`, storeKey), {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `QC policy API returned ${response.status}`);
  }
  return response.json() as Promise<TaskDetail>;
}

export async function fetchConfigSnapshot(storeKey: string | null = null): Promise<ConfigSnapshot> {
  const response = await fetch(withStore("/api/config", storeKey));
  if (!response.ok) {
    throw new Error(`Config API returned ${response.status}`);
  }
  return response.json() as Promise<ConfigSnapshot>;
}

export async function fetchConfigFile(
  id: string,
  storeKey: string | null = null,
): Promise<{ id: string; path: string; content: string; exists: boolean }> {
  const snap = await fetchConfigSnapshot(storeKey);
  const file = snap.files.find((f) => f.id === id);
  if (!file) {
    throw new Error(`Config file '${id}' not in snapshot`);
  }
  return { id: file.id, path: file.path, content: file.content, exists: file.exists };
}

export async function saveConfigFile(id: string, content: string, storeKey: string | null = null): Promise<void> {
  const response = await fetch(withStore(`/api/config/${encodeURIComponent(id)}`, storeKey), {
    method: "PUT",
    headers: { "content-type": "application/yaml; charset=utf-8" },
    body: content,
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(payload?.detail ?? payload?.error ?? `Config save returned ${response.status}`);
  }
}

export async function fetchEventLog(
  projectId: string | null = null,
  storeKey: string | null = null,
  options: { limit?: number; offset?: number } = {},
): Promise<EventLog> {
  const limit = options.limit ?? 100;
  const offset = options.offset ?? 0;
  let base = `/api/events?limit=${limit}&offset=${offset}`;
  if (projectId) base += `&project=${encodeURIComponent(projectId)}`;
  const response = await fetch(withStore(base, storeKey));
  if (!response.ok) {
    throw new Error(`Event log API returned ${response.status}`);
  }
  return response.json() as Promise<EventLog>;
}

export interface AlertEntry {
  ts: string;
  kind?: string;
  target?: string;
  api_error_status?: number | null;
  exception_class?: string | null;
  message?: string;
  task_id?: string;
  dropped?: Record<string, number>;
  [key: string]: unknown;
}

export interface AlertsResponse {
  alerts: AlertEntry[];
  total_lines: number;
  alerts_path: string;
}

export async function fetchAlerts(
  storeKey: string | null = null,
  options: { limit?: number } = {},
): Promise<AlertsResponse> {
  const limit = options.limit ?? 100;
  const response = await fetch(withStore(`/api/alerts?limit=${limit}`, storeKey));
  if (!response.ok) {
    throw new Error(`Alerts API returned ${response.status}`);
  }
  return response.json() as Promise<AlertsResponse>;
}

export async function fetchRuntimeSnapshot(storeKey: string | null = null): Promise<RuntimeSnapshot> {
  const response = await fetch(withStore("/api/runtime", storeKey));
  if (!response.ok) {
    throw new Error(`Runtime API returned ${response.status}`);
  }
  return response.json() as Promise<RuntimeSnapshot>;
}

export async function fetchRuntimeMonitor(storeKey: string | null = null): Promise<RuntimeMonitorReport> {
  const response = await fetch(withStore("/api/runtime/monitor", storeKey));
  if (!response.ok) {
    throw new Error(`Runtime monitor API returned ${response.status}`);
  }
  return response.json() as Promise<RuntimeMonitorReport>;
}

export async function fetchReadinessReport(projectId: string, storeKey: string | null = null): Promise<ReadinessReport> {
  const base = `/api/readiness?project=${encodeURIComponent(projectId)}`;
  const response = await fetch(withStore(base, storeKey));
  if (!response.ok) {
    throw new Error(`Readiness API returned ${response.status}`);
  }
  return response.json() as Promise<ReadinessReport>;
}

export async function fetchOutboxSummary(projectId: string | null = null, storeKey: string | null = null): Promise<OutboxSummary> {
  const base = `/api/outbox${projectQuery(projectId)}`;
  const response = await fetch(withStore(base, storeKey));
  if (!response.ok) {
    throw new Error(`Outbox API returned ${response.status}`);
  }
  return response.json() as Promise<OutboxSummary>;
}

export async function runRuntimeOnce(storeKey: string | null = null): Promise<RuntimeRunOnceResponse> {
  const response = await fetch(withStore("/api/runtime/run-once", storeKey), { method: "POST", body: "{}" });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { error?: string } | null;
    throw new Error(payload?.error ?? `Runtime run-once API returned ${response.status}`);
  }
  return response.json() as Promise<RuntimeRunOnceResponse>;
}

export async function fetchProviderConfig(storeKey: string | null = null): Promise<ProviderConfigSnapshot> {
  const response = await fetch(withStore("/api/providers", storeKey));
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(payload?.detail ?? payload?.error ?? `Provider API returned ${response.status}`);
  }
  return response.json() as Promise<ProviderConfigSnapshot>;
}

export async function saveProviderConfig(payload: {
  profiles: ProviderConfigSnapshot["profiles"];
  targets: ProviderConfigSnapshot["targets"];
  limits: ProviderConfigSnapshot["limits"];
}, storeKey: string | null = null): Promise<ProviderConfigSnapshot> {
  const response = await fetch(withStore("/api/providers", storeKey), {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Provider save returned ${response.status}`);
  }
  return response.json() as Promise<ProviderConfigSnapshot>;
}

export async function fetchDocuments(storeKey: string | null = null): Promise<DocumentsSnapshot> {
  const response = await fetch(withStore("/api/documents", storeKey));
  if (!response.ok) {
    throw new Error(`Documents API returned ${response.status}`);
  }
  return response.json() as Promise<DocumentsSnapshot>;
}

export async function fetchDocumentDetail(docId: string, storeKey: string | null = null): Promise<DocumentDetail> {
  const response = await fetch(withStore(`/api/documents/${encodeURIComponent(docId)}`, storeKey));
  if (!response.ok) {
    throw new Error(`Document detail API returned ${response.status}`);
  }
  return response.json() as Promise<DocumentDetail>;
}

export async function createDocument(
  payload: { title: string; description: string; created_by: string },
  storeKey: string | null = null,
): Promise<AnnotationDocument> {
  const response = await fetch(withStore("/api/documents", storeKey), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Create document returned ${response.status}`);
  }
  return response.json() as Promise<AnnotationDocument>;
}

export async function createDocumentVersion(
  docId: string,
  payload: { version: string; content: string; changelog: string; created_by: string },
  storeKey: string | null = null,
): Promise<AnnotationDocumentVersion> {
  const response = await fetch(withStore(`/api/documents/${encodeURIComponent(docId)}/versions`, storeKey), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Create document version returned ${response.status}`);
  }
  return response.json() as Promise<AnnotationDocumentVersion>;
}

export interface AnnotationRulesDocumentSnapshot {
  document: AnnotationDocument;
  versions: AnnotationDocumentVersion[];
  latest_version_id: string | null;
}

export async function fetchAnnotationRulesDocument(
  storeKey: string | null = null,
): Promise<AnnotationRulesDocumentSnapshot> {
  const response = await fetch(withStore("/api/annotation-rules-document", storeKey));
  if (!response.ok) {
    throw new Error(`Annotation rules document API returned ${response.status}`);
  }
  return response.json() as Promise<AnnotationRulesDocumentSnapshot>;
}

export async function createAnnotationRulesDocumentVersion(
  payload: { version?: string; content: string; changelog: string; created_by?: string },
  storeKey: string | null = null,
): Promise<AnnotationDocumentVersion> {
  const response = await fetch(withStore("/api/annotation-rules-document/versions", storeKey), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Create annotation rules version returned ${response.status}`);
  }
  return response.json() as Promise<AnnotationDocumentVersion>;
}

export async function fetchProjectSchema(storeKey: string | null = null): Promise<{ schema: Record<string, unknown> | null }> {
  const response = await fetch(withStore("/api/schema", storeKey));
  if (!response.ok) throw new Error(`Schema fetch returned ${response.status}`);
  return response.json() as Promise<{ schema: Record<string, unknown> | null }>;
}

export async function saveProjectSchema(
  schema: Record<string, unknown>,
  storeKey: string | null = null,
): Promise<{ schema: Record<string, unknown>; path: string }> {
  const response = await fetch(withStore("/api/schema", storeKey), {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ schema }),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Schema save returned ${response.status}`);
  }
  return response.json() as Promise<{ schema: Record<string, unknown>; path: string }>;
}

export interface Guideline {
  label: string;
  path: string;
  filename: string;
  exists: boolean;
  content: string | null;
}

export async function fetchGuidelines(storeKey: string | null = null): Promise<{ guidelines: Guideline[] }> {
  const response = await fetch(withStore("/api/guidelines", storeKey));
  if (!response.ok) throw new Error(`Guidelines fetch returned ${response.status}`);
  return response.json() as Promise<{ guidelines: Guideline[] }>;
}

export interface AnnotatorConfig {
  id: string;
  display_name: string;
  provider_target: string;
  llm_profile: string;
  enabled: boolean;
  modalities: string[];
  annotation_types: string[];
  input_artifact_kinds: string[];
  output_artifact_kinds: string[];
  preview_renderer_id: string | null;
}

export interface AnnotatorsSnapshot {
  annotators: AnnotatorConfig[];
  sampling: Record<string, Record<string, unknown>>;
  available_profiles: string[];
  stage_targets: Record<string, string>;
}

export async function fetchAnnotatorsConfig(storeKey: string | null = null): Promise<AnnotatorsSnapshot> {
  const response = await fetch(withStore("/api/annotators", storeKey));
  if (!response.ok) throw new Error(`Annotators fetch returned ${response.status}`);
  return response.json() as Promise<AnnotatorsSnapshot>;
}

export async function saveAnnotatorsConfig(
  payload: {
    annotators?: AnnotatorConfig[];
    sampling: Record<string, Record<string, unknown>>;
    stage_targets?: Record<string, string>;
  },
  storeKey: string | null = null,
): Promise<void> {
  const response = await fetch(withStore("/api/annotators", storeKey), {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string; error?: string } | null;
    throw new Error(errorPayload?.detail ?? errorPayload?.error ?? `Annotators save returned ${response.status}`);
  }
}

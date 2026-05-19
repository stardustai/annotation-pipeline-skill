import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  fetchKanbanSnapshot,
  fetchProjects,
  fetchStores,
  fetchTaskDetail,
  postHumanReviewDecision,
  postTaskMove,
} from "./api";
import { ConfigPanel } from "./components/ConfigPanel";
import { DashboardStatsBar } from "./components/DashboardStatsBar";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { SchemaPanel } from "./components/SchemaPanel";
import { EventLogPanel } from "./components/EventLogPanel";
import { EntityKnowledgePanel } from "./components/EntityKnowledgePanel";
import { TypeStatisticsPanel } from "./components/TypeStatisticsPanel";
import { KanbanBoard } from "./components/KanbanBoard";
import { OutputPanel } from "./components/OutputPanel";
// Lazy-loaded: the Distribution panel pulls plotly.js-dist-min (~3 MB) into the
// bundle. Keep it off the critical path so the rest of the dashboard loads fast;
// the plotly chunk only ships when the operator clicks the Distribution tab.
const DistributionPanel = lazy(() =>
  import("./components/DistributionPanel").then((m) => ({ default: m.DistributionPanel })),
);
import { PosteriorAuditPanel } from "./components/PosteriorAuditPanel";
import { ProvidersPanel } from "./components/ProvidersPanel";
import { RuntimePanel } from "./components/RuntimePanel";
import { TaskDrawer } from "./components/TaskDrawer";
import { countCards } from "./kanban";
import type { KanbanSnapshot, ProjectSummary, StoreInfo, TaskCard, TaskDetail } from "./types";
import { useUrlState, type UrlState } from "./url_state";

const emptySnapshot: KanbanSnapshot = { project_id: null, columns: [] };
type ViewMode = "kanban" | "runtime" | "output" | "providers" | "config" | "events" | "documents" | "schema" | "posterior-audit" | "entity-knowledge" | "distribution" | "statistics";

const urlDefaults: UrlState = { view: "kanban", store: null, project: null, task: null };

function findTaskInSnapshot(snapshot: KanbanSnapshot, taskId: string | null): TaskCard | null {
  if (!taskId) return null;
  for (const column of snapshot.columns) {
    for (const card of column.cards) {
      if (card.task_id === taskId) return card;
    }
  }
  return null;
}

export default function App() {
  const [snapshot, setSnapshot] = useState<KanbanSnapshot>(emptySnapshot);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [stores, setStores] = useState<StoreInfo[]>([]);
  const [workspacePath, setWorkspacePath] = useState<string | null>(null);
  const [urlState, { setView, setStore, setProject, setTask }] = useUrlState(urlDefaults);
  const selectedStoreKey = urlState.store;
  const selectedProjectId = urlState.project;
  const selectedTaskId = urlState.task;
  const viewMode = urlState.view as ViewMode;

  const [selectedDetail, setSelectedDetail] = useState<TaskDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailSaving, setDetailSaving] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Derive the full TaskCard from the kanban snapshot. URL only holds task_id.
  const selectedTask = useMemo<TaskCard | null>(
    () => findTaskInSnapshot(snapshot, selectedTaskId),
    [snapshot, selectedTaskId],
  );

  useEffect(() => {
    fetchStores()
      .then((snap) => {
        setStores(snap.stores);
        setWorkspacePath(snap.workspace_path ?? null);
        if (snap.stores.length > 0) {
          const valid = snap.stores.some((s) => s.key === selectedStoreKey);
          if (!valid) {
            setStore(snap.stores[0].key);
          }
        }
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let active = true;

    async function refresh(showLoading: boolean) {
      if (showLoading) setLoading(true);
      try {
        const [projectSnapshot, nextSnapshot] = await Promise.all([
          fetchProjects(selectedStoreKey),
          fetchKanbanSnapshot(selectedProjectId, selectedStoreKey),
        ]);
        if (!active) return;
        setProjects(projectSnapshot.projects);
        setSnapshot(nextSnapshot);
        setError(null);
        if (!selectedProjectId && projectSnapshot.projects.length === 1) {
          setProject(projectSnapshot.projects[0].project_id);
        }
      } catch (reason: unknown) {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "Unable to load dashboard data");
      } finally {
        if (active && showLoading) setLoading(false);
      }
    }

    refresh(true);
    const timer = setInterval(() => refresh(false), 5000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProjectId, selectedStoreKey]);

  useEffect(() => {
    if (!selectedTaskId) {
      setSelectedDetail(null);
      setDetailError(null);
      setDetailLoading(false);
      return;
    }

    let active = true;
    setDetailLoading(true);
    setDetailError(null);
    fetchTaskDetail(selectedTaskId, selectedStoreKey)
      .then((detail) => {
        if (!active) return;
        setSelectedDetail(detail);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setSelectedDetail(null);
        setDetailError(reason instanceof Error ? reason.message : "Unable to load task detail");
      })
      .finally(() => {
        if (active) setDetailLoading(false);
      });

    return () => {
      active = false;
    };
  }, [selectedTaskId, selectedStoreKey]);

  function handleStoreChange(key: string) {
    setStore(key || null);
    setTask(null);
  }

  function handleProjectChange(value: string) {
    setProject(value || null);
    setTask(null);
  }

  async function submitHumanReviewDecision(payload: Record<string, unknown>) {
    if (!selectedTaskId) return;
    setDetailSaving(true);
    setDetailError(null);
    try {
      const detail = await postHumanReviewDecision(selectedTaskId, payload, selectedStoreKey);
      setSelectedDetail(detail);
      setSnapshot(await fetchKanbanSnapshot(selectedProjectId, selectedStoreKey));
    } catch (reason: unknown) {
      setDetailError(reason instanceof Error ? reason.message : "Unable to save Human Review decision");
    } finally {
      setDetailSaving(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Annotation Pipeline</h1>
          <p>{countCards(snapshot)} tasks across operational stages</p>
          {workspacePath ? (
            <p className="workspace-path" title="serve --workspace argument">
              Workspace: <code>{workspacePath}</code>
            </p>
          ) : null}
        </div>
        <div className="topbar-actions">
          {stores.length > 0 ? (
            <label className="project-selector">
              <span>Project</span>
              <select
                value={selectedStoreKey ?? ""}
                onChange={(event) => handleStoreChange(event.target.value)}
              >
                {stores.map((s) => (
                  <option key={s.key} value={s.key}>
                    {s.name} ({s.task_count} tasks)
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <label className="project-selector">
            <span>Pipeline</span>
            <select
              value={selectedProjectId ?? ""}
              onChange={(event) => handleProjectChange(event.target.value)}
            >
              <option value="">All pipelines</option>
              {projects.map((project) => (
                <option key={project.project_id} value={project.project_id}>
                  {project.project_id} ({project.task_count} tasks)
                </option>
              ))}
            </select>
          </label>
          {loading || error ? (
            <div className={`status-pill ${error ? "error" : ""}`}>
              {loading ? "Loading" : "API error"}
            </div>
          ) : null}
        </div>
      </header>

      <DashboardStatsBar projectId={selectedProjectId} storeKey={selectedStoreKey} />

      <nav className="view-tabs" aria-label="Dashboard views">
        <button className={viewMode === "kanban" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("kanban")}>
          Kanban
        </button>
        <button className={viewMode === "runtime" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("runtime")}>
          Runtime
        </button>
        <button className={viewMode === "output" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("output")}>
          Export
        </button>
        <button className={viewMode === "providers" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("providers")}>
          Providers
        </button>
        <button className={viewMode === "config" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("config")}>
          Configuration
        </button>
        <button className={viewMode === "events" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("events")}>
          Event Log
        </button>
        <button className={viewMode === "documents" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("documents")}>
          Documents
        </button>
        <button className={viewMode === "schema" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("schema")}>
          Schema
        </button>
        <button className={viewMode === "posterior-audit" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("posterior-audit")}>
          Posterior Audit
        </button>
        <button className={viewMode === "entity-knowledge" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("entity-knowledge")}>
          Entity Knowledge
        </button>
        <button className={viewMode === "distribution" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("distribution")}>
          Distribution
        </button>
        <button className={viewMode === "statistics" ? "view-tab selected" : "view-tab"} type="button" onClick={() => setView("statistics")}>
          Statistics
        </button>
      </nav>

      {error ? <div className="notice">{error}</div> : null}
      {viewMode === "kanban" ? (
        <KanbanBoard
          snapshot={snapshot}
          selectedTaskId={selectedTaskId}
          onSelectTask={(card) => setTask(card.task_id)}
          onMoveTask={async (card, targetStatus, reason) => {
            await postTaskMove(card.task_id, targetStatus, reason, selectedStoreKey);
            setSnapshot(await fetchKanbanSnapshot(selectedProjectId, selectedStoreKey));
          }}
        />
      ) : null}
      {viewMode === "runtime" ? <RuntimePanel storeKey={selectedStoreKey} /> : null}
      {viewMode === "output" ? <OutputPanel projectId={selectedProjectId} storeKey={selectedStoreKey} storePath={stores.find((s) => s.key === selectedStoreKey)?.path ?? null} /> : null}
      {viewMode === "providers" ? <ProvidersPanel /> : null}
      {viewMode === "config" ? <ConfigPanel storeKey={selectedStoreKey} /> : null}
      {viewMode === "events" ? <EventLogPanel projectId={selectedProjectId} storeKey={selectedStoreKey} /> : null}
      {viewMode === "documents" ? <DocumentsPanel storeKey={selectedStoreKey} /> : null}
      {viewMode === "schema" ? <SchemaPanel storeKey={selectedStoreKey} /> : null}
      {viewMode === "posterior-audit" ? (
        <PosteriorAuditPanel
          projectId={selectedProjectId}
          storeKey={selectedStoreKey}
          onSendToHr={async (taskId) => {
            await postTaskMove(taskId, "human_review", "posterior_audit", selectedStoreKey);
            setSnapshot(await fetchKanbanSnapshot(selectedProjectId, selectedStoreKey));
          }}
          onDeclareCanonical={async (span, entityType) => {
            if (!selectedProjectId) return;
            const storeQ = selectedStoreKey ? `?store=${encodeURIComponent(selectedStoreKey)}` : "";
            await fetch(`/api/conventions${storeQ}`, {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({
                project_id: selectedProjectId,
                span,
                entity_type: entityType,
                actor: "operator_declaration",
              }),
            });
          }}
        />
      ) : null}
      {viewMode === "entity-knowledge" ? (
        <EntityKnowledgePanel projectId={selectedProjectId} storeKey={selectedStoreKey} />
      ) : null}
      {viewMode === "distribution" ? (
        <Suspense fallback={<div className="runtime-muted" style={{ padding: "2rem" }}>Loading Distribution panel…</div>}>
          <DistributionPanel
            projectId={selectedProjectId}
            storeKey={selectedStoreKey}
            onSelectTask={(tid) => setTask(tid)}
          />
        </Suspense>
      ) : null}
      {viewMode === "statistics" ? (
        <TypeStatisticsPanel projectId={selectedProjectId} storeKey={selectedStoreKey} />
      ) : null}
      <TaskDrawer
        task={selectedTask}
        detail={selectedDetail}
        loading={detailLoading}
        saving={detailSaving}
        error={detailError}
        onSubmitHumanReviewDecision={submitHumanReviewDecision}
        onClose={() => setTask(null)}
      />
    </main>
  );
}

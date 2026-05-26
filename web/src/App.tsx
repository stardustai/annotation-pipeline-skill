import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  fetchKanbanSnapshot,
  fetchProjects,
  fetchRuntimeMonitor,
  fetchRuntimeSnapshot,
  fetchStores,
  fetchTaskDetail,
  postHumanReviewDecision,
  postTaskMove,
  startRuntime,
  stopRuntime,
} from "./api";
import { ConfigPanel } from "./components/ConfigPanel";
import { DashboardStatsBar } from "./components/DashboardStatsBar";
import { AnnotationRulesPanel } from "./components/AnnotationRulesPanel";
import { SchemaPanel } from "./components/SchemaPanel";
import { EntityKnowledgePanel } from "./components/EntityKnowledgePanel";
import { KanbanBoard } from "./components/KanbanBoard";
import { OutputPanel } from "./components/OutputPanel";
// Lazy-loaded: the Statistics panel pulls plotly.js-dist-min (~3 MB) into the
// bundle (via the Scatter plot sub-tab). Keep it off the critical path so the
// rest of the dashboard loads fast; plotly only ships when the operator opens
// the Statistics tab.
const DistributionPanel = lazy(() =>
  import("./components/DistributionPanel").then((m) => ({ default: m.DistributionPanel })),
);
import { PosteriorAuditPanel } from "./components/PosteriorAuditPanel";
import { ProvidersPanel } from "./components/ProvidersPanel";
import { AlertsMarquee } from "./components/AlertsMarquee";
import { RuntimePanel, type RuntimeSubtab } from "./components/RuntimePanel";
import { TaskDrawer } from "./components/TaskDrawer";
import { countCards } from "./kanban";
import type { KanbanSnapshot, ProjectSummary, StoreInfo, TaskCard, TaskDetail } from "./types";
import { useUrlState, type UrlState } from "./url_state";

const emptySnapshot: KanbanSnapshot = { project_id: null, columns: [] };
// Top-level dashboard tabs. "statistics" hosts Duplicates+Scatter+Statistics
// sub-tabs; "annotation-rules" hosts Guidelines+Schema sub-tabs.
type ViewMode =
  | "kanban"
  | "statistics"
  | "annotation-rules"
  | "runtime"
  | "providers"
  | "config"
  | "events"
  | "entity-knowledge"
  | "posterior-audit"
  | "output";

// Backward-compatible aliases for old URLs that still link to the
// since-merged standalone tabs. `alerts` and `events` used to be top-level
// tabs; they now live as sub-tabs inside Runtime.
function canonicalizeViewMode(raw: string): ViewMode {
  if (raw === "distribution") return "statistics";
  if (raw === "schema") return "annotation-rules";
  if (raw === "alerts") return "runtime";
  if (raw === "events") return "runtime";
  return raw as ViewMode;
}

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
  const viewMode = canonicalizeViewMode(urlState.view);
  // Sub-tab state for Annotation Rules (Guidelines vs Schema). Kept in App
  // so the top-tab restructure doesn't force AnnotationRulesPanel to grow
  // a Schema-aware mode.
  const [annotationRulesSubtab, setAnnotationRulesSubtab] = useState<"guidelines" | "schema">("guidelines");
  // Aliased URL ?view=schema → land directly on the Schema sub-tab.
  useEffect(() => {
    if (urlState.view === "schema") setAnnotationRulesSubtab("schema");
  }, [urlState.view]);

  // Sub-tab state for Runtime (Overview vs Alerts). Aliased URL ?view=alerts
  // lands on the Alerts sub-tab; the AlertsMarquee onClick also jumps here.
  const [runtimeSubtab, setRuntimeSubtab] = useState<RuntimeSubtab>("overview");
  useEffect(() => {
    if (urlState.view === "alerts") setRuntimeSubtab("alerts");
    else if (urlState.view === "events") setRuntimeSubtab("events");
  }, [urlState.view]);

  // ── Runtime issue badge ───────────────────────────────────────────────
  // Poll the runtime monitor every 30s so the Runtime tab can show a red
  // dot + failure count even when the operator is on a different tab.
  const [runtimeIssueCount, setRuntimeIssueCount] = useState(0);
  const [runtimeHealthy, setRuntimeHealthy] = useState<boolean | null>(null);
  const [runtimeStarting, setRuntimeStarting] = useState(false);
  const [runtimeStopping, setRuntimeStopping] = useState(false);
  useEffect(() => {
    let active = true;
    let timer: number | null = null;
    setRuntimeHealthy(null);
    async function poll() {
      try {
        const [report, snap] = await Promise.all([
          fetchRuntimeMonitor(selectedStoreKey),
          fetchRuntimeSnapshot(selectedStoreKey),
        ]);
        if (!active) return;
        setRuntimeIssueCount(report.ok ? 0 : report.failures.length);
        setRuntimeHealthy(snap.runtime_status.healthy);
      } catch {
        if (active) setRuntimeIssueCount(0);
      } finally {
        if (active) timer = window.setTimeout(poll, 30000);
      }
    }
    poll();
    return () => {
      active = false;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [selectedStoreKey]);

  async function handleRuntimeStart() {
    setRuntimeStarting(true);
    try {
      await startRuntime(selectedStoreKey);
      await new Promise((r) => setTimeout(r, 1500));
      const snap = await fetchRuntimeSnapshot(selectedStoreKey);
      setRuntimeHealthy(snap.runtime_status.healthy);
    } catch { /* ignore */ } finally {
      setRuntimeStarting(false);
    }
  }

  async function handleRuntimeStop() {
    setRuntimeStopping(true);
    try {
      await stopRuntime(selectedStoreKey);
      await new Promise((r) => setTimeout(r, 1500));
      const snap = await fetchRuntimeSnapshot(selectedStoreKey);
      setRuntimeHealthy(snap.runtime_status.healthy);
    } catch { /* ignore */ } finally {
      setRuntimeStopping(false);
    }
  }

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
          <div className="topbar-title-row">
            <h1>Annotation Pipeline</h1>
            {runtimeHealthy === false ? (
              <button className="primary-button topbar-runtime-btn" type="button" disabled={runtimeStarting} onClick={handleRuntimeStart}>
                {runtimeStarting ? "Starting…" : "▶ Start"}
              </button>
            ) : runtimeHealthy === true ? (
              <button className="view-tab danger topbar-runtime-btn" type="button" disabled={runtimeStopping} onClick={handleRuntimeStop}>
                {runtimeStopping ? "Stopping…" : "⏹ Stop"}
              </button>
            ) : null}
          </div>
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

      <AlertsMarquee
        storeKey={selectedStoreKey}
        projectId={selectedProjectId}
        onClick={() => {
          setRuntimeSubtab("alerts");
          setView("runtime");
        }}
      />
      <DashboardStatsBar projectId={selectedProjectId} storeKey={selectedStoreKey} runtimeHealthy={runtimeHealthy} />

      <nav className="view-tabs" aria-label="Dashboard views" role="tablist">
        <button
          className={viewMode === "kanban" ? "view-tab kanban-tab selected" : "view-tab kanban-tab"}
          role="tab"
          aria-selected={viewMode === "kanban"}
          type="button"
          onClick={() => setView("kanban")}
        >
          Kanban
        </button>
        <button
          className={viewMode === "statistics" ? "view-tab selected" : "view-tab"}
          role="tab"
          aria-selected={viewMode === "statistics"}
          type="button"
          onClick={() => setView("statistics")}
        >
          Statistics
        </button>
        <button
          className={viewMode === "annotation-rules" ? "view-tab selected" : "view-tab"}
          role="tab"
          aria-selected={viewMode === "annotation-rules"}
          type="button"
          onClick={() => setView("annotation-rules")}
        >
          Annotation Rules
        </button>
        <button
          className={viewMode === "runtime" ? "view-tab selected" : "view-tab"}
          role="tab"
          aria-selected={viewMode === "runtime"}
          type="button"
          onClick={() => setView("runtime")}
        >
          Runtime
          {runtimeIssueCount > 0 ? (
            <span className="view-tab-badge" aria-label={`${runtimeIssueCount} runtime issues`}>
              {runtimeIssueCount}
            </span>
          ) : null}
        </button>
        <button
          className={viewMode === "providers" ? "view-tab selected" : "view-tab"}
          role="tab"
          aria-selected={viewMode === "providers"}
          type="button"
          onClick={() => setView("providers")}
        >
          Providers
        </button>
        <button
          className={viewMode === "config" ? "view-tab selected" : "view-tab"}
          role="tab"
          aria-selected={viewMode === "config"}
          type="button"
          onClick={() => setView("config")}
        >
          Configuration
        </button>
        <button
          className={viewMode === "entity-knowledge" ? "view-tab selected" : "view-tab"}
          role="tab"
          aria-selected={viewMode === "entity-knowledge"}
          type="button"
          onClick={() => setView("entity-knowledge")}
        >
          Entity Knowledge
        </button>
        <button
          className={viewMode === "posterior-audit" ? "view-tab selected" : "view-tab"}
          role="tab"
          aria-selected={viewMode === "posterior-audit"}
          type="button"
          onClick={() => setView("posterior-audit")}
        >
          Posterior Audit
        </button>
        <button
          className={viewMode === "output" ? "view-tab selected" : "view-tab"}
          role="tab"
          aria-selected={viewMode === "output"}
          type="button"
          onClick={() => setView("output")}
        >
          Export
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
      {viewMode === "runtime" ? (
        <RuntimePanel
          storeKey={selectedStoreKey}
          projectId={selectedProjectId}
          subtab={runtimeSubtab}
          onSubtabChange={setRuntimeSubtab}
        />
      ) : null}
      {viewMode === "output" ? <OutputPanel projectId={selectedProjectId} storeKey={selectedStoreKey} storePath={stores.find((s) => s.key === selectedStoreKey)?.path ?? null} /> : null}
      {viewMode === "providers" ? <ProvidersPanel /> : null}
      {viewMode === "config" ? <ConfigPanel storeKey={selectedStoreKey} /> : null}
      {viewMode === "annotation-rules" ? (
        <section className="runtime-panel" aria-label="Annotation rules and schema">
          <nav className="sub-tabs" aria-label="Annotation rules sections" role="tablist">
            <button
              className={annotationRulesSubtab === "guidelines" ? "sub-tab selected" : "sub-tab"}
              role="tab"
              aria-selected={annotationRulesSubtab === "guidelines"}
              type="button"
              onClick={() => setAnnotationRulesSubtab("guidelines")}
            >
              Guidelines
            </button>
            <button
              className={annotationRulesSubtab === "schema" ? "sub-tab selected" : "sub-tab"}
              role="tab"
              aria-selected={annotationRulesSubtab === "schema"}
              type="button"
              onClick={() => setAnnotationRulesSubtab("schema")}
            >
              Schema
            </button>
          </nav>
          {annotationRulesSubtab === "guidelines" ? <AnnotationRulesPanel storeKey={selectedStoreKey} /> : null}
          {annotationRulesSubtab === "schema" ? <SchemaPanel storeKey={selectedStoreKey} /> : null}
        </section>
      ) : null}
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
      {viewMode === "statistics" ? (
        <Suspense fallback={<div className="runtime-muted" style={{ padding: "2rem" }}>Loading Statistics panel…</div>}>
          <DistributionPanel
            projectId={selectedProjectId}
            storeKey={selectedStoreKey}
            onSelectTask={(tid) => setTask(tid)}
          />
        </Suspense>
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

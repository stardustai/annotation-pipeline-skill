import { useEffect, useState } from "react";
import {
  fetchAlerts,
  fetchRuntimeMonitor,
  fetchRuntimeSnapshot,
  runRuntimeOnce,
  type AlertEntry,
} from "../api";
import { formatRuntimeDate, orderedQueueCounts } from "../runtime";
import type { ActiveRun, RuntimeMonitorReport, RuntimeSnapshot } from "../types";
import { AlertsPanel } from "./AlertsPanel";

export type RuntimeSubtab = "overview" | "alerts";

interface RuntimePanelProps {
  storeKey: string | null;
  subtab: RuntimeSubtab;
  onSubtabChange: (subtab: RuntimeSubtab) => void;
}

const REFRESH_MS = 5000;
const RECENT_ALERTS_LIMIT = 20;

export function RuntimePanel({ storeKey, subtab, onSubtabChange }: RuntimePanelProps) {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [monitor, setMonitor] = useState<RuntimeMonitorReport | null>(null);
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    const [s, m, a] = await Promise.all([
      fetchRuntimeSnapshot(storeKey),
      fetchRuntimeMonitor(storeKey),
      fetchAlerts(storeKey, { limit: RECENT_ALERTS_LIMIT }).catch(
        () => ({ alerts: [] as AlertEntry[], total_lines: 0, alerts_path: "" }),
      ),
    ]);
    setSnapshot(s);
    setMonitor(m);
    setAlerts(a.alerts ?? []);
  }

  useEffect(() => {
    let active = true;
    setLoading(true);
    load()
      .then(() => { if (active) setError(null); })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "Unable to load runtime data");
      })
      .finally(() => { if (active) setLoading(false); });
    const id = window.setInterval(() => {
      if (!active) return;
      load().catch(() => {});
    }, REFRESH_MS);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [storeKey]);

  async function runOnce() {
    setRunning(true);
    setError(null);
    try {
      const result = await runRuntimeOnce(storeKey);
      setSnapshot(result.snapshot);
      const next = await fetchRuntimeMonitor(storeKey);
      setMonitor(next);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Unable to drain runtime queue");
    } finally {
      setRunning(false);
    }
  }

  if (loading && !snapshot) return <section className="runtime-panel">Loading runtime…</section>;
  if (!snapshot) return <section className="runtime-panel notice compact">{error ?? "Runtime unavailable"}</section>;

  const failureCount = monitor && !monitor.ok ? monitor.failures.length : 0;
  const alertCount = alerts.length;

  return (
    <section className="runtime-panel">
      <nav className="sub-tabs" aria-label="Runtime sections" role="tablist">
        <button
          className={subtab === "overview" ? "sub-tab selected" : "sub-tab"}
          role="tab"
          type="button"
          aria-selected={subtab === "overview"}
          onClick={() => onSubtabChange("overview")}
        >
          Overview
          {failureCount > 0 ? <span className="sub-tab-badge sub-tab-badge--bad">{failureCount}</span> : null}
        </button>
        <button
          className={subtab === "alerts" ? "sub-tab selected" : "sub-tab"}
          role="tab"
          type="button"
          aria-selected={subtab === "alerts"}
          onClick={() => onSubtabChange("alerts")}
        >
          Alerts
          {alertCount > 0 ? <span className="sub-tab-badge sub-tab-badge--warn">{alertCount}</span> : null}
        </button>
      </nav>

      {subtab === "alerts" ? (
        <AlertsPanel storeKey={storeKey} />
      ) : (
        <RuntimeOverview
          snapshot={snapshot}
          monitor={monitor}
          recentAlerts={alerts}
          totalAlerts={alertCount}
          running={running}
          error={error}
          onRunOnce={runOnce}
          onViewAllAlerts={() => onSubtabChange("alerts")}
        />
      )}
    </section>
  );
}

interface OverviewProps {
  snapshot: RuntimeSnapshot;
  monitor: RuntimeMonitorReport | null;
  recentAlerts: AlertEntry[];
  totalAlerts: number;
  running: boolean;
  error: string | null;
  onRunOnce: () => void;
  onViewAllAlerts: () => void;
}

function RuntimeOverview({
  snapshot,
  monitor,
  recentAlerts,
  totalAlerts,
  running,
  error,
  onRunOnce,
  onViewAllAlerts,
}: OverviewProps) {
  const queueItems = orderedQueueCounts(snapshot);
  const queueTotal = queueItems.reduce((acc, item) => acc + item.value, 0);
  const terminal = (snapshot.queue_counts.accepted ?? 0) + (snapshot.queue_counts.rejected ?? 0);
  const terminalPct = queueTotal > 0 ? Math.round((terminal / queueTotal) * 100) : 0;
  const heartbeatAge = snapshot.runtime_status.heartbeat_age_seconds;
  const healthy = snapshot.runtime_status.healthy;
  const cap = snapshot.capacity;
  const failureCount = monitor && !monitor.ok ? monitor.failures.length : 0;
  const alertCount = totalAlerts;

  return (
    <>
      {error ? <div className="notice compact">{error}</div> : null}

      <div className="runtime-status-banner">
        <div className="runtime-status-pills">
          <span className={`runtime-pill ${healthy ? "ok" : "bad"}`} title="Scheduler health">
            <span className="runtime-pill-dot" /> {healthy ? "Healthy" : "Unhealthy"}
          </span>
          <span className="runtime-pill muted" title={`Heartbeat ${formatRuntimeDate(snapshot.runtime_status.heartbeat_at)}`}>
            Heartbeat {heartbeatAge != null ? `${heartbeatAge}s` : "missing"}
          </span>
          <span className="runtime-pill muted" title="Active runs / max concurrent">
            Capacity <strong>{cap.active_count}</strong> / {cap.max_concurrent_tasks}
            <small> ({cap.available_slots} free)</small>
          </span>
          <span className="runtime-pill muted" title="Accepted + rejected vs total">
            Terminal <strong>{terminal}</strong> / {queueTotal} ({terminalPct}%)
          </span>
          {failureCount > 0 ? (
            <span className="runtime-pill bad">
              ⚠ {failureCount} monitor failure{failureCount === 1 ? "" : "s"}
            </span>
          ) : null}
          {alertCount > 0 ? (
            <button type="button" className="runtime-pill warn runtime-pill-button" onClick={onViewAllAlerts}>
              🚨 {alertCount} alert{alertCount === 1 ? "" : "s"}
            </button>
          ) : null}
          <span className="runtime-pill muted" title={snapshot.generated_at}>
            as of {ageSince(snapshot.generated_at)} ago
          </span>
        </div>
        <button className="primary-button" type="button" disabled={running} onClick={onRunOnce}>
          {running ? "Running…" : "Drain queue"}
        </button>
      </div>

      <QueueBar items={queueItems} total={queueTotal} />

      <div className="runtime-grid-3">
        <ActiveRunsCard runs={snapshot.active_runs} />
        <StaleAndRetriesCard stale={snapshot.stale_tasks} retries={snapshot.due_retries} />
        <MonitorAlertsCard
          monitor={monitor}
          recentAlerts={recentAlerts}
          totalAlerts={totalAlerts}
          onViewAllAlerts={onViewAllAlerts}
        />
      </div>
    </>
  );
}

const QUEUE_COLORS: Record<string, string> = {
  pending: "#9aa6ad",
  annotating: "#4f8fd1",
  qc: "#7b6cb8",
  arbitrating: "#d68b3a",
  human_review: "#d23a2a",
  accepted: "#3aa563",
  rejected: "#7a5848",
  blocked: "#52616b",
  cancelled: "#a9b3b9",
  draft: "#c0c7cc",
};

function QueueBar({ items, total }: { items: Array<{ key: string; value: number }>; total: number }) {
  const visible = items.filter((i) => i.value > 0);
  return (
    <div className="runtime-queue-bar">
      <div className="runtime-queue-bar-stack" role="img" aria-label="Queue distribution">
        {total === 0 ? <div className="runtime-queue-bar-empty">No tasks</div> : null}
        {visible.map((i) => (
          <div
            key={i.key}
            className="runtime-queue-bar-seg"
            style={{ width: `${(i.value / total) * 100}%`, background: QUEUE_COLORS[i.key] ?? "#9aa6ad" }}
            title={`${i.key}: ${i.value} (${((i.value / total) * 100).toFixed(1)}%)`}
          />
        ))}
      </div>
      <div className="runtime-queue-bar-legend">
        {items.map((i) => (
          <span
            key={i.key}
            className={`runtime-queue-bar-legend-item ${i.value === 0 ? "zero" : ""}`}
          >
            <span
              className="runtime-queue-bar-swatch"
              style={{ background: QUEUE_COLORS[i.key] ?? "#9aa6ad" }}
            />
            <span className="runtime-queue-bar-key">{i.key}</span>
            <strong>{i.value.toLocaleString()}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function ActiveRunsCard({ runs }: { runs: ActiveRun[] }) {
  return (
    <div className="runtime-card">
      <h3>
        Active Runs <span className="runtime-card-count">{runs.length}</span>
      </h3>
      {runs.length === 0 ? (
        <p className="runtime-muted">No active runs</p>
      ) : (
        <table className="runtime-table">
          <thead>
            <tr>
              <th>Task</th>
              <th>Stage</th>
              <th>Target</th>
              <th className="ar">Age</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id}>
                <td title={r.task_id}>{shortId(r.task_id)}</td>
                <td>{r.stage}</td>
                <td>{r.provider_target}</td>
                <td className="ar">{ageSince(r.started_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function StaleAndRetriesCard({ stale, retries }: { stale: string[]; retries: string[] }) {
  return (
    <div className="runtime-card">
      <h3>Stale &amp; Retries</h3>
      <div className="runtime-stale-group">
        <h4>
          Stale tasks <span className="runtime-card-count">{stale.length}</span>
        </h4>
        {stale.length === 0 ? (
          <p className="runtime-muted">No stale tasks</p>
        ) : (
          <ul className="runtime-id-list">
            {stale.map((id) => (
              <li key={id} title={id}>{shortId(id)}</li>
            ))}
          </ul>
        )}
      </div>
      <div className="runtime-stale-group">
        <h4>
          Due retries <span className="runtime-card-count">{retries.length}</span>
        </h4>
        {retries.length === 0 ? (
          <p className="runtime-muted">No due retries</p>
        ) : (
          <ul className="runtime-id-list">
            {retries.map((id) => (
              <li key={id} title={id}>{shortId(id)}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function MonitorAlertsCard({
  monitor,
  recentAlerts,
  totalAlerts,
  onViewAllAlerts,
}: {
  monitor: RuntimeMonitorReport | null;
  recentAlerts: AlertEntry[];
  totalAlerts: number;
  onViewAllAlerts: () => void;
}) {
  const failures = monitor?.failures ?? [];
  const topAlerts = recentAlerts.slice(0, 5);

  return (
    <div className="runtime-card">
      <h3>Monitor &amp; Alerts</h3>
      {failures.length === 0 && topAlerts.length === 0 ? (
        <p className="runtime-muted">All clear — no failures, no recent alerts.</p>
      ) : null}

      {failures.length > 0 ? (
        <div className="runtime-monitor-failures">
          {failures.map((f) => {
            const details = (monitor?.details[f] ?? {}) as Record<string, unknown>;
            const ids = Array.isArray(details.task_ids) ? (details.task_ids as string[]) : undefined;
            const count = typeof details.count === "number" ? (details.count as number) : undefined;
            return (
              <div key={f} className="runtime-monitor-failure">
                <span className="runtime-failure-tag">⚠ {f.replace(/_/g, " ")}</span>
                {count != null ? <span className="runtime-failure-count">×{count}</span> : null}
                {ids && ids.length > 0 ? (
                  <span className="runtime-failure-ids" title={ids.join(", ")}>
                    {ids.slice(0, 3).map(shortId).join(", ")}
                    {ids.length > 3 ? ` +${ids.length - 3}` : ""}
                  </span>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}

      {topAlerts.length > 0 ? (
        <div className="runtime-recent-alerts">
          <h4>Recent alerts</h4>
          <ul className="runtime-alert-list">
            {topAlerts.map((a, idx) => (
              <li key={`${a.ts}-${idx}`} className={`runtime-alert-row alert-${slug(a.kind ?? "other")}`}>
                <span className="runtime-alert-age">{ageSince(a.ts)}</span>
                <span className="runtime-alert-kind">{a.kind ?? "?"}</span>
                <span className="runtime-alert-target">{a.target ?? a.task_id ?? "-"}</span>
                <span className="runtime-alert-msg" title={a.message ?? ""}>
                  {shorten(a.message ?? droppedSummary(a.dropped), 60)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {totalAlerts > 0 || failures.length > 0 ? (
        <button type="button" className="runtime-link" onClick={onViewAllAlerts}>
          {totalAlerts > topAlerts.length
            ? `View all ${totalAlerts} alerts →`
            : "View Alerts tab →"}
        </button>
      ) : null}
    </div>
  );
}

function shortId(id: string): string {
  if (!id) return "";
  const m = id.match(/-(\d+)$/);
  return m ? m[1] : id.slice(-6);
}

function ageSince(tsIso: string | null | undefined): string {
  if (!tsIso) return "—";
  const t = Date.parse(tsIso);
  if (!Number.isFinite(t)) return "—";
  const dSec = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (dSec < 60) return `${dSec}s`;
  if (dSec < 3600) return `${Math.floor(dSec / 60)}m`;
  if (dSec < 86400) return `${Math.floor(dSec / 3600)}h`;
  return `${Math.floor(dSec / 86400)}d`;
}

function shorten(s: string | undefined, n = 80): string {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function droppedSummary(dropped: Record<string, number> | undefined): string {
  if (!dropped) return "";
  return Object.entries(dropped)
    .map(([k, v]) => `${k}×${v}`)
    .join(", ");
}

function slug(s: string): string {
  return s.replace(/[^A-Za-z0-9-]+/g, "-").toLowerCase();
}

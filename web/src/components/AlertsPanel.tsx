import { useEffect, useMemo, useState } from "react";
import { fetchAlerts, type AlertEntry } from "../api";

interface AlertsPanelProps {
  storeKey: string | null;
}

const REFRESH_INTERVAL_MS = 15000;
const LIMIT = 200;

export function AlertsPanel({ storeKey }: AlertsPanelProps) {
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [path, setPath] = useState<string>("alerts.jsonl");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState<number>(() => Date.now());

  // Auto-refresh every 15s so a 402-storm gets surfaced without a manual reload.
  useEffect(() => {
    let active = true;
    const load = () => {
      fetchAlerts(storeKey, { limit: LIMIT })
        .then((payload) => {
          if (!active) return;
          setAlerts(payload.alerts);
          setPath(payload.alerts_path);
          setError(null);
        })
        .catch((reason: unknown) => {
          if (!active) return;
          setError(reason instanceof Error ? reason.message : "Unable to load alerts");
        })
        .finally(() => {
          if (active) setLoading(false);
        });
    };
    load();
    const id = window.setInterval(load, REFRESH_INTERVAL_MS);
    const tick = window.setInterval(() => setNow(Date.now()), 1000);
    return () => {
      active = false;
      window.clearInterval(id);
      window.clearInterval(tick);
    };
  }, [storeKey]);

  const summary = useMemo(() => summarize(alerts), [alerts]);

  return (
    <section className="work-panel alerts-panel" aria-label="Provider alerts">
      <div className="panel-header">
        <div>
          <h2>Provider Alerts</h2>
          <p>
            {alerts.length} entr{alerts.length === 1 ? "y" : "ies"} from <code>{path}</code>
            {summary.lastTs ? ` · most recent ${ageDescription(summary.lastTs, now)}` : ""}
          </p>
        </div>
        <div className="alert-summary-chips">
          {summary.kinds.map(([kind, count]) => (
            <span key={kind} className={`alert-chip alert-chip-${slug(kind)}`}>
              {kind}: {count}
            </span>
          ))}
        </div>
      </div>
      {loading && alerts.length === 0 ? <div className="drawer-state">Loading alerts</div> : null}
      {error ? <div className="drawer-error">{error}</div> : null}
      {!loading && alerts.length === 0 && !error ? (
        <div className="drawer-state empty">
          No alerts on file. This is good — provider health probes pass and no
          permanent provider errors have been raised since the last alerts.jsonl
          rotation.
        </div>
      ) : null}
      <div className="alerts-table">
        {alerts.map((a, idx) => (
          <details className={`alert-row alert-${slug(a.kind ?? "other")}`} key={`${a.ts}-${idx}`}>
            <summary>
              <span className="alert-ts" title={a.ts}>
                {ageDescription(a.ts, now)}
              </span>
              <span className="alert-kind">{a.kind ?? "?"}</span>
              <span className="alert-target">{a.target ?? a.task_id ?? "-"}</span>
              <span className="alert-status">{statusLabel(a)}</span>
              <span className="alert-message">{shorten(a.message ?? droppedSummary(a.dropped))}</span>
            </summary>
            <pre className="json-block">{JSON.stringify(a, null, 2)}</pre>
          </details>
        ))}
      </div>
    </section>
  );
}

function summarize(alerts: AlertEntry[]): { kinds: Array<[string, number]>; lastTs: string | null } {
  const counts = new Map<string, number>();
  let lastTs: string | null = null;
  for (const a of alerts) {
    const k = a.kind ?? "other";
    counts.set(k, (counts.get(k) ?? 0) + 1);
    if (!lastTs || (a.ts && a.ts > lastTs)) lastTs = a.ts;
  }
  return {
    kinds: Array.from(counts.entries()).sort((x, y) => y[1] - x[1]),
    lastTs,
  };
}

function statusLabel(a: AlertEntry): string {
  if (a.api_error_status != null) return `${a.api_error_status}`;
  if (a.exception_class) return a.exception_class;
  return "-";
}

function droppedSummary(dropped: Record<string, number> | undefined): string {
  if (!dropped) return "";
  return Object.entries(dropped).map(([k, v]) => `${k}×${v}`).join(", ");
}

function shorten(s: string | undefined, n = 120): string {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function slug(s: string): string {
  return s.replace(/[^A-Za-z0-9-]+/g, "-").toLowerCase();
}

function ageDescription(tsIso: string, nowMs: number): string {
  const t = Date.parse(tsIso);
  if (!Number.isFinite(t)) return tsIso;
  const dSec = Math.max(0, Math.floor((nowMs - t) / 1000));
  if (dSec < 60) return `${dSec}s ago`;
  if (dSec < 3600) return `${Math.floor(dSec / 60)}m ago`;
  if (dSec < 86400) return `${Math.floor(dSec / 3600)}h ago`;
  return `${Math.floor(dSec / 86400)}d ago`;
}

import { useEffect, useMemo, useState } from "react";
import { fetchAlerts, fetchEventLog, type AlertEntry } from "../api";

interface EventLogPanelProps {
  projectId: string | null;
  storeKey: string | null;
}

// Unified log entry merging audit transitions and system alerts
interface LogEntry {
  ts: string;
  rowKind: "transition" | "alert";
  // transition fields
  event_id?: string;
  task_id?: string;
  previous_status?: string;
  next_status?: string;
  actor?: string;
  // alert fields
  alert_kind?: string;
  target?: string;
  message?: string;
  api_error_status?: number | null;
}

const ENTRIES_LIMIT = 500;
const ALERTS_LIMIT = 200;
const PAGE_SIZE = 100;
const REFRESH_MS = 30_000;

// SVG histogram dimensions
const VW = 960;
const VH = 150;
const PAD = { top: 10, right: 10, bottom: 28, left: 38 };
const N_BINS = 32;

const STATUS_COLORS: Record<string, string> = {
  // task statuses (by next_status)
  accepted: "#3aa563",
  rejected: "#7a5848",
  human_review: "#d23a2a",
  arbitrating: "#d68b3a",
  qc: "#7b6cb8",
  annotating: "#4f8fd1",
  pending: "#9aa6ad",
  blocked: "#52616b",
  cancelled: "#a9b3b9",
  // alert kinds
  provider_health: "#e03131",
  provider_alert: "#e8590c",
  arbiter_enum_coerce: "#ae3ec9",
  config_reload: "#1971c2",
};
const STATUS_ORDER = Object.keys(STATUS_COLORS);

function toLogEntry(e: Record<string, unknown>): LogEntry {
  return {
    ts: String(e.created_at ?? ""),
    rowKind: "transition",
    event_id: String(e.event_id ?? ""),
    task_id: String(e.task_id ?? ""),
    previous_status: String(e.previous_status ?? ""),
    next_status: String(e.next_status ?? ""),
    actor: String(e.actor ?? ""),
  };
}

function alertToLogEntry(a: AlertEntry): LogEntry {
  return {
    ts: a.ts,
    rowKind: "alert",
    alert_kind: a.kind ?? "alert",
    target: a.target ?? a.task_id ?? "–",
    message: a.message,
    api_error_status: a.api_error_status,
  };
}

function mergeEntries(transitions: LogEntry[], alerts: LogEntry[]): LogEntry[] {
  const merged = [...transitions, ...alerts];
  merged.sort((a, b) => (b.ts > a.ts ? 1 : b.ts < a.ts ? -1 : 0));
  return merged;
}

export function EventLogPanel({ projectId, storeKey }: EventLogPanelProps) {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { setPage(1); }, [projectId, storeKey]);

  useEffect(() => {
    let active = true;
    setLoading(true);

    const load = () => {
      Promise.all([
        fetchEventLog(projectId, storeKey, { limit: ENTRIES_LIMIT, offset: 0 }),
        fetchAlerts(storeKey, { limit: ALERTS_LIMIT }).catch(() => ({ alerts: [] as AlertEntry[], total_lines: 0, alerts_path: "" })),
      ])
        .then(([evPayload, alPayload]) => {
          if (!active) return;
          const transitions = evPayload.events.map(toLogEntry);
          const alerts = (alPayload.alerts ?? []).map(alertToLogEntry);
          setEntries(mergeEntries(transitions, alerts));
          setError(null);
        })
        .catch((reason: unknown) => {
          if (!active) return;
          setError(reason instanceof Error ? reason.message : "Unable to load event log");
        })
        .finally(() => { if (active) setLoading(false); });
    };

    load();
    const id = window.setInterval(load, REFRESH_MS);
    return () => { active = false; window.clearInterval(id); };
  }, [projectId, storeKey]);

  const totalPages = Math.max(1, Math.ceil(entries.length / PAGE_SIZE));
  const pageEntries = useMemo(
    () => entries.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE),
    [entries, page],
  );
  const pageNumbers = useMemo(() => buildPageList(page, totalPages), [page, totalPages]);

  const transitions = entries.filter((e) => e.rowKind === "transition");
  const alerts = entries.filter((e) => e.rowKind === "alert");

  return (
    <section className="event-log" aria-label="Event log">
      <div className="panel-header">
        <div>
          <h2>Event Log</h2>
          <p>
            {transitions.length.toLocaleString()} transitions · {alerts.length} alerts
            {" from "}{projectId ?? "all projects"}
            {totalPages > 1 ? ` · page ${page} of ${totalPages}` : ""}
          </p>
        </div>
      </div>

      <EventHistogram entries={entries} />

      {loading ? <div className="drawer-state">Loading events</div> : null}
      {error ? <div className="drawer-error">{error}</div> : null}
      {!loading && entries.length === 0 && !error ? (
        <div className="drawer-state empty">No events found.</div>
      ) : null}

      <div className="event-table-wrap">
        <table className="event-log-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>ID / Target</th>
              <th>Event</th>
              <th>Actor</th>
            </tr>
          </thead>
          <tbody>
            {pageEntries.map((entry, idx) =>
              entry.rowKind === "alert" ? (
                <tr key={`alert-${entry.ts}-${idx}`} className="event-tr event-tr-alert">
                  <td className="event-td-ts">{fmtTs(entry.ts)}</td>
                  <td className="event-td-task">{entry.target ?? "–"}</td>
                  <td className="event-td-transition">
                    <span
                      className="event-status-chip event-status-chip--alert"
                      style={chipStyle(entry.alert_kind ?? "alert")}
                    >
                      {entry.alert_kind ?? "alert"}
                    </span>
                    {entry.api_error_status != null ? (
                      <span className="event-alert-code">{entry.api_error_status}</span>
                    ) : null}
                    {entry.message ? (
                      <span className="event-alert-msg" title={entry.message}>
                        {shorten(entry.message, 80)}
                      </span>
                    ) : null}
                  </td>
                  <td className="event-td-actor">–</td>
                </tr>
              ) : (
                <tr key={entry.event_id ?? `ev-${idx}`} className="event-tr">
                  <td className="event-td-ts">{fmtTs(entry.ts)}</td>
                  <td className="event-td-task" title={entry.task_id ?? ""}>
                    {shortId(entry.task_id ?? "")}
                  </td>
                  <td className="event-td-transition">
                    <StatusChip status={entry.previous_status ?? "–"} />
                    <span className="transition-arrow">→</span>
                    <StatusChip status={entry.next_status ?? "–"} />
                  </td>
                  <td className="event-td-actor">{entry.actor ?? "–"}</td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 ? (
        <nav className="event-log-pager" aria-label="Event log pagination">
          <button
            type="button"
            className="page-arrow"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
          >
            ‹ prev
          </button>
          {pageNumbers.map((entry, idx) =>
            entry === "…" ? (
              <span key={`gap-${idx}`} className="page-gap">…</span>
            ) : (
              <button
                key={entry}
                type="button"
                className={entry === page ? "page-num selected" : "page-num"}
                onClick={() => setPage(entry)}
              >
                {entry}
              </button>
            ),
          )}
          <button
            type="button"
            className="page-arrow"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
          >
            next ›
          </button>
        </nav>
      ) : null}
    </section>
  );
}

function StatusChip({ status }: { status: string }) {
  return (
    <span className="event-status-chip" style={chipStyle(status)}>
      {status}
    </span>
  );
}

function chipStyle(key: string): React.CSSProperties {
  const color = STATUS_COLORS[key];
  if (!color) return {};
  return { background: color + "28", color, borderColor: color + "60" };
}

// ── Histogram ────────────────────────────────────────────────────────────────

interface BinData {
  ts: number;
  counts: Record<string, number>;
  total: number;
}

interface HistData {
  bins: BinData[];
  keys: string[];
  minTs: number;
  rangeMs: number;
  maxBinTotal: number;
}

function buildHistogram(entries: LogEntry[], nBins: number): HistData | null {
  if (entries.length === 0) return null;
  const times = entries.map((e) => Date.parse(e.ts)).filter(Number.isFinite);
  if (times.length === 0) return null;

  const minTs = Math.min(...times);
  const maxTs = Math.max(...times);
  const rangeMs = maxTs - minTs || 1;

  const bins: BinData[] = Array.from({ length: nBins }, (_, i) => ({
    ts: minTs + (i / nBins) * rangeMs,
    counts: {},
    total: 0,
  }));

  const keySet = new Set<string>();
  for (const e of entries) {
    const t = Date.parse(e.ts);
    if (!Number.isFinite(t)) continue;
    const key = e.rowKind === "transition"
      ? (e.next_status ?? "unknown")
      : (e.alert_kind ?? "alert");
    keySet.add(key);
    const binIdx = Math.min(nBins - 1, Math.floor(((t - minTs) / rangeMs) * nBins));
    bins[binIdx].counts[key] = (bins[binIdx].counts[key] ?? 0) + 1;
    bins[binIdx].total += 1;
  }

  const maxBinTotal = Math.max(1, ...bins.map((b) => b.total));
  const keys = [
    ...STATUS_ORDER.filter((k) => keySet.has(k)),
    ...Array.from(keySet).filter((k) => !STATUS_ORDER.includes(k)),
  ];

  return { bins, keys, minTs, rangeMs, maxBinTotal };
}

function EventHistogram({ entries }: { entries: LogEntry[] }) {
  const data = useMemo(() => buildHistogram(entries, N_BINS), [entries]);
  if (!data) return null;

  const { bins, keys, minTs, rangeMs, maxBinTotal } = data;
  const innerW = VW - PAD.left - PAD.right;
  const innerH = VH - PAD.top - PAD.bottom;
  const binW = innerW / N_BINS;

  const xTicks = Array.from({ length: 6 }, (_, i) => {
    const frac = i / 5;
    return { x: PAD.left + frac * innerW, label: fmtTick(minTs + frac * rangeMs, rangeMs) };
  });

  const yTicks = [0, Math.round(maxBinTotal / 2), maxBinTotal].map((v) => ({
    y: PAD.top + innerH - (v / maxBinTotal) * innerH,
    label: String(v),
  }));

  return (
    <div className="event-histogram-wrap">
      <div className="event-histogram-caption">
        Last {entries.length.toLocaleString()} entries (transitions + alerts) · binned by time · colored by event type
      </div>
      <svg
        viewBox={`0 0 ${VW} ${VH}`}
        className="event-histogram-svg"
        aria-label="Event log over time"
        preserveAspectRatio="xMidYMid meet"
      >
        {yTicks.map(({ y, label }) => (
          <g key={label}>
            <line x1={PAD.left} x2={VW - PAD.right} y1={y} y2={y} stroke="#e8eef1" strokeWidth="1" />
            <text x={PAD.left - 5} y={y + 4} textAnchor="end" fontSize="11" fill="#8fa3af">
              {label}
            </text>
          </g>
        ))}

        {bins.map((bin, i) => {
          const x = PAD.left + i * binW + 1;
          const w = Math.max(0, binW - 2);
          let cumH = 0;
          return keys.map((key) => {
            const count = bin.counts[key] ?? 0;
            if (count === 0) return null;
            const barH = (count / maxBinTotal) * innerH;
            const y = PAD.top + innerH - cumH - barH;
            cumH += barH;
            return (
              <rect
                key={`${i}-${key}`}
                x={x}
                y={y}
                width={w}
                height={barH}
                fill={STATUS_COLORS[key] ?? "#9aa6ad"}
                opacity={0.88}
              >
                <title>{key}: {count} · {fmtTick(bin.ts, rangeMs)}</title>
              </rect>
            );
          });
        })}

        <line
          x1={PAD.left}
          x2={VW - PAD.right}
          y1={PAD.top + innerH}
          y2={PAD.top + innerH}
          stroke="#d7e0e5"
          strokeWidth="1"
        />
        {xTicks.map(({ x, label }) => (
          <text key={label} x={x} y={VH - 6} textAnchor="middle" fontSize="10" fill="#8fa3af">
            {label}
          </text>
        ))}
      </svg>

      <div className="event-histogram-legend">
        {keys.map((key) => (
          <span key={key} className="event-legend-item">
            <span
              className="event-legend-swatch"
              style={{ background: STATUS_COLORS[key] ?? "#9aa6ad" }}
            />
            {key}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtTs(tsIso: string): string {
  const d = new Date(tsIso);
  if (!Number.isFinite(d.getTime())) return tsIso;
  return `${pad2(d.getMonth() + 1)}/${pad2(d.getDate())} ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

function fmtTick(tsMs: number, rangeMs: number): string {
  const d = new Date(tsMs);
  if (rangeMs < 3_600_000) return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
  if (rangeMs < 86_400_000) return `${d.getMonth() + 1}/${d.getDate()} ${pad2(d.getHours())}h`;
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function pad2(n: number): string {
  return n.toString().padStart(2, "0");
}

function shortId(id: string): string {
  if (!id) return "";
  const m = id.match(/-(\d+)$/);
  return m ? m[1] : id.slice(-6);
}

function shorten(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function buildPageList(current: number, total: number): (number | "…")[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const out: (number | "…")[] = [1];
  const start = Math.max(2, current - 2);
  const end = Math.min(total - 1, current + 2);
  if (start > 2) out.push("…");
  for (let i = start; i <= end; i++) out.push(i);
  if (end < total - 1) out.push("…");
  out.push(total);
  return out;
}

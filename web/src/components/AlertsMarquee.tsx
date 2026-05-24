import { useEffect, useState } from "react";
import { fetchAlerts, fetchEventLog, type AlertEntry } from "../api";

interface AlertsMarqueeProps {
  storeKey: string | null;
  projectId?: string | null;
  /** Click target — defaults to switching to the alerts tab via a CustomEvent. */
  onClick?: () => void;
  /** Suppress alerts older than this many minutes from the marquee (default 30). */
  freshnessMinutes?: number;
}

const REFRESH_INTERVAL_MS = 15000;
const ALERT_LIMIT = 50;
const EVENT_LIMIT = 30;
/** Task transitions older than this are dropped from the ticker. */
const EVENT_FRESHNESS_MINUTES = 3;

interface TransitionItem {
  kind: "transition";
  ts: string;
  task_id: string;
  previous_status: string;
  next_status: string;
}

interface AlertItem {
  kind: "alert";
  ts: string;
  data: AlertEntry;
}

type MarqueeItem = TransitionItem | AlertItem;

/**
 * Top-bar scrolling marquee of recent system alerts and task transitions.
 *
 * Renders nothing when there are no fresh items. Polls /api/alerts every 15s
 * (freshness controlled by `freshnessMinutes`) and /api/events every 15s
 * (fixed 3-minute freshness window). Items are sorted newest-first.
 */
export function AlertsMarquee({ storeKey, projectId, onClick, freshnessMinutes = 30 }: AlertsMarqueeProps) {
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [transitions, setTransitions] = useState<TransitionItem[]>([]);

  useEffect(() => {
    let active = true;
    const load = () => {
      fetchAlerts(storeKey, { limit: ALERT_LIMIT })
        .then((payload) => {
          if (!active) return;
          const cutoff = Date.now() - freshnessMinutes * 60_000;
          setAlerts(
            payload.alerts.filter((a) => {
              const t = Date.parse(a.ts);
              return Number.isFinite(t) && t >= cutoff;
            }),
          );
        })
        .catch(() => {});
    };
    load();
    const id = window.setInterval(load, REFRESH_INTERVAL_MS);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [storeKey, freshnessMinutes]);

  useEffect(() => {
    let active = true;
    const load = () => {
      fetchEventLog(projectId ?? null, storeKey, { limit: EVENT_LIMIT })
        .then((payload) => {
          if (!active) return;
          const cutoff = Date.now() - EVENT_FRESHNESS_MINUTES * 60_000;
          const fresh: TransitionItem[] = [];
          for (const e of payload.events) {
            const ts = (e.created_at ?? "") as string;
            if (!ts || Date.parse(ts) < cutoff) continue;
            const prev = (e.previous_status ?? "") as string;
            const next = (e.next_status ?? "") as string;
            const task_id = (e.task_id ?? "") as string;
            if (prev && next && task_id) {
              fresh.push({ kind: "transition", ts, task_id, previous_status: prev, next_status: next });
            }
          }
          setTransitions(fresh);
        })
        .catch(() => {});
    };
    load();
    const id = window.setInterval(load, REFRESH_INTERVAL_MS);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [storeKey, projectId]);

  const items: MarqueeItem[] = [
    ...alerts.map((a): AlertItem => ({ kind: "alert", ts: a.ts, data: a })),
    ...transitions,
  ].sort((a, b) => Date.parse(b.ts) - Date.parse(a.ts));

  if (items.length === 0) return null;

  return (
    <div
      className="alerts-marquee"
      role="alert"
      aria-live="polite"
      title="Click to view all alerts"
      onClick={onClick}
    >
      <span className="alerts-marquee-badge">🚨 {alerts.length > 0 ? alerts.length : null}{alerts.length > 0 && transitions.length > 0 ? " · " : null}{transitions.length > 0 ? `▶ ${transitions.length}` : null}</span>
      <div className="alerts-marquee-viewport">
        <div className="alerts-marquee-track">
          {items.map((item, idx) => renderItem(item, idx, ""))}
          {items.map((item, idx) => renderItem(item, idx, "dup-"))}
        </div>
      </div>
    </div>
  );
}

function renderItem(item: MarqueeItem, idx: number, keyPrefix: string) {
  if (item.kind === "alert") {
    const a = item.data;
    return (
      <span key={`${keyPrefix}${a.ts}-${idx}`} className="alerts-marquee-item" aria-hidden={keyPrefix !== ""}>
        <strong>{a.kind ?? "alert"}</strong>
        {a.target ? <span> · {a.target}</span> : null}
        {a.api_error_status != null ? <span> · {a.api_error_status}</span> : null}
        <span> · {shorten(a.message ?? droppedSummary(a.dropped))}</span>
        <span className="alerts-marquee-sep">  •  </span>
      </span>
    );
  }
  // transition item
  const shortId = item.task_id.slice(-8);
  return (
    <span key={`${keyPrefix}${item.ts}-${idx}`} className="alerts-marquee-item alerts-marquee-item--transition" aria-hidden={keyPrefix !== ""}>
      <span className="alerts-marquee-transition-arrow">▶</span>
      <span> {shortId}: {item.previous_status} → <strong>{item.next_status}</strong></span>
      <span className="alerts-marquee-sep">  •  </span>
    </span>
  );
}

function shorten(s: string | undefined, n = 140): string {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function droppedSummary(dropped: Record<string, number> | undefined): string {
  if (!dropped) return "";
  return Object.entries(dropped).map(([k, v]) => `${k}×${v}`).join(", ");
}

import { useEffect, useState } from "react";
import { fetchAlerts, type AlertEntry } from "../api";

interface AlertsMarqueeProps {
  storeKey: string | null;
  /** Click target — defaults to switching to the alerts tab via a CustomEvent. */
  onClick?: () => void;
  /** Suppress alerts older than this many minutes from the marquee (default 30). */
  freshnessMinutes?: number;
}

const REFRESH_INTERVAL_MS = 15000;
const LIMIT = 50;

/**
 * Top-bar scrolling marquee of recent provider alerts.
 *
 * Renders nothing (`display: none`) when there are no fresh alerts —
 * the topbar layout is unaffected on a healthy pipeline. When alerts
 * exist, slides the most-recent N across the bar in a CSS animation.
 *
 * Polls /api/alerts every 15s. Click anywhere on the marquee to jump
 * to the full Alerts tab.
 */
export function AlertsMarquee({ storeKey, onClick, freshnessMinutes = 30 }: AlertsMarqueeProps) {
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);

  useEffect(() => {
    let active = true;
    const load = () => {
      fetchAlerts(storeKey, { limit: LIMIT })
        .then((payload) => {
          if (!active) return;
          const cutoff = Date.now() - freshnessMinutes * 60_000;
          const fresh = payload.alerts.filter((a) => {
            const t = Date.parse(a.ts);
            return Number.isFinite(t) && t >= cutoff;
          });
          setAlerts(fresh);
        })
        .catch(() => {
          // Don't surface fetch errors in the topbar — the user will
          // see them when they open the Alerts tab if persistent.
        });
    };
    load();
    const id = window.setInterval(load, REFRESH_INTERVAL_MS);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [storeKey, freshnessMinutes]);

  if (alerts.length === 0) {
    return null;
  }

  return (
    <div
      className="alerts-marquee"
      role="alert"
      aria-live="polite"
      title="Click to view all alerts"
      onClick={onClick}
    >
      <span className="alerts-marquee-badge">🚨 {alerts.length}</span>
      <div className="alerts-marquee-viewport">
        <div className="alerts-marquee-track">
          {alerts.map((a, idx) => (
            <span key={`${a.ts}-${idx}`} className="alerts-marquee-item">
              <strong>{a.kind ?? "alert"}</strong>
              {a.target ? <span> · {a.target}</span> : null}
              {a.api_error_status != null ? <span> · {a.api_error_status}</span> : null}
              <span> · {shorten(a.message ?? droppedSummary(a.dropped))}</span>
              <span className="alerts-marquee-sep">  •  </span>
            </span>
          ))}
          {/* Duplicate for seamless infinite scroll. CSS animates -50% of total width. */}
          {alerts.map((a, idx) => (
            <span key={`dup-${a.ts}-${idx}`} className="alerts-marquee-item" aria-hidden>
              <strong>{a.kind ?? "alert"}</strong>
              {a.target ? <span> · {a.target}</span> : null}
              {a.api_error_status != null ? <span> · {a.api_error_status}</span> : null}
              <span> · {shorten(a.message ?? droppedSummary(a.dropped))}</span>
              <span className="alerts-marquee-sep">  •  </span>
            </span>
          ))}
        </div>
      </div>
    </div>
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

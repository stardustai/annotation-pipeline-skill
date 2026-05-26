// Lazy-loaded by EventLogPanel — keeps the ~3 MB Plotly bundle off the
// critical path.  Do NOT import this file directly; use React.lazy().
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);

// ── Colours (kept in sync with EventLogPanel STATUS_COLORS) ─────────────────
const STATUS_COLORS: Record<string, string> = {
  accepted: "#3aa563",
  rejected: "#7a5848",
  human_review: "#d23a2a",
  arbitrating: "#d68b3a",
  qc: "#7b6cb8",
  annotating: "#4f8fd1",
  pending: "#9aa6ad",
  blocked: "#52616b",
  cancelled: "#a9b3b9",
  provider_health: "#e03131",
  provider_alert: "#e8590c",
  arbiter_enum_coerce: "#ae3ec9",
  config_reload: "#1971c2",
};
const STATUS_ORDER = Object.keys(STATUS_COLORS);

// ── Minimal entry shape the histogram needs ──────────────────────────────────
export interface HistLogEntry {
  ts: string;
  rowKind: "transition" | "alert";
  next_status?: string;
  alert_kind?: string;
}

// ── Bin helpers ──────────────────────────────────────────────────────────────
interface BinData {
  ts: number;
  counts: Record<string, number>;
}

interface HistData {
  bins: BinData[];
  keys: string[];
}

function buildHistogram(entries: HistLogEntry[], nBins: number): HistData | null {
  const times = entries.map((e) => Date.parse(e.ts)).filter(Number.isFinite);
  if (times.length === 0) return null;

  const minTs = Math.min(...times);
  const maxTs = Math.max(...times);
  const rangeMs = maxTs - minTs || 1;

  const bins: BinData[] = Array.from({ length: nBins }, (_, i) => ({
    ts: minTs + (i / nBins) * rangeMs,
    counts: {},
  }));

  const keySet = new Set<string>();
  for (const e of entries) {
    const t = Date.parse(e.ts);
    if (!Number.isFinite(t)) continue;
    const key =
      e.rowKind === "transition" ? (e.next_status ?? "unknown") : (e.alert_kind ?? "alert");
    keySet.add(key);
    const idx = Math.min(nBins - 1, Math.floor(((t - minTs) / rangeMs) * nBins));
    bins[idx].counts[key] = (bins[idx].counts[key] ?? 0) + 1;
  }

  const keys = [
    ...STATUS_ORDER.filter((k) => keySet.has(k)),
    ...Array.from(keySet).filter((k) => !STATUS_ORDER.includes(k)),
  ];

  return { bins, keys };
}

// ── Layout constants ─────────────────────────────────────────────────────────
const BAR_PX = 20;        // pixels per bin column
const MIN_CHART_W = 700;  // never narrower than this
const CHART_H = 180;      // fixed height
const DEFAULT_BINS = 64;
const MIN_BINS = 8;
const MAX_BINS = 256;

// ── Component ────────────────────────────────────────────────────────────────
interface EventHistogramChartProps {
  entries: HistLogEntry[];
  totalCount: number;
}

export default function EventHistogramChart({ entries, totalCount }: EventHistogramChartProps) {
  const [nBins, setNBins] = useState(DEFAULT_BINS);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Non-passive wheel listener: vertical scroll adjusts bin width (bar spacing).
  // Must be non-passive so preventDefault() works and the page doesn't scroll.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return; // ignore horizontal swipes
      e.preventDefault();
      setNBins((prev) => {
        const step = Math.max(4, Math.round(prev * 0.15));
        // scroll up (deltaY < 0) → more bins (narrower bars / more detail)
        // scroll down (deltaY > 0) → fewer bins (wider bars / less detail)
        const next = e.deltaY < 0 ? prev + step : prev - step;
        return Math.min(MAX_BINS, Math.max(MIN_BINS, next));
      });
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, []); // wrapRef.current is stable; nBins is read via functional updater

  const data = useMemo(() => buildHistogram(entries, nBins), [entries, nBins]);

  const resetBins = useCallback(() => setNBins(DEFAULT_BINS), []);

  if (!data) return null;

  const chartW = Math.max(MIN_CHART_W, nBins * BAR_PX);
  // Keep x-axis tick density at ~70 px/tick regardless of zoom level.
  const xNTicks = Math.max(6, Math.round(chartW / 70));

  const traces: Plotly.Data[] = data.keys.map((key) => ({
    type: "bar",
    name: key,
    x: data.bins.map((b) => new Date(b.ts).toISOString()),
    y: data.bins.map((b) => b.counts[key] ?? 0),
    marker: { color: STATUS_COLORS[key] ?? "#9aa6ad", opacity: 0.88 },
    hovertemplate: `%{x|%m/%d %H:%M}<br>${key}: <b>%{y}</b><extra></extra>`,
  }));

  const layout: Partial<Plotly.Layout> = {
    barmode: "stack",
    bargap: 0.05,
    width: chartW,
    height: CHART_H,
    margin: { t: 8, r: 8, b: 40, l: 40 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    showlegend: false,
    // Both axes fixed — interaction is handled by the CSS-scroll container +
    // the custom wheel handler above.
    dragmode: false,
    xaxis: {
      type: "date",
      fixedrange: true,
      nticks: xNTicks,
      tickfont: { size: 10, color: "#8fa3af" },
      gridcolor: "#e8eef1",
      linecolor: "#d7e0e5",
    },
    yaxis: {
      fixedrange: true,
      tickfont: { size: 10, color: "#8fa3af" },
      gridcolor: "#e8eef1",
      linecolor: "#d7e0e5",
    },
  };

  const config: Partial<Plotly.Config> = {
    displayModeBar: false,
    responsive: false,
  };

  return (
    <div className="event-histogram-wrap">
      <div className="event-histogram-caption">
        Last {totalCount.toLocaleString()} entries (transitions + alerts) · binned by time ·{" "}
        <strong>{nBins}</strong> bins ·{" "}
        <span className="dsb-metric-hint">scroll ↕ to adjust bar width</span>
        {nBins !== DEFAULT_BINS && (
          <button
            type="button"
            className="event-histogram-reset"
            onClick={resetBins}
            title="Reset to default bin count"
          >
            reset
          </button>
        )}
      </div>

      {/* overflow-x: auto gives the horizontal scrollbar.
          width:100% + min-width:0 keeps the container pinned to the panel
          width so that a wide chart scrolls inside it instead of pushing
          the page wider. */}
      <div
        ref={wrapRef}
        className="event-histogram-scroll"
      >
        <Plot data={traces} layout={layout} config={config} />
      </div>

      <div className="event-histogram-legend">
        {data.keys.map((key) => (
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

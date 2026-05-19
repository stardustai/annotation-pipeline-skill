import React, { useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";

import type { CoordEntry } from "./DistributionPanel";

// react-plotly.js factory pattern: needed so we can bundle plotly.js-dist-min
// instead of relying on a globally-loaded plotly. Building this Plot
// component is the side-effect that imports the ~3 MB plotly bundle —
// keeping it inside ScatterSubTab.tsx (which is lazy-loaded by
// DistributionPanel) means plotly only ships when the operator opens
// the Scatter sub-tab.
const Plot = createPlotlyComponent(Plotly);

const STATUS_COLORS: Record<string, string> = {
  accepted: "#047857",
  rejected: "#b91c1c",
  human_review: "#d97706",
  arbitrating: "#7c3aed",
  qc: "#2563eb",
  annotating: "#0891b2",
  pending: "#6b7280",
  draft: "#9ca3af",
  blocked: "#92400e",
  cancelled: "#1f2937",
};

// All 10 TaskStatus values; order is stable and drives trace + legend order.
const ALL_STATUSES: string[] = [
  "accepted",
  "rejected",
  "human_review",
  "arbitrating",
  "qc",
  "annotating",
  "pending",
  "draft",
  "blocked",
  "cancelled",
];

export type ScatterSubTabProps = {
  coords: CoordEntry[];
  onSelectTask?: (taskId: string) => void;
};

export default function ScatterSubTab({
  coords,
  onSelectTask,
}: ScatterSubTabProps): React.ReactElement {
  const [visible, setVisible] = useState<Record<string, boolean>>(() => {
    const init: Record<string, boolean> = {};
    for (const s of ALL_STATUSES) init[s] = true;
    return init;
  });

  const totalCountByStatus: Record<string, number> = {};
  for (const s of ALL_STATUSES) totalCountByStatus[s] = 0;
  for (const c of coords) {
    if (c.status in totalCountByStatus) {
      totalCountByStatus[c.status]++;
    }
  }

  function toggleStatus(status: string) {
    setVisible((prev) => ({ ...prev, [status]: !prev[status] }));
  }

  if (coords.length === 0) {
    return (
      <div className="runtime-muted" style={{ padding: "2rem", textAlign: "center" }}>
        No scan data yet — run <strong>[Re-]Scan</strong> first.
      </div>
    );
  }

  const filteredCoords = coords.filter((c) => visible[c.status] !== false);

  const traces = ALL_STATUSES.map((status) => {
    const points = filteredCoords.filter((c) => c.status === status);
    return {
      type: "scatter" as const,
      mode: "markers" as const,
      name: `${status} (${totalCountByStatus[status]})`,
      x: points.map((p) => p.x),
      y: points.map((p) => p.y),
      customdata: points.map((p): [string, string] => [
        p.task_id,
        p.text_preview.length > 80 ? p.text_preview.slice(0, 80) + "…" : p.text_preview,
      ]),
      hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
      marker: {
        color: STATUS_COLORS[status] ?? "#6b7280",
        size: 5,
        opacity: 0.7,
      },
      visible: visible[status] ? (true as const) : ("legendonly" as const),
    };
  });

  const uirevision = `${coords.length}:${coords[0]?.task_id ?? ""}`;
  const layout: Partial<Plotly.Layout> = {
    autosize: true,
    hovermode: "closest" as const,
    margin: { l: 50, r: 30, t: 30, b: 50 },
    xaxis: { title: { text: "UMAP-1" }, autorange: true },
    yaxis: { title: { text: "UMAP-2" }, autorange: true },
    legend: { itemsizing: "constant" as const },
    showlegend: true,
    uirevision,
  };

  const config: Partial<Plotly.Config> = {
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] as Plotly.ModeBarDefaultButtons[],
  };

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.4rem 0.75rem",
          padding: "0.6rem 0.75rem",
          background: "var(--surface2, #f8fafc)",
          borderRadius: "6px",
          border: "1px solid var(--border, #e5e7eb)",
          marginBottom: "0.75rem",
        }}
      >
        {ALL_STATUSES.map((status) => (
          <label
            key={status}
            style={{ display: "flex", alignItems: "center", gap: "0.3rem", fontSize: "0.82rem", cursor: "pointer" }}
          >
            <input
              type="checkbox"
              checked={visible[status] !== false}
              onChange={() => toggleStatus(status)}
            />
            <span
              style={{
                display: "inline-block",
                width: "10px",
                height: "10px",
                borderRadius: "50%",
                background: STATUS_COLORS[status] ?? "#6b7280",
                flexShrink: 0,
              }}
            />
            {status} ({totalCountByStatus[status]})
          </label>
        ))}
      </div>

      <Plot
        data={traces}
        layout={layout}
        config={config}
        useResizeHandler
        style={{ width: "100%", height: "600px" }}
        onClick={(event) => {
          const point = event.points?.[0];
          if (!point) return;
          const taskId = (point.customdata as unknown as [string, string])[0];
          onSelectTask?.(taskId);
        }}
      />
    </div>
  );
}

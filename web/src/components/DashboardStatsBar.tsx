import { useEffect, useState } from "react";
import { fetchDashboardStats, type DashboardStats } from "../api";

const QUEUE_COLORS: Record<string, string> = {
  draft: "#c0c7cc",
  pending: "#9aa6ad",
  annotating: "#4f8fd1",
  qc: "#7b6cb8",
  arbitrating: "#d68b3a",
  human_review: "#d23a2a",
  accepted: "#3aa563",
  rejected: "#7a5848",
  blocked: "#52616b",
  cancelled: "#a9b3b9",
};

const STAGE_LABELS: Record<string, string> = {
  pending: "Pending",
  annotating: "Annotating",
  qc: "QC",
  arbitrating: "Arbitrating",
  human_review: "HR",
  accepted: "Accepted",
  rejected: "Rejected",
  blocked: "Blocked",
};

const BAR_STAGES = ["pending", "annotating", "qc", "arbitrating", "human_review", "accepted", "rejected", "blocked"];
const LEGEND_STAGES = ["pending", "annotating", "qc", "arbitrating", "human_review", "accepted", "rejected", "blocked"];

interface DashboardStatsBarProps {
  projectId: string | null;
  storeKey: string | null;
  runtimeHealthy?: boolean | null;
}

export function DashboardStatsBar({ projectId, storeKey, runtimeHealthy }: DashboardStatsBarProps) {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const next = await fetchDashboardStats(projectId, storeKey);
        if (!active) return;
        setStats(next);
        setError(null);
      } catch (reason: unknown) {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "Unable to load stats");
      }
    }
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => { active = false; clearInterval(timer); };
  }, [projectId, storeKey]);

  if (!stats && !error) {
    return <div className="dashboard-stats-bar dashboard-stats-bar-loading">Loading…</div>;
  }
  if (!stats) {
    return <div className="dashboard-stats-bar dashboard-stats-bar-error">{error}</div>;
  }

  const counts = stats.status_counts;
  const winMin = stats.throughput_window_minutes || 1;
  const perMin = (stage: string): string => {
    const raw = stats.throughput_per_window[stage] ?? 0;
    const rate = raw / winMin;
    return rate >= 10 ? rate.toFixed(0) : rate.toFixed(1);
  };
  const hint = `attempts/min averaged over last ${winMin} min`;

  const barTotal = BAR_STAGES.reduce((s, k) => s + (counts[k] ?? 0), 0);
  const terminal = (counts["accepted"] ?? 0) + (counts["rejected"] ?? 0);
  const hrRate = terminal > 0 ? ((counts["human_review"] ?? 0) / terminal) * 100 : 0;
  const rejRate = terminal > 0 ? ((counts["rejected"] ?? 0) / terminal) * 100 : 0;

  const hrTone = hrRate > 10 ? "critical" : hrRate > 2 ? "warning" : "";
  const rejTone = rejRate > 30 ? "warning" : "";
  const feedbackTone = (stats.open_feedback_count ?? 0) > 0 ? "warning" : "";

  // Health metrics
  const acceptedCount = stats.accepted_count ?? 0;
  const firstPassRate = acceptedCount > 0
    ? ((stats.first_pass_count ?? 0) / acceptedCount) * 100
    : 0;
  const terminalCount = stats.terminal_count ?? 0;
  const arbRate = terminalCount > 0
    ? ((stats.arb_entered_count ?? 0) / terminalCount) * 100
    : 0;
  const avgLlmCalls = stats.avg_llm_calls ?? 0;

  const firstPassTone = acceptedCount > 0 && firstPassRate < 30 ? "warning" : "";
  const arbRateTone = terminalCount > 0 && arbRate > 20 ? "warning" : "";
  const avgLlmTone = acceptedCount > 0 && avgLlmCalls > 6 ? "warning" : "";

  // ETA: remaining active tasks / accepted rate (tasks/min).
  // "Active" = everything not yet terminal or blocked.
  const ACTIVE_STAGES = ["pending", "annotating", "qc", "arbitrating", "human_review"];
  const remaining = ACTIVE_STAGES.reduce((s, k) => s + (counts[k] ?? 0), 0);
  const acceptedRatePerMin = (stats.accepted_in_window ?? 0) / winMin;
  const etaLabel: string = (() => {
    if (remaining === 0) return "done";
    if (acceptedRatePerMin <= 0) return "—";
    const mins = Math.ceil(remaining / acceptedRatePerMin);
    if (mins < 60) return `${mins}m`;
    const h = Math.floor(mins / 60), m = mins % 60;
    return m === 0 ? `${h}h` : `${h}h ${m}m`;
  })();
  const etaTitle = remaining === 0
    ? "All tasks complete"
    : acceptedRatePerMin > 0
      ? `≈${remaining} tasks remaining at ${acceptedRatePerMin.toFixed(1)} accepted/min`
      : "No accepted throughput in the last window — ETA unavailable";

  return (
    <div className="dashboard-stats-bar">
      {/* row 1: colored bar — hover shows stage-count tooltip */}
      <div className="dsb-bar-wrap">
        <div className="dsb-bar-stack">
          {barTotal === 0 ? (
            <div className="dsb-bar-empty" />
          ) : (
            BAR_STAGES.map((stage) => {
              const count = counts[stage] ?? 0;
              if (count === 0) return null;
              return (
                <div
                  key={stage}
                  className="dsb-bar-seg"
                  style={{ width: `${(count / barTotal) * 100}%`, background: QUEUE_COLORS[stage] }}
                />
              );
            })
          )}
        </div>
        <div className="dsb-bar-tooltip">
          {LEGEND_STAGES.map((stage) => (
            <span
              key={stage}
              className={`dsb-legend-item${(counts[stage] ?? 0) === 0 ? " zero" : ""}`}
              style={{ "--dsb-color": QUEUE_COLORS[stage] } as React.CSSProperties}
            >
              <span className="dsb-legend-swatch" />
              <span className="dsb-legend-key">{STAGE_LABELS[stage]}</span>
              <strong>{counts[stage] ?? 0}</strong>
            </span>
          ))}
        </div>
      </div>

      {/* row 2: throughput metrics only */}
      <div className="dsb-metrics-row">
      <span className="dsb-divider" style={{ marginLeft: 0 }} />

      {/* metrics */}
      <span className={`dsb-metric${feedbackTone ? ` ${feedbackTone}` : ""}`} title="Open reviewer feedback">
        <span>Feedback</span>
        <strong>{stats.open_feedback_count ?? 0}</strong>
      </span>
      <span className="dsb-metric" title={hint}>
        <span>Ann/min</span>
        <strong>{perMin("annotation")}</strong>
      </span>
      <span className="dsb-metric" title={hint}>
        <span>QC/min</span>
        <strong>{perMin("qc")}</strong>
      </span>
      <span className="dsb-metric" title={hint}>
        <span>Arb/min</span>
        <strong>{perMin("arbitration")}</strong>
      </span>
      <span
        className={`dsb-metric${hrTone ? ` ${hrTone}` : ""}`}
        title="HR rate = human_review / (accepted+rejected). Target <3%"
      >
        <span>HR</span>
        <strong>{hrRate.toFixed(1)}%</strong>
      </span>
      <span
        className={`dsb-metric${rejTone ? ` ${rejTone}` : ""}`}
        title="Rejection rate = rejected / (accepted+rejected)"
      >
        <span>Rej</span>
        <strong>{rejRate.toFixed(1)}%</strong>
      </span>
      <span className="dsb-metric" title={etaTitle}>
        <span>ETA</span>
        <strong>{etaLabel}</strong>
      </span>
      <span className="dsb-divider" />
      <span
        className={`dsb-metric${firstPassTone ? ` ${firstPassTone}` : ""}`}
        title={`First-pass rate: accepted without arbitration / total accepted. Target >30%. (${stats.first_pass_count ?? 0}/${acceptedCount})`}
      >
        <span>1st-pass</span>
        <strong>{acceptedCount > 0 ? `${firstPassRate.toFixed(1)}%` : "—"}</strong>
      </span>
      <span
        className={`dsb-metric${arbRateTone ? ` ${arbRateTone}` : ""}`}
        title={`Arbitration entry rate: terminal tasks that entered arbitration. Target <20%. (${stats.arb_entered_count ?? 0}/${terminalCount})`}
      >
        <span>Arb%</span>
        <strong>{terminalCount > 0 ? `${arbRate.toFixed(1)}%` : "—"}</strong>
      </span>
      <span
        className={`dsb-metric${avgLlmTone ? ` ${avgLlmTone}` : ""}`}
        title={`Avg LLM calls per accepted task (annotation+QC+arbitration succeeded attempts). Target <6.`}
      >
        <span>LLM/task</span>
        <strong>{acceptedCount > 0 ? avgLlmCalls.toFixed(1) : "—"}</strong>
      </span>

      {runtimeHealthy !== undefined && runtimeHealthy !== null && (
        <span className={`runtime-pill${runtimeHealthy ? " ok" : " bad"}`}>
          <span className="runtime-pill-dot" />
          Runtime
        </span>
      )}
      </div>
    </div>
  );
}

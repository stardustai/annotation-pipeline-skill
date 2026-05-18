import React, { useEffect, useState } from "react";
import type { EntityConvention, EntityStatsItem } from "../types";
import { DistributionBar, OriginalTextCell, Pagination, TypePill } from "../entityHelpers";
import { clearConvention } from "../api";

const PAGE_SIZE = 100;

export type EntityKnowledgePanelProps = {
  projectId: string | null;
  storeKey: string | null;
};

type Subtab = "conventions" | "statistics";

const TABLE_STYLE: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: "0.85rem",
};
const THEAD_ROW: React.CSSProperties = {
  borderBottom: "1px solid var(--border, #2a2f3a)",
  textAlign: "left",
};
const TR: React.CSSProperties = { borderBottom: "1px solid var(--border, #2a2f3a)" };
const FORMULA_BLOCK_STYLE: React.CSSProperties = {
  background: "#f5f7fa",
  fontSize: "0.85em",
  padding: "0.5rem 0.75rem",
  margin: "0.5rem 0",
  borderRadius: "4px",
};

function fmtTime(date: Date): string {
  const pad = (n: number) => (n < 10 ? "0" + n : String(n));
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function EntityKnowledgePanel({
  projectId,
  storeKey,
}: EntityKnowledgePanelProps): React.ReactElement {
  const [subtab, setSubtab] = useState<Subtab>("conventions");
  const [conventions, setConventions] = useState<EntityConvention[] | null>(null);
  const [stats, setStats] = useState<EntityStatsItem[] | null>(null);
  const [loadedAt, setLoadedAt] = useState<{ conventions: Date | null; statistics: Date | null }>(
    { conventions: null, statistics: null },
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);
  const [unsetting, setUnsetting] = useState<string | null>(null);

  async function handleUnset(span: string | null) {
    if (!projectId || !span) return;
    if (
      !window.confirm(
        `Unset the project convention for '${span}'? This removes the rule from future prompts. Any task annotations already modified via Apply-to-all are NOT reverted.`,
      )
    ) {
      return;
    }
    setUnsetting(span);
    setError(null);
    try {
      await clearConvention(projectId, span, storeKey);
      // Optimistically drop from the in-memory list so the row vanishes
      // immediately; re-fetch in the background for the source of truth.
      setConventions((prev) =>
        prev ? prev.filter((c) => (c.span ?? "").toLowerCase() !== span.toLowerCase()) : prev,
      );
      fetchData("conventions");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUnsetting(null);
    }
  }

  function fetchData(which: Subtab) {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    const storeQ = storeKey ? `&store=${encodeURIComponent(storeKey)}` : "";
    const url =
      which === "conventions"
        ? `/api/conventions?project=${encodeURIComponent(projectId)}${storeQ}`
        : `/api/entity-statistics?project=${encodeURIComponent(projectId)}${storeQ}`;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => {
        if (which === "conventions") setConventions(d.conventions ?? []);
        else setStats(d.items ?? []);
        setLoadedAt((prev) => ({ ...prev, [which]: new Date() }));
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    fetchData(subtab);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, storeKey, subtab]);

  // Reset to first page whenever the filter or active tab changes.
  useEffect(() => { setPage(0); }, [filter, subtab]);

  if (!projectId) {
    return (
      <section className="runtime-panel" aria-label="Entity Knowledge">
        <p className="runtime-muted">Select a project first.</p>
      </section>
    );
  }

  const filterLower = filter.trim().toLowerCase();
  const filteredConvs = conventions
    ? conventions.filter(
        (c) =>
          !filterLower ||
          (c.span ?? "").toLowerCase().includes(filterLower) ||
          (c.entity_type ?? "").toLowerCase().includes(filterLower),
      )
    : null;
  const filteredStats = stats
    ? stats.filter(
        (s) =>
          !filterLower ||
          s.span.includes(filterLower) ||
          Object.keys(s.distribution).some((t) => t.toLowerCase().includes(filterLower)),
      )
    : null;

  const activeLoadedAt =
    subtab === "conventions" ? loadedAt.conventions : loadedAt.statistics;

  return (
    <section className="runtime-panel" aria-label="Entity Knowledge">
      <div className="runtime-header">
        <div>
          <h2 style={{ marginBottom: "0.25rem" }}>Entity Knowledge</h2>
          <p style={{ marginTop: 0 }}>
            Project-level entity-type dictionary (high-trust conventions used
            for prompt injection) and the broader observation statistics (all
            ACCEPTED decisions, weighted) used by the posterior verifier.
            {activeLoadedAt ? (
              <>
                {" "}
                <span className="runtime-muted" style={{ fontSize: "0.85em" }}>
                  · Last loaded: {fmtTime(activeLoadedAt)}
                </span>
              </>
            ) : null}
          </p>
        </div>
        <button
          className="primary-button"
          type="button"
          onClick={() => fetchData(subtab)}
          disabled={loading}
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      <nav className="view-tabs" aria-label="Knowledge tabs" style={{ marginBottom: 0 }}>
        <button
          className={subtab === "conventions" ? "view-tab selected" : "view-tab"}
          type="button"
          onClick={() => setSubtab("conventions")}
        >
          Conventions ({conventions?.length ?? "…"})
        </button>
        <button
          className={subtab === "statistics" ? "view-tab selected" : "view-tab"}
          type="button"
          onClick={() => setSubtab("statistics")}
        >
          Statistics ({stats?.length ?? "…"})
        </button>
      </nav>

      {subtab === "conventions" ? (
        <div style={FORMULA_BLOCK_STYLE}>
          <strong>Conventions table</strong> — high-trust dictionary injected
          into future annotator/QC prompts. <strong>Source</strong>:
          annotator+QC consensus that agreed with project statistics, plus
          HR-authored decisions. <em>Excludes arbiter-only decisions</em> to
          avoid the cascade where one LLM error reinforces itself.{" "}
          <strong>Injection criteria</strong>: <code>evidence_count ≥ 5</code>{" "}
          and span length <code>≥ 4</code> chars. Use the{" "}
          <strong>🚫 not entity</strong> type for spans that should{" "}
          <em>never</em> be tagged (stop-word style, e.g. placeholders like{" "}
          "XXX").
        </div>
      ) : (
        <div style={FORMULA_BLOCK_STYLE}>
          <strong>Statistics table</strong> — full empirical distribution over
          all <em>ACCEPTED</em> (span, type) decisions across the project,
          including arbiter-driven and HR (HR-authored decisions count{" "}
          <code>5×</code>). Used by the prior-driven verifier at QC-pass,
          first-arbiter, and HR-submit checkpoints to detect divergence.
        </div>
      )}

      <div style={{ margin: "0.4rem 0" }}>
        <input
          type="search"
          placeholder={
            subtab === "conventions"
              ? "Filter by span or type…"
              : "Filter span / type…"
          }
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ width: "min(360px, 60%)" }}
        />
        {filter && filteredConvs && subtab === "conventions" ? (
          <span className="runtime-muted" style={{ marginLeft: "0.75rem", fontSize: "0.85rem" }}>
            {filteredConvs.length} of {conventions?.length ?? 0}
          </span>
        ) : null}
        {filter && filteredStats && subtab === "statistics" ? (
          <span className="runtime-muted" style={{ marginLeft: "0.75rem", fontSize: "0.85rem" }}>
            {filteredStats.length} of {stats?.length ?? 0}
          </span>
        ) : null}
      </div>

      {error ? <div className="notice compact">{error}</div> : null}
      {loading ? <p className="runtime-muted">Loading…</p> : null}

      {!loading && subtab === "conventions" && filteredConvs ? (
        <>
        <Pagination
          total={filteredConvs.length}
          page={page}
          pageSize={PAGE_SIZE}
          onPageChange={setPage}
        />
        <div className="runtime-card">
          <table style={TABLE_STYLE}>
            <thead>
              <tr style={THEAD_ROW}>
                <th style={{ padding: "0.4rem 0.75rem 0.4rem 0" }}>Span</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Type</th>
                <th style={{ padding: "0.4rem 0.75rem", width: "35%" }}>
                  Typical text
                </th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Status</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Evidence</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Updated</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredConvs.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map((c) => (
                <tr key={c.convention_id} style={TR}>
                  <td
                    style={{
                      padding: "0.4rem 0.75rem 0.4rem 0",
                      fontFamily: "monospace",
                    }}
                  >
                    {c.span ?? <em className="runtime-muted">—</em>}
                  </td>
                  <td style={{ padding: "0.4rem 0.75rem" }}>
                    {c.entity_type ? <TypePill type={c.entity_type} /> : <em>—</em>}
                  </td>
                  <td style={{ padding: "0.4rem 0.75rem" }}>
                    {c.span ? (
                      <OriginalTextCell
                        projectId={projectId}
                        storeKey={storeKey}
                        span={c.span}
                      />
                    ) : (
                      <em className="runtime-muted">—</em>
                    )}
                  </td>
                  <td style={{ padding: "0.4rem 0.75rem" }}>
                    <span
                      style={{
                        color:
                          c.status === "active"
                            ? "var(--success, #047857)"
                            : c.status === "disputed"
                              ? "var(--danger, #b91c1c)"
                              : "var(--muted, #6b7280)",
                        fontWeight: 500,
                      }}
                    >
                      {c.status}
                    </span>
                  </td>
                  <td style={{ padding: "0.4rem 0.75rem" }}>{c.evidence_count}</td>
                  <td
                    style={{
                      padding: "0.4rem 0.75rem",
                      whiteSpace: "nowrap",
                      fontSize: "0.8rem",
                      color: "var(--muted, #6b7280)",
                    }}
                  >
                    {c.updated_at.replace("T", " ").slice(0, 19)}
                  </td>
                  <td style={{ padding: "0.4rem 0.75rem" }}>
                    <button
                      type="button"
                      disabled={!c.span || unsetting === c.span}
                      onClick={() => handleUnset(c.span)}
                      title="Remove this convention from the project. Future tasks won't see this rule injected. Already-modified task annotations are NOT reverted."
                      style={{
                        fontSize: "0.75rem",
                        color: c.span ? "var(--danger, #b91c1c)" : undefined,
                        background: "transparent",
                        border: "1px solid var(--border, #d1d5db)",
                        padding: "2px 8px",
                        borderRadius: "3px",
                        cursor: c.span ? "pointer" : "not-allowed",
                        opacity: unsetting === c.span ? 0.6 : 1,
                      }}
                    >
                      {unsetting === c.span ? "…" : "Unset"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Pagination
          total={filteredConvs.length}
          page={page}
          pageSize={PAGE_SIZE}
          onPageChange={setPage}
        />
        </>
      ) : null}

      {!loading && subtab === "statistics" && filteredStats ? (
        <>
        <Pagination
          total={filteredStats.length}
          page={page}
          pageSize={PAGE_SIZE}
          onPageChange={setPage}
        />
        <div className="runtime-card">
          <table style={TABLE_STYLE}>
            <thead>
              <tr style={THEAD_ROW}>
                <th style={{ padding: "0.4rem 0.75rem 0.4rem 0" }}>Span</th>
                <th style={{ padding: "0.4rem 0.75rem", width: "4rem" }}>Total</th>
                <th style={{ padding: "0.4rem 0.75rem", width: "30%" }}>Distribution</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Breakdown</th>
              </tr>
            </thead>
            <tbody>
              {filteredStats.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map((s) => (
                <tr key={s.span} style={TR}>
                  <td
                    style={{
                      padding: "0.4rem 0.75rem 0.4rem 0",
                      fontFamily: "monospace",
                    }}
                  >
                    {s.span}
                  </td>
                  <td style={{ padding: "0.4rem 0.75rem" }}>{s.total}</td>
                  <td style={{ padding: "0.4rem 0.75rem" }}>
                    <DistributionBar distribution={s.distribution} total={s.total} />
                  </td>
                  <td
                    style={{
                      padding: "0.4rem 0.75rem",
                      fontSize: "0.8rem",
                      color: "var(--muted, #6b7280)",
                    }}
                  >
                    {Object.entries(s.distribution)
                      .sort((a, b) => b[1] - a[1])
                      .map(([t, c]) => `${t}: ${c}`)
                      .join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Pagination
          total={filteredStats.length}
          page={page}
          pageSize={PAGE_SIZE}
          onPageChange={setPage}
        />
        </>
      ) : null}
    </section>
  );
}


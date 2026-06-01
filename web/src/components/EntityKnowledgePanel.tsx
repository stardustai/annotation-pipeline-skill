import React, { useEffect, useState } from "react";
import type { EntityConvention, EntityStatsItem } from "../types";
import { DistributionBar, OriginalTextCell, Pagination, TypePill } from "../entityHelpers";
import { clearConvention, fetchKnowledgeSummary, type KnowledgeSignal } from "../api";
import { RangeStepper } from "./RangeStepper";

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
  // For conventions, `conventions` holds only the CURRENT page of rows
  // (the server paginates); `convTotal` is the count matching the active
  // filter and `convMax` the project-wide max distinct_task_count (slider
  // bound). Statistics stays fully client-side.
  const [conventions, setConventions] = useState<EntityConvention[] | null>(null);
  const [convTotal, setConvTotal] = useState<number | null>(null);
  const [convMax, setConvMax] = useState(1);
  const [stats, setStats] = useState<EntityStatsItem[] | null>(null);
  const [statsTotal, setStatsTotal] = useState<number | null>(null);
  const [loadedAt, setLoadedAt] = useState<{ conventions: Date | null; statistics: Date | null }>(
    { conventions: null, statistics: null },
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  // Minimum distinct-task vote count for the conventions table. Drops the
  // long tail of low-evidence rows so the table mirrors the injection gate.
  const [minCount, setMinCount] = useState(0);
  // Debounced copy of (filter, minCount) that actually drives server queries,
  // so dragging the slider / typing doesn't fire a request per keystroke.
  const [query, setQuery] = useState({ filter: "", minCount: 0 });
  const [page, setPage] = useState(0);
  const [unsetting, setUnsetting] = useState<string | null>(null);
  // Spans cleared via Unset this session. We keep the row in place (greyed +
  // struck through) instead of removing it, so the table never reflows under
  // the operator. The marks reset on the next data (re)load.
  const [cleared, setCleared] = useState<Set<string>>(new Set());
  // Change-signal: `baseline` is the (count, latest_updated_at) fingerprint
  // captured when each subtab was last loaded; `live` is the latest polled
  // value. When they diverge for the active subtab, the background pipeline
  // has written new conventions/stats since you loaded, so we light up the
  // Refresh button — without auto-refetching the heavy paginated table.
  type SigPair = { conventions: KnowledgeSignal | null; statistics: KnowledgeSignal | null };
  const [baseline, setBaseline] = useState<SigPair>({ conventions: null, statistics: null });
  const [live, setLive] = useState<SigPair>({ conventions: null, statistics: null });

  async function handleUnset(span: string | null) {
    if (!projectId || !span) return;
    // No confirm dialog, no re-fetch: drop the row only AFTER the API
    // confirms the delete, so a failed call leaves the row visible with
    // an error rather than silently re-appearing on next refresh.
    setUnsetting(span);
    setError(null);
    try {
      await clearConvention(projectId, span, storeKey);
      // Don't re-fetch and don't remove the row: either one moves the table
      // the operator is reading. Just mark this span cleared in place; the
      // row stays put, greyed and struck through, until the next Refresh.
      setCleared((prev) => {
        const next = new Set(prev);
        next.add(span);
        return next;
      });
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
    const params = new URLSearchParams();
    params.set("project", projectId);
    if (storeKey) params.set("store", storeKey);
    // Both subtabs are server-paginated: push limit/offset/search into SQL.
    params.set("limit", String(PAGE_SIZE));
    params.set("offset", String(page * PAGE_SIZE));
    if (query.filter.trim()) params.set("q", query.filter.trim());
    let url: string;
    if (which === "conventions") {
      if (query.minCount > 0) params.set("min_count", String(query.minCount));
      url = `/api/conventions?${params.toString()}`;
    } else {
      url = `/api/entity-statistics?${params.toString()}`;
    }
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => {
        if (which === "conventions") {
          setConventions(d.conventions ?? []);
          setConvTotal(d.total ?? 0);
          setConvMax(Math.max(1, d.max_count ?? 0));
          // Fresh page: cleared spans are gone server-side, drop the marks.
          setCleared(new Set());
        } else {
          setStats(d.items ?? []);
          setStatsTotal(d.total ?? 0);
        }
        setLoadedAt((prev) => ({ ...prev, [which]: new Date() }));
        // Snapshot the project-level fingerprint at load time so the staleness
        // delta is filter-independent (it counts what the project gained since
        // you loaded, regardless of the active search/min_count). Reset both
        // baseline and live for this subtab so it reads "fresh" right away.
        fetchKnowledgeSummary(projectId, storeKey)
          .then((sig) => {
            setBaseline((prev) => ({ ...prev, [which]: sig[which] }));
            setLive((prev) => ({ ...prev, [which]: sig[which] }));
          })
          .catch(() => {});
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }

  // Debounce filter / slider into the server `query` (300ms).
  useEffect(() => {
    const t = setTimeout(() => setQuery({ filter, minCount }), 300);
    return () => clearTimeout(t);
  }, [filter, minCount]);

  // Reset to first page whenever the server query or active tab changes.
  useEffect(() => { setPage(0); }, [query, subtab]);

  // Both subtabs are server-paginated: re-fetch the active one on
  // project/store/page/query change.
  useEffect(() => {
    fetchData(subtab);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, storeKey, subtab, page, query]);

  // Poll only the cheap change-signal (NOT the heavy paginated tables) every
  // 15s, so an operator who keeps the panel open learns the pipeline wrote new
  // data without us re-sorting/re-paginating the table under them.
  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    const poll = () => {
      fetchKnowledgeSummary(projectId, storeKey)
        .then((sig) => { if (!cancelled) setLive(sig); })
        .catch(() => {});
    };
    const timer = setInterval(poll, 15000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [projectId, storeKey]);

  if (!projectId) {
    return (
      <section className="runtime-panel" aria-label="Entity Knowledge">
        <p className="runtime-muted">Select a project first.</p>
      </section>
    );
  }

  // Both tables are filtered + paginated server-side, so `conventions` and
  // `stats` already hold just the current page. The slider's upper bound
  // (`convMax`) and the pager totals (`convTotal` / `statsTotal`) come from
  // the API.
  const activeLoadedAt =
    subtab === "conventions" ? loadedAt.conventions : loadedAt.statistics;

  // Stale = the live fingerprint has moved past the snapshot taken when this
  // subtab last loaded (new rows inserted → count up; in-place edits → newer
  // updated_at). `newCount` is the positive count delta, shown as "+N new".
  const activeBaseline = subtab === "conventions" ? baseline.conventions : baseline.statistics;
  const activeLive = subtab === "conventions" ? live.conventions : live.statistics;
  const stale =
    !!activeBaseline &&
    !!activeLive &&
    (activeLive.count !== activeBaseline.count ||
      activeLive.latest_updated_at !== activeBaseline.latest_updated_at);
  const newCount = stale && activeLive && activeBaseline ? activeLive.count - activeBaseline.count : 0;

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
          title={
            stale
              ? "后台已写入新数据,点击加载最新(自动每 15s 检测,不会打乱当前表格)"
              : undefined
          }
          style={
            stale
              ? { boxShadow: "0 0 0 2px var(--accent, #2563eb)", fontWeight: 600 }
              : undefined
          }
        >
          {loading
            ? "Loading…"
            : stale
              ? newCount > 0
                ? `Refresh · +${newCount} new`
                : "Refresh · updated"
              : "Refresh"}
        </button>
      </div>

      <nav className="sub-tabs" aria-label="Knowledge tabs" role="tablist">
        <button
          className={subtab === "conventions" ? "sub-tab selected" : "sub-tab"}
          role="tab"
          aria-selected={subtab === "conventions"}
          type="button"
          onClick={() => setSubtab("conventions")}
        >
          Conventions ({convTotal ?? "…"})
        </button>
        <button
          className={subtab === "statistics" ? "sub-tab selected" : "sub-tab"}
          role="tab"
          aria-selected={subtab === "statistics"}
          type="button"
          onClick={() => setSubtab("statistics")}
        >
          Statistics ({statsTotal ?? "…"})
        </button>
      </nav>

      {subtab === "conventions" ? (
        <div style={FORMULA_BLOCK_STYLE}>
          <strong>Conventions table</strong> — high-trust dictionary injected
          into future annotator/QC prompts. <strong>Source</strong>:
          annotator+QC consensus that agreed with project statistics, plus
          HR-authored decisions. <em>Excludes arbiter-only decisions</em> to
          avoid the cascade where one LLM error reinforces itself.{" "}
          <strong>Injection criteria</strong>: <code>distinct tasks ≥ 5</code>{" "}
          (the <strong>Tasks</strong> column) and span length{" "}
          <code>≥ 4</code> chars. Use the{" "}
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

      <div
        style={{
          margin: "0.4rem 0",
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "0.75rem",
        }}
      >
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
        {subtab === "conventions" ? (
          <label
            style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.85rem" }}
            title="Hide conventions with fewer than this many distinct accepted-task votes. The injection gate requires ≥ 5."
          >
            <span className="runtime-muted">Min tasks</span>
            <RangeStepper
              min={0}
              max={convMax}
              step={1}
              value={Math.min(minCount, convMax)}
              onChange={setMinCount}
              width="160px"
              disabled={!conventions}
            />
            <code style={{ fontFamily: "monospace", minWidth: "2.5rem" }}>
              ≥ {minCount}
            </code>
          </label>
        ) : null}
        {(filter || minCount > 0) && convTotal != null && subtab === "conventions" ? (
          <span className="runtime-muted" style={{ fontSize: "0.85rem" }}>
            {convTotal} match{convTotal === 1 ? "" : "es"}
          </span>
        ) : null}
        {filter && statsTotal != null && subtab === "statistics" ? (
          <span className="runtime-muted" style={{ fontSize: "0.85rem" }}>
            {statsTotal} match{statsTotal === 1 ? "" : "es"}
          </span>
        ) : null}
      </div>

      {error ? <div className="notice compact">{error}</div> : null}
      {loading ? <p className="runtime-muted">Loading…</p> : null}

      {!loading && subtab === "conventions" && conventions ? (
        <>
        <Pagination
          total={convTotal ?? 0}
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
                <th style={{ padding: "0.4rem 0.75rem" }} title="Distinct accepted tasks voting for this type (injection gate keys off this)">
                  Tasks
                </th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Evidence</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Updated</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {conventions.map((c) => {
                const isCleared = c.span ? cleared.has(c.span) : false;
                return (
                <tr
                  key={c.convention_id}
                  style={{
                    ...TR,
                    ...(isCleared
                      ? { opacity: 0.45, textDecoration: "line-through" }
                      : null),
                  }}
                >
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
                  <td style={{ padding: "0.4rem 0.75rem" }}>{c.distinct_task_count ?? 0}</td>
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
                  <td style={{ padding: "0.4rem 0.75rem", textDecoration: "none" }}>
                    <button
                      type="button"
                      disabled={!c.span || unsetting === c.span || isCleared}
                      onClick={() => handleUnset(c.span)}
                      title={
                        isCleared
                          ? "Cleared. It will disappear on the next Refresh."
                          : "Remove this convention from the project. Future tasks won't see this rule injected. Already-modified task annotations are NOT reverted."
                      }
                      style={{
                        fontSize: "0.75rem",
                        color: isCleared
                          ? "var(--muted, #6b7280)"
                          : c.span
                            ? "var(--danger, #b91c1c)"
                            : undefined,
                        background: "transparent",
                        border: "1px solid var(--border, #d1d5db)",
                        padding: "2px 8px",
                        borderRadius: "3px",
                        cursor: c.span && !isCleared ? "pointer" : "not-allowed",
                        opacity: unsetting === c.span ? 0.6 : 1,
                        textDecoration: "none",
                      }}
                    >
                      {unsetting === c.span ? "…" : isCleared ? "Cleared" : "Unset"}
                    </button>
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <Pagination
          total={convTotal ?? 0}
          page={page}
          pageSize={PAGE_SIZE}
          onPageChange={setPage}
        />
        </>
      ) : null}

      {!loading && subtab === "statistics" && stats ? (
        <>
        <Pagination
          total={statsTotal ?? 0}
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
              {stats.map((s) => (
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
          total={statsTotal ?? 0}
          page={page}
          pageSize={PAGE_SIZE}
          onPageChange={setPage}
        />
        </>
      ) : null}
    </section>
  );
}


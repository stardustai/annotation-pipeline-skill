import React, { useEffect, useMemo, useState } from "react";
import type { PosteriorAudit, TaskDeviation, ContestedSpan } from "../types";
import {
  DistributionBar,
  OriginalTextCell,
  Pagination,
  TopNTypeSelector,
  NOT_ENTITY,
} from "../entityHelpers";

const PAGE_SIZE = 100;

export type PosteriorAuditPanelProps = {
  projectId: string | null;
  storeKey?: string | null;
  onSendToHr: (taskId: string) => Promise<void> | void;
  onDeclareCanonical: (span: string, entityType: string) => Promise<void> | void;
};

type CacheResponse = {
  cached: boolean;
  payload: PosteriorAudit | null;
  generated_at: string | null;
  cached_accepted_hash: string | null;
  current_accepted_hash: string;
  stale: boolean;
};

type Subtab = "deviations" | "contested";

const TABLE_STYLE: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: "0.85rem",
};
const TH_FIRST: React.CSSProperties = { padding: "0.4rem 0.75rem 0.4rem 0" };
const TH: React.CSSProperties = { padding: "0.4rem 0.75rem" };
const THEAD_ROW: React.CSSProperties = {
  borderBottom: "1px solid var(--border, #2a2f3a)",
  textAlign: "left",
};
const TR: React.CSSProperties = { borderBottom: "1px solid var(--border, #2a2f3a)" };
const TD: React.CSSProperties = { padding: "0.4rem 0.75rem" };
const TD_MONO: React.CSSProperties = {
  padding: "0.4rem 0.75rem 0.4rem 0",
  fontFamily: "monospace",
};

const FORMULA_BLOCK_STYLE: React.CSSProperties = {
  background: "#f5f7fa",
  fontSize: "0.85em",
  padding: "0.5rem 0.75rem",
  margin: "0.5rem 0",
  borderRadius: "4px",
};

function urlWithStore(base: string, projectId: string, storeKey: string | null | undefined): string {
  const storeQ = storeKey ? `&store=${encodeURIComponent(storeKey)}` : "";
  return `${base}?project=${encodeURIComponent(projectId)}${storeQ}`;
}

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 19) + " UTC";
}

export function PosteriorAuditPanel({
  projectId,
  storeKey = null,
  onSendToHr,
  onDeclareCanonical,
}: PosteriorAuditPanelProps): React.ReactElement {
  const [cache, setCache] = useState<CacheResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [subtab, setSubtab] = useState<Subtab>("deviations");

  useEffect(() => {
    if (!projectId) {
      setCache(null);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(urlWithStore("/api/posterior-audit", projectId, storeKey))
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => setCache(d))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [projectId, storeKey]);

  function reloadCache() {
    if (!projectId) return;
    fetch(urlWithStore("/api/posterior-audit", projectId, storeKey))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d) setCache(d); })
      .catch(() => {});
  }

  async function handleCheck() {
    if (!projectId) {
      setError("Select a project first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(urlWithStore("/api/posterior-audit", projectId, storeKey), {
        method: "POST",
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setCache(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const payload = cache?.payload ?? null;
  const deviations = payload?.task_deviations ?? [];
  const contested = payload?.contested_spans ?? [];
  const stale = cache?.stale ?? false;
  const generatedAt = cache?.generated_at ?? null;
  const cachedExists = cache?.cached ?? false;

  return (
    <section className="runtime-panel posterior-audit-panel" aria-label="Posterior audit">
      {/* Header: title + last-checked status inline, plus Check button. */}
      <div
        className="runtime-header"
        style={{
          borderLeft: cachedExists
            ? stale
              ? "3px solid var(--warning, #d97706)"
              : "3px solid var(--success, #047857)"
            : "3px solid var(--border, #d1d5db)",
          paddingLeft: "0.75rem",
          background: stale ? "#fff4e0" : undefined,
        }}
      >
        <div>
          <h2 style={{ marginBottom: "0.25rem" }}>Posterior Audit</h2>
          <p style={{ marginTop: 0, fontSize: "0.85rem" }}>
            Scan accepted tasks against current project statistics.
            {cachedExists ? (
              <>
                {" · "}
                <strong>Last checked:</strong> {fmtTime(generatedAt)}
                {" — "}
                {stale ? (
                  <span style={{ color: "var(--warning, #d97706)", fontWeight: 600 }}>
                    ACCEPTED data has changed since this scan{" "}
                    <span className="runtime-muted" style={{ fontSize: "0.8rem", fontWeight: 400 }}>
                      ({cache?.cached_accepted_hash} → {cache?.current_accepted_hash})
                    </span>
                  </span>
                ) : (
                  <span className="runtime-muted">in sync with current ACCEPTED data</span>
                )}
              </>
            ) : (
              <span className="runtime-muted"> · no cached scan yet</span>
            )}
          </p>
        </div>
        <button
          className="primary-button"
          type="button"
          onClick={handleCheck}
          disabled={loading || !projectId}
          style={
            stale
              ? { background: "var(--warning, #d97706)", borderColor: "var(--warning, #d97706)" }
              : undefined
          }
        >
          {loading ? "Checking…" : stale ? "Re-check (stale)" : cachedExists ? "Re-check" : "Check"}
        </button>
      </div>

      {error ? <div className="notice compact">{error}</div> : null}

      {!cachedExists && !loading ? (
        <p className="runtime-muted">
          No cached scan yet. Click <strong>Check</strong> to run the first scan.
        </p>
      ) : null}

      {cachedExists &&
       deviations.length === 0 &&
       contested.length === 0 ? (
        <p className="runtime-muted">
          All accepted tasks agree with current statistics; no contested spans.
        </p>
      ) : null}

      {cachedExists && (deviations.length > 0 || contested.length > 0) ? (
        <>
          <nav
            className="view-tabs"
            aria-label="Audit sections"
            style={{ marginTop: "0.5rem", marginBottom: 0 }}
          >
            <button
              className={subtab === "deviations" ? "view-tab selected" : "view-tab"}
              type="button"
              onClick={() => setSubtab("deviations")}
            >
              Task deviations ({deviations.length})
            </button>
            <button
              className={subtab === "contested" ? "view-tab selected" : "view-tab"}
              type="button"
              onClick={() => setSubtab("contested")}
            >
              Contested spans ({contested.length})
            </button>
          </nav>

          {subtab === "deviations" ? (
            <>
              <div style={FORMULA_BLOCK_STYLE}>
                <strong>Task deviation rule</strong> — an annotation disagrees with project
                history. Triggered when the span has at least <code>10</code> prior observations
                <em> and </em>one type accounts for <code>≥ 80%</code> of them (settled)
                <em> and </em>the task's current type is <em>not</em> that dominant type.
                <br />
                <span style={{ color: "var(--muted, #6b7280)" }}>
                  Formula:{" "}
                  <code>total ≥ 10  ∧  max_share ≥ 0.80  ∧  proposed_type ≠ dominant_type</code>
                </span>
              </div>
              {deviations.length > 0 ? (
                <DeviationsTable
                  items={deviations}
                  projectId={projectId!}
                  storeKey={storeKey ?? null}
                  onSendToHr={onSendToHr}
                  onAfterFix={() => {
                    // Re-fetch the (server-side surgically updated) cache
                    // so the list reflects the just-applied fix.
                    if (!projectId) return;
                    fetch(urlWithStore("/api/posterior-audit", projectId, storeKey))
                      .then((r) => (r.ok ? r.json() : null))
                      .then((d) => { if (d) setCache(d); })
                      .catch(() => {});
                  }}
                />
              ) : (
                <p className="runtime-muted">No task deviations.</p>
              )}
            </>
          ) : null}

          {subtab === "contested" ? (
            <>
              <div style={FORMULA_BLOCK_STYLE}>
                <strong>Contested span rule</strong> — the project itself is split on a span's
                type. Triggered when the span has at least <code>10</code> total observations
                <em> and </em>no type reaches the <code>80%</code> dominance threshold
                <em> and </em>the runner-up type holds <code>≥ 20%</code> share.
                <br />
                <span style={{ color: "var(--muted, #6b7280)" }}>
                  Formula:{" "}
                  <code>total ≥ 10  ∧  top_share &lt; 0.80  ∧  runner_up_share ≥ 0.20</code>
                </span>
              </div>
              {contested.length > 0 ? (
                <ContestedTable
                  items={contested}
                  projectId={projectId!}
                  storeKey={storeKey ?? null}
                  onDeclare={onDeclareCanonical}
                  onAfterRetroactiveFix={reloadCache}
                />
              ) : (
                <p className="runtime-muted">No contested spans.</p>
              )}
            </>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

type DedupedDeviation = TaskDeviation & { occurrences: number };

function DeviationsTable({
  items,
  projectId,
  storeKey,
  onSendToHr,
  onAfterFix,
}: {
  items: TaskDeviation[];
  projectId: string;
  storeKey: string | null;
  onSendToHr: (taskId: string) => Promise<void> | void;
  onAfterFix: () => void;
}): React.ReactElement {
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);
  // Per-row picked type (null = nothing picked yet)
  const [picked, setPicked] = useState<Record<string, string | null>>({});
  // Per-row status: "submitting" | "submitted" | "error: ..." | undefined
  const [rowStatus, setRowStatus] = useState<Record<string, string>>({});
  // Per-row "Save as project convention" checkbox — default ON; when off
  // the fix patches this task only and doesn't promote to a project rule.
  const [saveConv, setSaveConv] = useState<Record<string, boolean>>({});
  const getSaveConv = (key: string) => saveConv[key] ?? true;

  // Dedupe: backend may emit multiple rows for the same (task, span) when
  // the span occurs in multiple input rows of one task. The operator's
  // fix targets the whole task at once, so collapse and count.
  const deduped = useMemo<DedupedDeviation[]>(() => {
    const byKey = new Map<string, DedupedDeviation>();
    for (const d of items) {
      const k = `${d.task_id}|${d.span}|${d.current_type}`;
      const ex = byKey.get(k);
      if (ex) ex.occurrences += 1;
      else byKey.set(k, { ...d, occurrences: 1 });
    }
    return Array.from(byKey.values());
  }, [items]);

  const lower = filter.trim().toLowerCase();
  const filtered = lower
    ? deduped.filter(
        (d) =>
          d.task_id.toLowerCase().includes(lower) ||
          d.span.toLowerCase().includes(lower) ||
          d.current_type.toLowerCase().includes(lower) ||
          d.prior_dominant_type.toLowerCase().includes(lower),
      )
    : deduped;
  useEffect(() => { setPage(0); }, [filter, items]);
  const visible = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  async function submitFix(d: DedupedDeviation, newType: string | null) {
    const rowKey = `${d.task_id}|${d.span}|${d.current_type}`;
    setRowStatus((s) => ({ ...s, [rowKey]: "submitting" }));
    try {
      const storeQ = storeKey ? `?store=${encodeURIComponent(storeKey)}` : "";
      const body = JSON.stringify({
        span: d.span,
        current_type: d.current_type,
        new_type: newType, // null or "not_an_entity" = delete from entities
        actor: "posterior_audit_submit",
        save_as_convention: getSaveConv(rowKey),
      });
      const r = await fetch(`/api/tasks/${encodeURIComponent(d.task_id)}/posterior-fix${storeQ}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
      });
      if (!r.ok) {
        const txt = await r.text();
        throw new Error(`HTTP ${r.status}: ${txt.slice(0, 200)}`);
      }
      setRowStatus((s) => ({
        ...s,
        [rowKey]: `submitted: ${newType ?? "deleted"}`,
      }));
      // Re-pull cache so the row disappears from the list and the total
      // count updates. Server has already removed this deviation from
      // the cached payload (cache surgery in apply_posterior_fix path).
      onAfterFix();
    } catch (e) {
      setRowStatus((s) => ({
        ...s,
        [rowKey]: `error: ${e instanceof Error ? e.message : String(e)}`,
      }));
    }
  }

  return (
    <>
      <div style={{ margin: "0.4rem 0" }}>
        <input
          type="search"
          placeholder="Filter task / span / type…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ width: "min(360px, 60%)" }}
        />
        <span className="runtime-muted" style={{ marginLeft: "0.75rem", fontSize: "0.85rem" }}>
          {filter ? `${filtered.length} of ${deduped.length}` : `${deduped.length} unique`}
          {items.length !== deduped.length
            ? ` (${items.length} raw rows deduped)`
            : ""}
        </span>
      </div>
      <Pagination
        total={filtered.length}
        page={page}
        pageSize={PAGE_SIZE}
        onPageChange={setPage}
      />
      <div className="runtime-card">
        <table style={TABLE_STYLE}>
          <thead>
            <tr style={THEAD_ROW}>
              <th style={TH_FIRST}>Task</th>
              <th style={TH}>Span</th>
              <th style={{ ...TH, width: "24%" }}>Original text</th>
              <th style={TH}>Current</th>
              <th style={{ ...TH, width: "14%" }}>Distribution</th>
              <th style={{ ...TH, width: "26%" }}>Set type</th>
              <th style={TH}></th>
            </tr>
          </thead>
          <tbody>
            {visible.map((d) => {
              const rowKey = `${d.task_id}|${d.span}|${d.current_type}`;
              const sel = picked[rowKey] ?? null;
              const status = rowStatus[rowKey];
              const isSubmitting = status === "submitting";
              const isSubmitted = status?.startsWith("submitted");
              const err = status?.startsWith("error:") ? status.slice(7) : null;
              // Top-3 preferred: prior dominant, then current type (red),
              // then runner-up if any.
              const distEntries = Object.entries(d.prior_distribution).sort(
                (a, b) => b[1] - a[1],
              );
              const runnerUp = distEntries[1]?.[0];
              const preferred = [
                d.prior_dominant_type,
                d.current_type,
                runnerUp ?? "",
              ].filter(Boolean);
              return (
                <tr key={rowKey} style={TR}>
                  <td style={TD_MONO}>
                    {d.task_id}
                    {d.occurrences > 1 ? (
                      <span className="runtime-muted" style={{ marginLeft: "0.3rem", fontSize: "0.75rem" }}>
                        ×{d.occurrences}
                      </span>
                    ) : null}
                  </td>
                  <td style={{ ...TD, fontFamily: "monospace" }}>{d.span}</td>
                  <td style={TD}>
                    <OriginalTextCell
                      projectId={projectId}
                      storeKey={storeKey}
                      taskId={d.task_id}
                      span={d.span}
                    />
                  </td>
                  <td style={TD}>
                    <span style={{ color: "var(--danger, #b91c1c)" }}>{d.current_type}</span>
                  </td>
                  <td style={TD}>
                    <DistributionBar distribution={d.prior_distribution} total={d.prior_total} />
                  </td>
                  <td style={TD}>
                    {isSubmitted ? (
                      <span style={{ fontSize: "0.8rem", color: "var(--success, #047857)" }}>
                        ✓ {status?.replace("submitted: ", "applied: ")}
                      </span>
                    ) : (
                      <TopNTypeSelector
                        selected={sel}
                        preferredOrder={preferred}
                        topN={3}
                        onSelect={(t) =>
                          setPicked((p) => ({ ...p, [rowKey]: t }))
                        }
                      />
                    )}
                  </td>
                  <td style={{ padding: "0.4rem 0", whiteSpace: "nowrap" }}>
                    {isSubmitted ? null : (
                      <>
                        <button
                          type="button"
                          disabled={!sel || isSubmitting}
                          onClick={() =>
                            submitFix(d, sel === NOT_ENTITY ? null : sel)
                          }
                          title={
                            sel
                              ? `Apply ${sel} as the corrected type for this task's span (operator-authored, HR weight 5×)${getSaveConv(rowKey) ? "; also saved as project convention" : "; one-off fix, NOT saved as convention"}`
                              : "Pick a type first"
                          }
                          style={{
                            fontSize: "0.8rem",
                            marginRight: "0.4rem",
                            background: sel ? "var(--success, #047857)" : undefined,
                            color: sel ? "white" : undefined,
                            opacity: !sel || isSubmitting ? 0.6 : 1,
                          }}
                        >
                          {isSubmitting ? "…" : "Submit"}
                        </button>
                        <button
                          type="button"
                          onClick={() => onSendToHr(d.task_id)}
                          style={{ fontSize: "0.8rem" }}
                          title="Route this task to HR queue without applying a fix"
                        >
                          Send to HR
                        </button>
                        <label
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "4px",
                            marginLeft: "0.6rem",
                            fontSize: "0.75rem",
                            color: "#4a5660",
                            cursor: "pointer",
                            userSelect: "none",
                          }}
                          title="When checked, the pick is saved as a project-wide convention. Uncheck for a one-off task fix."
                        >
                          <input
                            type="checkbox"
                            checked={getSaveConv(rowKey)}
                            disabled={isSubmitting}
                            onChange={(e) => {
                              const checked = e.target.checked;
                              setSaveConv((s) => ({ ...s, [rowKey]: checked }));
                            }}
                            style={{ margin: 0, cursor: "pointer" }}
                          />
                          <span>Save as convention</span>
                        </label>
                      </>
                    )}
                    {err ? (
                      <p style={{ margin: "0.3rem 0 0", color: "var(--danger, #b91c1c)", fontSize: "0.75rem" }}>
                        {err}
                      </p>
                    ) : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Pagination
        total={filtered.length}
        page={page}
        pageSize={PAGE_SIZE}
        onPageChange={setPage}
      />
    </>
  );
}

function ContestedTable({
  items,
  projectId,
  storeKey,
  onDeclare,
  onAfterRetroactiveFix,
}: {
  items: ContestedSpan[];
  projectId: string;
  storeKey: string | null;
  onDeclare: (span: string, entityType: string) => Promise<void> | void;
  onAfterRetroactiveFix?: () => void;
}): React.ReactElement {
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);
  // Per-row picked (pending confirm); per-row committed (after confirm).
  const [picked, setPicked] = useState<Record<string, string | null>>({});
  const [committed, setCommitted] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [retroResult, setRetroResult] = useState<
    Record<string, { fixed: number; skipped: number; errors: number }>
  >({});
  const [error, setError] = useState<string | null>(null);

  const lower = filter.trim().toLowerCase();
  const filtered = lower
    ? items.filter(
        (c) =>
          c.span.toLowerCase().includes(lower) ||
          Object.keys(c.prior_distribution).some((t) => t.toLowerCase().includes(lower)),
      )
    : items;
  useEffect(() => { setPage(0); }, [filter, items]);
  const visible = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  async function handleConfirm(span: string, type: string) {
    setSubmitting(span);
    setError(null);
    try {
      // Translate the UI's "NOT_ENTITY" sentinel to the backend constant.
      await onDeclare(span, type);
      setCommitted((prev) => ({ ...prev, [span]: type }));
      setPicked((prev) => ({ ...prev, [span]: null }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(null);
    }
  }

  async function handleConfirmAndApplyAll(span: string, type: string) {
    setSubmitting(span);
    setError(null);
    try {
      const storeQ = storeKey ? `?store=${encodeURIComponent(storeKey)}` : "";
      const r = await fetch(`/api/posterior-audit/retroactive-fix${storeQ}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          project_id: projectId,
          span,
          entity_type: type === NOT_ENTITY ? "not_an_entity" : type,
          actor: "posterior_audit_retroactive_ui",
        }),
      });
      if (!r.ok) {
        const txt = await r.text();
        throw new Error(`HTTP ${r.status}: ${txt.slice(0, 200)}`);
      }
      const data = (await r.json()) as {
        fixed: number;
        skipped: number;
        errors: { task_id: string; reason: string }[];
      };
      setCommitted((prev) => ({ ...prev, [span]: type }));
      setPicked((prev) => ({ ...prev, [span]: null }));
      setRetroResult((prev) => ({
        ...prev,
        [span]: {
          fixed: data.fixed,
          skipped: data.skipped,
          errors: data.errors?.length ?? 0,
        },
      }));
      onAfterRetroactiveFix?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(null);
    }
  }

  return (
    <>
      <div style={{ margin: "0.4rem 0" }}>
        <input
          type="search"
          placeholder="Filter span / type…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ width: "min(360px, 60%)" }}
        />
        <span className="runtime-muted" style={{ marginLeft: "0.75rem", fontSize: "0.85rem" }}>
          {filter ? `${filtered.length} of ${items.length}` : `${items.length} total`}
        </span>
      </div>
      {error ? <div className="notice compact">{error}</div> : null}
      <Pagination
        total={filtered.length}
        page={page}
        pageSize={PAGE_SIZE}
        onPageChange={setPage}
      />
      <div className="runtime-card">
        <table style={TABLE_STYLE}>
          <thead>
            <tr style={THEAD_ROW}>
              <th style={TH_FIRST}>Span</th>
              <th style={TH}>Total</th>
              <th style={{ ...TH, width: "24%" }}>Sample text</th>
              <th style={{ ...TH, width: "14%" }}>Distribution</th>
              <th style={{ ...TH, width: "32%" }}>Set Convention</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((c) => {
              const pickedType = picked[c.span] ?? null;
              const committedType = committed[c.span] ?? c.resolved_convention_type;
              const isSubmitting = submitting === c.span;
              // Top-3 from observed distribution + canonical fallbacks.
              const distEntries = Object.entries(c.prior_distribution).sort((a, b) => b[1] - a[1]);
              const preferred = distEntries.slice(0, 3).map(([t]) => t);
              return (
                <tr key={c.span} style={TR}>
                  <td style={{ ...TD_MONO }}>{c.span}</td>
                  <td style={TD}>{c.prior_total}</td>
                  <td style={TD}>
                    <OriginalTextCell
                      projectId={projectId}
                      storeKey={storeKey}
                      span={c.span}
                    />
                  </td>
                  <td style={TD}>
                    <DistributionBar distribution={c.prior_distribution} total={c.prior_total} />
                  </td>
                  <td style={TD}>
                    {committedType ? (
                      <div style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
                        <span style={{ fontSize: "0.8rem", color: "var(--success, #047857)" }}>
                          ✓ set convention:{" "}
                          <strong>
                            {committedType === NOT_ENTITY ? "🚫 not entity" : committedType}
                          </strong>
                        </span>
                        {retroResult[c.span] ? (
                          <span style={{ fontSize: "0.75rem", color: "var(--muted, #4b5563)" }}>
                            retroactively fixed {retroResult[c.span].fixed} task(s)
                            {retroResult[c.span].skipped > 0
                              ? `, skipped ${retroResult[c.span].skipped}`
                              : ""}
                            {retroResult[c.span].errors > 0
                              ? `, ${retroResult[c.span].errors} error(s)`
                              : ""}
                          </span>
                        ) : null}
                      </div>
                    ) : (
                      <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                        <TopNTypeSelector
                          selected={pickedType}
                          preferredOrder={preferred}
                          topN={3}
                          onSelect={(t) => setPicked((p) => ({ ...p, [c.span]: t }))}
                        />
                        <div style={{ display: "flex", gap: "0.4rem", alignItems: "center", flexWrap: "wrap" }}>
                          <button
                            type="button"
                            disabled={!pickedType || isSubmitting}
                            onClick={() => pickedType && handleConfirm(c.span, pickedType)}
                            title="Save as project convention only — future tasks; existing ACCEPTED tasks are NOT changed"
                            style={{
                              fontSize: "0.8rem",
                              background: pickedType ? "var(--success, #047857)" : undefined,
                              color: pickedType ? "white" : undefined,
                              opacity: !pickedType || isSubmitting ? 0.6 : 1,
                            }}
                          >
                            {isSubmitting ? "…" : "Confirm"}
                          </button>
                          <button
                            type="button"
                            disabled={!pickedType || isSubmitting}
                            onClick={() => pickedType && handleConfirmAndApplyAll(c.span, pickedType)}
                            title={
                              pickedType
                                ? `Declare convention AND retroactively patch every existing ACCEPTED task in the project that has '${c.span}' tagged differently`
                                : "Pick a type first"
                            }
                            style={{
                              fontSize: "0.8rem",
                              background: pickedType ? "var(--primary, #1e40af)" : undefined,
                              color: pickedType ? "white" : undefined,
                              opacity: !pickedType || isSubmitting ? 0.6 : 1,
                            }}
                          >
                            {isSubmitting ? "…" : "Confirm & apply to all"}
                          </button>
                          {pickedType ? (
                            <span className="runtime-muted" style={{ fontSize: "0.75rem" }}>
                              will set <strong>{c.span}</strong> →{" "}
                              <strong>{pickedType === NOT_ENTITY ? "not entity" : pickedType}</strong>
                            </span>
                          ) : null}
                        </div>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Pagination
        total={filtered.length}
        page={page}
        pageSize={PAGE_SIZE}
        onPageChange={setPage}
      />
    </>
  );
}

// Re-export so TaskDrawer or other panels can use the constant.
export { NOT_ENTITY };

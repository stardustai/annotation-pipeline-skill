import React, { useEffect, useMemo, useState } from "react";
import type { PosteriorAudit, TaskDeviation, ContestedSpan } from "../types";
import {
  DistributionBar,
  OriginalTextCell,
  Pagination,
  TopNTypeSelector,
  NOT_ENTITY,
} from "../entityHelpers";
import { clearConvention } from "../api";

const PAGE_SIZE = 100;
// Each Contested-spans row fires an OriginalTextCell fetch for its
// sample text. With PAGE_SIZE=100 that's 100 backend hits per page —
// noticeably slow. Contested gets its own smaller page size; the
// Deviations table is per-task (less work per row, fewer rows in
// typical projects) and keeps the default.
const CONTESTED_PAGE_SIZE = 20;

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
  // Parse the (UTC-tagged) backend timestamp and render in the browser's
  // local timezone with a stable YYYY-MM-DD HH:mm:ss TZ format. The
  // backend writes ISO 8601 with explicit offset (e.g. "+00:00"), so
  // `new Date(iso)` parses correctly without ambiguity.
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  const yyyy = d.getFullYear();
  const mm = pad(d.getMonth() + 1);
  const dd = pad(d.getDate());
  const hh = pad(d.getHours());
  const mi = pad(d.getMinutes());
  const ss = pad(d.getSeconds());
  // Pull the timezone abbreviation from Intl so the user sees their
  // actual zone (e.g. PDT / GMT+8) instead of a misleading "UTC".
  let tz = "";
  try {
    const parts = new Intl.DateTimeFormat(undefined, { timeZoneName: "short" }).formatToParts(d);
    tz = parts.find((p) => p.type === "timeZoneName")?.value ?? "";
  } catch {
    tz = "";
  }
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}${tz ? " " + tz : ""}`;
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
  const [filter, setFilter] = useState("");
  // Default ON: Contested-spans Apply-to-all also writes a project
  // convention. Operator can flip off (per-session) when they want a
  // bulk task fix WITHOUT introducing a project rule (e.g. context-
  // specific span where the type only applies in this corpus).
  const [saveAsConvention, setSaveAsConvention] = useState(true);

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
          All accepted tasks agree with current statistics; no divergent annotations.
        </p>
      ) : null}

      {cachedExists && (deviations.length > 0 || contested.length > 0) ? (
        <>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.75rem",
              marginTop: "0.5rem",
              flexWrap: "wrap",
            }}
          >
            <nav className="view-tabs" aria-label="Audit sections" style={{ margin: 0 }}>
              <button
                className={subtab === "deviations" ? "view-tab selected" : "view-tab"}
                type="button"
                onClick={() => { setSubtab("deviations"); setFilter(""); }}
              >
                Task deviations ({deviations.length})
              </button>
              <button
                className={subtab === "contested" ? "view-tab selected" : "view-tab"}
                type="button"
                onClick={() => { setSubtab("contested"); setFilter(""); }}
              >
                Divergent annotations ({contested.length})
              </button>
            </nav>
            <input
              type="search"
              placeholder={
                subtab === "deviations" ? "Filter task / span / type…" : "Filter span / type…"
              }
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              style={{ width: "min(300px, 40%)", marginLeft: "auto" }}
            />
            {subtab === "contested" ? (
              <label
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "5px",
                  fontSize: "0.8rem",
                  color: "#4a5660",
                  cursor: "pointer",
                  userSelect: "none",
                }}
                title="Shows the Confirm-convention button per row. Apply-to-all NEVER writes a convention regardless of this checkbox — it only patches matching task annotations. Confirm and Apply are independent actions; click both if you want both effects."
              >
                <input
                  type="checkbox"
                  checked={saveAsConvention}
                  onChange={(e) => setSaveAsConvention(e.target.checked)}
                  style={{ margin: 0, cursor: "pointer" }}
                />
                <span>Save as convention</span>
              </label>
            ) : null}
          </div>

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
                  externalFilter={filter}
                  onAfterFix={() => {
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
                <strong>Divergent annotations rule</strong> — the project itself is split on a span's
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
                  externalFilter={filter}
                  saveAsConvention={saveAsConvention}
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
  externalFilter,
}: {
  items: TaskDeviation[];
  projectId: string;
  storeKey: string | null;
  onSendToHr: (taskId: string) => Promise<void> | void;
  onAfterFix: () => void;
  externalFilter?: string;
}): React.ReactElement {
  const filter = externalFilter ?? "";
  const [page, setPage] = useState(0);
  // Per-row picked type (null = nothing picked yet)
  const [picked, setPicked] = useState<Record<string, string | null>>({});
  // Per-row status: "submitting" | "submitted" | "error: ..." | undefined
  const [rowStatus, setRowStatus] = useState<Record<string, string>>({});
  // Per-row "Save as project convention" checkbox — default ON; when off
  // the fix patches this task only and doesn't promote to a project rule.
  const [saveConv, setSaveConv] = useState<Record<string, boolean>>({});
  const getSaveConv = (key: string) => saveConv[key] ?? true;
  // Apply-to-all flow: confirmation dialog + per-span progress + final
  // summary. Keyed by span (not row key) since one apply-to-all affects
  // every row sharing the same span.
  const [pendingApplyAll, setPendingApplyAll] = useState<
    | { span: string; type: string; otherCount: number; totalCount: number; sharePct: number }
    | null
  >(null);
  const [retroProgress, setRetroProgress] = useState<
    Record<string, { processed: number; total: number }>
  >({});
  const [retroResult, setRetroResult] = useState<
    Record<string, { fixed: number; skipped: number; errors: number }>
  >({});
  const [applyingAllSpan, setApplyingAllSpan] = useState<string | null>(null);

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
  // Filter change resets to page 0 (the new filter might yield fewer
  // rows than the current page covers). Items-only changes (after a
  // fix, Apply-to-all, Unset, etc.) clamp instead — preserve scroll
  // position by staying on the same page if it's still in range.
  useEffect(() => { setPage(0); }, [filter]);
  useEffect(() => {
    const maxPage = Math.max(0, Math.ceil(filtered.length / PAGE_SIZE) - 1);
    if (page > maxPage) setPage(maxPage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtered.length]);
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

  function requestApplyAll(d: DedupedDeviation, type: string) {
    const otherCount = type === NOT_ENTITY
      ? d.prior_total
      : d.prior_total - (d.prior_distribution[type] || 0);
    if (otherCount <= 0) return;
    const sharePct = d.prior_total > 0
      ? Math.round((otherCount / d.prior_total) * 100)
      : 0;
    setPendingApplyAll({
      span: d.span,
      type,
      otherCount,
      totalCount: d.prior_total,
      sharePct,
    });
  }

  async function runApplyAllSweep(span: string, type: string, expectedTotal: number) {
    setApplyingAllSpan(span);
    setRetroProgress((prev) => ({ ...prev, [span]: { processed: 0, total: expectedTotal } }));
    let totalFixed = 0;
    let totalSkipped = 0;
    let totalErrors = 0;
    try {
      const storeQ = storeKey ? `?store=${encodeURIComponent(storeKey)}` : "";
      const baseBody = {
        project_id: projectId,
        span,
        entity_type: type === NOT_ENTITY ? "not_an_entity" : type,
        actor: "posterior_audit_retroactive_ui",
      };
      // Step 1: scan + first batch. Server returns full candidate list
      // so subsequent polls can skip the O(N_accepted_tasks) scan.
      const BATCH = 10;
      const initialResp = await fetch(`/api/posterior-audit/retroactive-fix${storeQ}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...baseBody, batch_size: BATCH }),
      });
      if (!initialResp.ok) {
        const txt = await initialResp.text();
        throw new Error(`HTTP ${initialResp.status}: ${txt.slice(0, 200)}`);
      }
      const initialData = (await initialResp.json()) as {
        fixed: number;
        skipped: number;
        errors: { task_id: string; reason: string }[];
        remaining: number;
        done: boolean;
        candidate_task_ids: string[] | null;
      };
      totalFixed += initialData.fixed;
      totalSkipped += initialData.skipped;
      totalErrors += initialData.errors?.length ?? 0;
      const allCandidates = initialData.candidate_task_ids ?? [];
      const serverTotal = allCandidates.length;
      setRetroProgress((prev) => ({
        ...prev,
        [span]: { processed: initialData.fixed + initialData.errors.length, total: serverTotal },
      }));
      // Server already processed the first BATCH of the candidate list,
      // so we skip those when iterating.
      let cursor = initialData.fixed + initialData.errors.length;
      while (cursor < allCandidates.length) {
        const slice = allCandidates.slice(cursor, cursor + BATCH);
        const r = await fetch(`/api/posterior-audit/retroactive-fix${storeQ}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ...baseBody, task_ids: slice, batch_size: BATCH }),
        });
        if (!r.ok) {
          const txt = await r.text();
          throw new Error(`HTTP ${r.status}: ${txt.slice(0, 200)}`);
        }
        const data = (await r.json()) as {
          fixed: number; skipped: number;
          errors: { task_id: string; reason: string }[];
        };
        totalFixed += data.fixed;
        totalSkipped += data.skipped;
        totalErrors += data.errors?.length ?? 0;
        cursor += slice.length;
        setRetroProgress((prev) => ({
          ...prev,
          [span]: { processed: cursor, total: serverTotal },
        }));
      }
      setRetroResult((prev) => ({
        ...prev,
        [span]: { fixed: totalFixed, skipped: totalSkipped, errors: totalErrors },
      }));
      // Recount entity_statistics so contested-spans classification
      // reflects current task state. Best-effort.
      try {
        await fetch(`/api/entity-statistics/recount${storeQ}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ project_id: projectId, span }),
        });
      } catch {
        // best-effort
      }
      onAfterFix();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setRowStatus((s) => {
        const next = { ...s };
        for (const r of deduped) {
          if (r.span === span) {
            const k = `${r.task_id}|${r.span}|${r.current_type}`;
            next[k] = `error: ${msg}`;
          }
        }
        return next;
      });
    } finally {
      setApplyingAllSpan(null);
      setRetroProgress((prev) => {
        const next = { ...prev };
        delete next[span];
        return next;
      });
    }
  }

  async function handleApplyAllConfirm() {
    if (!pendingApplyAll) return;
    const { span, type, otherCount } = pendingApplyAll;
    setPendingApplyAll(null);
    await runApplyAllSweep(span, type, otherCount);
  }

  async function handleUnsetDeviationConvention(span: string, rowKey: string) {
    if (!window.confirm(
      `Unset the project convention for '${span}'? Future tasks won't see this rule. The task's annotation that this Submit patched is NOT reverted.`,
    )) {
      return;
    }
    setRowStatus((s) => ({ ...s, [rowKey]: "submitting" }));
    try {
      await clearConvention(projectId, span, storeKey);
      // Drop the per-row "submitted" status so the row reverts to its
      // initial Type-picker state. Also drop the apply-to-all summary
      // (since the convention behind it is gone).
      setRowStatus((s) => {
        const next = { ...s };
        delete next[rowKey];
        return next;
      });
      setRetroResult((prev) => {
        const next = { ...prev };
        delete next[span];
        return next;
      });
      onAfterFix();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setRowStatus((s) => ({ ...s, [rowKey]: `error: ${msg}` }));
    }
  }

  return (
    <>
      {pendingApplyAll ? (
        <div
          role="dialog"
          aria-modal="true"
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.4)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
          }}
          onClick={() => setPendingApplyAll(null)}
        >
          <div
            style={{
              background: "white",
              borderRadius: "6px",
              padding: "1.25rem 1.5rem",
              maxWidth: "520px",
              boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ marginTop: 0, marginBottom: "0.75rem", fontSize: "1rem" }}>
              Confirm bulk retroactive fix
            </h3>
            <p style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", lineHeight: 1.5 }}>
              Apply <strong>'{pendingApplyAll.span}'</strong> → type{" "}
              <strong>
                {pendingApplyAll.type === NOT_ENTITY
                  ? "🚫 not_an_entity (delete)"
                  : pendingApplyAll.type}
              </strong>{" "}
              to all <strong>{pendingApplyAll.otherCount}</strong> occurrence(s) currently
              tagged differently ({pendingApplyAll.sharePct}% of {pendingApplyAll.totalCount} total)?
            </p>
            <p style={{ margin: "0 0 1rem", fontSize: "0.8rem", color: "var(--muted, #6b7280)" }}>
              This will declare a project convention AND rewrite each task's
              annotation. Original annotation_result preserved for audit.
            </p>
            <div style={{ display: "flex", gap: "0.5rem", justifyContent: "flex-end" }}>
              <button
                type="button"
                onClick={() => setPendingApplyAll(null)}
                style={{ fontSize: "0.85rem" }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleApplyAllConfirm}
                style={{
                  fontSize: "0.85rem",
                  background: "var(--primary, #1e40af)",
                  color: "white",
                }}
              >
                Apply to {pendingApplyAll.otherCount}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      <div style={{ margin: "0.4rem 0", fontSize: "0.8rem", color: "var(--muted, #6b7280)" }}>
        {filter ? `${filtered.length} of ${deduped.length}` : `${deduped.length} unique`}
        {items.length !== deduped.length
          ? ` (${items.length} raw rows deduped)`
          : ""}
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
              <th style={{ ...TH_FIRST, width: "11%", fontSize: "0.8rem" }}>Task</th>
              <th style={{ ...TH, width: "8%", fontSize: "0.8rem" }}>Span</th>
              <th style={{ ...TH, width: "18%", fontSize: "0.8rem" }}>Original text</th>
              <th
                style={{ ...TH, width: "16%", fontSize: "0.8rem" }}
                title="Current task type (red) + project-wide prior distribution"
              >
                Current / Distribution
              </th>
              <th style={{ ...TH, width: "20%" }}>Set type</th>
              <th
                style={{ ...TH, width: "27%" }}
                title="Save: patch THIS task only. Apply to all: patch every ACCEPTED task that has this span tagged differently."
              >
                Actions
              </th>
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
                  <td style={{ ...TD_MONO, fontSize: "0.75rem" }}>
                    {d.task_id}
                    {d.occurrences > 1 ? (
                      <span className="runtime-muted" style={{ marginLeft: "0.3rem", fontSize: "0.7rem" }}>
                        ×{d.occurrences}
                      </span>
                    ) : null}
                  </td>
                  <td style={{ ...TD, fontFamily: "monospace", fontSize: "0.78rem" }}>{d.span}</td>
                  <td style={{ ...TD, fontSize: "0.78rem" }}>
                    <OriginalTextCell
                      projectId={projectId}
                      storeKey={storeKey}
                      taskId={d.task_id}
                      span={d.span}
                    />
                  </td>
                  <td style={TD}>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                      <span style={{ fontSize: "0.75rem", color: "var(--danger, #b91c1c)", fontWeight: 500 }}>
                        {d.current_type}
                      </span>
                      <DistributionBar distribution={d.prior_distribution} total={d.prior_total} />
                    </div>
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
                  <td style={{ padding: "0.4rem 0.5rem", verticalAlign: "top" }}>
                    {isSubmitted ? (
                      <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem", alignItems: "flex-start" }}>
                        {getSaveConv(rowKey) ? (
                          <button
                            type="button"
                            onClick={() => handleUnsetDeviationConvention(d.span, rowKey)}
                            title="Remove the project convention. The task's annotation is already patched and is NOT reverted."
                            style={{
                              fontSize: "0.7rem",
                              color: "var(--danger, #b91c1c)",
                              background: "transparent",
                              border: "1px solid var(--border, #d1d5db)",
                              padding: "1px 6px",
                              borderRadius: "3px",
                              cursor: "pointer",
                            }}
                          >
                            Unset convention
                          </button>
                        ) : null}
                        {retroResult[d.span] ? (
                          <span style={{ fontSize: "0.7rem", color: "var(--muted, #4b5563)" }}>
                            (last sweep) fixed {retroResult[d.span].fixed} task(s)
                          </span>
                        ) : null}
                      </div>
                    ) : retroProgress[d.span] ? (
                      // Active sweep — render progress bar in the action cell.
                      (() => {
                        const { processed, total } = retroProgress[d.span];
                        const pct = total > 0 ? Math.round((processed / total) * 100) : 0;
                        return (
                          <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                            <div style={{ fontSize: "0.75rem", color: "var(--muted, #4b5563)" }}>
                              {processed} / {total} — {pct}%
                            </div>
                            <div
                              style={{
                                width: "100%",
                                height: "6px",
                                background: "var(--border, #e5e7eb)",
                                borderRadius: "3px",
                                overflow: "hidden",
                              }}
                            >
                              <div
                                style={{
                                  width: `${pct}%`,
                                  height: "100%",
                                  background: "var(--primary, #1e40af)",
                                  transition: "width 0.2s ease-out",
                                }}
                              />
                            </div>
                          </div>
                        );
                      })()
                    ) : (
                      <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
                        <div style={{ display: "flex", gap: "0.4rem", alignItems: "center", flexWrap: "wrap" }}>
                          <button
                            type="button"
                            disabled={!sel || isSubmitting || applyingAllSpan !== null}
                            onClick={() =>
                              submitFix(d, sel === NOT_ENTITY ? null : sel)
                            }
                            title={
                              sel
                                ? `Patch THIS task's '${d.span}' → ${sel === NOT_ENTITY ? "delete" : sel}${getSaveConv(rowKey) ? "; also saves as project convention" : "; one-off task fix, no convention"}`
                                : "Pick a type first"
                            }
                            style={{
                              fontSize: "0.8rem",
                              background: sel ? "var(--success, #047857)" : undefined,
                              color: sel ? "white" : undefined,
                              opacity: !sel || isSubmitting ? 0.6 : 1,
                            }}
                          >
                            {isSubmitting ? "…" : "Save"}
                          </button>
                          {(() => {
                            const otherCount = sel
                              ? (sel === NOT_ENTITY
                                  ? d.prior_total
                                  : d.prior_total - (d.prior_distribution[sel] || 0))
                              : 0;
                            const noun = otherCount === 1 ? "occurrence" : "occurrences";
                            const label = sel
                              ? `Apply to other ${otherCount} ${noun}`
                              : "Apply to all";
                            const disabled = !sel || otherCount === 0 || isSubmitting || applyingAllSpan !== null;
                            return (
                              <button
                                type="button"
                                disabled={disabled}
                                onClick={() => sel && requestApplyAll(d, sel)}
                                title={
                                  sel
                                    ? `Declare '${d.span}' → ${sel === NOT_ENTITY ? "not_an_entity" : sel} as project convention AND retroactively patch ${otherCount} other ACCEPTED task annotation(s)`
                                    : "Pick a type first"
                                }
                                style={{
                                  fontSize: "0.8rem",
                                  background: !disabled ? "var(--primary, #1e40af)" : undefined,
                                  color: !disabled ? "white" : undefined,
                                  opacity: disabled ? 0.6 : 1,
                                }}
                              >
                                {label}
                              </button>
                            );
                          })()}
                        </div>
                        <label
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "4px",
                            fontSize: "0.7rem",
                            color: "#4a5660",
                            cursor: "pointer",
                            userSelect: "none",
                          }}
                          title="When checked, the Save / Apply action also writes a project-wide convention. Uncheck to fix THIS task only without touching the convention table."
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
                      </div>
                    )}
                    {err ? (
                      <p style={{ margin: "0.3rem 0 0", color: "var(--danger, #b91c1c)", fontSize: "0.7rem" }}>
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
  externalFilter,
  saveAsConvention,
}: {
  items: ContestedSpan[];
  projectId: string;
  storeKey: string | null;
  onDeclare: (span: string, entityType: string) => Promise<void> | void;
  onAfterRetroactiveFix?: () => void;
  externalFilter?: string;
  saveAsConvention?: boolean;
}): React.ReactElement {
  const filter = externalFilter ?? "";
  const saveConv = saveAsConvention ?? true;
  const [page, setPage] = useState(0);
  // Per-row picked (pending confirm); per-row committed (after confirm).
  const [picked, setPicked] = useState<Record<string, string | null>>({});
  const [committed, setCommitted] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [retroResult, setRetroResult] = useState<
    Record<string, { fixed: number; skipped: number; errors: number }>
  >({});
  // Per-row live progress while the retroactive sweep runs: {processed, total}.
  // Rendered as a progress bar inside the Apply-to-all cell.
  const [retroProgress, setRetroProgress] = useState<
    Record<string, { processed: number; total: number }>
  >({});
  // Pending confirmation dialog: which (span, type, otherCount, totalCount)
  // are we asking the operator to confirm before kicking off the sweep?
  const [pendingConfirm, setPendingConfirm] = useState<
    | { span: string; type: string; otherCount: number; totalCount: number; sharePct: number }
    | null
  >(null);
  const [error, setError] = useState<string | null>(null);

  const lower = filter.trim().toLowerCase();
  const filtered = lower
    ? items.filter(
        (c) =>
          c.span.toLowerCase().includes(lower) ||
          Object.keys(c.prior_distribution).some((t) => t.toLowerCase().includes(lower)),
      )
    : items;
  // Filter change resets to page 0; items-only changes (after Unset
  // or Apply-to-all) clamp to stay on the same page when possible.
  useEffect(() => { setPage(0); }, [filter]);
  useEffect(() => {
    const maxPage = Math.max(0, Math.ceil(filtered.length / CONTESTED_PAGE_SIZE) - 1);
    if (page > maxPage) setPage(maxPage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtered.length]);
  const visible = filtered.slice(page * CONTESTED_PAGE_SIZE, (page + 1) * CONTESTED_PAGE_SIZE);

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

  // Open the confirmation dialog with the impact numbers; actual work
  // starts in runRetroactiveSweep after the operator confirms.
  function requestApplyToAll(c: ContestedSpan, type: string) {
    const otherCount = type === NOT_ENTITY
      ? c.prior_total
      : c.prior_total - (c.prior_distribution[type] || 0);
    if (otherCount <= 0) return;
    const sharePct = c.prior_total > 0
      ? Math.round((otherCount / c.prior_total) * 100)
      : 0;
    setPendingConfirm({
      span: c.span,
      type,
      otherCount,
      totalCount: c.prior_total,
      sharePct,
    });
  }

  // Two-phase sweep: first call scans + processes one batch and returns
  // the full candidate list; subsequent calls pass slices of that list
  // back so the server skips the O(N_accepted_tasks) scan. Otherwise
  // every poll re-reads every ACCEPTED task's latest annotation, which
  // adds 30-60s of overhead per poll on a project with thousands of
  // accepted tasks.
  async function runRetroactiveSweep(span: string, type: string, expectedTotal: number) {
    setSubmitting(span);
    setError(null);
    setRetroProgress((prev) => ({ ...prev, [span]: { processed: 0, total: expectedTotal } }));
    let totalFixed = 0;
    let totalSkipped = 0;
    let totalErrors = 0;
    try {
      const storeQ = storeKey ? `?store=${encodeURIComponent(storeKey)}` : "";
      const baseBody = {
        project_id: projectId,
        span,
        entity_type: type === NOT_ENTITY ? "not_an_entity" : type,
        actor: "posterior_audit_retroactive_ui",
        // Apply-to-all NEVER writes the project convention — convention
        // write is a separate action via the Confirm button. Decoupling
        // the two lets the operator patch tasks without committing to a
        // project rule (and vice versa).
        set_convention: false,
      };
      const BATCH = 10;
      const initialResp = await fetch(`/api/posterior-audit/retroactive-fix${storeQ}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...baseBody, batch_size: BATCH }),
      });
      if (!initialResp.ok) {
        const txt = await initialResp.text();
        throw new Error(`HTTP ${initialResp.status}: ${txt.slice(0, 200)}`);
      }
      const initialData = (await initialResp.json()) as {
        fixed: number;
        skipped: number;
        errors: { task_id: string; reason: string }[];
        remaining: number;
        done: boolean;
        candidate_task_ids: string[] | null;
      };
      totalFixed += initialData.fixed;
      totalSkipped += initialData.skipped;
      totalErrors += initialData.errors?.length ?? 0;
      const allCandidates = initialData.candidate_task_ids ?? [];
      const serverTotal = allCandidates.length;
      setRetroProgress((prev) => ({
        ...prev,
        [span]: { processed: initialData.fixed + initialData.errors.length, total: serverTotal },
      }));
      let cursor = initialData.fixed + initialData.errors.length;
      while (cursor < allCandidates.length) {
        const slice = allCandidates.slice(cursor, cursor + BATCH);
        const r = await fetch(`/api/posterior-audit/retroactive-fix${storeQ}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ...baseBody, task_ids: slice, batch_size: BATCH }),
        });
        if (!r.ok) {
          const txt = await r.text();
          throw new Error(`HTTP ${r.status}: ${txt.slice(0, 200)}`);
        }
        const data = (await r.json()) as {
          fixed: number; skipped: number;
          errors: { task_id: string; reason: string }[];
        };
        totalFixed += data.fixed;
        totalSkipped += data.skipped;
        totalErrors += data.errors?.length ?? 0;
        cursor += slice.length;
        setRetroProgress((prev) => ({
          ...prev,
          [span]: { processed: cursor, total: serverTotal },
        }));
      }
      setCommitted((prev) => ({ ...prev, [span]: type }));
      setPicked((prev) => ({ ...prev, [span]: null }));
      setRetroResult((prev) => ({
        ...prev,
        [span]: { fixed: totalFixed, skipped: totalSkipped, errors: totalErrors },
      }));
      // Recount entity_statistics for this span so the contested-spans
      // classification reflects current task state, not historical vote
      // accumulation. Best-effort: failures don't surface to the user.
      try {
        await fetch(`/api/entity-statistics/recount${storeQ}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ project_id: projectId, span }),
        });
      } catch {
        // best-effort
      }
      onAfterRetroactiveFix?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(null);
      setRetroProgress((prev) => {
        const next = { ...prev };
        delete next[span];
        return next;
      });
    }
  }

  async function handleConfirmDialog() {
    if (!pendingConfirm) return;
    const { span, type, otherCount } = pendingConfirm;
    setPendingConfirm(null);
    await runRetroactiveSweep(span, type, otherCount);
  }

  async function handleUnsetConvention(span: string) {
    if (!window.confirm(
      `Unset the project convention for '${span}'? Future tasks won't see this rule. Already-modified task annotations are NOT reverted.`,
    )) {
      return;
    }
    setSubmitting(span);
    setError(null);
    try {
      await clearConvention(projectId, span, storeKey);
      // Drop the committed/result entries so the row reverts to its
      // "pick a type" state.
      setCommitted((prev) => {
        const next = { ...prev };
        delete next[span];
        return next;
      });
      setRetroResult((prev) => {
        const next = { ...prev };
        delete next[span];
        return next;
      });
      onAfterRetroactiveFix?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(null);
    }
  }

  return (
    <>
      {pendingConfirm ? (
        <div
          role="dialog"
          aria-modal="true"
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.4)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
          }}
          onClick={() => setPendingConfirm(null)}
        >
          <div
            style={{
              background: "white",
              borderRadius: "6px",
              padding: "1.25rem 1.5rem",
              maxWidth: "520px",
              boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ marginTop: 0, marginBottom: "0.75rem", fontSize: "1rem" }}>
              Confirm bulk retroactive fix
            </h3>
            <p style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", lineHeight: 1.5 }}>
              Apply <strong>'{pendingConfirm.span}'</strong> → type{" "}
              <strong>
                {pendingConfirm.type === NOT_ENTITY ? "🚫 not_an_entity (delete)" : pendingConfirm.type}
              </strong>{" "}
              to all{" "}
              <strong>{pendingConfirm.otherCount}</strong> occurrence(s) currently tagged
              differently ({pendingConfirm.sharePct}% of {pendingConfirm.totalCount} total)?
            </p>
            <p style={{ margin: "0 0 1rem", fontSize: "0.8rem", color: "var(--muted, #6b7280)" }}>
              This will declare a project convention AND rewrite each task's
              annotation (writing new human_review_answer artifacts; original
              annotation_result preserved for audit).
            </p>
            <div style={{ display: "flex", gap: "0.5rem", justifyContent: "flex-end" }}>
              <button
                type="button"
                onClick={() => setPendingConfirm(null)}
                style={{ fontSize: "0.85rem" }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmDialog}
                style={{
                  fontSize: "0.85rem",
                  background: "var(--primary, #1e40af)",
                  color: "white",
                }}
              >
                Apply to {pendingConfirm.otherCount}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      <div style={{ margin: "0.4rem 0", fontSize: "0.8rem", color: "var(--muted, #6b7280)" }}>
        {filter ? `${filtered.length} of ${items.length}` : `${items.length} total`}
      </div>
      {error ? <div className="notice compact">{error}</div> : null}
      <Pagination
        total={filtered.length}
        page={page}
        pageSize={CONTESTED_PAGE_SIZE}
        onPageChange={setPage}
      />
      <div className="runtime-card">
        <table style={TABLE_STYLE}>
          <thead>
            <tr style={THEAD_ROW}>
              <th style={TH_FIRST}>Span</th>
              <th style={TH}>Total</th>
              <th style={{ ...TH, width: "22%" }}>Sample text</th>
              <th style={{ ...TH, width: "12%" }}>Distribution</th>
              <th style={{ ...TH, width: "22%" }}>Set Convention</th>
              <th style={{ ...TH, width: "18%" }} title="Declare convention AND retroactively patch every existing ACCEPTED task that tagged this span differently">
                Apply to all
              </th>
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
                      <div style={{ display: "flex", flexDirection: "column", gap: "0.2rem", alignItems: "flex-start" }}>
                        <span style={{ fontSize: "0.8rem", color: "var(--success, #047857)" }}>
                          ✓ set:{" "}
                          <strong>
                            {committedType === NOT_ENTITY ? "🚫 not entity" : committedType}
                          </strong>
                        </span>
                        <button
                          type="button"
                          disabled={submitting === c.span}
                          onClick={() => handleUnsetConvention(c.span)}
                          title="Remove this convention from the project. Future tasks won't see this rule. Already-modified task annotations (via Apply to all) are NOT reverted."
                          style={{
                            fontSize: "0.7rem",
                            color: "var(--danger, #b91c1c)",
                            background: "transparent",
                            border: "1px solid var(--border, #d1d5db)",
                            padding: "1px 6px",
                            borderRadius: "3px",
                            cursor: "pointer",
                            opacity: submitting === c.span ? 0.6 : 1,
                          }}
                        >
                          Unset
                        </button>
                      </div>
                    ) : (
                      <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                        <TopNTypeSelector
                          selected={pickedType}
                          preferredOrder={preferred}
                          topN={3}
                          onSelect={(t) => setPicked((p) => ({ ...p, [c.span]: t }))}
                        />
                        {saveConv ? (
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
                              alignSelf: "flex-start",
                            }}
                          >
                            {isSubmitting ? "…" : "Confirm"}
                          </button>
                        ) : (
                          <span style={{ fontSize: "0.7rem", color: "var(--muted, #6b7280)", fontStyle: "italic" }}>
                            (convention save off — use Apply to all to patch tasks)
                          </span>
                        )}
                      </div>
                    )}
                  </td>
                  <td style={TD}>
                    {retroResult[c.span] ? (
                      // Retroactive fix already ran — show the summary.
                      <span style={{ fontSize: "0.75rem", color: "var(--muted, #4b5563)" }}>
                        fixed {retroResult[c.span].fixed} task(s)
                        {retroResult[c.span].skipped > 0
                          ? `, skipped ${retroResult[c.span].skipped}`
                          : ""}
                        {retroResult[c.span].errors > 0
                          ? `, ${retroResult[c.span].errors} error(s)`
                          : ""}
                      </span>
                    ) : retroProgress[c.span] ? (
                      // Active sweep — render progress bar.
                      (() => {
                        const { processed, total } = retroProgress[c.span];
                        const pct = total > 0 ? Math.round((processed / total) * 100) : 0;
                        return (
                          <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                            <div style={{ fontSize: "0.75rem", color: "var(--muted, #4b5563)" }}>
                              {processed} / {total} task(s) — {pct}%
                            </div>
                            <div
                              style={{
                                width: "100%",
                                height: "6px",
                                background: "var(--border, #e5e7eb)",
                                borderRadius: "3px",
                                overflow: "hidden",
                              }}
                            >
                              <div
                                style={{
                                  width: `${pct}%`,
                                  height: "100%",
                                  background: "var(--primary, #1e40af)",
                                  transition: "width 0.2s ease-out",
                                }}
                              />
                            </div>
                          </div>
                        );
                      })()
                    ) : (() => {
                      // Decoupled from convention write: Apply-to-all
                      // ONLY patches task annotations now. Operator uses
                      // the Confirm button (in the Set Convention column)
                      // separately to write the project rule. This lets
                      // each side be exercised independently.
                      const triggerType = committedType ?? pickedType;
                      const otherCount = triggerType
                        ? (triggerType === NOT_ENTITY
                            ? c.prior_total
                            : c.prior_total - (c.prior_distribution[triggerType] || 0))
                        : 0;
                      const noun = otherCount === 1 ? "occurrence" : "occurrences";
                      const label = triggerType
                        ? `Apply to other ${otherCount} ${noun}`
                        : "Apply to all";
                      const title = triggerType
                        ? `Patch ${otherCount} ACCEPTED task annotation(s) where '${c.span}' is tagged as something other than ${triggerType === NOT_ENTITY ? "not_an_entity" : triggerType}. Does NOT write a project convention — use the Confirm button for that.`
                        : "Pick a type in the Set Convention column first";
                      const disabled = !triggerType || otherCount === 0 || isSubmitting;
                      return (
                        <button
                          type="button"
                          disabled={disabled}
                          onClick={() => triggerType && requestApplyToAll(c, triggerType)}
                          title={title}
                          style={{
                            fontSize: "0.8rem",
                            background: !disabled ? "var(--primary, #1e40af)" : undefined,
                            color: !disabled ? "white" : undefined,
                            opacity: disabled ? 0.6 : 1,
                          }}
                        >
                          {isSubmitting ? "…" : label}
                        </button>
                      );
                    })()}
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
        pageSize={CONTESTED_PAGE_SIZE}
        onPageChange={setPage}
      />
    </>
  );
}

// Re-export so TaskDrawer or other panels can use the constant.
export { NOT_ENTITY };

import React, { lazy, Suspense, useEffect, useMemo, useState } from "react";

// Lazy-loaded: ScatterSubTab brings in plotly.js-dist-min (~3 MB / 1.47 MB
// gzip). Off the critical path of the Distribution tab so opening the
// Duplicates view is instant; plotly only ships when the operator actually
// switches to the Scatter plot sub-tab.
const ScatterSubTab = lazy(() => import("./ScatterSubTab"));
import { TypeStatisticsPanel } from "./TypeStatisticsPanel";
import { RangeStepper } from "./RangeStepper";

// ── Types ────────────────────────────────────────────────────────────────────

type ClusterEntry = {
  cluster_id: string;
  task_ids: string[];
  method: string;
  similarity: number;
};

export type CoordEntry = {
  task_id: string;
  x: number;
  y: number;
  status: string;
  cluster_id: string | null;
  text_preview: string;
};

type DistributionPayload = {
  params: Record<string, unknown>;
  task_count: number;
  clusters: ClusterEntry[];
  coords: CoordEntry[];
};

type CacheResponse = {
  cached: boolean;
  payload: DistributionPayload | null;
  generated_at: string | null;
  cached_content_hash: string | null;
  current_content_hash: string;
  stale: boolean;
  available_profiles: string[];
};

// ── Row-dedup types ───────────────────────────────────────────────────────────

export type RowMember = {
  task_id: string;
  row_index: number;
  text_preview: string;
  // New: rows that are already masked in row_masks are kept in the
  // cluster output (so the operator sees what they've already
  // handled) but tagged with masked=true so the UI can render a
  // "masked" badge and disable per-row actions.
  // Optional for backward-compat with older cached payloads — those
  // had no masked field and use the `currently_masked` overlay below.
  masked?: boolean;
  // Per-member direct similarity (Jaccard for MinHash, cosine for
  // embedding) to the cluster's rep. The frontend slider filters
  // members in real time on this field: drag up to hide loosely-
  // related members; drag down (within scan-time threshold) to
  // reveal them again. Backward-compat: missing field is treated
  // as 1.0 so old caches render every member.
  sim_to_rep?: number;
  word_count?: number;
};

export type RowCluster = {
  cluster_id: string;
  members: RowMember[];
  similarity: number;
  method: string;
};

export type RowDedupPayload = {
  params: {
    profile: string;
    provider: string;
    model: string;
    jaccard_threshold: number;
    metric: string;
    statuses: string[] | null;
    generated_at: string;
    embedding_cache: { hits: number; misses: number };
  };
  clusters: RowCluster[];
  row_count: number;
  task_count: number;
};

export type RowDedupCacheResponse = {
  cached: boolean;
  payload: RowDedupPayload | null;
  generated_at: string | null;
  cached_content_hash: string | null;
  current_content_hash: string;
  stale: boolean;
  available_profiles: string[];
  // Tuples of [task_id, row_index] for rows that currently have an active
  // mask AND appear in the cached cluster payload. The scan filters out
  // pre-masked rows, but masks applied *after* the scan still show up in
  // the cached clusters as un-masked — we use this overlay to tag them
  // visually without forcing an expensive re-scan.
  currently_masked?: [string, number][];
};

// "duplicates" used to mean task-level (concat all rows, embed once, cluster
// tasks) — but in practice the rows of a single task vary enough by IDs /
// dates / numbers that real templates land in 0.05-0.10 Jaccard territory
// and aren't found. Row-level dedup catches the real cases, so the
// task-level UI was deleted; "duplicates" now IS row-level. The
// /api/distribution endpoints are still wired for the Scatter plot.
// "statistics" hosts the type-frequency panel that used to be its own tab.
type Subtab = "duplicates" | "scatter" | "statistics";

export type DistributionPanelProps = {
  projectId: string | null;
  storeKey?: string | null;
  onSelectTask?: (taskId: string) => void;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function buildDistributionUrl(
  projectId: string,
  storeKey: string | null | undefined,
  profile: string,
): string {
  const storeQ = storeKey ? `&store=${encodeURIComponent(storeKey)}` : "";
  return `/api/distribution?project=${encodeURIComponent(projectId)}&profile=${encodeURIComponent(profile)}${storeQ}`;
}

function buildStoreQs(storeKey: string | null | undefined): string {
  return storeKey ? `?store=${encodeURIComponent(storeKey)}` : "";
}

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  const yyyy = d.getFullYear();
  const mm = pad(d.getMonth() + 1);
  const dd = pad(d.getDate());
  const hh = pad(d.getHours());
  const mi = pad(d.getMinutes());
  const ss = pad(d.getSeconds());
  let tz = "";
  try {
    const parts = new Intl.DateTimeFormat(undefined, { timeZoneName: "short" }).formatToParts(d);
    tz = parts.find((p) => p.type === "timeZoneName")?.value ?? "";
  } catch {
    tz = "";
  }
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}${tz ? " " + tz : ""}`;
}

/**
 * Pick the representative row member for a row-dedup cluster.
 *
 * Sort key matches the backend: (str(task_id), int(row_index)).
 * We prefer the lex-smallest UN-MASKED member as the rep so a
 * masked row doesn't end up with the "rep" badge. If every member
 * is masked (rare — a fully-already-handled cluster), fall back to
 * the absolute lex-smallest.
 */
function pickRowRep(members: RowMember[]): RowMember {
  const sorted = [...members].sort((a, b) => {
    if (a.task_id < b.task_id) return -1;
    if (a.task_id > b.task_id) return 1;
    return a.row_index - b.row_index;
  });
  const firstUnmasked = sorted.find((m) => !m.masked);
  return firstUnmasked ?? sorted[0] ?? members[0];
}

// Word-aware truncation matching backend util.text.truncate_to_words.
// Each CJK ideograph / kana / hangul char counts as one word; whitespace-
// delimited Latin/number/punctuation runs count as one. Backend already
// truncates to 100 words server-side — this is a defensive client cap
// for legacy cache payloads that pre-date the backend change.
const WORD_TOKEN_RE = /[一-鿿぀-ヿ가-힯]|\S+/g;

function truncateToWords(text: string, maxWords: number): string {
  if (!text || maxWords <= 0) return text;
  let count = 0;
  let lastEnd = 0;
  for (const m of text.matchAll(WORD_TOKEN_RE)) {
    count++;
    lastEnd = (m.index ?? 0) + m[0].length;
    if (count >= maxWords) return text.slice(0, lastEnd);
  }
  return text;
}

// ── Main component ────────────────────────────────────────────────────────────

export function DistributionPanel({
  projectId,
  storeKey = null,
  onSelectTask,
}: DistributionPanelProps): React.ReactElement {
  const [cache, setCache] = useState<CacheResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [subtab, setSubtab] = useState<Subtab>("duplicates");

  // Default to "jina_small"; synced to available_profiles[0] after first GET.
  const [profile, setProfile] = useState<string>("jina_small");
  const [minClusterSize, setMinClusterSize] = useState<number>(5);

  // ── Auto-load cache on mount / when deps change ──────────────────────────
  useEffect(() => {
    if (!projectId) {
      setCache(null);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(buildDistributionUrl(projectId, storeKey, profile))
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<CacheResponse>;
      })
      .then((d) => {
        setCache(d);
        // Sync profile to first available if the current one isn't offered.
        if (d.available_profiles.length > 0 && !d.available_profiles.includes(profile)) {
          setProfile(d.available_profiles[0]);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, storeKey, profile]);

  // ── Scan (POST) ───────────────────────────────────────────────────────────
  async function handleScan() {
    if (!projectId) {
      setError("Select a project first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const storeQs = buildStoreQs(storeKey);
      const scanUrl = `/api/distribution/scan?project=${encodeURIComponent(projectId)}${storeQs ? "&" + storeQs.slice(1) : ""}`;
      const r = await fetch(scanUrl, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          profile,
          statuses: null,
          min_cluster_size: minClusterSize,
          umap_neighbors: 15,
          umap_min_dist: 0.1,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const launch = await r.json();
      const job = launch?.job;
      if (!job?.job_id) {
        setCache(launch as CacheResponse);
        return;
      }
      // Distribution scans can take minutes (UMAP + HDBSCAN on
      // thousands of embeddings). Poll every 3s, allow up to 15 min.
      const deadline = Date.now() + 15 * 60 * 1000;
      while (Date.now() < deadline) {
        await new Promise((res) => setTimeout(res, 3000));
        const jr = await fetch(`/api/jobs/${encodeURIComponent(job.job_id)}${storeQs}`);
        if (!jr.ok) continue;
        const jdata = await jr.json();
        if (jdata.status === "done") {
          const fresh = await fetch(buildDistributionUrl(projectId, storeKey, profile));
          if (fresh.ok) setCache((await fresh.json()) as CacheResponse);
          return;
        }
        if (jdata.status === "error") {
          throw new Error(jdata.error || "background scan failed");
        }
      }
      throw new Error("scan timed out (15 min)");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  // ── Derived state ─────────────────────────────────────────────────────────
  const payload = cache?.payload ?? null;
  const stale = cache?.stale ?? false;
  const generatedAt = cache?.generated_at ?? null;
  const cachedExists = cache?.cached ?? false;
  const availableProfiles = cache?.available_profiles ?? ["jina_small", "random_baseline"];

  // Row-dedup cluster count for the tab label — populated by RowDuplicatesSubTab
  // via a setState callback so the tab button can reflect live data.
  const [rowDedupClusterCount, setRowDedupClusterCount] = useState<number>(0);

  return (
    <section className="runtime-panel distribution-panel" aria-label="Distribution">
      {error ? <div className="notice compact">{error}</div> : null}

      {/* ── Sub-tab nav ──────────────────────────────────────────────── */}
      <nav className="sub-tabs" aria-label="Statistics sections" role="tablist">
          <button
            className={subtab === "duplicates" ? "sub-tab selected" : "sub-tab"}
            role="tab"
            aria-selected={subtab === "duplicates"}
            type="button"
            onClick={() => setSubtab("duplicates")}
          >
            Duplicates ({rowDedupClusterCount})
          </button>
          <button
            className={subtab === "scatter" ? "sub-tab selected" : "sub-tab"}
            role="tab"
            aria-selected={subtab === "scatter"}
            type="button"
            onClick={() => setSubtab("scatter")}
          >
            Scatter plot
          </button>
          <button
            className={subtab === "statistics" ? "sub-tab selected" : "sub-tab"}
            role="tab"
            aria-selected={subtab === "statistics"}
            type="button"
            onClick={() => setSubtab("statistics")}
          >
            Statistics
          </button>
      </nav>

      {/* ── Duplicates sub-tab (row-level) ──────────────────────────── */}
      {subtab === "duplicates" ? (
        <RowDuplicatesSubTab
          projectId={projectId}
          storeKey={storeKey}
          onSelectTask={onSelectTask}
          onClusterCountChange={setRowDedupClusterCount}
        />
      ) : null}

      {/* ── Scatter plot sub-tab (controls + lazy plotly) ───────────── */}
      {subtab === "scatter" ? (
        <div style={{ marginTop: "0.75rem" }}>
          {/* Scan controls live next to the plot for parallelism with the
              Duplicates sub-tab, which manages its own control bar. */}
          <div
            style={{
              display: "flex",
              gap: "1rem",
              alignItems: "center",
              flexWrap: "wrap",
              marginBottom: "0.75rem",
              padding: "0.6rem 0.75rem",
              background: "var(--surface2, #f8fafc)",
              borderRadius: "6px",
              border: "1px solid var(--border, #e5e7eb)",
            }}
          >
            <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.35rem" }}>
              Profile
              <select
                value={profile}
                onChange={(e) => setProfile(e.target.value)}
                style={{ fontSize: "0.85rem" }}
                disabled={loading}
              >
                {availableProfiles.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </label>
            <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.35rem" }}>
              Min cluster size
              <input
                type="number"
                min={2}
                max={100}
                value={minClusterSize}
                onChange={(e) => setMinClusterSize(Math.max(2, parseInt(e.target.value, 10) || 2))}
                style={{ width: "4.5rem", fontSize: "0.85rem" }}
                disabled={loading}
              />
            </label>
            {/* Re-Scan is disabled when a fresh cache exists (cache hash
                matches current dataset hash). UMAP+HDBSCAN is deterministic
                given the same inputs, so re-running with unchanged data just
                burns minutes. Stale → button re-enables with a warning color.
                Mouse-over the disabled button explains why. */}
            <button
              className="primary-button"
              type="button"
              onClick={handleScan}
              disabled={loading || !projectId || (cachedExists && !stale)}
              title={
                cachedExists && !stale
                  ? "Dataset hash unchanged since last scan — result would be identical. Disabled."
                  : stale
                    ? "Dataset changed since last scan; click to recompute"
                    : "Compute distribution"
              }
              style={
                stale
                  ? { background: "var(--warning, #d97706)", borderColor: "var(--warning, #d97706)" }
                  : undefined
              }
            >
              {loading ? "Scanning…" : stale ? "Re-Scan (stale)" : cachedExists ? "Up to date" : "Scan"}
            </button>
            {cachedExists ? (
              <span className="runtime-muted" style={{ fontSize: "0.8rem" }}>
                Last scan: {fmtTime(generatedAt)}
                {stale ? (
                  <span
                    style={{
                      marginLeft: "0.4rem",
                      background: "var(--warning, #d97706)",
                      color: "#fff",
                      borderRadius: "3px",
                      padding: "1px 5px",
                      fontSize: "0.75rem",
                      fontWeight: 600,
                    }}
                  >
                    stale
                  </span>
                ) : null}
              </span>
            ) : null}
          </div>

          {cachedExists ? (
            <Suspense
              fallback={
                <div className="runtime-muted" style={{ padding: "2rem", textAlign: "center" }}>
                  Loading scatter plot…
                </div>
              }
            >
              <ScatterSubTab
                coords={payload?.coords ?? []}
                onSelectTask={onSelectTask}
              />
            </Suspense>
          ) : (
            <div className="runtime-muted" style={{ padding: "2rem", textAlign: "center" }}>
              {loading
                ? "Loading…"
                : <>No scan yet. Click <strong>Scan</strong> above to embed all tasks (UMAP + HDBSCAN, may take minutes the first time).</>}
            </div>
          )}
        </div>
      ) : null}

      {/* ── Statistics sub-tab (entity / phrase type frequencies) ────── */}
      {subtab === "statistics" ? (
        <div style={{ marginTop: "0.75rem" }}>
          <TypeStatisticsPanel projectId={projectId} storeKey={storeKey} />
        </div>
      ) : null}
    </section>
  );
}


// ── RowDuplicatesSubTab ──────────────────────────────────────────────────────

type RowDuplicatesSubTabProps = {
  projectId: string | null;
  storeKey?: string | null;
  onSelectTask?: (taskId: string) => void;
  onClusterCountChange: (count: number) => void;
};

function RowDuplicatesSubTab({
  projectId,
  storeKey = null,
  onSelectTask,
  onClusterCountChange,
}: RowDuplicatesSubTabProps): React.ReactElement {
  const [rowCache, setRowCache] = useState<RowDedupCacheResponse | null>(null);
  const [rowProfile, setRowProfile] = useState<string>("MinHash");
  // Default to a low scan threshold so "scan once, drag freely" works:
  // the LSH/cluster step captures everything down to 0.10 Jaccard, and
  // the slider is a real-time member-level filter from there upward.
  // Operators who want a tighter scan can lower the slider before
  // clicking Scan; LSH at lower thresholds is slower but step-2
  // rep-anchored verification keeps the cluster shapes clean.
  const [jaccardThreshold, setJaccardThreshold] = useState<number>(0.4);
  const [selectedRows, setSelectedRows] = useState<Record<string, true>>({});
  const [applying, setApplying] = useState(false);
  const [applyStatus, setApplyStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Controls visibility of the metric-explainer popover next to the ?
  // icon. We use React state instead of native `title` (which has a
  // ~700ms delay and renders as plain browser chrome) so the bubble
  // appears instantly on hover or focus.
  const [metricHelpOpen, setMetricHelpOpen] = useState(false);
  const [showMasked, setShowMasked] = useState<boolean>(true);

  // ── URL helpers ────────────────────────────────────────────────────────────
  function buildRowDedupGetUrl(pid: string, profile: string): string {
    const storeQ = storeKey ? `&store=${encodeURIComponent(storeKey)}` : "";
    return `/api/row-dedup?project=${encodeURIComponent(pid)}&profile=${encodeURIComponent(profile)}${storeQ}`;
  }

  // ── Auto-load on mount and when profile changes ────────────────────────────
  // We use an AbortController + ignore flag to cancel stale fetches when
  // (projectId, storeKey, rowProfile) flip rapidly during initial mount —
  // e.g. URL state resolves the store key on the second render. Also
  // retries once on transient network failure, which can happen when the
  // dev server's --reload watcher restarts mid-request.
  useEffect(() => {
    if (!projectId) {
      setRowCache(null);
      onClusterCountChange(0);
      return;
    }
    let cancelled = false;
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    const url = buildRowDedupGetUrl(projectId, rowProfile);

    async function fetchOnce(): Promise<RowDedupCacheResponse> {
      const r = await fetch(url, { signal: ctrl.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return (await r.json()) as RowDedupCacheResponse;
    }

    (async () => {
      let d: RowDedupCacheResponse;
      try {
        d = await fetchOnce();
      } catch (e: unknown) {
        if (cancelled || ctrl.signal.aborted) return;
        // One retry after 500ms for transient failures (e.g. server reload).
        try {
          await new Promise((res) => setTimeout(res, 500));
          if (cancelled || ctrl.signal.aborted) return;
          d = await fetchOnce();
        } catch (e2: unknown) {
          if (cancelled || ctrl.signal.aborted) return;
          setError(e2 instanceof Error ? e2.message : String(e2));
          setLoading(false);
          return;
        }
      }
      if (cancelled || ctrl.signal.aborted) return;
      setRowCache(d);
      setError(null);
      if (d.available_profiles.length > 0 && !d.available_profiles.includes(rowProfile)) {
        setRowProfile(d.available_profiles[0]);
      }
      onClusterCountChange(d.payload?.clusters?.length ?? 0);
      setLoading(false);
    })();

    return () => {
      cancelled = true;
      ctrl.abort();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, storeKey, rowProfile]);

  // Keep cluster count in sync with cache changes.
  useEffect(() => {
    onClusterCountChange(rowCache?.payload?.clusters?.length ?? 0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rowCache]);

  function reloadRowCache() {
    if (!projectId) return;
    fetch(buildRowDedupGetUrl(projectId, rowProfile))
      .then((r) => (r.ok ? (r.json() as Promise<RowDedupCacheResponse>) : null))
      .then((d) => { if (d) setRowCache(d); })
      .catch(() => {});
  }

  // ── Scan handler (synchronous POST) ───────────────────────────────────────
  async function handleScan() {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    setApplyStatus(null);
    try {
      const storeQ = storeKey ? `&store=${encodeURIComponent(storeKey)}` : "";
      const url = `/api/row-dedup/scan?project=${encodeURIComponent(projectId)}${storeQ}`;
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile: rowProfile,
          statuses: null,
          jaccard_threshold: jaccardThreshold,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
      const result = await r.json() as RowDedupCacheResponse;
      setRowCache(result);
      // DO NOT clear `selectedRows` here. Keep selections are keyed by
      // (task_id, row_index) — those identities are stable across scans
      // at any threshold. Wiping them on re-scan silently discarded the
      // operator's intent and caused "all rows masked" after lowering
      // the threshold. Now: keeps you set at threshold A survive a
      // re-scan at threshold B. To clear them explicitly, use the
      // "Clear" bulk-action button.
      // Note: orphan keys (rows the new scan didn't put in any cluster)
      // are harmless — handleMask only consults keys for the clusters
      // it's currently iterating, so unused entries do nothing.
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  // ── Mask handler ──────────────────────────────────────────────────────────
  async function handleMask() {
    if (!projectId || !rowCache?.payload) return;
    setApplying(true);
    setApplyStatus(null);
    setError(null);

    const storeQ = storeKey ? `&store=${encodeURIComponent(storeKey)}` : "";
    const maskUrl = `/api/row-dedup/mask?project=${encodeURIComponent(projectId)}${storeQ}`;

    let totalMasked = 0;
    let clustersAffected = 0;

    try {
      // CRITICAL: iterate the slider-FILTERED `clusters` (with member-
      // level sim_to_rep filtering applied), NOT the raw cache. The
      // user's expectation is "Mask only what I see". Previously
      // iterating ``rowCache.payload.clusters`` masked rows in
      // clusters the slider had hidden — a footgun that produced
      // unexpected mass-masks (e.g. slider at 0.7 showed 30 rows but
      // Mask actually hit 1798 rows across clusters below 0.7).
      for (const cluster of clusters) {
        const rep = pickRowRep(cluster.members);
        const repKey = `${rep.task_id}::${rep.row_index}`;

        // Inverted semantics: checkbox = "Keep". Mask the UN-checked
        // non-rep members (default action). Checked members are
        // explicitly preserved alongside the rep. Skip rows already
        // masked — re-sending them is wasted and would clutter logs.
        const uncheckedNonReps = cluster.members.filter((m) => {
          const key = `${m.task_id}::${m.row_index}`;
          return key !== repKey
            && selectedRows[key] !== true
            && !maskedKeySet.has(key);
        });

        if (uncheckedNonReps.length === 0) continue;

        // Send rep + unchecked non-reps. Backend drops rep (smallest
        // key), masks the rest of what we sent. Checked "keep" rows
        // are simply omitted from the request → never masked.
        const membersToSend = [rep, ...uncheckedNonReps];

        const r = await fetch(maskUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            cluster_id: cluster.cluster_id,
            members: membersToSend.map((m) => ({ task_id: m.task_id, row_index: m.row_index })),
            cluster_similarity: cluster.similarity,
            embedding_profile: rowCache.payload.params.profile,
            embedding_model: rowCache.payload.params.model,
            actor: "operator",
          }),
        });
        if (!r.ok) {
          const txt = await r.text();
          throw new Error(`HTTP ${r.status}: ${txt.slice(0, 200)}`);
        }
        const result = await r.json() as { masked: number; skipped: number };
        totalMasked += result.masked;
        clustersAffected++;
      }

      setApplyStatus(
        `Masked ${totalMasked} row${totalMasked !== 1 ? "s" : ""} across ${clustersAffected} cluster${clustersAffected !== 1 ? "s" : ""}.`,
      );
      setSelectedRows({});
      reloadRowCache();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplying(false);
    }
  }

  // ── Derived state ──────────────────────────────────────────────────────────
  // The slider is a real-time client-side filter over the cached scan
  // result. Two-level filtering:
  //   1) MEMBER level — drop any member whose ``sim_to_rep`` falls below
  //      the current slider value. The rep itself stays (it has
  //      sim_to_rep = 1.0). Older cached payloads with no ``sim_to_rep``
  //      field treat every member as 1.0 so they still render.
  //   2) CLUSTER level — after member filtering, drop the cluster if
  //      fewer than 2 members survive (no longer a duplicate group).
  // To see members or clusters below the scan-time threshold, re-scan
  // at a looser threshold — the slider can't conjure new members below
  // what the LSH bucketed at scan time.
  const allClusters = rowCache?.payload?.clusters ?? [];
  const scanThreshold = rowCache?.payload?.params?.jaccard_threshold ?? jaccardThreshold;
  const rowStale = rowCache?.stale ?? false;
  const rowGeneratedAt = rowCache?.generated_at ?? null;
  const rowCached = rowCache?.cached ?? false;
  const availableRowProfiles = rowCache?.available_profiles ?? ["MinHash"];

  // Set of "task_id::row_index" keys for cluster members that are
  // currently masked. Two sources are merged:
  //
  // 1. ``member.masked === true`` on members from the new scan
  //    output (scan now includes masked rows tagged inline).
  // 2. The legacy ``currently_masked`` overlay for caches built
  //    before the scan change — those payloads have NO per-member
  //    flag, so we still need the overlay as a fallback.
  const maskedKeySet = useMemo(() => {
    const s = new Set<string>();
    for (const c of rowCache?.payload?.clusters ?? []) {
      for (const m of c.members) {
        if (m.masked) s.add(`${m.task_id}::${m.row_index}`);
      }
    }
    for (const [tid, ridx] of rowCache?.currently_masked ?? []) {
      s.add(`${tid}::${ridx}`);
    }
    return s;
  }, [rowCache?.payload?.clusters, rowCache?.currently_masked]);

  // Two-pass filtering so the "−N hidden" UI can attribute hides to
  // the right cause:
  //   Pass 1 (adj-threshold): drop members whose length-adjusted
  //     similarity to rep falls below the slider; rep always survives.
  //   Pass 2 (show-masked): if the operator unchecked "Show masked",
  //     drop members already in row_masks.
  // After each pass we drop clusters that no longer have ≥2 members.
  // ``hiddenByThreshold`` counts clusters lost to pass 1; ``hiddenByMaskHide``
  // counts the ADDITIONAL clusters lost to pass 2, so the two never
  // double-count.
  const clustersAfterThreshold = allClusters
    .map((c) => ({
      ...c,
      members: c.members.filter((m) => {
        if ((m.sim_to_rep ?? 1) >= 1 - 1e-9) return true; // rep always survives
        const wc = m.word_count ?? 10;
        const score = (m.sim_to_rep ?? 1) / Math.log(wc / 10 + Math.E);
        return score >= jaccardThreshold;
      }),
    }))
    .filter((c) => c.members.length >= 2);
  const clusters = showMasked
    ? clustersAfterThreshold
    : clustersAfterThreshold
        .map((c) => ({
          ...c,
          members: c.members.filter(
            (m) => !maskedKeySet.has(`${m.task_id}::${m.row_index}`),
          ),
        }))
        .filter((c) => c.members.length >= 2);
  const hiddenByThreshold = allClusters.length - clustersAfterThreshold.length;
  const hiddenByMaskHide = clustersAfterThreshold.length - clusters.length;
  const sliderBelowScan = jaccardThreshold < scanThreshold - 1e-9;

  // Inverted semantics: checkbox = Keep. `selectedRows` is the set of
  // explicitly-kept rows. Everything else (every non-rep without a check)
  // is queued to be masked when the user clicks "Mask".
  //
  // `keepRowCount` reflects ONLY keeps that map to currently-displayed
  // non-rep, non-masked rows. Orphan keeps (set at threshold A, but the
  // row dropped out at threshold B, or was masked since) are excluded
  // here so the "K kept" UI matches what's actually visible. The full
  // `selectedRows` is still preserved across re-scans — orphans become
  // effective again if the row reappears in a future cluster.
  const totalSelectedRows = Object.keys(selectedRows).length;

  // Total non-rep rows that AREN'T already masked across all clusters —
  // these are the candidates for masking. The rep of each cluster is
  // always kept implicitly. Already-masked rows are excluded so the
  // counts and bulk actions don't double-charge them.
  const allSelectableRowKeys: string[] = [];
  for (const cluster of clusters) {
    const rep = pickRowRep(cluster.members);
    const repKey = `${rep.task_id}::${rep.row_index}`;
    for (const m of cluster.members) {
      const key = `${m.task_id}::${m.row_index}`;
      if (key !== repKey && !maskedKeySet.has(key)) {
        allSelectableRowKeys.push(key);
      }
    }
  }
  const totalNonReps = allSelectableRowKeys.length;
  // Count keeps that actually map to a currently-displayed non-rep, non-masked row.
  const allSelectableRowKeySet = new Set(allSelectableRowKeys);
  const keepRowCount = Object.keys(selectedRows).filter(
    (k) => allSelectableRowKeySet.has(k),
  ).length;
  const orphanKeepCount = totalSelectedRows - keepRowCount;
  const rowsToMask = Math.max(0, totalNonReps - keepRowCount);

  // Count of clusters with at least one unchecked, not-yet-masked non-rep.
  const clustersToMask = clusters.filter((cluster) => {
    const rep = pickRowRep(cluster.members);
    const repKey = `${rep.task_id}::${rep.row_index}`;
    return cluster.members.some((m) => {
      const key = `${m.task_id}::${m.row_index}`;
      return key !== repKey && !maskedKeySet.has(key) && selectedRows[key] !== true;
    });
  }).length;

  // Total already-masked rows visible across the displayed clusters — shown
  // as a header summary so the operator sees the impact of past masks at a glance.
  const alreadyMaskedInView = clusters.reduce((sum, c) => {
    let n = 0;
    for (const m of c.members) {
      if (maskedKeySet.has(`${m.task_id}::${m.row_index}`)) n++;
    }
    return sum + n;
  }, 0);

  function handleSelectAll() {
    const next: Record<string, true> = {};
    for (const k of allSelectableRowKeys) next[k] = true;
    setSelectedRows(next);
  }

  function handleClearAll() {
    setSelectedRows({});
  }

  function toggleRow(key: string, checked: boolean) {
    setSelectedRows((prev) => {
      const next = { ...prev };
      if (checked) {
        next[key] = true;
      } else {
        delete next[key];
      }
      return next;
    });
  }

  const isEmpty = rowCached && clusters.length === 0;

  return (
    <div style={{ marginTop: "0.75rem" }}>
      {/* ── Filter + scan controls ────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          gap: "1rem",
          alignItems: "center",
          flexWrap: "wrap",
          marginBottom: "0.75rem",
          padding: "0.6rem 0.75rem",
          background: "var(--surface2, #f8fafc)",
          borderRadius: "6px",
          border: "1px solid var(--border, #e5e7eb)",
        }}
      >
        <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.35rem" }}>
          Profile
          <select
            value={rowProfile}
            onChange={(e) => {
              setRowProfile(e.target.value);
              // Keep selections survive profile changes — the (task_id,
              // row_index) keys still identify the same physical rows.
              // Use the explicit "Clear" bulk-action button to reset.
            }}
            style={{ fontSize: "0.85rem" }}
            disabled={loading}
          >
            {availableRowProfiles.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>

        {(() => {
          // Threshold semantics depend on the active profile's metric:
          //   - MinHash → Jaccard similarity of word shingles
          //   - embedding provider (e.g. jina) → cosine similarity
          // The slider label, the help tooltip, and the units in the
          // value pill all switch to match. The single `jaccardThreshold`
          // state variable holds the value regardless of metric.
          const scanMetric = rowCache?.payload?.params?.metric;
          const isJaccard = scanMetric ? scanMetric === "jaccard" : rowProfile === "MinHash";
          const metricLabel = isJaccard ? "Adj Jaccard" : "Cosine";
          const helpText = isJaccard
            ? "Length-adjusted Jaccard = J / ln(words/10 + e). " +
              "Raw Jaccard alone is biased toward long sentences: same proportional overlap, " +
              "longer sentence → lower score (still has many unique words → keep). " +
              "Short sentence → higher score (fewer unique words → mask sooner). " +
              "The cluster representative always survives regardless of score. " +
              "Typical threshold: 0.35–0.50."
            : "Cosine similarity of the row's embedding vector, in [−1, 1]. Two rows with " +
              "near-identical meaning sit at ~1.0; unrelated rows cluster around 0. " +
              "Higher threshold = stricter semantic match. Typical sweet spot: 0.80–0.92.";
          return (
            <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem" }}>
                {metricLabel} ≥
                <span
                  style={{ position: "relative", display: "inline-flex" }}
                  onMouseEnter={() => setMetricHelpOpen(true)}
                  onMouseLeave={() => setMetricHelpOpen(false)}
                  onFocus={() => setMetricHelpOpen(true)}
                  onBlur={() => setMetricHelpOpen(false)}
                >
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={() => setMetricHelpOpen((v) => !v)}
                    aria-label={`What is ${metricLabel} similarity?`}
                    aria-expanded={metricHelpOpen}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: "16px",
                      height: "16px",
                      borderRadius: "50%",
                      background: metricHelpOpen
                        ? "var(--accent, #3b82f6)"
                        : "var(--surface3, #e5e7eb)",
                      color: metricHelpOpen ? "#fff" : "var(--muted, #6b7280)",
                      fontSize: "0.7rem",
                      fontWeight: 700,
                      cursor: "help",
                      userSelect: "none",
                      transition: "background 0.1s, color 0.1s",
                    }}
                  >
                    ?
                  </span>
                  {metricHelpOpen ? (
                    <div
                      role="tooltip"
                      style={{
                        position: "absolute",
                        top: "calc(100% + 6px)",
                        left: "-8px",
                        zIndex: 50,
                        width: "360px",
                        padding: "0.6rem 0.75rem",
                        background: "var(--surface, #ffffff)",
                        color: "var(--fg, #111827)",
                        border: "1px solid var(--border, #d1d5db)",
                        borderRadius: "6px",
                        boxShadow: "0 6px 18px rgba(0,0,0,0.12)",
                        fontSize: "0.78rem",
                        lineHeight: 1.5,
                        fontWeight: 400,
                        whiteSpace: "normal",
                      }}
                    >
                      <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>
                        {metricLabel} similarity
                      </div>
                      {helpText}
                    </div>
                  ) : null}
                </span>
              </span>
              <RangeStepper
                min={0.01}
                max={0.99}
                step={0.01}
                decimals={2}
                value={jaccardThreshold}
                onChange={setJaccardThreshold}
                width="120px"
                disabled={loading}
                title={helpText}
              />
              <code style={{ fontFamily: "monospace", minWidth: "3rem" }}>
                {jaccardThreshold.toFixed(2)}
              </code>
              {/* Hint: the slider is a client-side filter on cached cluster
                  similarities. Raising it hides clusters; lowering it past the
                  scan-time threshold cannot reveal new ones — re-scan for that. */}
              {rowCached && hiddenByThreshold > 0 ? (
                <span
                  className="runtime-muted"
                  style={{ fontSize: "0.75rem" }}
                  title={`Hiding ${hiddenByThreshold} cluster(s) below the slider threshold. Scan threshold: ${scanThreshold.toFixed(2)}`}
                >
                  −{hiddenByThreshold} below threshold
                </span>
              ) : null}
              {rowCached && hiddenByMaskHide > 0 ? (
                <span
                  className="runtime-muted"
                  style={{ fontSize: "0.75rem" }}
                  title={`Hiding ${hiddenByMaskHide} additional cluster(s) because "Show masked" is off — every remaining non-rep member in these clusters is already masked.`}
                >
                  −{hiddenByMaskHide} all masked
                </span>
              ) : null}
              {rowCached && sliderBelowScan ? (
                <span
                  style={{
                    fontSize: "0.72rem",
                    background: "var(--warning, #d97706)",
                    color: "#fff",
                    borderRadius: "3px",
                    padding: "1px 5px",
                    fontWeight: 600,
                  }}
                  title={`Slider is below scan-time threshold (${scanThreshold.toFixed(2)}). Re-scan to recompute clusters at this threshold.`}
                >
                  re-scan to expand
                </span>
              ) : null}
            </label>
          );
        })()}

        <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.35rem", cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={showMasked}
            onChange={(e) => setShowMasked(e.target.checked)}
            style={{ cursor: "pointer" }}
          />
          Show masked
        </label>

        {/* ── Bulk actions: live next to Re-scan so the operator's scan +
            select + mask flow stays on one row. Buttons are inert (disabled
            + muted) when there's nothing to act on, so they're harmless
            even before a scan exists. ─────────────────────────────────── */}
        {clusters.length > 0 ? (
          <>
            <button
              type="button"
              onClick={handleSelectAll}
              disabled={allSelectableRowKeys.length === 0}
              style={{ fontSize: "0.8rem" }}
              title="Mark every non-representative row as Keep — nothing will be masked"
            >
              Keep all ({allSelectableRowKeys.length})
            </button>
            <button
              type="button"
              onClick={handleClearAll}
              disabled={totalSelectedRows === 0}
              style={{ fontSize: "0.8rem" }}
              title="Unkeep all (including any keeps on rows no longer in any cluster)"
            >
              Clear{orphanKeepCount > 0 ? ` (${keepRowCount}+${orphanKeepCount} orphan)` : ""}
            </button>
            <button
              type="button"
              disabled={rowsToMask === 0 || applying}
              onClick={handleMask}
              style={{
                fontSize: "0.85rem",
                background:
                  rowsToMask > 0 && !applying
                    ? "var(--danger, #b91c1c)"
                    : undefined,
                color: rowsToMask > 0 && !applying ? "white" : undefined,
                borderColor:
                  rowsToMask > 0 && !applying ? "var(--danger, #b91c1c)" : undefined,
                opacity: rowsToMask === 0 || applying ? 0.6 : 1,
              }}
              title="Mask every non-representative row that is NOT checked as Keep"
            >
              {applying
                ? "Masking…"
                : rowsToMask > 0
                  ? `Mask ${rowsToMask} row${rowsToMask !== 1 ? "s" : ""}`
                  : "Mask"}
            </button>
          </>
        ) : null}

        {rowCached ? (
          <span className="runtime-muted" style={{ fontSize: "0.8rem", marginLeft: "auto" }}>
            Last checked: {fmtTime(rowGeneratedAt)}
            {" · "}
            {rowCache?.payload?.row_count ?? 0} rows
            {rowStale ? (
              <span
                style={{
                  marginLeft: "0.4rem",
                  background: "var(--warning, #d97706)",
                  color: "#fff",
                  borderRadius: "3px",
                  padding: "1px 5px",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                }}
              >
                stale
              </span>
            ) : null}
          </span>
        ) : null}

        {/* Re-scan lives on the far right of the controls row. Disabled when
            a fresh cache exists (content-hash unchanged) — re-running on the
            same dataset would just rebuild the same clusters. Stale → button
            re-enables with a warning color so the operator knows to act. */}
        <button
          className="primary-button"
          type="button"
          onClick={handleScan}
          disabled={loading || !projectId || (rowCached && !rowStale)}
          title={
            rowCached && !rowStale
              ? "Dataset hash unchanged since last scan — would produce identical clusters. Disabled."
              : rowStale
                ? "Dataset (or masks) changed since last scan; click to recompute"
                : "Scan rows for near-duplicates"
          }
          style={
            rowStale
              ? {
                  background: "var(--warning, #d97706)",
                  borderColor: "var(--warning, #d97706)",
                  marginLeft: rowCached ? "0.5rem" : "auto",
                }
              : { marginLeft: rowCached ? "0.5rem" : "auto" }
          }
        >
          {loading ? "Scanning…" : rowStale ? "Re-scan (stale)" : rowCached ? "Up to date" : "Scan rows"}
        </button>
      </div>

      {error ? <div className="notice compact">{error}</div> : null}
      {applyStatus ? (
        <div className="notice compact" style={{ color: "var(--success, #047857)" }}>
          {applyStatus}
        </div>
      ) : null}

      {/* ── Empty / no-cache state ────────────────────────────────────── */}
      {!rowCached && !loading ? (
        <p className="runtime-muted">
          No cached scan yet. Click <strong>Scan rows</strong> to find near-duplicate rows across tasks.
        </p>
      ) : null}

      {isEmpty ? (
        <p className="runtime-muted">
          No row-level duplicates found. Lower the threshold or run a fresh scan.
        </p>
      ) : null}

      {clusters.length > 0 ? (
        <>
          {/* Already-masked summary: shows past-mask impact so the operator
              isn't confused why some rows are no longer actionable. */}
          {alreadyMaskedInView > 0 ? (
            <div style={{ fontSize: "0.8rem", marginBottom: "0.35rem", color: "var(--muted, #6b7280)" }}>
              <strong>{alreadyMaskedInView}</strong> row{alreadyMaskedInView !== 1 ? "s" : ""} already masked in displayed clusters
              {rowStale ? " · re-scan to recompute clusters without them" : ""}
            </div>
          ) : null}

          {/* Pending-mask summary: shows how many rows in how many clusters
              will be masked if the operator clicks "Mask". Inverted-keep
              semantics: every non-rep without a check is queued for masking. */}
          {rowsToMask > 0 ? (
            <div style={{ fontSize: "0.85rem", marginBottom: "0.5rem" }}>
              <strong>{rowsToMask}</strong> row{rowsToMask !== 1 ? "s" : ""} across{" "}
              <strong>{clustersToMask}</strong> cluster{clustersToMask !== 1 ? "s" : ""} will be masked
              {keepRowCount > 0 ? (
                <> · <strong>{keepRowCount}</strong> kept</>
              ) : null}
            </div>
          ) : null}

          {/* ── Cluster cards ─────────────────────────────────────────── */}
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            {clusters.map((cluster) => {
              const rep = pickRowRep(cluster.members);
              const repKey = `${rep.task_id}::${rep.row_index}`;
              return (
                <div
                  key={cluster.cluster_id}
                  className="runtime-card"
                  style={{ padding: "0.75rem 1rem" }}
                >
                  {/* ── Cluster header ──────────────────────────────── */}
                  <div
                    style={{
                      display: "flex",
                      gap: "0.75rem",
                      alignItems: "baseline",
                      marginBottom: "0.4rem",
                      flexWrap: "wrap",
                    }}
                  >
                    <code style={{ fontFamily: "monospace", fontWeight: 600, fontSize: "0.85rem" }}>
                      {cluster.cluster_id}
                    </code>
                    <span className="runtime-muted" style={{ fontSize: "0.8rem" }}>
                      size={cluster.members.length}
                    </span>
                    <span
                      className="runtime-muted"
                      style={{ fontSize: "0.8rem", cursor: "help" }}
                      title={
                        cluster.method === "minhash"
                          ? "Average pairwise MinHash-estimated Jaccard across all member pairs (not the minimum). " +
                            "Connected-components clustering means some chain-linked pairs may have actual Jaccard < this average."
                          : "Average pairwise cosine similarity across all member pairs."
                      }
                    >
                      avg {cluster.method === "minhash" ? "jaccard" : "cos"}={cluster.similarity.toFixed(3)}
                    </span>
                    <span className="runtime-muted" style={{ fontSize: "0.8rem" }}>
                      method={cluster.method}
                    </span>
                    {/* "Keep" column header — clarifies that the checkbox on each
                        non-rep row marks it for keeping. Unchecked rows below
                        carry a "masked" badge so the pending action is obvious. */}
                    <span
                      style={{
                        marginLeft: "auto",
                        display: "inline-flex",
                        alignItems: "center",
                        padding: "0 6px",
                        height: "18px",
                        background: "var(--surface3, #e5e7eb)",
                        borderRadius: "3px",
                        fontSize: "0.72rem",
                        fontWeight: 600,
                        color: "var(--muted, #6b7280)",
                        textTransform: "uppercase",
                        letterSpacing: "0.04em",
                        userSelect: "none",
                      }}
                      title="Check a row below to Keep it (exempt from masking). Unchecked non-rep rows show as 'masked' and will be masked when you click Mask."
                    >
                      Keep
                    </span>
                  </div>

                  {/* ── Member rows ─────────────────────────────────── */}
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                    {cluster.members.map((member) => {
                      const key = `${member.task_id}::${member.row_index}`;
                      const isRep = key === repKey;
                      const isChecked = selectedRows[key] === true;
                      const isMasked = maskedKeySet.has(key);
                      return (
                        <div
                          key={key}
                          style={{
                            display: "flex",
                            alignItems: "flex-start",
                            gap: "0.5rem",
                            fontSize: "0.82rem",
                            padding: "0.2rem 0",
                            opacity: isMasked ? 0.55 : 1,
                          }}
                        >
                          {isRep ? (
                            <span
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                padding: "0 5px",
                                height: "16px",
                                background: "var(--surface3, #e5e7eb)",
                                borderRadius: "3px",
                                fontSize: "0.7rem",
                                fontWeight: 600,
                                color: "var(--muted, #6b7280)",
                                flexShrink: 0,
                                marginTop: "2px",
                                userSelect: "none",
                              }}
                              title="Representative — kept, not masked"
                            >
                              rep
                            </span>
                          ) : isMasked ? (
                            <span
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                padding: "0 5px",
                                height: "16px",
                                background: "var(--danger, #b91c1c)",
                                color: "#fff",
                                borderRadius: "3px",
                                fontSize: "0.7rem",
                                fontWeight: 600,
                                flexShrink: 0,
                                marginTop: "2px",
                                userSelect: "none",
                              }}
                              title="This row is currently masked in row_masks. Re-scan to drop it from the cluster."
                            >
                              masked
                            </span>
                          ) : (
                            <input
                              type="checkbox"
                              checked={isChecked}
                              onChange={(e) => toggleRow(key, e.target.checked)}
                              title="Check to Keep this row (exempt from masking)"
                              style={{ marginTop: "2px", flexShrink: 0, cursor: "pointer" }}
                            />
                          )}
                          <button
                            type="button"
                            onClick={() => onSelectTask?.(member.task_id)}
                            title="Open task drawer"
                            style={{
                              flex: 1, textAlign: "left",
                              background: "transparent", border: "none", padding: 0,
                              color: isRep || isMasked ? "var(--muted, #6b7280)" : "inherit",
                              cursor: "pointer", font: "inherit",
                              textDecoration: isMasked ? "line-through" : undefined,
                            }}
                          >
                            <code
                              style={{
                                fontFamily: "monospace", fontSize: "0.8rem",
                                textDecoration: "underline", textUnderlineOffset: "2px",
                              }}
                            >
                              {member.task_id}:{member.row_index}
                            </code>
                            {member.text_preview ? (() => {
                              const truncated = truncateToWords(member.text_preview, 100);
                              return (
                                <span
                                  className="runtime-muted"
                                  style={{
                                    marginLeft: "0.4rem",
                                    whiteSpace: "normal",
                                    lineHeight: 1.45,
                                  }}
                                >
                                  — {truncated}
                                  {truncated.length < member.text_preview.length ? "…" : ""}
                                </span>
                              );
                            })() : null}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      ) : null}
    </div>
  );
}

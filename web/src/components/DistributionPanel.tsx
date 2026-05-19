import React, { lazy, Suspense, useEffect, useState } from "react";

// Lazy-loaded: ScatterSubTab brings in plotly.js-dist-min (~3 MB / 1.47 MB
// gzip). Off the critical path of the Distribution tab so opening the
// Duplicates view is instant; plotly only ships when the operator actually
// switches to the Scatter plot sub-tab.
const ScatterSubTab = lazy(() => import("./ScatterSubTab"));

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
    max_rows_per_task: number;
    generated_at: string;
    embedding_cache: { hits: number; misses: number };
    skipped_tasks_too_many_rows: number;
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
};

// "duplicates" used to mean task-level (concat all rows, embed once, cluster
// tasks) — but in practice the rows of a single task vary enough by IDs /
// dates / numbers that real templates land in 0.05-0.10 Jaccard territory
// and aren't found. Row-level dedup catches the real cases, so the
// task-level UI was deleted; "duplicates" now IS row-level. The
// /api/distribution endpoints are still wired for the Scatter plot.
type Subtab = "duplicates" | "scatter";

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
 * Mirrors backend RowDedupService.mask_duplicates sort key:
 * (str(task_id), int(row_index)) — string compare on task_id, then numeric on row_index.
 */
function pickRowRep(members: RowMember[]): RowMember {
  return [...members].sort((a, b) => {
    if (a.task_id < b.task_id) return -1;
    if (a.task_id > b.task_id) return 1;
    return a.row_index - b.row_index;
  })[0] ?? members[0];
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
      {/* ── Header ─────────────────────────────────────────────────────── */}
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
          <h2 style={{ marginBottom: "0.25rem" }}>Distribution</h2>
          <p style={{ marginTop: 0, fontSize: "0.85rem" }}>
            Find duplicate rows (Duplicates) and visualize the project's task
            distribution in 2-D (Scatter plot). Each sub-tab manages its own
            profile / threshold / scan.
          </p>
        </div>
        {/* Scatter-specific scan controls live in the panel header only when
            the Scatter plot sub-tab is active. The Duplicates sub-tab has
            its own independent control bar inside RowDuplicatesSubTab —
            putting both at this level was the source of the duplicate
            Profile/Scan UI that the operator complained about. */}
        {subtab === "scatter" ? (
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
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
            <span className="runtime-muted" style={{ fontSize: "0.78rem" }}>
              {cachedExists
                ? <>Last scan: {fmtTime(generatedAt)}{stale ? " (stale)" : ""}</>
                : "no cached scan yet"}
            </span>
            <button
              className="primary-button"
              type="button"
              onClick={handleScan}
              disabled={loading || !projectId}
              style={
                stale
                  ? { background: "var(--warning, #d97706)", borderColor: "var(--warning, #d97706)" }
                  : undefined
              }
            >
              {loading ? "Scanning…" : stale ? "Re-Scan (stale)" : cachedExists ? "Re-Scan" : "Scan"}
            </button>
          </div>
        ) : null}
      </div>

      {error ? <div className="notice compact">{error}</div> : null}

      {/* ── Sub-tab nav (always visible — each sub-tab owns its own cache) ── */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginTop: "0.5rem" }}>
        <nav className="view-tabs" aria-label="Distribution sections" style={{ margin: 0 }}>
          <button
            className={subtab === "duplicates" ? "view-tab selected" : "view-tab"}
            type="button"
            onClick={() => setSubtab("duplicates")}
          >
            Duplicates ({rowDedupClusterCount})
          </button>
          <button
            className={subtab === "scatter" ? "view-tab selected" : "view-tab"}
            type="button"
            onClick={() => setSubtab("scatter")}
          >
            Scatter plot
          </button>
        </nav>
      </div>

      {/* ── Duplicates sub-tab (row-level — task-level was retired) ─── */}
      {subtab === "duplicates" ? (
        <RowDuplicatesSubTab
          projectId={projectId}
          storeKey={storeKey}
          onSelectTask={onSelectTask}
          onClusterCountChange={setRowDedupClusterCount}
        />
      ) : null}

      {/* ── Scatter plot sub-tab (lazy: plotly only loads on demand) ─── */}
      {subtab === "scatter" ? (
        cachedExists ? (
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
        )
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
  const [jaccardThreshold, setJaccardThreshold] = useState<number>(0.5);
  const [maxRowsPerTask, setMaxRowsPerTask] = useState<number>(100);
  const [selectedRows, setSelectedRows] = useState<Record<string, true>>({});
  const [applying, setApplying] = useState(false);
  const [applyStatus, setApplyStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── URL helpers ────────────────────────────────────────────────────────────
  function buildRowDedupGetUrl(pid: string, profile: string): string {
    const storeQ = storeKey ? `&store=${encodeURIComponent(storeKey)}` : "";
    return `/api/row-dedup?project=${encodeURIComponent(pid)}&profile=${encodeURIComponent(profile)}${storeQ}`;
  }

  // ── Auto-load on mount and when profile changes ────────────────────────────
  useEffect(() => {
    if (!projectId) {
      setRowCache(null);
      onClusterCountChange(0);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(buildRowDedupGetUrl(projectId, rowProfile))
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<RowDedupCacheResponse>;
      })
      .then((d) => {
        setRowCache(d);
        // Sync profile to first available if current isn't offered.
        if (d.available_profiles.length > 0 && !d.available_profiles.includes(rowProfile)) {
          setRowProfile(d.available_profiles[0]);
        }
        onClusterCountChange(d.payload?.clusters?.length ?? 0);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
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
          max_rows_per_task: maxRowsPerTask,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
      const result = await r.json() as RowDedupCacheResponse;
      setRowCache(result);
      setSelectedRows({});
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
      for (const cluster of rowCache.payload.clusters) {
        const rep = pickRowRep(cluster.members);
        const repKey = `${rep.task_id}::${rep.row_index}`;

        // Collect selected non-rep members.
        const selectedNonReps = cluster.members.filter((m) => {
          const key = `${m.task_id}::${m.row_index}`;
          return key !== repKey && selectedRows[key] === true;
        });

        if (selectedNonReps.length === 0) continue;

        // Send rep + selected non-reps. Backend drops rep, masks the rest.
        const membersToSend = [rep, ...selectedNonReps];

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
  const clusters = rowCache?.payload?.clusters ?? [];
  const rowStale = rowCache?.stale ?? false;
  const rowGeneratedAt = rowCache?.generated_at ?? null;
  const rowCached = rowCache?.cached ?? false;
  const availableRowProfiles = rowCache?.available_profiles ?? ["MinHash"];

  const selectedRowCount = Object.keys(selectedRows).length;

  const selectedClusterCount = clusters.filter((cluster) => {
    const rep = pickRowRep(cluster.members);
    const repKey = `${rep.task_id}::${rep.row_index}`;
    return cluster.members.some((m) => {
      const key = `${m.task_id}::${m.row_index}`;
      return key !== repKey && selectedRows[key] === true;
    });
  }).length;

  const allSelectableRowKeys: string[] = [];
  for (const cluster of clusters) {
    const rep = pickRowRep(cluster.members);
    const repKey = `${rep.task_id}::${rep.row_index}`;
    for (const m of cluster.members) {
      const key = `${m.task_id}::${m.row_index}`;
      if (key !== repKey) allSelectableRowKeys.push(key);
    }
  }

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
              setSelectedRows({});
            }}
            style={{ fontSize: "0.85rem" }}
            disabled={loading}
          >
            {availableRowProfiles.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>

        <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <span>Jaccard ≥</span>
          <input
            type="range"
            min={0.01}
            max={0.99}
            step={0.01}
            value={jaccardThreshold}
            onChange={(e) => setJaccardThreshold(parseFloat(e.target.value))}
            style={{ width: "120px" }}
            disabled={loading}
          />
          <code style={{ fontFamily: "monospace", minWidth: "3rem" }}>
            {jaccardThreshold.toFixed(2)}
          </code>
        </label>

        <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.35rem" }}>
          Max rows/task
          <input
            type="number"
            min={1}
            max={10000}
            value={maxRowsPerTask}
            onChange={(e) => setMaxRowsPerTask(Math.max(1, parseInt(e.target.value, 10) || 100))}
            style={{ width: "5rem", fontSize: "0.85rem" }}
            disabled={loading}
          />
        </label>

        <button
          className="primary-button"
          type="button"
          onClick={handleScan}
          disabled={loading || !projectId}
          style={
            rowStale
              ? { background: "var(--warning, #d97706)", borderColor: "var(--warning, #d97706)" }
              : undefined
          }
        >
          {loading ? "Scanning…" : rowStale ? "Re-scan (stale)" : rowCached ? "Re-scan" : "Scan rows"}
        </button>

        {rowCached ? (
          <span className="runtime-muted" style={{ fontSize: "0.8rem" }}>
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
          {/* ── Selection summary + bulk actions ──────────────────────── */}
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
            <span style={{ fontSize: "0.85rem" }}>
              {selectedRowCount > 0 ? (
                <>
                  <strong>{selectedRowCount}</strong> row{selectedRowCount !== 1 ? "s" : ""} across{" "}
                  <strong>{selectedClusterCount}</strong> cluster{selectedClusterCount !== 1 ? "s" : ""} selected
                </>
              ) : (
                <span className="runtime-muted">No rows selected — check boxes below to select duplicates to mask</span>
              )}
            </span>
            <div style={{ display: "flex", gap: "0.4rem", marginLeft: "auto" }}>
              <button
                type="button"
                onClick={handleSelectAll}
                disabled={allSelectableRowKeys.length === 0}
                style={{ fontSize: "0.8rem" }}
                title="Select all non-representative rows in all clusters"
              >
                Select all ({allSelectableRowKeys.length})
              </button>
              <button
                type="button"
                onClick={handleClearAll}
                disabled={selectedRowCount === 0}
                style={{ fontSize: "0.8rem" }}
              >
                Clear
              </button>
            </div>
            <button
              type="button"
              disabled={selectedRowCount === 0 || applying}
              onClick={handleMask}
              style={{
                fontSize: "0.85rem",
                background:
                  selectedRowCount > 0 && !applying
                    ? "var(--danger, #b91c1c)"
                    : undefined,
                color: selectedRowCount > 0 && !applying ? "white" : undefined,
                borderColor: selectedRowCount > 0 && !applying ? "var(--danger, #b91c1c)" : undefined,
                opacity: selectedRowCount === 0 || applying ? 0.6 : 1,
              }}
            >
              {applying ? "Masking…" : `Mask ${selectedRowCount > 0 ? selectedRowCount + " row" + (selectedRowCount !== 1 ? "s" : "") : ""}`}
            </button>
          </div>

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
                    <span className="runtime-muted" style={{ fontSize: "0.8rem" }}>
                      {cluster.method === "minhash" ? "jaccard" : "cos"}={cluster.similarity.toFixed(3)}
                    </span>
                    <span className="runtime-muted" style={{ fontSize: "0.8rem" }}>
                      method={cluster.method}
                    </span>
                  </div>

                  {/* ── Member rows ─────────────────────────────────── */}
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                    {cluster.members.map((member) => {
                      const key = `${member.task_id}::${member.row_index}`;
                      const isRep = key === repKey;
                      const isChecked = selectedRows[key] === true;
                      return (
                        <div
                          key={key}
                          style={{
                            display: "flex",
                            alignItems: "flex-start",
                            gap: "0.5rem",
                            fontSize: "0.82rem",
                            padding: "0.2rem 0",
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
                          ) : (
                            <input
                              type="checkbox"
                              checked={isChecked}
                              onChange={(e) => toggleRow(key, e.target.checked)}
                              title="Include in batch mask"
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
                              color: isRep ? "var(--muted, #6b7280)" : "inherit",
                              cursor: "pointer", font: "inherit",
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
                            {member.text_preview ? (
                              <span
                                className="runtime-muted"
                                style={{
                                  marginLeft: "0.4rem",
                                  whiteSpace: "normal",
                                  lineHeight: 1.45,
                                }}
                              >
                                — {member.text_preview.slice(0, 200)}
                                {member.text_preview.length > 200 ? "…" : ""}
                              </span>
                            ) : null}
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

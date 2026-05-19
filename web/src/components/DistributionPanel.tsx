import React, { useEffect, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);

// ── Types ────────────────────────────────────────────────────────────────────

type ClusterEntry = {
  cluster_id: string;
  task_ids: string[];
  method: string;
  similarity: number;
};

type CoordEntry = {
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

type Subtab = "duplicates" | "scatter";

export type DistributionPanelProps = {
  projectId: string | null;
  storeKey?: string | null;
  onSelectTask?: (taskId: string) => void;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

// Map known profile names to their embedding model identifier for audit metadata.
const PROFILE_TO_MODEL: Record<string, string> = {
  jina_small: "jinaai/jina-embeddings-v5-text-small",
  random_baseline: "random_baseline",
};

function getEmbeddingModel(profile: string): string {
  return PROFILE_TO_MODEL[profile] ?? profile;
}

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

/** Pick the representative task for a cluster: lowest task_id lexicographically. */
function pickRepresentative(taskIds: string[]): string {
  return [...taskIds].sort()[0] ?? taskIds[0];
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
  // Client-side similarity threshold filter (does NOT affect scan params).
  // Semantics differ by provider — cosine for jina (~0.85 baseline) vs
  // Jaccard for MinHash (~0.10 is already a real duplicate). useEffect
  // below resets the default whenever the cache reports a new provider.
  const [cosineThreshold, setCosineThreshold] = useState<number>(0.85);

  // task_id → true for batch reject selection (non-rep tasks only)
  const [selected, setSelected] = useState<Record<string, true>>({});
  const [rejecting, setRejecting] = useState(false);
  const [rejectStatus, setRejectStatus] = useState<string | null>(null);

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
        // Reset similarity-threshold default based on whichever provider
        // the cached payload is from — jina cosine and MinHash Jaccard
        // live on very different scales.
        const providerKey = (d.payload?.params as { provider?: string } | undefined)?.provider;
        if (providerKey === "minhash") {
          setCosineThreshold((cur) => (cur > 0.5 ? 0.10 : cur));
        } else if (providerKey === "jina_http") {
          setCosineThreshold((cur) => (cur < 0.5 ? 0.85 : cur));
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, storeKey, profile]);

  function reloadCache() {
    if (!projectId) return;
    fetch(buildDistributionUrl(projectId, storeKey, profile))
      .then((r) => (r.ok ? (r.json() as Promise<CacheResponse>) : null))
      .then((d) => { if (d) setCache(d); })
      .catch(() => {});
  }

  // ── Scan (POST) ───────────────────────────────────────────────────────────
  async function handleScan() {
    if (!projectId) {
      setError("Select a project first.");
      return;
    }
    setLoading(true);
    setError(null);
    setRejectStatus(null);
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
        setSelected({});
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
          setSelected({});
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

  // ── Reject (per-cluster POST) ─────────────────────────────────────────────
  async function handleReject(filteredClusters: ClusterEntry[]) {
    if (!projectId) return;
    setRejecting(true);
    setRejectStatus(null);
    setError(null);
    let totalMoved = 0;
    let totalSkipped = 0;
    try {
      const storeQs = buildStoreQs(storeKey);
      for (const cluster of filteredClusters) {
        const rep = pickRepresentative(cluster.task_ids);
        const toReject = cluster.task_ids.filter(
          (id) => id !== rep && selected[id] === true,
        );
        if (toReject.length === 0) continue;
        const r = await fetch(
          `/api/distribution/reject?project=${encodeURIComponent(projectId)}${storeQs ? "&" + storeQs.slice(1) : ""}`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              task_ids: toReject,
              cluster_id: cluster.cluster_id,
              representative_task_id: rep,
              cluster_similarity: cluster.similarity,
              embedding_profile: profile,
              embedding_model: getEmbeddingModel(profile),
              actor: "operator",
            }),
          },
        );
        if (!r.ok) {
          const txt = await r.text();
          throw new Error(`HTTP ${r.status}: ${txt.slice(0, 200)}`);
        }
        const result = (await r.json()) as { moved: number; skipped: number; skipped_task_ids: string[] };
        totalMoved += result.moved;
        totalSkipped += result.skipped;
      }
      setRejectStatus(`Rejected ${totalMoved} task(s)${totalSkipped > 0 ? `, skipped ${totalSkipped}` : ""}.`);
      setSelected({});
      reloadCache();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRejecting(false);
    }
  }

  // ── Derived state ─────────────────────────────────────────────────────────
  const payload = cache?.payload ?? null;
  const stale = cache?.stale ?? false;
  const generatedAt = cache?.generated_at ?? null;
  const cachedExists = cache?.cached ?? false;
  const availableProfiles = cache?.available_profiles ?? ["jina_small", "random_baseline"];

  // Filter clusters for Duplicates sub-tab.
  const filteredClusters: ClusterEntry[] = (payload?.clusters ?? []).filter(
    (c) =>
      c.method === "embedding" &&
      c.task_ids.length >= 2 &&
      c.similarity >= cosineThreshold,
  );

  // Build task_id → CoordEntry lookup for text previews.
  const coordMap = new Map<string, CoordEntry>();
  for (const coord of payload?.coords ?? []) {
    coordMap.set(coord.task_id, coord);
  }

  // All selectable task_ids (non-reps in filtered clusters).
  const allSelectableIds: string[] = [];
  for (const cluster of filteredClusters) {
    const rep = pickRepresentative(cluster.task_ids);
    for (const id of cluster.task_ids) {
      if (id !== rep) allSelectableIds.push(id);
    }
  }

  const selectedCount = Object.keys(selected).length;
  const affectedClusters = filteredClusters.filter((c) => {
    const rep = pickRepresentative(c.task_ids);
    return c.task_ids.some((id) => id !== rep && selected[id] === true);
  });

  function handleSelectAll() {
    const next: Record<string, true> = {};
    for (const id of allSelectableIds) {
      next[id] = true;
    }
    setSelected(next);
  }

  function handleClearAll() {
    setSelected({});
  }

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
            Embed tasks, project to 2-D with UMAP, and find duplicate clusters with HDBSCAN.
            {cachedExists ? (
              <>
                {" · "}
                <strong>Last scan:</strong> {fmtTime(generatedAt)}
                {" — "}
                {stale ? (
                  <span style={{ color: "var(--warning, #d97706)", fontWeight: 600 }}>
                    content has changed since this scan{" "}
                    <span className="runtime-muted" style={{ fontSize: "0.8rem", fontWeight: 400 }}>
                      ({cache?.cached_content_hash} → {cache?.current_content_hash})
                    </span>
                  </span>
                ) : (
                  <span className="runtime-muted">in sync with current content</span>
                )}
              </>
            ) : (
              <span className="runtime-muted"> · no cached scan yet</span>
            )}
          </p>
        </div>
        {/* ── Controls ─────────────────────────────────────────────────── */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
          <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.35rem" }}>
            Profile
            <select
              value={profile}
              onChange={(e) => {
                setProfile(e.target.value);
                setSelected({});
              }}
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
      </div>

      {error ? <div className="notice compact">{error}</div> : null}
      {rejectStatus ? (
        <div className="notice compact" style={{ color: "var(--success, #047857)" }}>
          {rejectStatus}
        </div>
      ) : null}

      {!cachedExists && !loading ? (
        <p className="runtime-muted">
          No cached scan yet. Click <strong>Scan</strong> to run the first embedding + UMAP + HDBSCAN pass.
        </p>
      ) : null}

      {cachedExists ? (
        <>
          {/* ── Sub-tab nav ─────────────────────────────────────────────── */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginTop: "0.5rem" }}>
            <nav className="view-tabs" aria-label="Distribution sections" style={{ margin: 0 }}>
              <button
                className={subtab === "duplicates" ? "view-tab selected" : "view-tab"}
                type="button"
                onClick={() => setSubtab("duplicates")}
              >
                Duplicates ({filteredClusters.length} clusters)
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

          {/* ── Duplicates sub-tab ────────────────────────────────────── */}
          {subtab === "duplicates" ? (
            <DuplicatesSubTab
              filteredClusters={filteredClusters}
              coordMap={coordMap}
              cosineThreshold={cosineThreshold}
              setCosineThreshold={setCosineThreshold}
              minClusterSize={minClusterSize}
              setMinClusterSize={setMinClusterSize}
              selected={selected}
              setSelected={setSelected}
              selectedCount={selectedCount}
              affectedClusters={affectedClusters}
              allSelectableIds={allSelectableIds}
              rejecting={rejecting}
              onSelectAll={handleSelectAll}
              onClearAll={handleClearAll}
              onReject={() => handleReject(filteredClusters)}
              onSelectTask={onSelectTask}
            />
          ) : null}

          {/* ── Scatter sub-tab (T6) ─────────────────────────────────── */}
          {subtab === "scatter" ? (
            <ScatterSubTab
              coords={payload?.coords ?? []}
              onSelectTask={onSelectTask}
            />
          ) : null}
        </>
      ) : null}
    </section>
  );
}

// ── DuplicatesSubTab ─────────────────────────────────────────────────────────

type DuplicatesSubTabProps = {
  filteredClusters: ClusterEntry[];
  coordMap: Map<string, CoordEntry>;
  cosineThreshold: number;
  setCosineThreshold: (v: number) => void;
  minClusterSize: number;
  setMinClusterSize: (v: number) => void;
  selected: Record<string, true>;
  setSelected: React.Dispatch<React.SetStateAction<Record<string, true>>>;
  selectedCount: number;
  affectedClusters: ClusterEntry[];
  allSelectableIds: string[];
  rejecting: boolean;
  onSelectAll: () => void;
  onClearAll: () => void;
  onReject: () => void;
  onSelectTask?: (taskId: string) => void;
};

function DuplicatesSubTab({
  filteredClusters,
  coordMap,
  cosineThreshold,
  setCosineThreshold,
  selected,
  setSelected,
  selectedCount,
  affectedClusters,
  allSelectableIds,
  rejecting,
  onSelectAll,
  onClearAll,
  onReject,
  onSelectTask,
}: DuplicatesSubTabProps): React.ReactElement {
  function toggleTask(taskId: string, checked: boolean) {
    setSelected((prev) => {
      const next = { ...prev };
      if (checked) {
        next[taskId] = true;
      } else {
        delete next[taskId];
      }
      return next;
    });
  }

  return (
    <div style={{ marginTop: "0.75rem" }}>
      {/* ── Filter controls ──────────────────────────────────────────── */}
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
        <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <span>Similarity ≥</span>
          {/* Slider range covers both regimes: 0.01-0.99 so MinHash (Jaccard,
              real duplicates around 0.10-0.30) and jina (cosine, real
              duplicates around 0.85+) both have headroom. */}
          <input
            type="range"
            min={0.01}
            max={0.99}
            step={0.01}
            value={cosineThreshold}
            onChange={(e) => setCosineThreshold(parseFloat(e.target.value))}
            style={{ width: "160px" }}
          />
          <code style={{ fontFamily: "monospace", minWidth: "3rem" }}>
            {cosineThreshold.toFixed(2)}
          </code>
        </label>

        <div style={{ marginLeft: "auto", display: "flex", gap: "0.4rem" }}>
          <button
            type="button"
            onClick={onSelectAll}
            disabled={allSelectableIds.length === 0}
            style={{ fontSize: "0.8rem" }}
            title="Select all non-representative tasks in visible clusters"
          >
            Select all ({allSelectableIds.length})
          </button>
          <button
            type="button"
            onClick={onClearAll}
            disabled={selectedCount === 0}
            style={{ fontSize: "0.8rem" }}
          >
            Clear
          </button>
        </div>
      </div>

      {/* ── Empty state ───────────────────────────────────────────────── */}
      {filteredClusters.length === 0 ? (
        <p className="runtime-muted">
          No duplicate clusters at cosine ≥ {cosineThreshold.toFixed(2)}. Lower the threshold or re-scan.
        </p>
      ) : null}

      {/* ── Cluster cards ─────────────────────────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        {filteredClusters.map((cluster) => {
          const rep = pickRepresentative(cluster.task_ids);
          const members = cluster.task_ids.filter((id) => id !== rep);
          const preview5 = members.slice(0, 5);
          return (
            <div
              key={cluster.cluster_id}
              className="runtime-card"
              style={{ padding: "0.75rem 1rem" }}
            >
              {/* ── Cluster header ────────────────────────────────────── */}
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
                  size={cluster.task_ids.length}
                </span>
                <span className="runtime-muted" style={{ fontSize: "0.8rem" }}>
                  cos={cluster.similarity.toFixed(3)}
                </span>
                <span className="runtime-muted" style={{ fontSize: "0.8rem" }}>
                  rep=
                  <button
                    type="button"
                    onClick={() => onSelectTask?.(rep)}
                    title="Open task drawer"
                    style={{
                      fontFamily: "monospace", fontSize: "0.8rem",
                      background: "transparent", border: "none", padding: 0,
                      color: "inherit", textDecoration: "underline",
                      textUnderlineOffset: "2px", cursor: "pointer",
                    }}
                  >
                    {rep}
                  </button>
                </span>
              </div>

              {/* ── Member rows ──────────────────────────────────────── */}
              {/* Each row: checkbox toggles batch-reject selection (independent);
                  the task-id button + preview opens the drawer for inspection. */}
              <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                {preview5.map((taskId) => {
                  const coord = coordMap.get(taskId);
                  const preview = coord?.text_preview ?? "";
                  const isChecked = selected[taskId] === true;
                  return (
                    <div
                      key={taskId}
                      style={{
                        display: "flex",
                        alignItems: "flex-start",
                        gap: "0.5rem",
                        fontSize: "0.82rem",
                        padding: "0.2rem 0",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={(e) => toggleTask(taskId, e.target.checked)}
                        title="Include in batch reject"
                        style={{ marginTop: "2px", flexShrink: 0, cursor: "pointer" }}
                      />
                      <button
                        type="button"
                        onClick={() => onSelectTask?.(taskId)}
                        title="Open task drawer"
                        style={{
                          flex: 1, textAlign: "left",
                          background: "transparent", border: "none", padding: 0,
                          color: "inherit", cursor: "pointer", font: "inherit",
                        }}
                      >
                        <code
                          style={{
                            fontFamily: "monospace", fontSize: "0.8rem",
                            textDecoration: "underline", textUnderlineOffset: "2px",
                          }}
                        >
                          {taskId}
                        </code>
                        {preview ? (
                          <span className="runtime-muted" style={{ marginLeft: "0.4rem" }}>
                            — {preview.slice(0, 80)}
                            {preview.length > 80 ? "…" : ""}
                          </span>
                        ) : null}
                      </button>
                    </div>
                  );
                })}
                {members.length > 5 ? (
                  <span className="runtime-muted" style={{ fontSize: "0.78rem", paddingLeft: "1.6rem" }}>
                    … and {members.length - 5} more member(s)
                  </span>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Batch reject footer ───────────────────────────────────────── */}
      {filteredClusters.length > 0 ? (
        <div
          style={{
            marginTop: "1rem",
            padding: "0.6rem 0.75rem",
            background: "var(--surface2, #f8fafc)",
            borderRadius: "6px",
            border: "1px solid var(--border, #e5e7eb)",
            display: "flex",
            alignItems: "center",
            gap: "1rem",
            flexWrap: "wrap",
          }}
        >
          <span style={{ fontSize: "0.85rem" }}>
            {selectedCount > 0 ? (
              <>
                <strong>{selectedCount}</strong> task{selectedCount !== 1 ? "s" : ""} across{" "}
                <strong>{affectedClusters.length}</strong> cluster{affectedClusters.length !== 1 ? "s" : ""} will be rejected
              </>
            ) : (
              <span className="runtime-muted">No tasks selected — check boxes above to select duplicates to reject</span>
            )}
          </span>
          <button
            type="button"
            disabled={selectedCount === 0 || rejecting}
            onClick={onReject}
            style={{
              fontSize: "0.85rem",
              background:
                selectedCount > 0 && !rejecting
                  ? "var(--danger, #b91c1c)"
                  : undefined,
              color: selectedCount > 0 && !rejecting ? "white" : undefined,
              borderColor: selectedCount > 0 && !rejecting ? "var(--danger, #b91c1c)" : undefined,
              opacity: selectedCount === 0 || rejecting ? 0.6 : 1,
            }}
          >
            {rejecting ? "Rejecting…" : `Reject ${selectedCount > 0 ? selectedCount + " task" + (selectedCount !== 1 ? "s" : "") : ""}`}
          </button>
        </div>
      ) : null}
    </div>
  );
}

// ── ScatterSubTab ─────────────────────────────────────────────────────────────

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

type ScatterSubTabProps = {
  coords: CoordEntry[];
  onSelectTask?: (taskId: string) => void;
};

function ScatterSubTab({ coords, onSelectTask }: ScatterSubTabProps): React.ReactElement {
  // Per-status visibility toggles; default all on.
  const [visible, setVisible] = useState<Record<string, boolean>>(() => {
    const init: Record<string, boolean> = {};
    for (const s of ALL_STATUSES) init[s] = true;
    return init;
  });

  // Count all coords per status (unfiltered), used for toggle labels.
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

  // Empty state — shown when no scan data is available yet.
  if (coords.length === 0) {
    return (
      <div className="runtime-muted" style={{ padding: "2rem", textAlign: "center" }}>
        No scan data yet — run <strong>[Re-]Scan</strong> first.
      </div>
    );
  }

  // Filter coords client-side using visibility toggles.
  const filteredCoords = coords.filter((c) => visible[c.status] !== false);

  // Build one trace per status (always 10 traces — stable legend order).
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

  // Force axis auto-range on every render so a fresh scan (especially
  // when switching profiles between MinHash and jina_small — the coord
  // ranges differ by ~100×) re-fits the screen instead of keeping the
  // previous zoom. `uirevision` ties this re-fit to the data identity:
  // toggling status filters doesn't reset zoom (same uirevision), but
  // running a new scan does (different first task_id / coord count
  // produce a new revision string).
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
      {/* ── Status filter checkboxes ──────────────────────────────────── */}
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

      {/* ── Plotly scatter ────────────────────────────────────────────── */}
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

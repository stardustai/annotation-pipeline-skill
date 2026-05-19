import React, { useEffect, useState } from "react";

export type TypeStatisticsPanelProps = {
  projectId: string | null;
  storeKey: string | null;
};

type TypeRow = { tasks: number; occurrences?: number; phrases?: number };

type Payload = {
  entities: Record<string, TypeRow>;
  json_structures: Record<string, TypeRow>;
  scanned_tasks: number;
  skipped_tasks: number;
};

type Response = {
  cached: boolean;
  payload: Payload | null;
  generated_at: string | null;
};

const SECTION_STYLE: React.CSSProperties = { marginTop: "1rem" };
const TABLE_STYLE: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: "0.85rem",
};
const TH: React.CSSProperties = { padding: "0.4rem 0.6rem", textAlign: "left" };
const TD: React.CSSProperties = { padding: "0.35rem 0.6rem" };
const TR: React.CSSProperties = { borderBottom: "1px solid var(--border, #e5e7eb)" };

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  let tz = "";
  try {
    const parts = new Intl.DateTimeFormat(undefined, { timeZoneName: "short" }).formatToParts(d);
    tz = parts.find((p) => p.type === "timeZoneName")?.value ?? "";
  } catch {
    tz = "";
  }
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}${tz ? " " + tz : ""}`;
}

function Bar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div
      style={{
        width: "100%",
        height: "8px",
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
        }}
      />
    </div>
  );
}

function DistributionTable({
  title,
  rows,
  occurrenceLabel,
  occurrenceKey,
}: {
  title: string;
  rows: Record<string, TypeRow>;
  occurrenceLabel: string;
  occurrenceKey: "occurrences" | "phrases";
}) {
  const entries = Object.entries(rows);
  const maxOcc = entries.reduce((m, [, r]) => Math.max(m, (r[occurrenceKey] ?? 0)), 0);
  const maxTasks = entries.reduce((m, [, r]) => Math.max(m, r.tasks), 0);
  const totalOcc = entries.reduce((s, [, r]) => s + (r[occurrenceKey] ?? 0), 0);
  const totalTasks = entries.reduce((s, [, r]) => s + r.tasks, 0);
  return (
    <div style={SECTION_STYLE}>
      <h3 style={{ marginBottom: "0.25rem", fontSize: "1rem" }}>
        {title}{" "}
        <span className="runtime-muted" style={{ fontWeight: 400, fontSize: "0.8rem" }}>
          · {entries.length} types · {totalOcc.toLocaleString()} total {occurrenceLabel}
        </span>
      </h3>
      {entries.length === 0 ? (
        <p className="runtime-muted">No data.</p>
      ) : (
        <div className="runtime-card">
          <table style={TABLE_STYLE}>
            <thead>
              <tr style={{ ...TR, fontWeight: 600 }}>
                <th style={{ ...TH, width: "18%" }}>Type</th>
                <th style={{ ...TH, width: "14%" }}>{occurrenceLabel}</th>
                <th style={{ ...TH, width: "30%" }}>{occurrenceLabel} share</th>
                <th style={{ ...TH, width: "14%" }}>Tasks</th>
                <th style={{ ...TH, width: "24%" }}>Task coverage</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([typ, row]) => {
                const occ = row[occurrenceKey] ?? 0;
                return (
                  <tr key={typ} style={TR}>
                    <td style={{ ...TD, fontFamily: "monospace" }}>{typ}</td>
                    <td style={TD}>{occ.toLocaleString()}</td>
                    <td style={TD}>
                      <Bar value={occ} max={maxOcc} />
                      <span className="runtime-muted" style={{ fontSize: "0.7rem" }}>
                        {totalOcc > 0 ? Math.round((occ / totalOcc) * 100) : 0}% of all
                      </span>
                    </td>
                    <td style={TD}>{row.tasks.toLocaleString()}</td>
                    <td style={TD}>
                      <Bar value={row.tasks} max={maxTasks} />
                      <span className="runtime-muted" style={{ fontSize: "0.7rem" }}>
                        {totalTasks > 0 ? Math.round((row.tasks / maxTasks) * 100) : 0}% of max
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function TypeStatisticsPanel({
  projectId,
  storeKey,
}: TypeStatisticsPanelProps): React.ReactElement {
  const [data, setData] = useState<Response | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function urlWithStore(base: string): string {
    if (!projectId) return base;
    const storeQ = storeKey ? `&store=${encodeURIComponent(storeKey)}` : "";
    return `${base}?project=${encodeURIComponent(projectId)}${storeQ}`;
  }

  useEffect(() => {
    if (!projectId) {
      setData(null);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(urlWithStore("/api/type-statistics"))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => setData(d))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, storeKey]);

  async function rebuild() {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(urlWithStore("/api/type-statistics"), { method: "POST" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const launch = await r.json();
      const job = launch?.job;
      if (!job?.job_id) {
        // Legacy sync response — treat as already-done.
        setData(launch);
        return;
      }
      // Background job started. Poll status every 2s until done; UI
      // stays responsive (operator can navigate elsewhere and come
      // back). Cap polling at ~5 min so we don't loop forever on a
      // stuck job.
      const deadline = Date.now() + 5 * 60 * 1000;
      while (Date.now() < deadline) {
        await new Promise((res) => setTimeout(res, 2000));
        const jr = await fetch(urlWithStore("/api/jobs/" + encodeURIComponent(job.job_id)));
        if (!jr.ok) continue;
        const jdata = await jr.json();
        if (jdata.status === "done") {
          // Refetch the GET endpoint to pull the freshly-written cache.
          const fresh = await fetch(urlWithStore("/api/type-statistics"));
          if (fresh.ok) setData(await fresh.json());
          return;
        }
        if (jdata.status === "error") {
          throw new Error(jdata.error || "background job failed");
        }
      }
      throw new Error("background job timed out (5 min); check server logs");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  if (!projectId) {
    return (
      <section className="runtime-panel" aria-label="Statistics">
        <p className="runtime-muted">Select a project first.</p>
      </section>
    );
  }

  return (
    <section className="runtime-panel" aria-label="Statistics">
      <div className="runtime-header">
        <div>
          <h2 style={{ marginBottom: "0.25rem" }}>Statistics</h2>
          <p style={{ marginTop: 0 }}>
            Project-wide distribution of annotation type usage across ACCEPTED
            tasks. <strong>Occurrences</strong> = every (span, type) pair
            counted, including duplicates across rows.{" "}
            <strong>Tasks</strong> = distinct tasks where the type appears at
            least once. Rebuild after major imports or bulk operations to
            re-scan annotations.
            {data?.generated_at ? (
              <>
                {" "}
                <span className="runtime-muted" style={{ fontSize: "0.85em" }}>
                  · Last computed: {fmtTime(data.generated_at)}
                </span>
              </>
            ) : null}
            {data?.payload ? (
              <>
                {" "}
                <span className="runtime-muted" style={{ fontSize: "0.85em" }}>
                  · {data.payload.scanned_tasks.toLocaleString()} tasks scanned
                  {data.payload.skipped_tasks > 0
                    ? ` (${data.payload.skipped_tasks} skipped)`
                    : ""}
                </span>
              </>
            ) : null}
          </p>
        </div>
        <button
          className="primary-button"
          type="button"
          onClick={rebuild}
          disabled={loading}
        >
          {loading ? "Scanning…" : data?.payload ? "Rebuild" : "Build"}
        </button>
      </div>

      {error ? <div className="notice compact">{error}</div> : null}

      {!data?.payload && !loading ? (
        <p className="runtime-muted" style={{ marginTop: "1rem" }}>
          No cached statistics yet. Click <strong>Build</strong> to scan
          ACCEPTED tasks.
        </p>
      ) : null}

      {data?.payload ? (
        <>
          <DistributionTable
            title="Entity types"
            rows={data.payload.entities}
            occurrenceLabel="Occurrences"
            occurrenceKey="occurrences"
          />
          <DistributionTable
            title="JSON-structure phrase types"
            rows={data.payload.json_structures}
            occurrenceLabel="Phrases"
            occurrenceKey="phrases"
          />
        </>
      ) : null}
    </section>
  );
}

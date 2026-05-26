import { useEffect, useState } from "react";
import { fetchOutboxSummary, fetchReadinessReport } from "../api";
import { outboxFacts, outboxRecordTitle } from "../outbox";
import { readinessFacts, readinessTitle } from "../readiness";
import type { OutboxSummary, ReadinessReport } from "../types";

interface OutputPanelProps {
  projectId: string | null;
  storeKey: string | null;
  storePath: string | null;
}

export function OutputPanel({ projectId, storeKey, storePath }: OutputPanelProps) {
  const [readiness, setReadiness] = useState<ReadinessReport | null>(null);
  const [outbox, setOutbox] = useState<OutboxSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    const readinessPromise = projectId
      ? fetchReadinessReport(projectId, storeKey)
      : Promise.resolve(null);
    Promise.all([readinessPromise, fetchOutboxSummary(projectId, storeKey)])
      .then(([nextReadiness, nextOutbox]) => {
        if (!active) return;
        setReadiness(nextReadiness);
        setOutbox(nextOutbox);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "Unable to load output data");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [projectId, storeKey]);

  if (loading) return <section className="runtime-panel">Loading output…</section>;

  return (
    <section className="runtime-panel" aria-label="Export and delivery">
      {error ? <div className="notice compact">{error}</div> : null}
      <div className="runtime-header">
        <div>
          <h2>Output</h2>
          <p>{projectId ?? "All projects"}</p>
        </div>
      </div>

      {/* Data Storage */}
      <h3 style={{ marginTop: "1.5rem", borderBottom: "1px solid var(--border, #2a2f3a)", paddingBottom: "0.5rem" }}>Data Storage</h3>
      {storePath ? (
        <div className="runtime-card">
          <dl className="runtime-facts">
            <div>
              <dt>Project root</dt>
              <dd><code>{storePath}</code></dd>
            </div>
            <div>
              <dt>Database</dt>
              <dd><code>{storePath}/.annotation-pipeline/db.sqlite</code></dd>
            </div>
            <div>
              <dt>Annotation artifacts</dt>
              <dd><code>{storePath}/.annotation-pipeline/artifact_payloads/</code></dd>
            </div>
            <div>
              <dt>Export outputs</dt>
              <dd><code>{storePath}/.annotation-pipeline/exports/</code></dd>
            </div>
          </dl>
          {projectId ? (
            <>
              <p className="runtime-muted" style={{ marginTop: "0.75rem" }}>Export accepted tasks to JSONL training data:</p>
              <pre className="json-block">{`annotation-pipeline export training-data --project-root ${storePath} --project-id ${projectId}`}</pre>
            </>
          ) : null}
        </div>
      ) : (
        <p className="runtime-muted">Select a project store to see storage paths.</p>
      )}

      {/* Readiness */}
      <h3 style={{ marginTop: "1.5rem", borderBottom: "1px solid var(--border, #2a2f3a)", paddingBottom: "0.5rem" }}>Export Readiness</h3>
      {!projectId ? (
        <p className="runtime-muted">Select a project to see export readiness.</p>
      ) : !readiness ? (
        <p className="runtime-muted">Readiness report unavailable.</p>
      ) : (
        <>
          <div className="runtime-card">
              <dl className="runtime-facts">
                {readinessFacts(readiness).map((fact) => (
                  <div key={fact.label}>
                    <dt>{fact.label}</dt>
                    <dd>
                      {fact.value}
                      <small style={{ display: "block", color: "var(--muted, #6b7280)", fontWeight: "normal", marginTop: "0.15rem" }}>{fact.description}</small>
                    </dd>
                  </div>
                ))}
              </dl>
            </div>

          <div className="runtime-card">
            <h3>Next Action</h3>
            <p className="runtime-muted">{readinessTitle(readiness)}</p>
            {readiness.next_command ? <pre className="json-block">{readiness.next_command}</pre> : null}
            {readiness.validation_blockers.length > 0 ? (
              <ul className="runtime-list compact-list">
                {readiness.validation_blockers.map((blocker) => (
                  <li key={`${String(blocker.task_id)}-${String(blocker.reason)}`}>
                    {String(blocker.task_id)}: {String(blocker.reason)}
                    {Array.isArray(blocker.errors) && blocker.errors.length > 0 ? (
                      <small>{blocker.errors.map((e) => String(e)).join(", ")}</small>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        </>
      )}

      {/* Export History */}
      <h3 style={{ marginTop: "1.5rem", borderBottom: "1px solid var(--border, #2a2f3a)", paddingBottom: "0.5rem" }}>Export History</h3>
      {!projectId ? (
        <p className="runtime-muted">Select a project to see export history.</p>
      ) : !readiness ? (
        <p className="runtime-muted">Export history unavailable.</p>
      ) : readiness.exports.length === 0 ? (
        <p className="runtime-muted">No exports recorded.</p>
      ) : (
        <div className="runtime-card">
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border, #2a2f3a)", textAlign: "left" }}>
                <th style={{ padding: "0.4rem 0.75rem 0.4rem 0" }}>Export ID</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Created</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Included</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Excluded</th>
                <th style={{ padding: "0.4rem 0.75rem" }}>Output path</th>
                <th style={{ padding: "0.4rem 0" }}></th>
              </tr>
            </thead>
            <tbody>
              {readiness.exports.map((exp) => {
                const relPath = exp.output_paths[0];
                const zipHref = `/api/export-zip?export_id=${encodeURIComponent(exp.export_id)}${storeKey ? `&store=${storeKey}` : ""}`;
                return (
                  <tr key={exp.export_id} style={{ borderBottom: "1px solid var(--border, #2a2f3a)" }}>
                    <td style={{ padding: "0.4rem 0.75rem 0.4rem 0", fontFamily: "monospace" }}>{exp.export_id}</td>
                    <td style={{ padding: "0.4rem 0.75rem", whiteSpace: "nowrap" }}>{exp.created_at.replace("T", " ").slice(0, 19)}</td>
                    <td style={{ padding: "0.4rem 0.75rem" }}>{exp.included}</td>
                    <td style={{ padding: "0.4rem 0.75rem" }}>{exp.excluded}</td>
                    <td style={{ padding: "0.4rem 0.75rem", fontFamily: "monospace", fontSize: "0.8rem" }}>{relPath ?? "—"}</td>
                    <td style={{ padding: "0.4rem 0" }}>
                      <a href={zipHref} download={`${exp.export_id}.zip`} style={{ fontSize: "0.8rem", whiteSpace: "nowrap" }}>
                        Download
                      </a>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Outbox */}
      <h3 style={{ marginTop: "1.5rem", borderBottom: "1px solid var(--border, #2a2f3a)", paddingBottom: "0.5rem" }}>External Delivery</h3>
      <p className="runtime-muted" style={{ marginBottom: "0.75rem" }}>
        用于把导出的训练数据通过 webhook 推送到外部系统（如训练平台）。工作流程分三步：
      </p>
      <ol className="runtime-muted" style={{ margin: "0 0 0.75rem 1.25rem", lineHeight: "1.8" }}>
        <li>
          Export 时加 <code>--enqueue-external-submit</code>，系统会为每个有外部来源（<code>external_ref</code>）的任务在 outbox 里写入一条待投递记录，payload 为完整的 training row。
        </li>
        <li>
          手动跑 <code>annotation-pipeline outbox drain --project-root &lt;path&gt;</code> 触发投递——系统读取 <code>callbacks.yaml</code> 里配置的 <code>submit</code> webhook URL，HTTP POST 每条记录，失败自动重试最多 3 次，耗尽重试后进入 Dead Letters。
        </li>
        <li>
          在 <code>callbacks.yaml</code> 里配置目标地址：
          <pre className="json-block" style={{ marginTop: "0.4rem" }}>{`callbacks:\n  submit:\n    enabled: true\n    url: https://your-platform/api/submit\n    secret_env: MY_TOKEN   # 可选，用作 Bearer token`}</pre>
        </li>
      </ol>
      <p className="runtime-muted" style={{ marginBottom: "0.75rem" }}>
        如果任务不是从外部系统导入的，或者不需要自动推送，这里会一直为空。
      </p>
      {outbox ? (
        <div className="runtime-grid">
            <div className="runtime-card">
              <h3>Delivery State</h3>
              <dl className="runtime-facts">
                {outboxFacts(outbox).map((fact) => (
                  <div key={fact.label}>
                    <dt>{fact.label}</dt>
                    <dd>
                      {fact.value}
                      <small style={{ display: "block", color: "var(--muted, #6b7280)", fontWeight: "normal", marginTop: "0.15rem" }}>{fact.description}</small>
                    </dd>
                  </div>
                ))}
              </dl>
            </div>

            <div className="runtime-card">
              <h3>Records</h3>
              <div className="outbox-list">
                {outbox.records.map((record) => (
                  <details className="timeline-item" key={record.record_id}>
                    <summary>
                      <span>{record.task_id}</span>
                      <small>{outboxRecordTitle(record)} · retries {record.retry_count}</small>
                    </summary>
                    <dl className="runtime-facts compact">
                      <div>
                        <dt>Next retry</dt>
                        <dd>{record.next_retry_at ?? "none"}</dd>
                      </div>
                      <div>
                        <dt>Last error</dt>
                        <dd>{record.last_error ?? "none"}</dd>
                      </div>
                    </dl>
                    <pre className="json-block">{JSON.stringify(record.payload, null, 2)}</pre>
                  </details>
                ))}
              </div>
            </div>
          </div>
      ) : <p className="runtime-muted">Outbox unavailable.</p>}
    </section>
  );
}

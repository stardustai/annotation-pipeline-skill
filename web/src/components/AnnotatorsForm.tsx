import { useEffect, useState } from "react";
import { fetchAnnotatorsConfig, saveAnnotatorsConfig } from "../api";

interface AnnotatorsFormProps {
  storeKey: string | null;
}

interface QcSampling {
  strategy: string;
  ratio: number;
  batch_size: number;
  threshold: number;
  require_all_batches_pass: boolean;
}

const defaultQcSampling: QcSampling = {
  strategy: "stratified",
  ratio: 1.0,
  batch_size: 10,
  threshold: 16,
  require_all_batches_pass: true,
};

function parseQcSampling(raw: Record<string, unknown> | undefined): QcSampling {
  if (!raw) return { ...defaultQcSampling };
  return {
    strategy: typeof raw.strategy === "string" ? raw.strategy : defaultQcSampling.strategy,
    ratio: typeof raw.ratio === "number" ? raw.ratio : defaultQcSampling.ratio,
    batch_size: typeof raw.batch_size === "number" ? raw.batch_size : defaultQcSampling.batch_size,
    threshold: typeof raw.threshold === "number" ? raw.threshold : defaultQcSampling.threshold,
    require_all_batches_pass:
      typeof raw.require_all_batches_pass === "boolean"
        ? raw.require_all_batches_pass
        : defaultQcSampling.require_all_batches_pass,
  };
}

export function AnnotatorsForm({ storeKey }: AnnotatorsFormProps) {
  const [qcSampling, setQcSampling] = useState<QcSampling>(defaultQcSampling);
  const [otherSampling, setOtherSampling] = useState<Record<string, Record<string, unknown>>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchAnnotatorsConfig(storeKey)
      .then((snap) => {
        if (!active) return;
        const { qc, ...rest } = snap.sampling;
        setQcSampling(parseQcSampling(qc as Record<string, unknown> | undefined));
        setOtherSampling(rest);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "Unable to load annotators");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [storeKey]);

  async function submit() {
    setSaving(true);
    setMessage(null);
    setError(null);
    try {
      const sampling: Record<string, Record<string, unknown>> = { ...otherSampling };
      sampling.qc = { ...qcSampling };
      // stage_targets is NOT sent from this form anymore — it lives in
      // the Providers tab as the single editing point. Backend's save
      // handler tolerates the field being absent (only updates
      // llm_profiles.yaml when explicitly supplied).
      await saveAnnotatorsConfig({ sampling }, storeKey);
      setMessage("Saved");
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Unable to save");
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <div className="drawer-state">Loading annotators…</div>;

  return (
    <div className="annotators-form">
      <div className="config-editor-header">
        <div>
          <h2>Annotation Agents</h2>
          <p>
            Tune QC sampling here. Stage → profile routing moved to the{" "}
            <strong>Providers</strong> tab (single source of truth).
          </p>
        </div>
        <button className="primary-button" type="button" disabled={saving} onClick={submit}>
          {saving ? "Saving" : "Save"}
        </button>
      </div>

      {error ? <div className="drawer-error">{error}</div> : null}
      {message ? <div className="notice compact">{message}</div> : null}

      <section className="annotators-section">
        <h3>QC Sampling</h3>
        <div className="annotator-card-fields">
          <label>
            <span>Strategy</span>
            <select
              value={qcSampling.strategy}
              onChange={(e) => setQcSampling({ ...qcSampling, strategy: e.target.value })}
            >
              <option value="stratified">stratified</option>
              <option value="random">random</option>
              <option value="sample_all">sample_all</option>
            </select>
          </label>
          <label>
            <span>Ratio (0–1)</span>
            <input
              type="number"
              step="0.01"
              min={0}
              max={1}
              value={qcSampling.ratio}
              onChange={(e) => setQcSampling({ ...qcSampling, ratio: Number(e.target.value) })}
            />
          </label>
          <label>
            <span>Batch size</span>
            <input
              type="number"
              min={1}
              value={qcSampling.batch_size}
              onChange={(e) => setQcSampling({ ...qcSampling, batch_size: Number(e.target.value) })}
            />
          </label>
          <label>
            <span>Pass threshold</span>
            <input
              type="number"
              min={0}
              value={qcSampling.threshold}
              onChange={(e) => setQcSampling({ ...qcSampling, threshold: Number(e.target.value) })}
            />
          </label>
          <label className="checkbox-row wide">
            <input
              type="checkbox"
              checked={qcSampling.require_all_batches_pass}
              onChange={(e) => setQcSampling({ ...qcSampling, require_all_batches_pass: e.target.checked })}
            />
            Require all batches to pass
          </label>
        </div>
      </section>
    </div>
  );
}

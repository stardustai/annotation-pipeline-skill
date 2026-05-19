import React, { useEffect, useState } from "react";
import { fetchConfigFile, saveConfigFile } from "../api";

export type AnnotationRulesPanelProps = {
  storeKey: string | null;
};

const CONFIG_ID = "annotation_rules.yaml";

export function AnnotationRulesPanel({ storeKey }: AnnotationRulesPanelProps): React.ReactElement {
  const [content, setContent] = useState<string>("");
  const [originalContent, setOriginalContent] = useState<string>("");
  const [path, setPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    setMessage(null);
    fetchConfigFile(CONFIG_ID, storeKey)
      .then((file) => {
        if (!active) return;
        setContent(file.content);
        setOriginalContent(file.content);
        setPath(file.path);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "Unable to load annotation_rules.yaml");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [storeKey]);

  async function save() {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      await saveConfigFile(CONFIG_ID, content, storeKey);
      setOriginalContent(content);
      setMessage("Saved. Next annotator / QC / arbiter pass will see the updated rules in their prompt.");
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  const dirty = content !== originalContent;

  return (
    <section className="runtime-panel" aria-label="Annotation Rules">
      <div className="runtime-header">
        <div>
          <h2 style={{ marginBottom: "0.25rem" }}>Annotation Rules</h2>
          <p style={{ marginTop: 0 }}>
            Project-level structured rules. The runtime injects this YAML's raw
            content into the annotator / QC / arbiter prompt as the
            "Annotation rules (project)" preamble — edits take effect on the
            next pass without restart.
            {path ? (
              <>
                {" "}
                <span className="runtime-muted" style={{ fontSize: "0.85em" }}>
                  · File: <code>{path}</code>
                </span>
              </>
            ) : null}
          </p>
        </div>
        <button
          className="primary-button"
          type="button"
          onClick={save}
          disabled={saving || loading || !dirty}
          title={!dirty ? "No changes to save" : "Write the YAML to disk"}
        >
          {saving ? "Saving…" : dirty ? "Save" : "Saved"}
        </button>
      </div>

      {error ? <div className="notice compact">{error}</div> : null}
      {message ? <div className="notice compact">{message}</div> : null}

      {loading ? (
        <p className="runtime-muted">Loading…</p>
      ) : (
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          spellCheck={false}
          style={{
            width: "100%",
            minHeight: "70vh",
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            fontSize: "0.85rem",
            lineHeight: 1.45,
            padding: "0.75rem",
            border: "1px solid var(--border, #d1d5db)",
            borderRadius: "4px",
            background: "#fafafa",
            resize: "vertical",
            boxSizing: "border-box",
          }}
        />
      )}
    </section>
  );
}

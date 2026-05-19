import React, { useEffect, useMemo, useState } from "react";
import {
  fetchAnnotationRulesDocument,
  createAnnotationRulesDocumentVersion,
  type AnnotationRulesDocumentSnapshot,
} from "../api";

export type AnnotationRulesPanelProps = {
  storeKey: string | null;
};

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export function AnnotationRulesPanel({ storeKey }: AnnotationRulesPanelProps): React.ReactElement {
  const [snapshot, setSnapshot] = useState<AnnotationRulesDocumentSnapshot | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [draftContent, setDraftContent] = useState<string>("");
  const [versionLabel, setVersionLabel] = useState<string>("");
  const [changelog, setChangelog] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const snap = await fetchAnnotationRulesDocument(storeKey);
      setSnapshot(snap);
      const latest = snap.versions[0] ?? null;
      setSelectedVersionId((prev) => prev ?? latest?.version_id ?? null);
      // Seed draft with latest content if user has not started editing.
      setDraftContent((prev) => (prev ? prev : latest?.content ?? ""));
      // Suggest next version label.
      setVersionLabel((prev) => (prev ? prev : suggestNextLabel(snap.versions.map((v) => v.version))));
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Unable to load annotation rules document");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let active = true;
    setSnapshot(null);
    setSelectedVersionId(null);
    setDraftContent("");
    setVersionLabel("");
    setChangelog("");
    setMessage(null);
    fetchAnnotationRulesDocument(storeKey)
      .then((snap) => {
        if (!active) return;
        setSnapshot(snap);
        const latest = snap.versions[0] ?? null;
        setSelectedVersionId(latest?.version_id ?? null);
        setDraftContent(latest?.content ?? "");
        setVersionLabel(suggestNextLabel(snap.versions.map((v) => v.version)));
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "Unable to load annotation rules document");
      });
    return () => {
      active = false;
    };
  }, [storeKey]);

  const selectedVersion = useMemo(() => {
    if (!snapshot) return null;
    return snapshot.versions.find((v) => v.version_id === selectedVersionId) ?? null;
  }, [snapshot, selectedVersionId]);

  const latestId = snapshot?.latest_version_id ?? null;
  const editingLatest = selectedVersionId !== null && selectedVersionId === latestId;
  const dirty = selectedVersion ? draftContent !== selectedVersion.content : draftContent.length > 0;

  async function saveNewVersion() {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const created = await createAnnotationRulesDocumentVersion(
        {
          version: versionLabel.trim() || undefined,
          content: draftContent,
          changelog: changelog.trim(),
          created_by: "operator",
        },
        storeKey,
      );
      setMessage(`Saved as ${created.version}. New annotator / QC / arbiter tasks will be stamped with this version.`);
      setChangelog("");
      await refresh();
      setSelectedVersionId(created.version_id);
      setVersionLabel(suggestNextLabel([created.version]));
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  function selectVersion(versionId: string) {
    const v = snapshot?.versions.find((x) => x.version_id === versionId) ?? null;
    setSelectedVersionId(versionId);
    setDraftContent(v?.content ?? "");
    setMessage(null);
    setError(null);
  }

  return (
    <section className="runtime-panel" aria-label="Annotation Rules">
      <div className="runtime-header">
        <div>
          <h2 style={{ marginBottom: "0.25rem" }}>Annotation Rules</h2>
          <p style={{ marginTop: 0 }}>
            Versioned, project-level annotation rules. The runtime injects the
            latest version into annotator / QC / arbiter prompts, and stamps
            every new task with the active version_id so future audits can
            reproduce which rule set the task was annotated against.
          </p>
        </div>
      </div>

      {error ? <div className="notice compact">{error}</div> : null}
      {message ? <div className="notice compact">{message}</div> : null}

      {loading && !snapshot ? (
        <p className="runtime-muted">Loading…</p>
      ) : !snapshot ? null : (
        <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: "1rem", marginTop: "0.5rem" }}>
          <aside
            style={{
              border: "1px solid var(--border, #d1d5db)",
              borderRadius: 4,
              padding: "0.5rem",
              maxHeight: "70vh",
              overflowY: "auto",
              background: "#fafafa",
            }}
          >
            <div style={{ fontWeight: 600, fontSize: "0.85rem", marginBottom: "0.4rem" }}>
              Versions ({snapshot.versions.length})
            </div>
            {snapshot.versions.length === 0 ? (
              <p className="runtime-muted" style={{ fontSize: "0.85rem" }}>
                No versions yet. Save the first version below.
              </p>
            ) : (
              snapshot.versions.map((v) => {
                const isSel = v.version_id === selectedVersionId;
                const isLatest = v.version_id === latestId;
                return (
                  <button
                    key={v.version_id}
                    type="button"
                    onClick={() => selectVersion(v.version_id)}
                    style={{
                      display: "block",
                      width: "100%",
                      textAlign: "left",
                      padding: "0.4rem 0.5rem",
                      marginBottom: "0.25rem",
                      border: "1px solid",
                      borderColor: isSel ? "#2563eb" : "#e5e7eb",
                      background: isSel ? "#eff6ff" : "#fff",
                      borderRadius: 3,
                      cursor: "pointer",
                      fontSize: "0.82rem",
                    }}
                  >
                    <div style={{ fontWeight: 600 }}>
                      {v.version}
                      {isLatest ? (
                        <span
                          style={{
                            marginLeft: 6,
                            fontSize: "0.7rem",
                            background: "#16a34a",
                            color: "#fff",
                            padding: "1px 5px",
                            borderRadius: 3,
                          }}
                        >
                          active
                        </span>
                      ) : null}
                    </div>
                    <div style={{ color: "#6b7280", fontSize: "0.72rem" }}>
                      {formatTimestamp(v.created_at)} · {v.created_by}
                    </div>
                    {v.changelog ? (
                      <div style={{ color: "#374151", fontSize: "0.75rem", marginTop: 2 }}>
                        {v.changelog.length > 80 ? `${v.changelog.slice(0, 80)}…` : v.changelog}
                      </div>
                    ) : null}
                  </button>
                );
              })
            )}
          </aside>

          <div>
            <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.4rem" }}>
              <strong style={{ fontSize: "0.9rem" }}>
                {selectedVersion
                  ? `Viewing ${selectedVersion.version}${editingLatest ? " (active)" : " (historical)"}`
                  : "New version"}
              </strong>
              {!editingLatest && selectedVersion ? (
                <span className="runtime-muted" style={{ fontSize: "0.78rem" }}>
                  Edits create a brand-new version; historical versions are immutable.
                </span>
              ) : null}
            </div>

            <textarea
              value={draftContent}
              onChange={(e) => setDraftContent(e.target.value)}
              spellCheck={false}
              style={{
                width: "100%",
                minHeight: "55vh",
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

            <div style={{ display: "grid", gridTemplateColumns: "160px 1fr auto", gap: "0.5rem", marginTop: "0.6rem" }}>
              <input
                type="text"
                placeholder="Version label (e.g. v2)"
                value={versionLabel}
                onChange={(e) => setVersionLabel(e.target.value)}
                style={{ padding: "0.4rem 0.55rem", border: "1px solid #d1d5db", borderRadius: 3, fontSize: "0.85rem" }}
              />
              <input
                type="text"
                placeholder="Changelog (what changed in this version?)"
                value={changelog}
                onChange={(e) => setChangelog(e.target.value)}
                style={{ padding: "0.4rem 0.55rem", border: "1px solid #d1d5db", borderRadius: 3, fontSize: "0.85rem" }}
              />
              <button
                className="primary-button"
                type="button"
                onClick={saveNewVersion}
                disabled={saving || !dirty || !draftContent.trim()}
                title={!dirty ? "No changes vs selected version" : "Create a new version"}
              >
                {saving ? "Saving…" : "Save as new version"}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function suggestNextLabel(existing: string[]): string {
  let maxN = 0;
  for (const v of existing) {
    const m = /^v(\d+)$/.exec(v.trim());
    if (m) {
      const n = parseInt(m[1], 10);
      if (n > maxN) maxN = n;
    }
  }
  return `v${maxN + 1}`;
}

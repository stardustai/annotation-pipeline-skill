import { useEffect, useMemo, useRef, useState } from "react";
import Editor, { type OnMount } from "@monaco-editor/react";
import { fetchProjectSchema, saveProjectSchema } from "../api";

interface SchemaPanelProps {
  storeKey: string | null;
}

function pretty(obj: unknown): string {
  return JSON.stringify(obj, null, 2);
}

export function SchemaPanel({ storeKey }: SchemaPanelProps) {
  const [originalText, setOriginalText] = useState<string>("");
  const [draftText, setDraftText] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    setSavedAt(null);
    fetchProjectSchema(storeKey)
      .then((result) => {
        if (!active) return;
        const text = result.schema ? pretty(result.schema) : "";
        setOriginalText(text);
        setDraftText(text);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "Unable to load schema");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [storeKey]);

  const dirty = draftText !== originalText;

  // Validate draft JSON parses (Monaco itself shows inline errors; we surface for the Save button).
  useEffect(() => {
    if (!draftText.trim()) {
      setParseError(null);
      return;
    }
    try {
      JSON.parse(draftText);
      setParseError(null);
    } catch (err) {
      setParseError(err instanceof Error ? err.message : "Invalid JSON");
    }
  }, [draftText]);

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    // Disable schema-against-self validation (this IS a schema document, not data).
    monaco.languages.json.jsonDefaults.setDiagnosticsOptions({
      validate: true,
      allowComments: false,
      schemaValidation: "warning",
    });
    // Default fold level 2 so the user sees top-level structure on open.
    editor.getAction("editor.foldLevel2")?.run();
  };

  async function handleSave() {
    if (!dirty || parseError) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(draftText);
    } catch (err) {
      setParseError(err instanceof Error ? err.message : "Invalid JSON");
      return;
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      setError("Schema must be a JSON object");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const result = await saveProjectSchema(parsed as Record<string, unknown>, storeKey);
      const newText = pretty(result.schema);
      setOriginalText(newText);
      setDraftText(newText);
      setSavedAt(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  function handleRevert() {
    setDraftText(originalText);
    setParseError(null);
    setError(null);
  }

  function handleFormat() {
    if (parseError) return;
    try {
      const parsed = JSON.parse(draftText);
      setDraftText(pretty(parsed));
    } catch {
      /* ignore — parseError already set by effect */
    }
  }

  const saveDisabled = !dirty || saving || parseError != null || loading;

  const statusText = useMemo(() => {
    if (loading) return "Loading…";
    if (saving) return "Saving…";
    if (parseError) return `JSON error: ${parseError}`;
    if (dirty) return "Unsaved changes";
    if (savedAt) return `Saved at ${savedAt}`;
    return "Up to date";
  }, [loading, saving, parseError, dirty, savedAt]);

  return (
    <section className="runtime-panel schema-panel" aria-label="Output schema">
      <div className="runtime-header">
        <div>
          <h2>Output Schema</h2>
          <p>Project-level JSON Schema used for annotation and QC validation</p>
        </div>
        <div className="schema-toolbar">
          <span className={`schema-status${parseError ? " error" : dirty ? " dirty" : ""}`}>
            {statusText}
          </span>
          <button
            type="button"
            className="json-viewer-link"
            onClick={handleFormat}
            disabled={loading || parseError != null}
            title="Reformat JSON with 2-space indent"
          >
            Format
          </button>
          <button
            type="button"
            className="json-viewer-link"
            onClick={handleRevert}
            disabled={!dirty || saving}
          >
            Revert
          </button>
          <button
            type="button"
            className="primary-button"
            onClick={handleSave}
            disabled={saveDisabled}
          >
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
      {error ? <div className="notice compact error">{error}</div> : null}
      <div className="schema-editor">
        <Editor
          height="calc(100vh - 260px)"
          defaultLanguage="json"
          language="json"
          theme="vs-dark"
          value={draftText}
          onChange={(value) => setDraftText(value ?? "")}
          onMount={handleMount}
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            lineNumbers: "on",
            folding: true,
            foldingStrategy: "indentation",
            showFoldingControls: "always",
            tabSize: 2,
            insertSpaces: true,
            formatOnPaste: true,
            formatOnType: true,
            scrollBeyondLastLine: false,
            wordWrap: "off",
            bracketPairColorization: { enabled: true },
            renderWhitespace: "selection",
            automaticLayout: true,
          }}
        />
      </div>
    </section>
  );
}

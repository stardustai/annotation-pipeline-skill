import { useMemo, useState } from "react";
import { JsonPopup, tokenize, tokensToString, unwrapJson } from "./JsonViewer";
import type { TaskDetailArtifact } from "../types";

const MAX_ROWS_RENDERED = 20;
const AUTO_COLLAPSE_THRESHOLD = 5;

export interface BatchRow {
  row_index: number;
  row_id?: string;
  source_id?: string;
  input?: unknown;
}

export interface AnnotationOutput {
  entities?: Record<string, unknown>;
  classifications?: unknown;
  json_structures?: unknown;
  relations?: unknown;
}

export interface PairedRow {
  row_index: number;
  row_id?: string;
  source_id?: string;
  input?: unknown;
  output?: AnnotationOutput;
}

/**
 * Extract `payload.rows` from a task `source_ref`.
 * Returns an empty array if rows are missing or malformed.
 */
export function extractBatchRows(sourceRef: unknown): BatchRow[] {
  if (sourceRef === null || typeof sourceRef !== "object") return [];
  const payload = (sourceRef as Record<string, unknown>).payload;
  if (payload === null || typeof payload !== "object") return [];
  const rows = (payload as Record<string, unknown>).rows;
  if (!Array.isArray(rows)) return [];
  const out: BatchRow[] = [];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (r === null || typeof r !== "object") continue;
    const rec = r as Record<string, unknown>;
    const idxRaw = rec.row_index;
    const row_index =
      typeof idxRaw === "number" && Number.isFinite(idxRaw) ? idxRaw : i;
    out.push({
      row_index,
      row_id: typeof rec.row_id === "string" ? rec.row_id : undefined,
      source_id: typeof rec.source_id === "string" ? rec.source_id : undefined,
      input: rec.input,
    });
  }
  return out;
}

/**
 * Recursively strip any `<think>...</think>` reasoning blocks from string
 * fields (the legacy artifacts wrote the raw LLM output, which can prefix
 * the JSON payload with a chain-of-thought block).
 */
const THINK_BLOCK_RE = /<think>[\s\S]*?<\/think>/gi;

function stripThinkBlocks(value: unknown): unknown {
  if (typeof value === "string") {
    if (!value.includes("<think>")) return value;
    return value.replace(THINK_BLOCK_RE, "").trim();
  }
  if (Array.isArray(value)) {
    return value.map(stripThinkBlocks);
  }
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      out[k] = stripThinkBlocks(v);
    }
    return out;
  }
  return value;
}

/**
 * Find the most recent NON-EMPTY `annotation_result` artifact and pull
 * its `rows[].output` keyed by row_index. Strings that wrap JSON are
 * unwrapped.
 *
 * Walks artifacts in reverse order, skipping any with an empty/missing
 * `text` field or whose unwrapped payload contains no rows. The runtime
 * sometimes writes an empty annotation_result on certain failure modes
 * (post-rate-limit retry, arbiter mechanical fail) — without this
 * fallback the AnnotationView shows "No annotation rows to render"
 * even though earlier artifacts have perfectly good data.
 */
export function extractOutputsByIndex(
  artifacts: TaskDetailArtifact[],
): Map<number, AnnotationOutput> {
  const annotation = artifacts.filter((a) => a.kind === "annotation_result");
  for (let i = annotation.length - 1; i >= 0; i--) {
    const candidate = annotation[i];
    const cleaned = stripThinkBlocks(candidate.payload);
    const unwrapped = unwrapJson(cleaned);
    const rows = locateRowsArray(unwrapped);
    if (!rows || rows.length === 0) continue;  // empty / unparseable → keep walking back
    const out = new Map<number, AnnotationOutput>();
    for (let j = 0; j < rows.length; j++) {
      const r = rows[j];
      if (r === null || typeof r !== "object") continue;
      const rec = r as Record<string, unknown>;
      const idxRaw = rec.row_index;
      const row_index =
        typeof idxRaw === "number" && Number.isFinite(idxRaw) ? idxRaw : j;
      const output = rec.output;
      if (output !== null && typeof output === "object") {
        out.set(row_index, output as AnnotationOutput);
      }
    }
    if (out.size > 0) return out;  // found a populated artifact — done
  }
  return new Map();
}

/**
 * Walk every `annotation_result` artifact in order and return a per-actor
 * (span → type) map. The annotator's first artifact is tagged "annotator";
 * artifacts whose metadata.source is "arbiter_correction" are tagged
 * "arbiter". Used by Manual Review to surface side-by-side picks.
 */
export function extractProposalsByActor(
  artifacts: TaskDetailArtifact[],
): Array<{ actor: "annotator" | "arbiter"; spanType: Map<string, string> }> {
  const result: Array<{
    actor: "annotator" | "arbiter";
    spanType: Map<string, string>;
  }> = [];
  let sawAnnotator = false;
  for (const art of artifacts) {
    if (art.kind !== "annotation_result") continue;
    const cleaned = stripThinkBlocks(art.payload);
    const unwrapped = unwrapJson(cleaned);
    const rows = locateRowsArray(unwrapped);
    if (!rows) continue;
    const meta = (art.metadata ?? {}) as Record<string, unknown>;
    const isArbiter = meta.source === "arbiter_correction";
    const actor: "annotator" | "arbiter" = isArbiter ? "arbiter" : "annotator";
    if (!isArbiter) sawAnnotator = true;
    const spanType = new Map<string, string>();
    for (const r of rows) {
      if (r === null || typeof r !== "object") continue;
      const output = (r as Record<string, unknown>).output;
      if (output === null || typeof output !== "object") continue;
      const entities = (output as Record<string, unknown>).entities;
      if (!entities || typeof entities !== "object") continue;
      for (const [type, spans] of Object.entries(entities as Record<string, unknown>)) {
        if (!Array.isArray(spans)) continue;
        for (const s of spans) {
          if (typeof s !== "string") continue;
          // Last write wins per span — fine for a single payload, and
          // collisions within a single artifact are flagged elsewhere.
          spanType.set(s, type);
        }
      }
    }
    result.push({ actor, spanType });
  }
  // If the first artifact happened to carry arbiter metadata (rare/legacy
  // tasks), the annotator's view is missing — leave the list as-is.
  void sawAnnotator;
  return result;
}

function locateRowsArray(value: unknown): unknown[] | null {
  if (value === null || typeof value !== "object") return null;
  if (Array.isArray(value)) return value;
  const rec = value as Record<string, unknown>;
  if (Array.isArray(rec.rows)) return rec.rows;
  // Common nesting: payload may wrap a `text` field whose unwrapped form has rows.
  if (rec.text && typeof rec.text === "object") {
    const nested = locateRowsArray(rec.text);
    if (nested) return nested;
  }
  if (rec.output && typeof rec.output === "object") {
    const nested = locateRowsArray(rec.output);
    if (nested) return nested;
  }
  return null;
}

/**
 * Pair extracted rows with their outputs by `row_index`.
 */
export function pairRowsAndOutputs(
  rows: BatchRow[],
  outputs: Map<number, AnnotationOutput>,
): PairedRow[] {
  return rows.map((r) => ({ ...r, output: outputs.get(r.row_index) }));
}

interface JsonStructuresSummary {
  legacyCount: number | null;
  /** Non-empty type entries for new-schema form: e.g. [["status", 1], ["goal", 2]]. */
  newSchemaTypes: Array<[string, number]>;
  empty: boolean;
}

/**
 * Summarize json_structures, accepting either:
 *   - the new schema form: an object whose values are arrays (phrase lists)
 *   - the legacy form: an array of structure records
 */
export function summarizeJsonStructures(value: unknown): JsonStructuresSummary {
  if (value === null || value === undefined) {
    return { legacyCount: null, newSchemaTypes: [], empty: true };
  }
  if (Array.isArray(value)) {
    return {
      legacyCount: value.length,
      newSchemaTypes: [],
      empty: value.length === 0,
    };
  }
  if (typeof value === "object") {
    const types: Array<[string, number]> = [];
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (Array.isArray(v)) {
        if (v.length > 0) types.push([k, v.length]);
      } else if (v !== null && v !== undefined && typeof v === "object") {
        // Treat any non-empty object as a single entry; rare.
        const inner = Object.keys(v as Record<string, unknown>).length;
        if (inner > 0) types.push([k, inner]);
      }
    }
    return {
      legacyCount: null,
      newSchemaTypes: types,
      empty: types.length === 0,
    };
  }
  return { legacyCount: null, newSchemaTypes: [], empty: true };
}

export function countRelations(value: unknown): number {
  if (Array.isArray(value)) return value.length;
  return 0;
}

interface ClassificationItem {
  task: string;
  final_label: string;
}

export function extractClassifications(value: unknown): ClassificationItem[] {
  if (!Array.isArray(value)) return [];
  const out: ClassificationItem[] = [];
  for (const entry of value) {
    if (entry === null || typeof entry !== "object") continue;
    const rec = entry as Record<string, unknown>;
    const task = typeof rec.task === "string" ? rec.task : undefined;
    const label =
      typeof rec.final_label === "string"
        ? rec.final_label
        : typeof rec.label === "string"
          ? rec.label
          : undefined;
    if (task && label) out.push({ task, final_label: label });
  }
  return out;
}

interface EntityGroup {
  type: string;
  values: string[];
}

export function extractEntities(value: unknown): EntityGroup[] {
  if (value === null || value === undefined || typeof value !== "object") return [];
  if (Array.isArray(value)) {
    // Tolerate legacy entity arrays of {type, text} records.
    const grouped = new Map<string, string[]>();
    for (const entry of value) {
      if (entry === null || typeof entry !== "object") continue;
      const rec = entry as Record<string, unknown>;
      const t =
        typeof rec.type === "string"
          ? rec.type
          : typeof rec.entity_type === "string"
            ? rec.entity_type
            : "entity";
      const txt =
        typeof rec.text === "string"
          ? rec.text
          : typeof rec.value === "string"
            ? rec.value
            : undefined;
      if (!txt) continue;
      const arr = grouped.get(t) ?? [];
      arr.push(txt);
      grouped.set(t, arr);
    }
    return Array.from(grouped.entries()).map(([type, values]) => ({ type, values }));
  }
  const out: EntityGroup[] = [];
  for (const [type, raw] of Object.entries(value as Record<string, unknown>)) {
    if (!Array.isArray(raw)) continue;
    const values: string[] = [];
    for (const v of raw) {
      if (typeof v === "string") values.push(v);
      else if (v !== null && typeof v === "object") {
        const rec = v as Record<string, unknown>;
        if (typeof rec.text === "string") values.push(rec.text);
        else if (typeof rec.value === "string") values.push(rec.value);
      }
    }
    if (values.length > 0) out.push({ type, values });
  }
  return out;
}

function inputAsText(input: unknown): string {
  if (typeof input === "string") return input;
  if (input === null || input === undefined) return "";
  try {
    return JSON.stringify(input, null, 2);
  } catch {
    return String(input);
  }
}

interface PerRowViewProps {
  sourceRef: unknown;
  artifacts: TaskDetailArtifact[];
}

export function PerRowView({ sourceRef, artifacts }: PerRowViewProps) {
  const rows = useMemo(() => extractBatchRows(sourceRef), [sourceRef]);
  const outputsByIndex = useMemo(() => extractOutputsByIndex(artifacts), [artifacts]);
  const paired = useMemo(
    () => pairRowsAndOutputs(rows, outputsByIndex),
    [rows, outputsByIndex],
  );

  if (paired.length === 0) return null;

  const visible = paired.slice(0, MAX_ROWS_RENDERED);
  const truncatedCount = paired.length - visible.length;
  const defaultExpanded = paired.length <= AUTO_COLLAPSE_THRESHOLD;

  return (
    <section className="detail-section per-row-section">
      <h3>Per-Row Content ({paired.length})</h3>
      {truncatedCount > 0 ? (
        <p className="per-row-truncated">
          Showing first {MAX_ROWS_RENDERED} of {paired.length} rows.
        </p>
      ) : null}
      <div className="per-row-stack">
        {visible.map((row) => (
          <PerRowCard
            key={`${row.row_index}-${row.row_id ?? ""}`}
            row={row}
            defaultExpanded={defaultExpanded}
          />
        ))}
      </div>
    </section>
  );
}

function PerRowCard({
  row,
  defaultExpanded,
}: {
  row: PairedRow;
  defaultExpanded: boolean;
}) {
  const [open, setOpen] = useState(defaultExpanded);
  const [popupOpen, setPopupOpen] = useState(false);

  const inputText = useMemo(() => inputAsText(row.input), [row.input]);
  const charCount = inputText.length;

  const popupValue = useMemo(
    () => ({ input: row.input, output: row.output ?? null }),
    [row.input, row.output],
  );
  const popupTokens = useMemo(() => tokenize(unwrapJson(popupValue)), [popupValue]);
  const popupText = useMemo(() => tokensToString(popupTokens), [popupTokens]);

  const entities = extractEntities(row.output?.entities);
  const classifications = extractClassifications(row.output?.classifications);
  const jsonStruct = summarizeJsonStructures(row.output?.json_structures);
  const relationCount = countRelations(row.output?.relations);

  return (
    <div className="per-row-card">
      <button
        className="per-row-card-header"
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="per-row-card-title">
          <span className="per-row-card-index">Row {row.row_index}</span>
          {row.row_id ? <span className="per-row-card-id">id: {row.row_id}</span> : null}
        </span>
        <span className="per-row-card-meta">
          <span className="per-row-char-badge">{charCount} chars</span>
          <span className="per-row-toggle">{open ? "−" : "+"}</span>
        </span>
      </button>
      {open ? (
        <div className="per-row-card-body">
          <div className="per-row-field">
            <div className="per-row-field-label">INPUT</div>
            {inputText ? (
              <pre className="per-row-input">{inputText}</pre>
            ) : (
              <p className="empty-detail">—</p>
            )}
          </div>
          <div className="per-row-field">
            <div className="per-row-field-label">ANNOTATION</div>
            {row.output ? (
              <ul className="per-row-annotation-list">
                <li>
                  <strong>Entities:</strong>
                  {entities.length === 0 ? (
                    <span className="per-row-empty"> —</span>
                  ) : (
                    <ul className="per-row-entity-list">
                      {entities.map((group) => (
                        <li key={group.type}>
                          <span className="per-row-entity-type">{group.type}:</span>{" "}
                          {group.values.join(", ")}
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
                <li>
                  <strong>Classifications:</strong>
                  {classifications.length === 0 ? (
                    <span className="per-row-empty"> —</span>
                  ) : (
                    <ul className="per-row-class-list">
                      {classifications.map((c) => (
                        <li key={c.task}>
                          {c.task} → <em>{c.final_label}</em>
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
                <li>
                  <strong>JSON structures:</strong>{" "}
                  {jsonStruct.empty ? (
                    <span className="per-row-empty">—</span>
                  ) : jsonStruct.legacyCount !== null ? (
                    <span>
                      {jsonStruct.legacyCount} records (legacy format)
                    </span>
                  ) : (
                    <span>
                      {jsonStruct.newSchemaTypes
                        .map(([t, n]) => `${t} (${n} phrase${n === 1 ? "" : "s"})`)
                        .join(", ")}
                    </span>
                  )}
                </li>
                <li>
                  <strong>Relations:</strong> {relationCount}
                </li>
              </ul>
            ) : (
              <p className="empty-detail">No annotation output for this row.</p>
            )}
          </div>
          <div className="per-row-actions">
            <button
              className="json-viewer-link"
              type="button"
              onClick={() => setPopupOpen(true)}
            >
              View raw JSON
            </button>
          </div>
        </div>
      ) : null}
      {popupOpen ? (
        <JsonPopup
          tokens={popupTokens}
          fullText={popupText}
          onClose={() => setPopupOpen(false)}
        />
      ) : null}
    </div>
  );
}

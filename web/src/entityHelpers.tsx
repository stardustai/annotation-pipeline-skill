import React from "react";

export const ENTITY_TYPES = [
  "person", "organization", "project", "document", "time",
  "number", "event", "location", "technology", "entity",
] as const;

// Pseudo-type for "this span should NOT be tagged as any entity". Stored
// in the same convention table; the runtime formats it as a negative
// instruction when injecting into prompts.
export const NOT_ENTITY = "not_an_entity";

const TYPE_PALETTE = [
  "#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c",
  "#0891b2", "#65a30d", "#db2777", "#4f46e5", "#0d9488",
  "#ca8a04", "#7c3aed", "#059669", "#c2410c", "#0369a1",
];
export function colorForType(type: string): string {
  if (type === NOT_ENTITY) return "#6b7280";
  let h = 0;
  for (let i = 0; i < type.length; i++) h = (h * 31 + type.charCodeAt(i)) >>> 0;
  return TYPE_PALETTE[h % TYPE_PALETTE.length];
}

// Find the first occurrence of `span` in `text`, return at least
// ``minTokens`` tokens of surrounding context on each side, snapped to a
// word/character boundary. Returns null when the span isn't in the text.
export function spanContext(text: string, span: string, minTokens = 10): {
  before: string;
  match: string;
  after: string;
} | null {
  const idx = text.toLowerCase().indexOf(span.toLowerCase());
  if (idx === -1) return null;
  const spanEnd = idx + span.length;
  return {
    before: text.slice(walkLeftTokens(text, idx, minTokens), idx),
    match: text.slice(idx, spanEnd),
    after: text.slice(spanEnd, walkRightTokens(text, spanEnd, minTokens)),
  };
}

const CJK_RE = /[　-鿿가-힯＀-￯]/;
const WS_RE = /\s/;

function walkLeftTokens(text: string, end: number, minTokens: number): number {
  let pos = end;
  let count = 0;
  while (pos > 0 && count < minTokens) {
    while (pos > 0 && WS_RE.test(text[pos - 1])) pos--;
    if (pos === 0) break;
    if (CJK_RE.test(text[pos - 1])) { pos--; count++; }
    else {
      while (pos > 0 && !WS_RE.test(text[pos - 1]) && !CJK_RE.test(text[pos - 1])) pos--;
      count++;
    }
  }
  while (pos < end && WS_RE.test(text[pos])) pos++;
  return pos;
}

function walkRightTokens(text: string, start: number, minTokens: number): number {
  let pos = start;
  let count = 0;
  while (pos < text.length && count < minTokens) {
    while (pos < text.length && WS_RE.test(text[pos])) pos++;
    if (pos >= text.length) break;
    if (CJK_RE.test(text[pos])) { pos++; count++; }
    else {
      while (pos < text.length && !WS_RE.test(text[pos]) && !CJK_RE.test(text[pos])) pos++;
      count++;
    }
  }
  while (pos > start && WS_RE.test(text[pos - 1])) pos--;
  return pos;
}

/** Horizontal stacked bar showing the proportion of each type. ≥80%
 *  segments get a green dominance ring; <2% segments collapse into a grey
 *  "other" sliver so tiny ones stay readable. Hover tooltip shows the
 *  full breakdown.
 */
export function DistributionBar({
  distribution,
  total,
  height = 14,
}: {
  distribution: Record<string, number>;
  total: number;
  height?: number;
}): React.ReactElement {
  if (total <= 0) return <span className="runtime-muted">—</span>;
  const entries = Object.entries(distribution).sort((a, b) => b[1] - a[1]);
  const visible: Array<[string, number]> = [];
  let otherCount = 0;
  for (const [t, c] of entries) {
    if (c / total >= 0.02) visible.push([t, c]);
    else otherCount += c;
  }
  if (otherCount > 0) visible.push(["other", otherCount]);
  const topShare = entries[0] ? entries[0][1] / total : 0;
  const dominant = topShare >= 0.8;

  return (
    <div
      title={entries.map(([t, c]) => `${t}: ${c} (${((c / total) * 100).toFixed(1)}%)`).join("\n")}
      style={{
        display: "flex",
        width: "100%",
        height: `${height}px`,
        borderRadius: "3px",
        overflow: "hidden",
        border: dominant
          ? "1px solid var(--success, #047857)"
          : "1px solid var(--border, #d1d5db)",
        background: "#e5e7eb",
      }}
    >
      {visible.map(([t, c]) => {
        const pct = (c / total) * 100;
        return (
          <div
            key={t}
            style={{
              width: `${pct}%`,
              background: t === "other" ? "#9ca3af" : colorForType(t),
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "white",
              fontSize: "0.65rem",
              fontWeight: 600,
              overflow: "hidden",
              whiteSpace: "nowrap",
            }}
          >
            {pct >= 12 ? `${t} ${Math.round(pct)}%` : ""}
          </div>
        );
      })}
    </div>
  );
}

/** A pill that renders the entity type with stable per-type color, plus
 *  a special grey "🚫 not entity" pill for the NOT_ENTITY pseudo-type.
 */
export function TypePill({ type }: { type: string }): React.ReactElement {
  if (type === NOT_ENTITY) {
    return (
      <span
        style={{
          display: "inline-block",
          padding: "0.1rem 0.5rem",
          borderRadius: "999px",
          background: "#6b7280",
          color: "white",
          fontSize: "0.75rem",
          fontWeight: 500,
        }}
        title="Stop-word: this span should NOT be tagged as any entity"
      >
        🚫 not entity
      </span>
    );
  }
  return (
    <span
      style={{
        display: "inline-block",
        padding: "0.1rem 0.5rem",
        borderRadius: "999px",
        background: colorForType(type),
        color: "white",
        fontSize: "0.75rem",
        fontWeight: 500,
      }}
    >
      {type}
    </span>
  );
}

/** Simple prev/next pagination control. Caller owns the page state and
 *  the underlying items array; this component only renders controls and
 *  reports the current visible range.
 */
export function Pagination({
  total,
  page,
  pageSize,
  onPageChange,
}: {
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (next: number) => void;
}): React.ReactElement | null {
  if (total <= pageSize) return null;
  const lastPage = Math.max(0, Math.ceil(total / pageSize) - 1);
  const from = page * pageSize + 1;
  const to = Math.min(total, (page + 1) * pageSize);

  const btnStyle: React.CSSProperties = {
    fontSize: "0.8rem",
    padding: "0.15rem 0.55rem",
    border: "1px solid var(--border, #d1d5db)",
    background: "white",
    borderRadius: "3px",
    cursor: "pointer",
  };
  const disabledStyle: React.CSSProperties = {
    ...btnStyle,
    cursor: "not-allowed",
    color: "var(--muted, #9ca3af)",
  };
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        margin: "0.4rem 0",
        fontSize: "0.85rem",
      }}
    >
      <button
        type="button"
        style={page === 0 ? disabledStyle : btnStyle}
        disabled={page === 0}
        onClick={() => onPageChange(0)}
        title="First"
      >
        ≪
      </button>
      <button
        type="button"
        style={page === 0 ? disabledStyle : btnStyle}
        disabled={page === 0}
        onClick={() => onPageChange(page - 1)}
        title="Previous"
      >
        ‹ Prev
      </button>
      <span className="runtime-muted" style={{ minWidth: "10rem", textAlign: "center" }}>
        {from}–{to} of {total} · page {page + 1} / {lastPage + 1}
      </span>
      <button
        type="button"
        style={page === lastPage ? disabledStyle : btnStyle}
        disabled={page === lastPage}
        onClick={() => onPageChange(page + 1)}
        title="Next"
      >
        Next ›
      </button>
      <button
        type="button"
        style={page === lastPage ? disabledStyle : btnStyle}
        disabled={page === lastPage}
        onClick={() => onPageChange(lastPage)}
        title="Last"
      >
        ≫
      </button>
    </div>
  );
}

/** Top-N type buttons (default 3) + dropdown for the remainder + the
 *  "🚫 not entity" pseudo-button. Caller owns the selected state; this
 *  component renders the controls only — no commit on click. The caller
 *  needs a separate Confirm button before applying the change.
 *
 *  ``preferredOrder`` puts those types first (e.g. prior dominant +
 *  current annotation type). Remaining ENTITY_TYPES fill the dropdown.
 */
export function TopNTypeSelector({
  selected,
  preferredOrder,
  topN = 3,
  onSelect,
  includeNotEntity = true,
}: {
  selected: string | null;
  preferredOrder: string[];
  topN?: number;
  onSelect: (type: string | null) => void;
  includeNotEntity?: boolean;
}): React.ReactElement {
  // Deduplicate preferredOrder; drop empty/falsy.
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const t of preferredOrder) {
    if (t && !seen.has(t)) { seen.add(t); ordered.push(t); }
  }
  // Fall back to canonical ENTITY_TYPES order for the rest.
  for (const t of ENTITY_TYPES) {
    if (!seen.has(t)) { seen.add(t); ordered.push(t); }
  }
  const top = ordered.slice(0, topN);
  const rest = ordered.slice(topN);

  function btnStyle(isSelected: boolean, color: string): React.CSSProperties {
    return {
      fontSize: "0.75rem",
      padding: "0.15rem 0.55rem",
      borderRadius: "4px",
      border: isSelected ? `1px solid ${color}` : "1px solid var(--border, #d1d5db)",
      background: isSelected ? color : "white",
      color: isSelected ? "white" : "#374151",
      cursor: "pointer",
      fontWeight: isSelected ? 600 : 400,
    };
  }

  // Whether the current selection is in `rest` — controls the dropdown
  // "selected" visual.
  const restSelected = selected && rest.includes(selected) ? selected : "";

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.25rem", alignItems: "center" }}>
      {top.map((t) => {
        const isSel = selected === t;
        return (
          <button
            key={t}
            type="button"
            onClick={() => onSelect(isSel ? null : t)}
            style={btnStyle(isSel, colorForType(t))}
          >
            {t}
          </button>
        );
      })}
      {rest.length > 0 ? (
        <select
          value={restSelected}
          onChange={(e) => onSelect(e.target.value || null)}
          style={{
            fontSize: "0.75rem",
            padding: "0.1rem 0.3rem",
            borderRadius: "4px",
            border: restSelected
              ? `1px solid ${colorForType(restSelected)}`
              : "1px solid var(--border, #d1d5db)",
            background: restSelected ? colorForType(restSelected) : "white",
            color: restSelected ? "white" : "#374151",
            fontWeight: restSelected ? 600 : 400,
          }}
        >
          <option value="">other type ▾</option>
          {rest.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      ) : null}
      {includeNotEntity ? (
        (() => {
          const isSel = selected === NOT_ENTITY;
          return (
            <button
              type="button"
              onClick={() => onSelect(isSel ? null : NOT_ENTITY)}
              style={{
                ...btnStyle(isSel, "#6b7280"),
                fontStyle: "italic",
                borderStyle: isSel ? "solid" : "dashed",
              }}
              title="Stop-word: this span should NOT be tagged as any entity"
            >
              🚫 not entity
            </button>
          );
        })()
      ) : null}
    </div>
  );
}

/** Shared cell: fetches a random example text containing the span from an
 *  ACCEPTED task. Refresh button cycles to the next match (excluding
 *  already-shown ones). When ``taskId`` is provided the search is
 *  restricted to that specific task (used by Task deviations to show
 *  context from THE task in question).
 */
export function OriginalTextCell({
  projectId,
  storeKey,
  span,
  taskId,
}: {
  projectId: string;
  storeKey: string | null;
  span: string;
  taskId?: string;
}): React.ReactElement {
  const [example, setExample] = React.useState<
    { task_id: string; row_index: number; text: string } | null
  >(null);
  const [exclude, setExclude] = React.useState<string[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [loaded, setLoaded] = React.useState(false);

  function load(excludeNow: string[]) {
    setLoading(true);
    const storeQ = storeKey ? `&store=${encodeURIComponent(storeKey)}` : "";
    const taskQ = taskId ? `&task=${encodeURIComponent(taskId)}` : "";
    // When taskId is set, exclude is on (task_id:row_index) keys;
    // otherwise exclude is on task_ids.
    const excludeParam = taskId ? "exclude_key" : "exclude";
    const excludeQ = excludeNow
      .map((k) => `&${excludeParam}=${encodeURIComponent(k)}`)
      .join("");
    fetch(
      `/api/typical-text?project=${encodeURIComponent(projectId)}&span=${encodeURIComponent(span)}${storeQ}${taskQ}${excludeQ}`,
    )
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d && d.found) {
          setExample({ task_id: d.task_id, row_index: d.row_index, text: d.text });
        } else if (excludeNow.length > 0) {
          // Cycled past all examples — reset exclusion and try again on next click.
          setExample(null);
          setExclude([]);
        } else {
          setExample(null);
        }
      })
      .catch(() => setExample(null))
      .finally(() => {
        setLoading(false);
        setLoaded(true);
      });
  }

  React.useEffect(() => {
    setExample(null);
    setExclude([]);
    setLoaded(false);
    load([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, storeKey, span, taskId]);

  function handleRefresh() {
    const nextKey = example
      ? taskId
        ? `${example.task_id}:${example.row_index}`
        : example.task_id
      : null;
    const nextExclude = nextKey ? [...exclude, nextKey].slice(-50) : exclude;
    setExclude(nextExclude);
    load(nextExclude);
  }

  if (!loaded && loading) return <span className="runtime-muted">…</span>;
  if (!example) {
    // No example found this round. Could be: empty stats / transient
    // load failure / cycled past examples and reset. Surface a click-
    // to-retry so the operator can recover without remounting the row.
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
        <span className="runtime-muted">—</span>
        <button
          type="button"
          onClick={() => load([])}
          disabled={loading}
          title="Retry — search for a sample annotation that mentions this span"
          style={{
            fontSize: "0.7rem",
            background: "transparent",
            border: "1px solid var(--border, #d1d5db)",
            padding: "0 5px",
            borderRadius: "3px",
            cursor: "pointer",
            color: "var(--muted, #6b7280)",
          }}
        >
          {loading ? "…" : "↻"}
        </button>
      </span>
    );
  }

  const ctx = spanContext(example.text, span, 8);
  return (
    <div style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
      <p
        style={{ margin: 0, fontSize: "0.8rem", color: "var(--muted, #4b5563)", flex: 1 }}
        title={`From task ${example.task_id} row ${example.row_index}`}
      >
        {ctx ? (
          <>
            {ctx.before ? <>…{ctx.before}</> : null}
            <mark style={{ background: "#fef3c7", padding: "0 2px", borderRadius: "2px" }}>
              {ctx.match}
            </mark>
            {ctx.after ? <>{ctx.after}…</> : null}
          </>
        ) : (
          example.text.slice(0, 120)
        )}
      </p>
      {/* Hide refresh when the cell is constrained to a single task —
          cycling through rows of the same task is rarely what the
          operator wants and the button suggests there are "other"
          examples that don't actually exist. */}
      {taskId ? null : (
        <button
          type="button"
          onClick={handleRefresh}
          disabled={loading}
          title="Show a different example"
          style={{
            fontSize: "0.7rem",
            padding: "0.1rem 0.4rem",
            background: "transparent",
            border: "1px solid var(--border, #d1d5db)",
            borderRadius: "3px",
            cursor: "pointer",
            color: "var(--muted, #6b7280)",
            whiteSpace: "nowrap",
          }}
        >
          ↻
        </button>
      )}
    </div>
  );
}

/** Inline button row to declare canonical type for a span. Selected type
 *  has a "selected" class; clicking the selected type toggles it off
 *  (clears the convention).
 */
export function TypeSelector({
  span,
  effectiveType,
  busyKey,
  pendingKey,
  onPick,
  includeNotEntity = true,
}: {
  span: string;
  effectiveType: string | null;
  busyKey?: string | null;
  pendingKey?: string | null;
  onPick: (span: string, type: string, effectiveType: string | null) => void;
  includeNotEntity?: boolean;
}): React.ReactElement {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.25rem" }}>
      {ENTITY_TYPES.map((t) => {
        const key = `${span}|${t}`;
        const isSelected = t === effectiveType;
        const pending = pendingKey === key;
        return (
          <button
            key={t}
            type="button"
            disabled={busyKey === key || pending}
            onClick={() => onPick(span, t, effectiveType)}
            title={
              isSelected
                ? "Current selection — click again to clear (fallback to annotator)"
                : `Set ${span} → ${t}`
            }
            style={{
              fontSize: "0.75rem",
              padding: "0.15rem 0.5rem",
              borderRadius: "4px",
              border: isSelected
                ? `1px solid ${colorForType(t)}`
                : "1px solid var(--border, #d1d5db)",
              background: isSelected ? colorForType(t) : "white",
              color: isSelected ? "white" : "#374151",
              cursor: "pointer",
              fontWeight: isSelected ? 600 : 400,
            }}
          >
            {pending ? "…" : t}
          </button>
        );
      })}
      {includeNotEntity ? (
        (() => {
          const key = `${span}|${NOT_ENTITY}`;
          const isSelected = NOT_ENTITY === effectiveType;
          const pending = pendingKey === key;
          return (
            <button
              type="button"
              disabled={busyKey === key || pending}
              onClick={() => onPick(span, NOT_ENTITY, effectiveType)}
              title={
                isSelected
                  ? "Current selection — click again to clear"
                  : `Set ${span} → not an entity (stop word)`
              }
              style={{
                fontSize: "0.75rem",
                padding: "0.15rem 0.5rem",
                borderRadius: "4px",
                border: isSelected
                  ? "1px solid #6b7280"
                  : "1px dashed var(--muted, #9ca3af)",
                background: isSelected ? "#6b7280" : "white",
                color: isSelected ? "white" : "#6b7280",
                cursor: "pointer",
                fontStyle: "italic",
              }}
            >
              {pending ? "…" : "🚫 not entity"}
            </button>
          );
        })()
      ) : null}
    </div>
  );
}

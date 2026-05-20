import { useEffect, useMemo, useState } from "react";

const MAX_UNWRAP_DEPTH = 4;
const INLINE_MAX_LINES = 10;
const INLINE_MAX_CHARS = 1500;

export function unwrapJson(value: unknown, depth = 0): unknown {
  if (depth > MAX_UNWRAP_DEPTH) return value;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.length >= 2) {
      const first = trimmed[0];
      const last = trimmed[trimmed.length - 1];
      if ((first === "{" && last === "}") || (first === "[" && last === "]")) {
        try {
          const parsed = JSON.parse(trimmed);
          if (parsed !== null && typeof parsed === "object") {
            return unwrapJson(parsed, depth + 1);
          }
        } catch {
          /* fall through and return original string */
        }
      }
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => unwrapJson(entry, depth + 1));
  }
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      out[k] = unwrapJson(v, depth + 1);
    }
    return out;
  }
  return value;
}

type TokenType = "key" | "string" | "number" | "boolean" | "null" | "punct";

interface Token {
  type: TokenType;
  text: string;
}

export function tokenize(value: unknown, indent = 0): Token[] {
  const tokens: Token[] = [];
  const pad = (n: number) => "  ".repeat(n);

  function pushScalar(v: unknown): void {
    if (v === null) {
      tokens.push({ type: "null", text: "null" });
      return;
    }
    if (typeof v === "string") {
      tokens.push({ type: "string", text: JSON.stringify(v) });
      return;
    }
    if (typeof v === "number") {
      tokens.push({ type: "number", text: String(v) });
      return;
    }
    if (typeof v === "boolean") {
      tokens.push({ type: "boolean", text: String(v) });
      return;
    }
    // Fallback (e.g., undefined)
    tokens.push({ type: "null", text: JSON.stringify(v ?? null) });
  }

  function walk(v: unknown, depth: number): void {
    if (Array.isArray(v)) {
      if (v.length === 0) {
        tokens.push({ type: "punct", text: "[]" });
        return;
      }
      tokens.push({ type: "punct", text: "[\n" });
      v.forEach((entry, idx) => {
        tokens.push({ type: "punct", text: pad(depth + 1) });
        walk(entry, depth + 1);
        tokens.push({ type: "punct", text: idx < v.length - 1 ? ",\n" : "\n" });
      });
      tokens.push({ type: "punct", text: `${pad(depth)}]` });
      return;
    }
    if (v !== null && typeof v === "object") {
      const entries = Object.entries(v);
      if (entries.length === 0) {
        tokens.push({ type: "punct", text: "{}" });
        return;
      }
      tokens.push({ type: "punct", text: "{\n" });
      entries.forEach(([k, val], idx) => {
        tokens.push({ type: "punct", text: pad(depth + 1) });
        tokens.push({ type: "key", text: JSON.stringify(k) });
        tokens.push({ type: "punct", text: ": " });
        walk(val, depth + 1);
        tokens.push({ type: "punct", text: idx < entries.length - 1 ? ",\n" : "\n" });
      });
      tokens.push({ type: "punct", text: `${pad(depth)}}` });
      return;
    }
    pushScalar(v);
  }

  walk(value, indent);
  return tokens;
}

export function tokensToString(tokens: Token[]): string {
  return tokens.map((t) => t.text).join("");
}

export function truncateTokens(tokens: Token[]): { tokens: Token[]; truncated: boolean } {
  let charBudget = INLINE_MAX_CHARS;
  let lineBudget = INLINE_MAX_LINES;
  const out: Token[] = [];
  for (const token of tokens) {
    const linesInToken = (token.text.match(/\n/g) ?? []).length;
    if (linesInToken > lineBudget || token.text.length > charBudget) {
      // Try to keep a partial slice of this token.
      const maxLineCut = lineBudget > 0
        ? token.text.split("\n").slice(0, lineBudget).join("\n")
        : token.text;
      const slice = maxLineCut.slice(0, charBudget);
      if (slice.length > 0) out.push({ ...token, text: slice });
      out.push({ type: "punct", text: "\n…" });
      return { tokens: out, truncated: true };
    }
    out.push(token);
    charBudget -= token.text.length;
    lineBudget -= linesInToken;
  }
  return { tokens: out, truncated: false };
}

interface JsonViewerProps {
  value: unknown;
  /** When true, render full JSON inline with no popup button and no truncation. */
  embedded?: boolean;
}

export function JsonViewer({ value, embedded = false }: JsonViewerProps) {
  const [popupOpen, setPopupOpen] = useState(false);
  const [showFullInline, setShowFullInline] = useState(false);

  const unwrapped = useMemo(() => unwrapJson(value), [value]);
  const tokens = useMemo(() => tokenize(unwrapped), [unwrapped]);
  const fullText = useMemo(() => tokensToString(tokens), [tokens]);
  const { tokens: inlineTokens, truncated } = useMemo(() => {
    if (embedded || showFullInline) return { tokens, truncated: false };
    return truncateTokens(tokens);
  }, [tokens, showFullInline, embedded]);

  // ESC closes popup; only attach listener while open.
  useEffect(() => {
    if (!popupOpen) return;
    if (typeof window === "undefined") return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setPopupOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [popupOpen]);

  if (embedded) {
    return (
      <div className="json-viewer json-viewer-embedded">
        <pre className="json-block">{renderTokens(inlineTokens)}</pre>
      </div>
    );
  }

  return (
    <>
      <div className="json-viewer">
        <div className="json-viewer-toolbar">
          {truncated && !showFullInline ? (
            <button
              className="json-viewer-link"
              type="button"
              onClick={() => setShowFullInline(true)}
            >
              Show more
            </button>
          ) : null}
          {showFullInline ? (
            <button
              className="json-viewer-link"
              type="button"
              onClick={() => setShowFullInline(false)}
            >
              Show less
            </button>
          ) : null}
          <button
            className="json-viewer-link"
            type="button"
            onClick={() => setPopupOpen(true)}
          >
            Open in popup
          </button>
        </div>
        <pre className="json-block">{renderTokens(inlineTokens)}</pre>
      </div>
      {popupOpen ? (
        <JsonPopup tokens={tokens} fullText={fullText} onClose={() => setPopupOpen(false)} />
      ) : null}
    </>
  );
}

function renderTokens(tokens: Token[]) {
  return tokens.map((token, idx) => (
    <span className={`json-token json-${token.type}`} key={idx}>
      {token.text}
    </span>
  ));
}

interface JsonPopupProps {
  tokens: Token[];
  fullText: string;
  onClose: () => void;
}

export function JsonPopup({ tokens, fullText, onClose }: JsonPopupProps) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(fullText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <div
      className="json-popup-backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      role="presentation"
    >
      <div className="json-popup" role="dialog" aria-label="JSON viewer">
        <div className="json-popup-header">
          <strong>JSON</strong>
          <div className="json-popup-actions">
            <button className="json-viewer-link" type="button" onClick={copy}>
              {copied ? "Copied" : "Copy"}
            </button>
            <button className="icon-button" type="button" aria-label="Close popup" onClick={onClose}>
              ×
            </button>
          </div>
        </div>
        <pre className="json-block json-popup-body">{renderTokens(tokens)}</pre>
      </div>
    </div>
  );
}

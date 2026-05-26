import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cardSubtitle } from "../kanban";
import { previewArtifacts, previewBoxes, previewImageSource, previewTitle } from "../preview";
import {
  DRAWER_DEFAULT_WIDTH,
  clampDrawerWidth,
  loadDrawerWidth,
  saveDrawerWidth,
} from "../drawer_state";
import { AnnotationView } from "./AnnotationView";
import { JsonViewer } from "./JsonViewer";
import { PerRowView, extractOutputsByIndex, extractProposalsByActor } from "./PerRowView";
import type { TaskCard, TaskDetail, TaskDetailArtifact } from "../types";
import type { ReactNode } from "react";
import {
  clearConvention,
  declareConvention,
  fetchConventions,
  resolveConventionDispute,
  type EntityConvention,
} from "../api";

interface TaskDrawerProps {
  task: TaskCard | null;
  detail: TaskDetail | null;
  loading: boolean;
  saving: boolean;
  error: string | null;
  onSubmitHumanReviewDecision: (payload: Record<string, unknown>) => Promise<void>;
  onClose: () => void;
}

export function TaskDrawer({
  task,
  detail,
  loading,
  saving,
  error,
  onSubmitHumanReviewDecision,
  onClose,
}: TaskDrawerProps) {
  const [width, setWidth] = useState<number>(DRAWER_DEFAULT_WIDTH);
  const [drawerTab, setDrawerTab] = useState<"raw" | "annotation" | "discussions" | "logs" | "manual_review">("annotation");
  const [annotationFormat, setAnnotationFormat] = useState<"structured" | "json">("structured");
  const dragStateRef = useRef<{ startX: number; startWidth: number } | null>(null);

  // Default to the Manual Review tab whenever a task enters HR, so the
  // operator's quick-pick UI is the first thing they see.
  const hrStatus = detail?.task.status === "human_review";
  useEffect(() => {
    if (hrStatus) setDrawerTab("manual_review");
    else if (drawerTab === "manual_review") setDrawerTab("annotation");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hrStatus, detail?.task.task_id]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setWidth(loadDrawerWidth(window.localStorage ?? null, window.innerWidth));
  }, []);

  useEffect(() => {
    if (!task) return;
    if (typeof window === "undefined") return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [task, onClose]);

  const onResizeMouseDown = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      event.preventDefault();
      if (typeof window === "undefined") return;
      dragStateRef.current = { startX: event.clientX, startWidth: width };
      function onMouseMove(ev: MouseEvent) {
        const state = dragStateRef.current;
        if (!state) return;
        const delta = state.startX - ev.clientX;
        const next = clampDrawerWidth(state.startWidth + delta, window.innerWidth);
        setWidth(next);
      }
      function onMouseUp() {
        dragStateRef.current = null;
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
      }
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
    },
    [width],
  );

  // Persist width whenever it changes (covers drag-end and programmatic updates).
  useEffect(() => {
    if (typeof window === "undefined") return;
    saveDrawerWidth(window.localStorage ?? null, width);
  }, [width]);

  if (!task) return null;

  const annotationArtifacts = detail?.artifacts.filter((artifact) => artifact.kind === "annotation_result") ?? [];
  const previewEvidence = detail ? previewArtifacts(detail.artifacts) : [];

  return (
    <>
      <div className="task-drawer-backdrop" onClick={onClose} aria-hidden="true" />
      <aside className="task-drawer" aria-label="Task detail" style={{ width }}>
        <div
          className="task-drawer-resize-handle"
          onMouseDown={onResizeMouseDown}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize task drawer"
        />
      <div className="drawer-header">
        <div>
          <h2>{task.task_id}</h2>
          <p>{cardSubtitle(task)}</p>
        </div>
        <button className="icon-button" type="button" aria-label="Close task detail" onClick={onClose}>
          ×
        </button>
      </div>

      <dl className="detail-grid">
        <div>
          <dt>Status</dt>
          <dd>{task.status}</dd>
        </div>
        <div>
          <dt>Annotator</dt>
          <dd>{task.selected_annotator_id ?? "unassigned"}</dd>
        </div>
        <div>
          <dt>Latest Attempt</dt>
          <dd>{task.latest_attempt_status ?? "none"}</dd>
        </div>
        <div>
          <dt>Feedback</dt>
          <dd>{task.feedback_count}</dd>
        </div>
        <div>
          <dt>Retry</dt>
          <dd>{task.retry_pending ? "pending" : "none"}</dd>
        </div>
        <div>
          <dt>External Sync</dt>
          <dd>{task.external_sync_pending ? "pending" : "clear"}</dd>
        </div>
        {detail?.task.document_version_id ? (
          <div>
            <dt>Guideline Version</dt>
            <dd><span className="agreement-pill">{detail.task.document_version_id}</span></dd>
          </div>
        ) : null}
      </dl>

      {loading ? <div className="drawer-state">Loading task detail</div> : null}
      {error ? <div className="drawer-error">{error}</div> : null}

      {detail ? (
        <>
          {detail.task.status === "human_review" ? (
            <HumanReviewReasonBanner events={detail.events} />
          ) : null}
          <div className="drawer-tabs" role="tablist">
            {(detail.task.status === "human_review"
              ? (["manual_review", "annotation", "raw", "discussions", "logs"] as const)
              : (["annotation", "raw", "discussions", "logs"] as const)
            ).map((tab) => (
              <button
                key={tab}
                role="tab"
                aria-selected={drawerTab === tab}
                className={drawerTab === tab ? "drawer-tab selected" : "drawer-tab"}
                type="button"
                onClick={() => setDrawerTab(tab)}
              >
                {tab === "raw"
                  ? "Raw Data"
                  : tab === "annotation"
                  ? "Annotation"
                  : tab === "discussions"
                  ? "Discussions"
                  : tab === "manual_review"
                  ? "Manual Review"
                  : "Logs"}
              </button>
            ))}
          </div>

          <div className="detail-sections">
            {drawerTab === "manual_review" ? (
              <ManualReviewTab
                projectId={detail.task.pipeline_id}
                taskId={detail.task.task_id}
                sourceRef={detail.task.source_ref}
                artifacts={detail.artifacts}
                feedback={detail.feedback}
                feedbackDiscussions={detail.feedback_discussions}
                saving={saving}
                onSubmitHumanReviewDecision={onSubmitHumanReviewDecision}
              />
            ) : null}

            {drawerTab === "raw" ? (
              <>
                <PerRowView sourceRef={detail.task.source_ref} artifacts={detail.artifacts} />
                <DetailSection title="Raw Source">
                  <JsonViewer value={detail.task.source_ref} />
                </DetailSection>
              </>
            ) : null}

            {drawerTab === "annotation" ? (
              <>
                {annotationArtifacts.length === 0 ? (
                  <p className="empty-detail">No annotation artifacts recorded.</p>
                ) : (
                  <>
                    <div className="annotation-format-toggle">
                      <button
                        type="button"
                        className={annotationFormat === "structured" ? "segment selected" : "segment"}
                        onClick={() => setAnnotationFormat("structured")}
                      >
                        Structured
                      </button>
                      <button
                        type="button"
                        className={annotationFormat === "json" ? "segment selected" : "segment"}
                        onClick={() => setAnnotationFormat("json")}
                      >
                        JSON
                      </button>
                    </div>
                    {annotationFormat === "structured" ? (
                      <div className="artifact-panel">
                        <div className="artifact-title">
                          <span>Latest</span>
                          <span>
                            {annotationArtifacts[annotationArtifacts.length - 1].metadata.provider
                              ? String(annotationArtifacts[annotationArtifacts.length - 1].metadata.provider)
                              : annotationArtifacts[annotationArtifacts.length - 1].content_type}
                          </span>
                        </div>
                        <AnnotationView
                          artifacts={annotationArtifacts}
                          sourceRef={detail.task.source_ref}
                        />
                      </div>
                    ) : (
                      annotationArtifacts.map((artifact, index) => {
                        const isLatest = index === annotationArtifacts.length - 1;
                        const label = artifact.metadata.provider
                          ? String(artifact.metadata.provider)
                          : artifact.content_type;
                        return isLatest ? (
                          <div className="artifact-panel" key={artifact.artifact_id}>
                            <div className="artifact-title">
                              <span>Latest</span>
                              <span>{label}</span>
                            </div>
                            <JsonViewer value={artifact.payload} />
                          </div>
                        ) : (
                          <details className="artifact-panel artifact-collapsed" key={artifact.artifact_id}>
                            <summary className="artifact-title">
                              <span>#{index + 1}</span>
                              <span>{label}</span>
                            </summary>
                            <JsonViewer value={artifact.payload} />
                          </details>
                        );
                      })
                    )}
                  </>
                )}
                {previewEvidence.length > 0 ? (
                  <DetailSection title="Preview Evidence">
                    <div className="preview-stack">
                      {previewEvidence.map((artifact) => (
                        <PreviewArtifact key={artifact.artifact_id} artifact={artifact} />
                      ))}
                    </div>
                  </DetailSection>
                ) : null}
              </>
            ) : null}

            {drawerTab === "discussions" ? (
              <>
                <DetailSection title={`Feedback (${detail.feedback.length})`}>
                  {detail.feedback.length === 0 ? (
                    <p className="empty-detail">No QC or Human Review feedback recorded.</p>
                  ) : (
                    <>
                      <ConsensusSummary detail={detail} />
                      {detail.feedback.map((item) => (
                        <FeedbackAgreementCard
                          key={String(item.feedback_id)}
                          feedback={item}
                          discussions={detail.feedback_discussions.filter(
                            (entry) => entry.feedback_id === item.feedback_id,
                          )}
                        />
                      ))}
                    </>
                  )}
                </DetailSection>
              </>
            ) : null}

            {drawerTab === "logs" ? (
              <LogsTab
                attempts={detail.attempts}
                events={detail.events}
                feedback={detail.feedback}
              />
            ) : null}
          </div>
        </>
      ) : null}
      </aside>
    </>
  );
}

function HumanReviewReasonBanner({ events }: { events: Array<Record<string, unknown>> }) {
  // Find the most recent transition that landed in human_review.
  let entry: Record<string, unknown> | null = null;
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e?.next_status === "human_review") {
      entry = e;
      break;
    }
  }
  if (!entry) return null;

  const reason = String(entry.reason ?? "");
  const meta = (entry.metadata && typeof entry.metadata === "object")
    ? (entry.metadata as Record<string, unknown>)
    : {};
  const arbiterRan = meta.arbiter_ran === true;
  const arbiterUnresolved = typeof meta.arbiter_unresolved === "number" ? meta.arbiter_unresolved : 0;
  const roundCount = typeof meta.round_count === "number" ? meta.round_count : null;
  const maxRounds = typeof meta.max_qc_rounds === "number" ? meta.max_qc_rounds : null;
  const autoEscalated = meta.auto_escalated === true;

  let detail: string | null = null;
  let tone: "warning" | "critical" = "warning";
  if (autoEscalated && !arbiterRan) {
    tone = "critical";
    detail =
      "Arbiter was skipped because the annotator never posted a rebuttal " +
      "(no discussion_replies emitted). The retry loop ran out without " +
      "anyone disputing QC's complaints.";
  } else if (autoEscalated && arbiterRan && arbiterUnresolved > 0) {
    detail = `Arbiter ran but ${arbiterUnresolved} disputes remained unresolved after the retry loop exhausted.`;
  } else if (autoEscalated) {
    detail = "Auto-escalated after the retry loop exhausted.";
  }

  return (
    <div className={`hr-reason-banner ${tone}`}>
      <strong>Why this is in Human Review</strong>
      <p className="hr-reason-quote">{reason}</p>
      {detail ? <p className="hr-reason-detail">{detail}</p> : null}
      {roundCount !== null && maxRounds !== null ? (
        <p className="hr-reason-meta">
          Rounds: {roundCount} / {maxRounds} · Arbiter ran: {arbiterRan ? "yes" : "no"}
          {arbiterRan ? ` · unresolved: ${arbiterUnresolved}` : ""}
        </p>
      ) : null}
    </div>
  );
}

function PreviewArtifact({ artifact }: { artifact: TaskDetailArtifact }) {
  const imageSource = previewImageSource(artifact);
  const boxes = previewBoxes(artifact);
  return (
    <div className="preview-panel">
      <div className="artifact-title">
        <span>{previewTitle(artifact)}</span>
        <span>{boxes.length} boxes</span>
      </div>
      {imageSource ? (
        <div className="image-preview-frame">
          <img alt="" src={imageSource} />
          {boxes.map((box, index) => (
            <span
              className="bbox-overlay"
              key={`${box.label}-${index}`}
              style={{
                left: `${box.left}%`,
                top: `${box.top}%`,
                width: `${box.width}%`,
                height: `${box.height}%`,
              }}
              title={`${box.label}${box.score === null ? "" : ` ${box.score}`}`}
            >
              <span>{box.label}</span>
            </span>
          ))}
        </div>
      ) : null}
      {boxes.length > 0 ? (
        <div className="bbox-list">
          {boxes.map((box, index) => (
            <span key={`${box.label}-${index}`}>
              {box.label}{box.score === null ? "" : ` ${box.score.toFixed(2)}`}
            </span>
          ))}
        </div>
      ) : null}
      <JsonViewer value={artifact.payload} />
    </div>
  );
}

function HumanReviewDecisionForm({
  saving,
  onSubmit,
}: {
  saving: boolean;
  onSubmit: (payload: Record<string, unknown>) => Promise<void>;
}) {
  const [action, setAction] = useState("request_changes");
  const [correctionMode, setCorrectionMode] = useState("manual_annotation");
  const [feedback, setFeedback] = useState("");

  async function submit() {
    await onSubmit({
      action,
      correction_mode: correctionMode,
      feedback,
      actor: "algorithm-engineer",
    });
    setFeedback("");
  }

  return (
    <div className="human-review-form">
      <div className="segmented-row" aria-label="Human Review action">
        <button
          className={action === "request_changes" ? "segment selected" : "segment"}
          type="button"
          onClick={() => setAction("request_changes")}
        >
          Request Changes
        </button>
        <button
          className={action === "accept" ? "segment selected" : "segment"}
          type="button"
          onClick={() => setAction("accept")}
        >
          Accept
        </button>
        <button
          className={action === "reject" ? "segment selected" : "segment"}
          type="button"
          onClick={() => setAction("reject")}
        >
          Reject
        </button>
      </div>
      <select value={correctionMode} onChange={(event) => setCorrectionMode(event.target.value)}>
        <option value="manual_annotation">Manual annotation</option>
        <option value="batch_code_update">Batch code update</option>
      </select>
      <textarea
        placeholder="Decision feedback for the annotator, QC agent, or project record."
        value={feedback}
        onChange={(event) => setFeedback(event.target.value)}
      />
      <button className="primary-button" type="button" disabled={saving || !feedback.trim()} onClick={submit}>
        {saving ? "Saving" : "Submit Decision"}
      </button>
    </div>
  );
}

const ROLE_LABELS: Record<string, string> = {
  annotator: "Annotator",
  qc: "QC Reviewer",
  coordinator: "Coordinator",
};

const STANCE_LABELS: Record<string, string> = {
  agree: "Agree",
  partial_agree: "Partially agree",
  disagree: "Disagree",
  proposal: "Proposal",
};

const STANCE_COLORS: Record<string, string> = {
  agree: "stance-agree",
  partial_agree: "stance-partial",
  disagree: "stance-disagree",
  proposal: "stance-proposal",
};

const SOURCE_LABELS: Record<string, string> = {
  qc: "QC Agent",
  annotation: "Annotation Agent",
  human_review: "Human Reviewer",
};

function ConsensusSummary({ detail }: { detail: TaskDetail }) {
  const c = detail.feedback_consensus;
  return (
    <div className={c.can_accept_by_consensus ? "consensus-box accepted" : "consensus-box"}>
      <strong>
        {c.can_accept_by_consensus ? "All feedback resolved" : `${c.consensus_feedback} of ${c.total_feedback} items resolved`}
      </strong>
      <span>
        {c.can_accept_by_consensus
          ? "Annotator and QC reached agreement on all items — task can pass QC."
          : "Some feedback still needs a response from the annotator or QC reviewer."}
      </span>
    </div>
  );
}

function FeedbackAgreementCard({
  feedback,
  discussions,
}: {
  feedback: Record<string, unknown>;
  discussions: Array<Record<string, unknown>>;
}) {
  const consensusReached = useMemo(() => discussions.some((entry) => entry.consensus === true), [discussions]);

  const sourceLabel = SOURCE_LABELS[String(feedback.source_stage ?? "")] ?? "QC Agent";
  const severityClass = String(feedback.severity) === "critical" ? "severity-critical"
    : String(feedback.severity) === "warning" ? "severity-warning" : "severity-info";

  return (
    <div className="feedback-card">
      <div className="feedback-issue">
        <div className="feedback-issue-meta">
          <span className="feedback-from">{sourceLabel}</span>
          <span className={`feedback-severity ${severityClass}`}>{String(feedback.severity)}</span>
          <span className="feedback-category">{String(feedback.category)}</span>
          <span className={consensusReached ? "agreement-pill accepted" : "agreement-pill"}>
            {consensusReached ? "Resolved" : "Open"}
          </span>
        </div>
        <p className="feedback-message">{highlightQuotedSpans(String(feedback.message))}</p>
      </div>

      {discussions.length === 0 ? (
        <p className="discussion-empty">No responses yet.</p>
      ) : (
        <div className="discussion-thread">
          {discussions.map((entry) => (
            <div key={String(entry.entry_id)} className="discussion-message">
              <div className="discussion-message-meta">
                <span className="discussion-role">{ROLE_LABELS[String(entry.role)] ?? String(entry.role)}</span>
                <span className={`discussion-stance ${STANCE_COLORS[String(entry.stance)] ?? ""}`}>
                  {STANCE_LABELS[String(entry.stance)] ?? String(entry.stance)}
                </span>
                {entry.consensus ? <span className="discussion-consensus-badge">✓ Consensus</span> : null}
              </div>
              <p className="discussion-message-body">{highlightQuotedSpans(String(entry.message))}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Render `text` with any quoted entity references highlighted. Catches the
// most common patterns reviewers use: 'span', "span", `span`, and the
// curly / CJK pairs “span”, ‘span’, 「span」, 『span』. Anything else is
// plain text.
//
// The lookbehind `(?<![A-Za-z0-9])` rules out apostrophes inside words
// (annotator's, don't, U.S.A.'s) so we don't grab the apostrophe in a
// contraction as the start of a "quoted" run. Lookahead does the mirror
// for typographic / CJK closing quotes.
const QUOTE_RE = new RegExp(
  [
    String.raw`(?<![A-Za-z0-9])(['"\`])([^'"\`\n]{1,60}?)\1`,
    String.raw`“([^”\n]{1,60}?)”`,
    String.raw`‘([^’\n]{1,60}?)’`,
    String.raw`「([^」\n]{1,60}?)」`,
    String.raw`『([^』\n]{1,60}?)』`,
  ].join("|"),
  "g",
);

function highlightQuotedSpans(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  QUOTE_RE.lastIndex = 0;
  while ((m = QUOTE_RE.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const inner = m[2] ?? m[3] ?? m[4] ?? m[5] ?? m[6] ?? "";
    parts.push(<mark className="discussion-span-hl" key={key++}>{inner}</mark>);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function DetailSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="detail-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function fmtLogTs(iso: string | undefined | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (!Number.isFinite(d.getTime())) return "";
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const da = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${mo}/${da} ${hh}:${mm}:${ss}`;
}

// ── Unified Logs Tab ──────────────────────────────────────────────────────────

type LogItemKind = "attempt" | "event" | "feedback";

interface UnifiedLogItem {
  kind: LogItemKind;
  ts: string;
  title: string;
  meta: string;
  value: unknown;
}

function buildLogItems(
  attempts: Array<Record<string, unknown>>,
  events: Array<Record<string, unknown>>,
  feedback: Array<Record<string, unknown>>,
): UnifiedLogItem[] {
  const all: UnifiedLogItem[] = [];

  for (const a of attempts) {
    const m = String(a.model ?? "");
    const p = String(a.provider_id ?? "");
    const modelMeta = m && p && m !== p ? `${m} / ${p}` : m || p;
    all.push({
      kind: "attempt",
      ts: String(a.started_at ?? ""),
      title: `${String(a.stage ?? "")} · ${String(a.status ?? "")}`,
      meta: modelMeta,
      value: a,
    });
  }
  for (const e of events) {
    all.push({
      kind: "event",
      ts: String(e.created_at ?? ""),
      title: `${String(e.previous_status ?? "")} → ${String(e.next_status ?? "")}`,
      meta: [e.stage, e.reason].filter(Boolean).join(" · "),
      value: e,
    });
  }
  for (const f of feedback) {
    const cat = String(f.category ?? "");
    const msg = String(f.message ?? "");
    all.push({
      kind: "feedback",
      ts: String(f.created_at ?? ""),
      title: cat,
      meta: msg.length > 90 ? msg.slice(0, 90) + "…" : msg,
      value: f,
    });
  }

  all.sort((a, b) => a.ts.localeCompare(b.ts));
  return all;
}

function LogsTab({
  attempts,
  events,
  feedback,
}: {
  attempts: Array<Record<string, unknown>>;
  events: Array<Record<string, unknown>>;
  feedback: Array<Record<string, unknown>>;
}) {
  const items = useMemo(
    () => buildLogItems(attempts, events, feedback),
    [attempts, events, feedback],
  );

  if (items.length === 0) {
    return <p className="empty-detail">No log entries recorded.</p>;
  }

  return (
    <div className="log-items-list">
      {items.map((item, idx) => (
        <LogItemRow key={idx} item={item} />
      ))}
    </div>
  );
}

// Status → color mapping (matches pipeline stage palette)
const LOG_STATUS_COLORS: Record<string, string> = {
  draft: "#c0c7cc",
  pending: "#9aa6ad",
  annotating: "#4f8fd1",
  qc: "#7b6cb8",
  arbitrating: "#d68b3a",
  human_review: "#d23a2a",
  accepted: "#3aa563",
  rejected: "#7a5848",
  blocked: "#52616b",
  cancelled: "#a9b3b9",
  // attempt statuses
  succeeded: "#3aa563",
  failed: "#d23a2a",
  running: "#4f8fd1",
  // feedback severities
  critical: "#d23a2a",
  warning: "#d68b3a",
  info: "#4f8fd1",
};

function getItemTag(item: UnifiedLogItem): { label: string; color: string } | null {
  const v = item.value as Record<string, unknown>;
  let label = "";
  if (item.kind === "event") label = String(v.next_status ?? "");
  else if (item.kind === "attempt") label = String(v.status ?? "");
  else if (item.kind === "feedback") label = String(v.severity ?? "");
  if (!label) return null;
  return { label, color: LOG_STATUS_COLORS[label] ?? "#9aa6ad" };
}

function LogItemRow({ item }: { item: UnifiedLogItem }) {
  const tag = getItemTag(item);
  return (
    <details className={`log-item log-item-${item.kind}`}>
      <summary className="log-item-summary">
        <span className="log-item-title">{item.title}</span>
        {item.meta ? (
          <>
            <span className="log-item-sep">--</span>
            <span className="log-item-meta">{item.meta}</span>
          </>
        ) : null}
        {tag ? (
          <span
            className="log-item-tag"
            style={{ background: tag.color }}
          >
            {tag.label}
          </span>
        ) : null}
        <span className="log-item-sep">--</span>
        <time className="log-item-ts">{fmtLogTs(item.ts)}</time>
      </summary>
      <div className="log-item-body">
        <JsonViewer value={item.value} />
      </div>
    </details>
  );
}


const ENTITY_TYPES = [
  "person", "organization", "project", "document", "time",
  "number", "event", "location", "technology", "entity",
] as const;

// Pseudo-type for "this span should NOT be tagged as any entity". Stored
// in the same convention table; the runtime formats it as a negative
// instruction when injecting into prompts.
const NOT_ENTITY = "not_an_entity";

function ManualReviewTab({
  projectId,
  taskId,
  sourceRef,
  artifacts,
  feedback,
  feedbackDiscussions,
  saving,
  onSubmitHumanReviewDecision,
}: {
  projectId: string;
  taskId: string;
  sourceRef: unknown;
  artifacts: TaskDetailArtifact[];
  feedback: Array<Record<string, unknown>>;
  feedbackDiscussions: Array<Record<string, unknown>>;
  saving: boolean;
  onSubmitHumanReviewDecision: (payload: Record<string, unknown>) => Promise<void>;
}) {
  return (
    <div className="manual-review-tab">
      <DeviationsBox projectId={projectId} taskId={taskId} />
      <EntityConventionForm
        projectId={projectId}
        taskId={taskId}
        sourceRef={sourceRef}
        artifacts={artifacts}
        feedback={feedback}
        feedbackDiscussions={feedbackDiscussions}
        hrSaving={saving}
        onSubmitHumanReviewDecision={onSubmitHumanReviewDecision}
      />
    </div>
  );
}

type TaskDeviation = {
  span: string;
  current_type: string;
  prior_dominant_type: string;
  prior_total: number;
  prior_distribution: Record<string, number>;
  has_convention: boolean;
};

function DeviationsBox({
  projectId,
  taskId,
}: {
  projectId: string;
  taskId: string;
}) {
  const [data, setData] = useState<TaskDeviation[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [submittingKey, setSubmittingKey] = useState<string | null>(null);
  const [rowStatus, setRowStatus] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  function reload() {
    setLoading(true);
    fetch(`/api/tasks/${encodeURIComponent(taskId)}/deviations`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        setData(d?.deviations ?? []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId]);

  async function applyFix(d: TaskDeviation, newType: string | null) {
    const key = `${d.span}|${d.current_type}`;
    setSubmittingKey(key);
    setError(null);
    try {
      const r = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/posterior-fix`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          span: d.span,
          current_type: d.current_type,
          new_type: newType,
          actor: "manual_review_inline",
        }),
      });
      if (!r.ok) {
        const txt = await r.text();
        throw new Error(`HTTP ${r.status}: ${txt.slice(0, 200)}`);
      }
      setRowStatus((s) => ({ ...s, [key]: `applied: ${newType ?? "deleted"}` }));
      reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmittingKey(null);
    }
  }

  if (loading && data === null) return null;
  if (!data || data.length === 0) return null;

  return (
    <section
      style={{
        background: "#fff4e0",
        border: "1px solid #d97706",
        borderRadius: "4px",
        padding: "0.6rem 0.85rem",
        marginBottom: "0.75rem",
      }}
      aria-label="Posterior audit deviations"
    >
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.4rem" }}>
        <strong style={{ color: "#92400e" }}>
          ⚠ Posterior audit: {data.length} span{data.length === 1 ? "" : "s"} diverge from project prior
        </strong>
        <span style={{ fontSize: "0.75rem", color: "#92400e" }}>
          (this task's annotation disagrees with the project's empirical distribution / operator convention)
        </span>
      </header>
      {error ? (
        <div style={{ color: "var(--danger, #b91c1c)", fontSize: "0.8rem", marginBottom: "0.4rem" }}>
          {error}
        </div>
      ) : null}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #d97706", textAlign: "left" }}>
            <th style={{ padding: "0.3rem 0.5rem 0.3rem 0" }}>Span</th>
            <th style={{ padding: "0.3rem 0.5rem" }}>Now</th>
            <th style={{ padding: "0.3rem 0.5rem" }}>
              Prior dominant{" "}
              <span style={{ fontWeight: 400, fontSize: "0.7rem" }} title="Operator-declared convention if set, else empirical stats">
                ⓘ
              </span>
            </th>
            <th style={{ padding: "0.3rem 0.5rem" }}>Distribution</th>
            <th style={{ padding: "0.3rem 0.5rem" }}>Apply fix</th>
          </tr>
        </thead>
        <tbody>
          {data.map((d) => {
            const key = `${d.span}|${d.current_type}`;
            const status = rowStatus[key];
            const isSubmitting = submittingKey === key;
            const isDone = status?.startsWith("applied:");
            const pct =
              d.prior_total > 0
                ? Math.round((d.prior_distribution[d.prior_dominant_type] / d.prior_total) * 100)
                : 0;
            return (
              <tr key={key} style={{ borderBottom: "1px solid #fcd34d" }}>
                <td style={{ padding: "0.3rem 0.5rem 0.3rem 0", fontFamily: "monospace" }}>
                  {d.span}
                </td>
                <td style={{ padding: "0.3rem 0.5rem", color: "var(--danger, #b91c1c)" }}>
                  {d.current_type}
                </td>
                <td style={{ padding: "0.3rem 0.5rem" }}>
                  <strong>{d.prior_dominant_type}</strong>{" "}
                  <span style={{ color: "#6b7280", fontSize: "0.75rem" }}>
                    {d.has_convention ? "(convention)" : `(${pct}%)`}
                  </span>
                </td>
                <td style={{ padding: "0.3rem 0.5rem", fontSize: "0.75rem", color: "#6b7280" }}>
                  {Object.entries(d.prior_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([t, c]) => `${t}:${c}`)
                    .join(", ")}
                </td>
                <td style={{ padding: "0.3rem 0.5rem", whiteSpace: "nowrap" }}>
                  {isDone ? (
                    <span style={{ color: "var(--success, #047857)" }}>✓ {status?.replace("applied:", "applied")}</span>
                  ) : (
                    <>
                      <button
                        type="button"
                        disabled={isSubmitting}
                        onClick={() => applyFix(d, d.prior_dominant_type)}
                        style={{
                          fontSize: "0.75rem",
                          padding: "0.15rem 0.5rem",
                          marginRight: "0.3rem",
                          background: "var(--success, #047857)",
                          color: "white",
                          border: "none",
                          borderRadius: "3px",
                          cursor: isSubmitting ? "wait" : "pointer",
                          opacity: isSubmitting ? 0.6 : 1,
                        }}
                        title={`Change this task's '${d.span}' from ${d.current_type} → ${d.prior_dominant_type}`}
                      >
                        {isSubmitting ? "…" : `→ ${d.prior_dominant_type}`}
                      </button>
                      <button
                        type="button"
                        disabled={isSubmitting}
                        onClick={() => applyFix(d, null)}
                        style={{
                          fontSize: "0.75rem",
                          padding: "0.15rem 0.5rem",
                          border: "1px dashed #6b7280",
                          background: "white",
                          color: "#6b7280",
                          borderRadius: "3px",
                          cursor: isSubmitting ? "wait" : "pointer",
                        }}
                        title={`Delete this span from the annotation (mark as not an entity)`}
                      >
                        🚫 delete
                      </button>
                    </>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ margin: "0.4rem 0 0", fontSize: "0.7rem", color: "#92400e" }}>
        These are spans in <em>this task</em> whose type disagrees with the project prior (or an
        operator-declared convention). Click <strong>→ type</strong> to overwrite the annotation,{" "}
        <strong>🚫 delete</strong> to drop the span. The fix is applied in place; the task stays
        ACCEPTED with the corrected artifact.
      </p>
    </section>
  );
}

// Find the first occurrence of `span` in `text`, return at least
// ``minTokens`` tokens of surrounding context on each side, snapped to a
// word/character boundary. A token is either a whitespace-delimited word
// (English / Latin scripts) or a single CJK character. Returns null when
// the span isn't in the text.
function spanContext(text: string, span: string, minTokens = 10): {
  before: string;
  match: string;
  after: string;
} | null {
  const idx = text.indexOf(span);
  if (idx === -1) return null;
  const spanEnd = idx + span.length;
  return {
    before: text.slice(walkLeftTokens(text, idx, minTokens), idx),
    match: text.slice(idx, spanEnd),
    after: text.slice(spanEnd, walkRightTokens(text, spanEnd, minTokens)),
  };
}

// CJK / Korean Hangul / fullwidth ranges. Each character in these ranges
// is treated as a single "token" since CJK text rarely uses whitespace
// word boundaries.
const CJK_RE = /[　-鿿가-힯＀-￯]/;
const WS_RE = /\s/;

function walkLeftTokens(text: string, end: number, minTokens: number): number {
  let pos = end;
  let count = 0;
  while (pos > 0 && count < minTokens) {
    while (pos > 0 && WS_RE.test(text[pos - 1])) pos--;
    if (pos === 0) break;
    if (CJK_RE.test(text[pos - 1])) {
      pos--;
      count++;
    } else {
      while (pos > 0 && !WS_RE.test(text[pos - 1]) && !CJK_RE.test(text[pos - 1])) {
        pos--;
      }
      count++;
    }
  }
  // Trim leading whitespace so the excerpt doesn't start with a space.
  while (pos < end && WS_RE.test(text[pos])) pos++;
  return pos;
}

function walkRightTokens(text: string, start: number, minTokens: number): number {
  let pos = start;
  let count = 0;
  while (pos < text.length && count < minTokens) {
    while (pos < text.length && WS_RE.test(text[pos])) pos++;
    if (pos >= text.length) break;
    if (CJK_RE.test(text[pos])) {
      pos++;
      count++;
    } else {
      while (pos < text.length && !WS_RE.test(text[pos]) && !CJK_RE.test(text[pos])) {
        pos++;
      }
      count++;
    }
  }
  // Trim trailing whitespace so the excerpt doesn't end with a space.
  while (pos > start && WS_RE.test(text[pos - 1])) pos--;
  return pos;
}

function extractInputRows(sourceRef: unknown): Array<{ label: string | null; text: string }> {
  if (!sourceRef || typeof sourceRef !== "object") return [];
  const payload = (sourceRef as { payload?: unknown }).payload;
  if (!payload || typeof payload !== "object") return [];
  const rec = payload as Record<string, unknown>;
  if (typeof rec.text === "string" && rec.text.trim()) {
    return [{ label: null, text: rec.text }];
  }
  const rows = rec.rows;
  if (!Array.isArray(rows)) return [];
  const out: Array<{ label: string | null; text: string }> = [];
  for (const r of rows) {
    if (!r || typeof r !== "object") continue;
    const rr = r as Record<string, unknown>;
    let text: string | null = null;
    if (typeof rr.input === "string") text = rr.input;
    else if (rr.input && typeof rr.input === "object") {
      const inner = (rr.input as Record<string, unknown>).text;
      if (typeof inner === "string") text = inner;
    } else if (typeof rr.text === "string") text = rr.text;
    if (!text) continue;
    const id =
      (typeof rr.row_id === "string" && rr.row_id) ||
      (typeof rr.source_id === "string" && rr.source_id) ||
      (typeof rr.row_index === "number" ? `row ${rr.row_index}` : null);
    out.push({ label: id, text });
  }
  return out;
}

function HumanReviewSubmitGate({
  totalSpans,
  addressedCount,
  saving,
  onSubmit,
}: {
  totalSpans: number;
  addressedCount: number;
  saving: boolean;
  onSubmit: (payload: Record<string, unknown>) => Promise<void>;
}) {
  const [feedback, setFeedback] = useState("");

  async function submit(action: "request_changes" | "accept" | "reject") {
    await onSubmit({
      action,
      correction_mode: "manual_annotation",
      feedback: feedback.trim() || (
        action === "accept"
          ? "Accepted with operator's per-span convention picks applied."
          : action === "request_changes"
          ? "Operator requested annotator changes."
          : "Rejected by operator."
      ),
      actor: "operator",
    });
    setFeedback("");
  }

  return (
    <section className="hr-submit-gate">
      <header className="hr-submit-gate-header">
        <h4>Submit Human Review</h4>
        <span
          className="hr-submit-progress ok"
          title="Operator picks are optional — unaddressed spans keep the annotator's call"
        >
          {totalSpans === 0
            ? "no disputed spans"
            : `${addressedCount} of ${totalSpans} span(s) picked`}
        </span>
      </header>
      <textarea
        className="hr-submit-feedback"
        placeholder="Optional note to attach to this decision (e.g. summary of rule change you applied)."
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        rows={2}
      />
      <div className="hr-submit-actions">
        <button
          type="button"
          className="primary-button"
          disabled={saving}
          title="Accept the task; operator picks (if any) are applied to the annotation"
          onClick={() => submit("accept")}
        >
          {saving ? "Submitting…" : "Accept"}
        </button>
        <button
          type="button"
          className="view-tab"
          disabled={saving}
          onClick={() => submit("request_changes")}
        >
          Request Changes
        </button>
        <button
          type="button"
          className="view-tab danger"
          disabled={saving}
          onClick={() => submit("reject")}
        >
          Reject
        </button>
      </div>
    </section>
  );
}

function EntityConventionForm({
  projectId,
  taskId,
  sourceRef,
  artifacts,
  feedback,
  feedbackDiscussions,
  hrSaving,
  onSubmitHumanReviewDecision,
}: {
  projectId: string;
  taskId: string;
  sourceRef: unknown;
  artifacts: TaskDetailArtifact[];
  feedback?: Array<Record<string, unknown>>;
  feedbackDiscussions?: Array<Record<string, unknown>>;
  hrSaving?: boolean;
  onSubmitHumanReviewDecision?: (payload: Record<string, unknown>) => Promise<void>;
}) {
  const inputRows = useMemo(() => extractInputRows(sourceRef), [sourceRef]);
  const [conventions, setConventions] = useState<EntityConvention[]>([]);
  const [span, setSpan] = useState("");
  const [entityType, setEntityType] = useState<string>(ENTITY_TYPES[1]);
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [pendingPick, setPendingPick] = useState<string | null>(null);
  // Spans the operator has explicitly clicked at least once this session.
  // Used to gate the Submit button: every disputed span must be addressed
  // before the operator can accept the task out of Human Review.
  const [addressed, setAddressed] = useState<Set<string>>(() => new Set());
  // One-off picks: lowercase span → chosen type (or null when explicitly
  // cleared). Used when "Save as project convention" is unchecked so the
  // pick is visible in the UI without polluting the project-wide convention
  // table.
  const [localPicks, setLocalPicks] = useState<Map<string, string | null>>(() => new Map());
  // Per-card "Save as project convention" checkbox state, keyed by
  // lowercase span. Default OFF.
  const [saveAsConvention, setSaveAsConvention] = useState<Map<string, boolean>>(() => new Map());
  const getSaveFlag = useCallback(
    (lower: string) => saveAsConvention.get(lower) ?? false,
    [saveAsConvention],
  );

  const reload = useCallback(async () => {
    try {
      const list = await fetchConventions(projectId);
      setConventions(list);
    } catch (e) {
      // silent — listing is best-effort
    }
  }, [projectId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  // IDs of feedback items that have been resolved by consensus — their
  // spans should no longer appear in Manual Review.
  const resolvedFeedbackIds = useMemo(() => {
    const ids = new Set<unknown>();
    if (feedbackDiscussions) {
      for (const d of feedbackDiscussions) {
        if (d.consensus === true) ids.add(d.feedback_id);
      }
    }
    return ids;
  }, [feedbackDiscussions]);

  // Spans explicitly named in unresolved feedback target fields. These are
  // the exact entities reviewers flagged and need operator attention. We
  // prefer structured target.span over free-text matching so that resolved
  // spans don't bleed through via mentions in unrelated messages.
  const unresolvedTargetSpans = useMemo(() => {
    const spans = new Set<string>();
    if (!feedback) return spans;
    for (const f of feedback) {
      if (resolvedFeedbackIds.has(f.feedback_id)) continue;
      if ((f as { category?: unknown }).category === "non_verbatim_span") continue;
      const tgt = (f as { target?: unknown }).target;
      if (tgt && typeof tgt === "object") {
        const span = (tgt as Record<string, unknown>).span;
        if (typeof span === "string") spans.add(span.toLowerCase());
      }
    }
    return spans;
  }, [feedback, resolvedFeedbackIds]);

  // Fallback blob: free-text of unresolved feedback, used only when no
  // structured target.span data exists (older feedback formats).
  const feedbackBlob = useMemo(() => {
    if (!feedback || feedback.length === 0) return "";
    const parts: string[] = [];
    for (const f of feedback) {
      if (resolvedFeedbackIds.has(f.feedback_id)) continue;
      const msg = (f as { message?: unknown }).message;
      if (typeof msg === "string") parts.push(msg);
      const tgt = (f as { target?: unknown }).target;
      if (tgt && typeof tgt === "object") parts.push(JSON.stringify(tgt));
    }
    return parts.join("\n").toLowerCase();
  }, [feedback, resolvedFeedbackIds]);

  // Categories that require span DELETION rather than type selection.
  // These must never drive quickPick display — showing a type-selector for a
  // non-verbatim span would confuse the operator into thinking the issue is
  // about the entity type when the real problem is the span text itself.
  const DELETION_CATEGORIES = new Set(["non_verbatim_span", "arbiter_correction_failed"]);

  // True when there is an unresolved feedback item that is actually about
  // entity type selection (not just deletion of a bad span).
  const hasActionableUnresolvedFeedback = useMemo(
    () => !!feedback && feedback.some(
      (f) => !resolvedFeedbackIds.has(f.feedback_id)
        && !DELETION_CATEGORIES.has(String((f as { category?: unknown }).category ?? "")),
    ),
    [feedback, resolvedFeedbackIds],
  );

  // Extract entity (span, current_type) pairs from this task's latest
  // annotation, filtered to only spans with unresolved feedback.
  // Priority order:
  //   1. Structured target.span from unresolved feedback (most precise).
  //   2. Free-text feedbackBlob match — only from actionable (type-selection)
  //      feedback, never from non_verbatim_span / arbiter_correction_failed.
  //   3. If all feedback is resolved OR only deletion-category feedback remains
  //      → show nothing (empty list).
  //   4. If there is no feedback at all → show everything (discovery mode).
  const quickPicks = useMemo(() => {
    const outputs = extractOutputsByIndex(artifacts);
    const seen = new Set<string>();
    const pairs: Array<{ span: string; currentType: string }> = [];
    for (const out of outputs.values()) {
      const entities = (out as { entities?: Record<string, unknown> }).entities;
      if (!entities || typeof entities !== "object") continue;
      for (const [type, spans] of Object.entries(entities)) {
        if (!Array.isArray(spans)) continue;
        for (const s of spans) {
          if (typeof s !== "string") continue;
          const key = `${s}|${type}`;
          if (seen.has(key)) continue;
          seen.add(key);
          if (unresolvedTargetSpans.size > 0) {
            // Structured unresolved spans available — exact match only.
            if (!unresolvedTargetSpans.has(s.toLowerCase())) continue;
          } else if (hasActionableUnresolvedFeedback) {
            // Actionable unresolved feedback (type-selection) but no structured
            // span target — fall back to free-text matching.
            if (feedbackBlob && !feedbackBlob.includes(s.toLowerCase())) continue;
          } else if (feedback && feedback.length > 0) {
            // Only deletion-category feedback, or all resolved — nothing to select.
            continue;
          }
          pairs.push({ span: s, currentType: type });
        }
      }
    }
    return pairs;
  }, [artifacts, unresolvedTargetSpans, hasActionableUnresolvedFeedback, feedbackBlob, feedback]);

  // Build per-span proposal lists from (a) every annotation_result
  // artifact tagged by actor (annotator vs arbiter) and (b) any QC
  // feedback whose target embeds a structured (span, type) hint.
  // Keys are lowercase span; values are de-duplicated `{actor, type}`
  // entries in display order: annotator → QC → arbiter.
  const proposalsBySpan = useMemo(() => {
    const map = new Map<string, Array<{ actor: "annotator" | "qc" | "arbiter"; type: string }>>();
    const push = (span: string, actor: "annotator" | "qc" | "arbiter", type: string) => {
      const lower = span.toLowerCase();
      const list = map.get(lower) ?? [];
      // De-dupe by actor — keep the first type each actor proposed for a
      // span (extra rounds of the same actor are noisy).
      if (list.some((p) => p.actor === actor)) return;
      list.push({ actor, type });
      map.set(lower, list);
    };
    for (const { actor, spanType } of extractProposalsByActor(artifacts)) {
      for (const [s, t] of spanType.entries()) push(s, actor, t);
    }
    // QC's proposals (when available) live in feedback[].target with
    // varying shapes. Probe the common ones — single-span, types-array,
    // and proposed_type variants.
    for (const f of feedback ?? []) {
      const tgt = (f as { target?: unknown }).target;
      if (!tgt || typeof tgt !== "object") continue;
      const t = tgt as Record<string, unknown>;
      const span = typeof t.span === "string" ? t.span : undefined;
      if (!span) continue;
      // three_way_disagreement / prior_divergence: first_arbiter = proposed_type
      // (already visible from annotation artifacts). Show the competing signal as
      // the "QC" dissent so the operator sees there IS a real disagreement.
      const cat = (f as { category?: unknown }).category;
      if (cat === "three_way_disagreement") {
        const secondType = typeof t.second_arbiter_type === "string" ? t.second_arbiter_type : undefined;
        if (secondType) push(span, "qc", secondType);
        continue;
      }
      if (cat === "prior_divergence") {
        const priorType = typeof t.prior_dominant_type === "string" ? t.prior_dominant_type : undefined;
        if (priorType) push(span, "qc", priorType);
        continue;
      }
      const type =
        typeof t.proposed_type === "string"
          ? t.proposed_type
          : Array.isArray(t.types) && typeof t.types[0] === "string"
            ? (t.types[0] as string)
            : typeof t.type === "string"
              ? (t.type as string)
              : undefined;
      if (type) push(span, "qc", type);
    }
    return map;
  }, [artifacts, feedback]);

  // Index conventions by lowercase span — backend matches on span_lower,
  // so an annotation that produced "LockBit 2.0" should resolve to a
  // convention previously declared as "lockbit 2.0" too.
  const conventionBySpan = useMemo(() => {
    const map = new Map<string, EntityConvention>();
    for (const c of conventions) map.set(c.span.toLowerCase(), c);
    return map;
  }, [conventions]);

  // Toggle semantics for picking a type:
  //  - Click a type that's NOT currently the effective convention →
  //    declare/switch the convention to that type.
  //  - Click the effective type (operator-set OR the original annotator
  //    type when no convention exists) → clear the convention; future
  //    runtime falls back to the annotator's call.
  const togglePick = useCallback(
    async (pickSpan: string, pickType: string, effectiveType: string | null) => {
      const isCancel = pickType === effectiveType;
      const key = `${pickSpan}|${pickType}`;
      const lower = pickSpan.toLowerCase();
      const save = getSaveFlag(lower);
      setPendingPick(key);
      setBusy(true);
      setError(null);
      setMessage(null);
      try {
        if (save) {
          // Project-convention path (default): persist to backend so the
          // pick propagates to future tasks via convention injection.
          if (isCancel) {
            await clearConvention(projectId, pickSpan);
            setMessage(`Cleared "${pickSpan}" — runtime falls back to annotator's call.`);
          } else {
            await declareConvention({
              project_id: projectId,
              span: pickSpan,
              entity_type: pickType,
              task_id: taskId,
              actor: "operator",
            });
            setMessage(
              pickType === NOT_ENTITY
                ? `Set "${pickSpan}" → not an entity.`
                : `Set "${pickSpan}" → ${pickType}.`,
            );
          }
          // Drop any local-only pick for this span — the convention now
          // owns the effective type.
          setLocalPicks((prev) => {
            if (!prev.has(lower)) return prev;
            const next = new Map(prev);
            next.delete(lower);
            return next;
          });
          await reload();
        } else {
          // One-off path: update local state only — no project convention
          // recorded. Pick is still visible (highlighted) and counts as
          // addressed for the gate.
          setLocalPicks((prev) => {
            const next = new Map(prev);
            next.set(lower, isCancel ? null : pickType);
            return next;
          });
          setMessage(
            isCancel
              ? `Cleared "${pickSpan}" (one-off, not saved as convention).`
              : pickType === NOT_ENTITY
                ? `Set "${pickSpan}" → not an entity (one-off, not saved as convention).`
                : `Set "${pickSpan}" → ${pickType} (one-off, not saved as convention).`,
          );
        }
        // Mark this span as "addressed by the operator" regardless of
        // whether they picked a type or cleared it, and regardless of
        // whether the pick was saved as a convention.
        setAddressed((prev) => {
          const next = new Set(prev);
          next.add(lower);
          return next;
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
        setPendingPick(null);
      }
    },
    [projectId, taskId, reload, getSaveFlag],
  );

  const onSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = span.trim();
      if (!trimmed) {
        setError("Span is required");
        return;
      }
      setBusy(true);
      setError(null);
      setMessage(null);
      try {
        const conv = await declareConvention({
          project_id: projectId,
          span: trimmed,
          entity_type: entityType,
          task_id: taskId,
          notes: notes.trim() || undefined,
          actor: "operator",
        });
        setMessage(
          conv.status === "disputed"
            ? `Recorded — now disputed (${conv.evidence_count} evidence, conflicting types in history)`
            : `Recorded "${trimmed}" → ${entityType} (evidence_count=${conv.evidence_count})`,
        );
        setSpan("");
        setNotes("");
        await reload();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [span, entityType, projectId, taskId, notes, reload],
  );

  const onResolveDispute = useCallback(
    async (convId: string, type: string) => {
      setBusy(true);
      setError(null);
      try {
        await resolveConventionDispute(convId, type, null, "operator");
        await reload();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [reload],
  );

  const disputed = conventions.filter((c) => c.status === "disputed");
  const active = conventions.filter((c) => c.status === "active");

  // Picks payload sent to the backend on Accept. We surface a pick for
  // every span the operator clicked this session — one-off picks live in
  // localPicks, saved-as-convention picks come from the refreshed
  // conventionBySpan. The backend rewrites the task's annotation payload
  // so the operator's calls actually land in the exported data.
  const picksPayload = useMemo<Array<{ span: string; entity_type: string | null }>>(() => {
    const out: Array<{ span: string; entity_type: string | null }> = [];
    const seen = new Set<string>();
    for (const { span: pickSpan } of quickPicks) {
      const lower = pickSpan.toLowerCase();
      if (seen.has(lower)) continue;
      const localPick = localPicks.get(lower);
      if (localPick !== undefined) {
        out.push({ span: pickSpan, entity_type: localPick });
        seen.add(lower);
        continue;
      }
      const conv = conventionBySpan.get(lower);
      if (conv) {
        out.push({ span: pickSpan, entity_type: conv.entity_type });
        seen.add(lower);
      }
    }
    return out;
  }, [quickPicks, localPicks, conventionBySpan]);

  const wrappedSubmit = useCallback(
    async (payload: Record<string, unknown>) => {
      if (!onSubmitHumanReviewDecision) return;
      await onSubmitHumanReviewDecision({ ...payload, picks: picksPayload });
    },
    [onSubmitHumanReviewDecision, picksPayload],
  );

  return (
    <section className="entity-convention-box">
      <p className="hint">
        Click a type to set the operator's pick for that span. By default the
        pick is saved as a project-wide convention (injected into future
        annotator / QC / arbiter prompts). Uncheck <em>Save as project
        convention</em> on a card to mark it as a one-off fix instead.
      </p>
      {error ? <p className="convention-error">{error}</p> : null}
      {message ? <p className="convention-ok">{message}</p> : null}

      {quickPicks.length === 0 && feedback && feedback.length > 0 ? (() => {
        // Show unresolved deletion-category items so the operator understands
        // why there are no entity type choices to make.
        const deletionItems = feedback.filter(
          (f) =>
            !resolvedFeedbackIds.has(f.feedback_id) &&
            DELETION_CATEGORIES.has(String((f as { category?: unknown }).category ?? "")),
        );
        if (deletionItems.length === 0) return null;
        // Deduplicate by span text so 39 non_verbatim_span records don't flood the UI.
        const seen = new Set<string>();
        const unique = deletionItems.filter((f) => {
          const tgt = (f as { target?: unknown }).target as Record<string, unknown> | null;
          const span = tgt && typeof tgt.span === "string" ? tgt.span : String(f.feedback_id);
          if (seen.has(span)) return false;
          seen.add(span);
          return true;
        });
        return (
          <div className="convention-correction-failed">
            <div className="correction-failed-card">
              <strong>Non-verbatim spans — no type selection needed</strong>
              <p>The following span(s) do not appear verbatim in the source text and cannot be fixed by choosing a type:</p>
              <ul style={{ margin: "4px 0", paddingLeft: 18 }}>
                {unique.slice(0, 8).map((f) => {
                  const tgt = (f as { target?: unknown }).target as Record<string, unknown> | null;
                  const span = tgt && typeof tgt.span === "string" ? tgt.span : null;
                  const field = tgt && typeof tgt.field === "string" ? tgt.field : null;
                  return span ? (
                    <li key={String(f.feedback_id)}>
                      <code>{span}</code>{field ? <> in <code>{field}</code></> : null}
                    </li>
                  ) : null;
                })}
                {unique.length > 8 ? <li>…and {unique.length - 8} more</li> : null}
              </ul>
              <p className="correction-failed-hint">Use <strong>Request Changes</strong> to re-queue for annotation, or <strong>Reject</strong> if unfixable.</p>
            </div>
          </div>
        );
      })() : null}

      {quickPicks.length > 0 ? (
        <div className="convention-quick-picks">
          {quickPicks.map(({ span: pickSpan, currentType }) => {
            const lower = pickSpan.toLowerCase();
            const existing = conventionBySpan.get(lower);
            const localPick = localPicks.get(lower);
            const saveFlag = getSaveFlag(lower);
            // First row whose input text contains this span — show its
            // surrounding sentence so the operator doesn't have to read the
            // whole task.
            let ctx: { before: string; match: string; after: string } | null = null;
            for (const r of inputRows) {
              ctx = spanContext(r.text, pickSpan);
              if (ctx) break;
            }
            const proposals = proposalsBySpan.get(lower) ?? [];
            return (
              <div className="convention-pick-card" key={`${pickSpan}|${currentType}`}>
                {ctx ? (
                  <p className="convention-pick-context">
                    {ctx.before ? <>…{ctx.before}</> : null}
                    <mark>{ctx.match}</mark>
                    {ctx.after ? <>{ctx.after}…</> : null}
                  </p>
                ) : null}
                {proposals.length > 0 ? (
                  <div className="convention-pick-actors">
                    {proposals.map(({ actor, type }) => {
                      const effectiveType =
                        localPick !== undefined
                          ? localPick
                          : existing?.entity_type ?? currentType;
                      const isSelected = type === effectiveType;
                      const key = `actor:${pickSpan}|${actor}|${type}`;
                      const pending = pendingPick === `${pickSpan}|${type}`;
                      const cls = [
                        "convention-pick-actor-btn",
                        `convention-pick-actor-${actor}`,
                        isSelected ? "selected" : "",
                      ].filter(Boolean).join(" ");
                      const label =
                        actor === "annotator" ? "Annotator"
                          : actor === "arbiter" ? "Arbiter"
                            : "QC";
                      return (
                        <button
                          type="button"
                          key={key}
                          className={cls}
                          disabled={busy}
                          title={
                            isSelected
                              ? `Currently the effective type (set by ${actor}) — click again to clear`
                              : `Adopt ${actor}'s call: ${pickSpan} → ${type}`
                          }
                          onClick={() => togglePick(pickSpan, type, effectiveType)}
                        >
                          {pending ? "…" : `${label}: ${type}`}
                        </button>
                      );
                    })}
                  </div>
                ) : null}
                <div className="convention-pick-row">
                  <code className="convention-pick-span">{pickSpan}</code>
                  <span className="convention-pick-sep">→</span>
                  {(() => {
                    // The "effective" selection is whatever the runtime would
                    // use right now: a one-off operator pick wins, else the
                    // operator's saved convention, else the annotator's call.
                    // Clicking it toggles off.
                    const effectiveType =
                      localPick !== undefined
                        ? localPick
                        : existing?.entity_type ?? currentType;
                    return (
                      <>
                        {ENTITY_TYPES.map((t) => {
                          const isSelected = t === effectiveType;
                          const key = `${pickSpan}|${t}`;
                          const pending = pendingPick === key;
                          const cls = [
                            "convention-pick-btn",
                            isSelected ? "selected" : "",
                          ].filter(Boolean).join(" ");
                          return (
                            <button
                              type="button"
                              key={t}
                              className={cls}
                              disabled={busy}
                              title={
                                isSelected
                                  ? `Current selection — click again to clear (fallback to annotator)`
                                  : `Set ${pickSpan} → ${t}`
                              }
                              onClick={() => togglePick(pickSpan, t, effectiveType)}
                            >
                              {pending ? "…" : t}
                            </button>
                          );
                        })}
                        {(() => {
                          const isSelected = NOT_ENTITY === effectiveType;
                          const key = `${pickSpan}|${NOT_ENTITY}`;
                          const pending = pendingPick === key;
                          const cls = [
                            "convention-pick-btn",
                            "convention-pick-btn-negative",
                            isSelected ? "selected" : "",
                          ].filter(Boolean).join(" ");
                          return (
                            <button
                              type="button"
                              className={cls}
                              disabled={busy}
                              title={
                                isSelected
                                  ? `Current selection — click again to clear`
                                  : `Set ${pickSpan} → not an entity`
                              }
                              onClick={() => togglePick(pickSpan, NOT_ENTITY, effectiveType)}
                            >
                              {pending ? "…" : "✗ not entity"}
                            </button>
                          );
                        })()}
                      </>
                    );
                  })()}
                </div>
                <label
                  className="convention-pick-save-label"
                  title="When checked, the pick is recorded as a project-wide convention. Uncheck for a one-off fix that won't affect future tasks."
                >
                  <input
                    type="checkbox"
                    checked={saveFlag}
                    disabled={busy}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      setSaveAsConvention((prev) => {
                        const next = new Map(prev);
                        next.set(lower, checked);
                        return next;
                      });
                    }}
                  />
                  <span>Save as project convention</span>
                </label>
                {existing?.status === "disputed" ? (
                  <p className="convention-pick-history">
                    <span className="convention-pick-disputed-tag">disputed</span>
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : (
        <p className="hint">
          {feedbackBlob
            ? "No annotated entity matches anything in the QC / HR feedback for this task. Use the manual form below to declare a convention for a span not yet annotated."
            : "No entities found in this task's annotation."}
        </p>
      )}

      <details className="convention-manual">
        <summary>+ declare a span not in the annotation</summary>
        <form onSubmit={onSubmit} className="convention-form">
          <input
            type="text"
            placeholder="span (e.g. Gmail, Apple)"
            value={span}
            onChange={(e) => setSpan(e.target.value)}
            disabled={busy}
          />
          <select value={entityType} onChange={(e) => setEntityType(e.target.value)} disabled={busy}>
            {ENTITY_TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="notes (optional)"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            disabled={busy}
          />
          <button type="submit" disabled={busy || !span.trim()}>
            {busy ? "Saving..." : "Declare convention"}
          </button>
        </form>
      </details>

      {onSubmitHumanReviewDecision ? (
        <HumanReviewSubmitGate
          totalSpans={quickPicks.length}
          addressedCount={addressed.size}
          saving={!!hrSaving}
          onSubmit={wrappedSubmit}
        />
      ) : null}
      {disputed.length > 0 ? (
        <div className="convention-disputed">
          <strong>Disputed conventions ({disputed.length})</strong>
          <ul>
            {disputed.map((c) => {
              const proposed = Array.from(
                new Set(c.proposals.map((p) => String((p as { type?: unknown }).type ?? ""))),
              ).filter(Boolean);
              return (
                <li key={c.convention_id}>
                  <code>{c.span}</code> — proposed:{" "}
                  {proposed.map((t) => (
                    <button
                      key={t}
                      type="button"
                      className="convention-resolve-btn"
                      onClick={() => onResolveDispute(c.convention_id, t)}
                      disabled={busy}
                    >
                      keep {t}
                    </button>
                  ))}
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
      {active.length > 0 ? (
        <details className="convention-list">
          <summary>Active conventions ({active.length})</summary>
          <ul>
            {active.map((c) => (
              <li key={c.convention_id}>
                <code>{c.span}</code> → <em>{c.entity_type}</em>{" "}
                <small>
                  (×{c.evidence_count}, {c.created_by})
                </small>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}

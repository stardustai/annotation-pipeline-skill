import { useState } from "react";
import { cardSubtitle, visibleColumns } from "../kanban";
import type { KanbanSnapshot, TaskCard } from "../types";

interface KanbanBoardProps {
  snapshot: KanbanSnapshot;
  selectedTaskId: string | null;
  onSelectTask: (task: TaskCard) => void;
  onMoveTask?: (task: TaskCard, targetStatus: string, reason: string) => Promise<void> | void;
}

// Mirror of the backend whitelist in DashboardApi._MANUAL_MOVE_WHITELIST.
// Each entry maps a card's current task.status to the columns it may be
// dragged into. Cards whose status is not a key here are not draggable.
const MANUAL_MOVE_TARGETS: Record<string, string[]> = {
  rejected: ["arbitrating"],
  human_review: ["arbitrating", "pending"],
  accepted: ["human_review"],
};

interface ColumnState {
  page: number;
  filter: string;
}

const PAGE_SIZE = 20;

function filterCard(card: TaskCard, query: string): boolean {
  const q = query.toLowerCase();
  return (
    card.task_id.toLowerCase().includes(q) ||
    card.modality.toLowerCase().includes(q) ||
    (card.annotator_model?.toLowerCase().includes(q) ?? false) ||
    (card.qc_model?.toLowerCase().includes(q) ?? false)
  );
}

export function KanbanBoard({ snapshot, selectedTaskId, onSelectTask, onMoveTask }: KanbanBoardProps) {
  const [columnStates, setColumnStates] = useState<Record<string, ColumnState>>({});
  const [dragSource, setDragSource] = useState<{ taskId: string; sourceStatus: string } | null>(null);
  const [hoverColumn, setHoverColumn] = useState<string | null>(null);

  function getState(columnId: string): ColumnState {
    return columnStates[columnId] ?? { page: 0, filter: "" };
  }

  function setFilter(columnId: string, filter: string) {
    setColumnStates((prev) => ({ ...prev, [columnId]: { page: 0, filter } }));
  }

  function setPage(columnId: string, page: number) {
    setColumnStates((prev) => ({ ...prev, [columnId]: { ...getState(columnId), page } }));
  }

  function allowedTargets(sourceStatus: string): string[] {
    return MANUAL_MOVE_TARGETS[sourceStatus] ?? [];
  }

  function isValidDropTarget(columnId: string): boolean {
    if (!dragSource) return false;
    return allowedTargets(dragSource.sourceStatus).includes(columnId);
  }

  async function handleDrop(columnId: string, card: TaskCard | null) {
    setHoverColumn(null);
    if (!dragSource || !onMoveTask) return;
    if (!allowedTargets(dragSource.sourceStatus).includes(columnId)) {
      setDragSource(null);
      return;
    }
    const reason = window.prompt(
      `Move ${dragSource.taskId} from ${dragSource.sourceStatus} to ${columnId}.\nReason (required):`,
      "",
    );
    setDragSource(null);
    if (!reason || !reason.trim()) return;
    // The card object lookup is for the optional drop target case; we already
    // have what we need from dragSource. card is unused for now but kept for
    // future drop-on-card actions (reordering, etc.).
    void card;
    try {
      await onMoveTask(
        snapshotCardOrShim(snapshot, dragSource.taskId),
        columnId,
        reason.trim(),
      );
    } catch (err) {
      console.error("move failed", err);
      window.alert(err instanceof Error ? err.message : "Move failed");
    }
  }

  return (
    <section className="kanban-board" aria-label="Task Kanban board">
      {visibleColumns(snapshot).map((column) => {
        const { page, filter } = getState(column.id);
        const sorted = [...column.cards].sort((a, b) => a.status_age_seconds - b.status_age_seconds);
        const filtered = filter ? sorted.filter((card) => filterCard(card, filter)) : sorted;
        const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
        const safePage = Math.min(page, pageCount - 1);
        const pageCards = filtered.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);
        const dropOk = isValidDropTarget(column.id);
        const isHovered = hoverColumn === column.id && dropOk;

        return (
          <div
            className={`kanban-column${dropOk ? " drop-allowed" : ""}${isHovered ? " drop-active" : ""}`}
            key={column.id}
            onDragOver={(e) => {
              if (!dropOk) return;
              e.preventDefault();
              e.dataTransfer.dropEffect = "move";
              if (hoverColumn !== column.id) setHoverColumn(column.id);
            }}
            onDragLeave={() => {
              if (hoverColumn === column.id) setHoverColumn(null);
            }}
            onDrop={(e) => {
              e.preventDefault();
              void handleDrop(column.id, null);
            }}
          >
            <div className="column-header">
              <h2>{column.title}</h2>
              <span>{column.cards.length}</span>
            </div>

            <div className="column-pagination">
              <button
                className="page-arrow"
                type="button"
                disabled={safePage === 0}
                aria-label="Previous page"
                onClick={() => setPage(column.id, safePage - 1)}
              >
                ‹
              </button>
              <input
                className="column-filter"
                type="text"
                value={filter}
                placeholder={pageCount > 1 ? `${safePage + 1} / ${pageCount}` : "filter…"}
                aria-label={`Filter ${column.title} tasks`}
                onChange={(e) => setFilter(column.id, e.target.value)}
              />
              <button
                className="page-arrow"
                type="button"
                disabled={safePage >= pageCount - 1}
                aria-label="Next page"
                onClick={() => setPage(column.id, safePage + 1)}
              >
                ›
              </button>
            </div>

            <div className="card-stack">
              {pageCards.map((card) => {
                const isDraggable = card.status in MANUAL_MOVE_TARGETS;
                return (
                  <button
                    className={card.task_id === selectedTaskId ? "task-card selected" : "task-card"}
                    key={card.task_id}
                    type="button"
                    draggable={isDraggable}
                    onDragStart={(e) => {
                      if (!isDraggable) return;
                      e.dataTransfer.effectAllowed = "move";
                      e.dataTransfer.setData("text/plain", card.task_id);
                      setDragSource({ taskId: card.task_id, sourceStatus: card.status });
                    }}
                    onDragEnd={() => {
                      setDragSource(null);
                      setHoverColumn(null);
                    }}
                    onClick={() => onSelectTask(card)}
                  >
                    <span className="task-id">{card.task_id}</span>
                    <span className="task-subtitle">{cardSubtitle(card)}</span>
                    <span className="task-meta">{formatAge(card.status_age_seconds)}</span>
                    <span className="task-meta">
                      {card.row_count !== null ? `${card.row_count} rows · ` : ""}
                      {card.attempt_count} {card.attempt_count === 1 ? "attempt" : "attempts"}
                    </span>
                    <span className="badges">
                      {card.annotator_model ? <span className="badge model-a" title="Annotator model">A: {card.annotator_model}</span> : null}
                      {card.qc_model ? <span className="badge model-q" title="QC model">Q: {card.qc_model}</span> : null}
                      {card.feedback_count > 0 ? <span className="badge warn">{card.feedback_count} feedback</span> : null}
                      {card.retry_pending ? (
                        <span className={`badge ${(card.retry_wait_seconds ?? 0) > 0 ? "retry-wait" : "retry-ready"}`}>
                          {(card.retry_wait_seconds ?? 0) > 0
                            ? `retry ${card.retry_wait_seconds}s`
                            : "retry"}
                        </span>
                      ) : null}
                      {card.external_sync_pending ? <span className="badge">sync</span> : null}
                    </span>
                  </button>
                );
              })}
              {filtered.length === 0 && filter ? (
                <p className="column-empty">No matches</p>
              ) : null}
            </div>
          </div>
        );
      })}
    </section>
  );
}

function snapshotCardOrShim(snapshot: KanbanSnapshot, taskId: string): TaskCard {
  for (const column of snapshot.columns) {
    for (const card of column.cards) {
      if (card.task_id === taskId) return card;
    }
  }
  // Should not happen — drag sources always come from a visible card. Fall
  // back to a minimal shim so the callback signature is preserved.
  return {
    task_id: taskId,
    status: "",
    operator_stage: "",
    pipeline_chain: "",
    modality: "",
    annotation_types: [],
    selected_annotator_id: null,
    annotator_model: null,
    qc_model: null,
    status_age_seconds: 0,
    latest_attempt_status: null,
    feedback_count: 0,
    retry_pending: false,
    retry_wait_seconds: null,
    blocked: false,
    external_sync_pending: false,
    row_count: null,
    attempt_count: 0,
  };
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return `${(seconds / 3600).toFixed(1)}h ago`;
}

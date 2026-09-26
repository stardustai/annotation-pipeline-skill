import type { KanbanColumn, KanbanSnapshot, TaskCard } from "./types";

export function countCards(snapshot: KanbanSnapshot): number {
  // Dedupe by task_id: in multi-annotation an in-flight task appears in every
  // annotator column at once (parallel), so a plain sum would over-count.
  const ids = new Set<string>();
  for (const column of snapshot.columns) {
    for (const card of column.cards) ids.add(card.task_id);
  }
  return ids.size;
}

export function visibleColumns(snapshot: KanbanSnapshot): KanbanColumn[] {
  return snapshot.columns;
}

export function cardSubtitle(card: Pick<TaskCard, "modality">): string {
  return card.modality;
}

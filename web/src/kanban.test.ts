import { describe, expect, it } from "vitest";
import { cardSubtitle, countCards, visibleColumns } from "./kanban";
import type { KanbanSnapshot } from "./types";

const snapshot: KanbanSnapshot = {
  project_id: null,
  columns: [
    {
      id: "pending",
      title: "Pending",
      cards: [
        {
          task_id: "task-1",
          status: "pending",
          operator_stage: "pending",
          pipeline_chain: "",
          modality: "text",
          annotation_types: ["entity_span"],
          selected_annotator_id: null,
          annotator_model: null,
          qc_model: null,
          status_age_seconds: 3,
          latest_attempt_status: null,
          feedback_count: 0,
          retry_pending: false,
          retry_wait_seconds: null,
          blocked: false,
          external_sync_pending: false,
          row_count: null,
          attempt_count: 0,
        },
      ],
    },
    { id: "human_review", title: "Human Review", cards: [] },
  ],
};

describe("kanban helpers", () => {
  it("counts cards across columns", () => {
    expect(countCards(snapshot)).toBe(1);
  });

  it("keeps empty operational columns visible", () => {
    expect(visibleColumns(snapshot).map((column) => column.id)).toEqual(["pending", "human_review"]);
  });

  it("returns modality as the card subtitle", () => {
    expect(cardSubtitle({ modality: "image" })).toBe("image");
  });
});

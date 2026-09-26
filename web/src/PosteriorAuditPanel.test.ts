import { describe, it, expect, vi } from "vitest";
import { renderToString } from "react-dom/server";
import React from "react";
import { PosteriorAuditPanel } from "./components/PosteriorAuditPanel";

describe("PosteriorAuditPanel", () => {
  it("renders header + Count button without a project", () => {
    const html = renderToString(
      React.createElement(PosteriorAuditPanel, {
        projectId: null,
        onSendToHr: vi.fn(),
        onDeclareCanonical: vi.fn(),
      })
    );
    // Header + Count button always render (the button reads "Re-count" once a scan is cached).
    expect(html).toContain("Posterior Audit");
    expect(html).toContain("Count");
    // No project -> no cached result, panel shows the "no cache yet" hint.
    expect(html).toContain("No cached scan yet");
  });
});

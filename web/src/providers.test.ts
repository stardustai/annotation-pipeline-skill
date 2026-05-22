import { describe, expect, it } from "vitest";
import { createProviderProfile, profileTitle, providerConfigPayload } from "./providers";
import type { ProviderConfigSnapshot } from "./types";

const snapshot: ProviderConfigSnapshot = {
  config_valid: true,
  profiles: [
    {
      name: "local_codex",
      runtime: "codex_cli",
      model: "gpt-5.4-mini",
      base_url: "https://api.openai.com",
      api_key_env: "OPENAI_API_KEY",
      reasoning_effort: "none",
      permission_mode: null,
      timeout_seconds: 900,
      max_retries: null,
      concurrency_limit: null,
      no_progress_timeout_seconds: 30,
      disable_continuity: null,
    },
  ],
  targets: { annotation: "local_codex", qc: "local_codex" },
  limits: { max_concurrent_tasks: 4 },
  diagnostics: {
    local_codex: {
      status: "ok",
      checks: [{ id: "cli_binary_found", status: "ok", message: "codex is available" }],
    },
  },
};

describe("provider config helpers", () => {
  it("creates explicit provider profiles for selected runtime kinds", () => {
    expect(createProviderProfile("codex_cli", 2)).toMatchObject({
      name: "profile_2",
      runtime: "codex_cli",
      model: "gpt-5.5",
      base_url: "https://api.openai.com",
      api_key_env: "OPENAI_API_KEY",
    });
  });

  it("builds a compact save payload without diagnostics", () => {
    expect(providerConfigPayload(snapshot)).toEqual({
      profiles: snapshot.profiles,
      targets: snapshot.targets,
      limits: snapshot.limits,
    });
  });

  it("formats profile titles for operator scanning", () => {
    expect(profileTitle(snapshot.profiles[0])).toBe("local_codex · codex_cli · gpt-5.4-mini");
  });
});

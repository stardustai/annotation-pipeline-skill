import type { ProviderConfigSnapshot, Runtime, ProviderProfileConfig } from "./types";

export function createProviderProfile(runtime: Runtime, index: number): ProviderProfileConfig {
  const defaults: Record<Runtime, { model: string; base_url: string; api_key_env: string }> = {
    claude_cli:     { model: "claude-sonnet-4-5",         base_url: "https://api.anthropic.com",  api_key_env: "ANTHROPIC_API_KEY" },
    codex_cli:      { model: "gpt-5.5",                   base_url: "https://api.openai.com",     api_key_env: "OPENAI_API_KEY" },
    anthropic_sdk:  { model: "claude-sonnet-4-5",         base_url: "https://api.anthropic.com",  api_key_env: "ANTHROPIC_API_KEY" },
    openai_sdk:     { model: "gpt-4o",                    base_url: "https://api.openai.com/v1",  api_key_env: "OPENAI_API_KEY" },
  };
  const { model, base_url, api_key_env } = defaults[runtime];
  return {
    name: `profile_${index}`,
    runtime,
    model,
    base_url,
    api_key_env,
    reasoning_effort: null,
    permission_mode: null,
    timeout_seconds: null,
    max_retries: null,
    concurrency_limit: null,
    no_progress_timeout_seconds: null,
    disable_continuity: null,
  };
}

export function providerConfigPayload(snapshot: ProviderConfigSnapshot) {
  return {
    profiles: snapshot.profiles,
    targets: snapshot.targets,
    limits: snapshot.limits,
  };
}

export function profileTitle(profile: ProviderProfileConfig): string {
  return `${profile.name} · ${profile.runtime} · ${profile.model}`;
}

export function profileStatusLabel(snapshot: ProviderConfigSnapshot, name: string): string {
  const diag = snapshot.diagnostics[name];
  if (!diag) return "unknown";
  return diag.status === "ok" ? "ok" : "error";
}

import type { ProviderConfigSnapshot, Runtime, ProviderProfileConfig } from "./types";

export function createProviderProfile(runtime: Runtime, index: number): ProviderProfileConfig {
  return {
    name: `profile_${index}`,
    runtime,
    model: runtime === "codex_cli" ? "gpt-5.5" : "deepseek-v4-flash",
    base_url: runtime === "codex_cli" ? "https://api.openai.com" : "https://api.deepseek.com/anthropic",
    api_key_env: runtime === "codex_cli" ? "OPENAI_API_KEY" : "DEEPSEEK_API_KEY",
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

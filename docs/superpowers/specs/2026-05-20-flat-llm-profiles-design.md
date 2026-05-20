# Flat LLM Profiles Schema

**Date:** 2026-05-20  
**Status:** Approved

## Problem

The existing `llm_profiles.yaml` schema was designed around `provider: openai_compatible` and carries several layers that no longer reflect reality:

- `provider` / `provider_flavor` / `cli_kind` are a three-level hierarchy that partially overlaps and requires conditional UI rendering
- `base_url` and `api_key_env` are only shown/validated for `openai_compatible` profiles, but the new architecture puts them at the center of every profile (they supply `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` to Claude CLI)
- All providers now route through Claude CLI via the Anthropic protocol — `http_api` (direct REST calls) is no longer used because CLI provides continuity, tool use, and MCP support

## Decision

Replace the layered schema with a flat, uniform structure. Every profile has the same fields. No inference — routing is controlled by an explicit `runtime` field.

## New Schema

```yaml
profiles:
  deepseek_claude:
    runtime: claude_cli              # claude_cli | codex_cli — determines which binary to exec
    model: deepseek-v4-flash
    base_url: https://api.deepseek.com/anthropic
    api_key_env: DEEPSEEK_API_KEY    # string or list[string]
    timeout_seconds: 120
    # optional tuning — null means "use default"
    reasoning_effort: null
    permission_mode: null
    concurrency_limit: null
    max_retries: null
    no_progress_timeout_seconds: null
    disable_continuity: null

targets:
  annotation: minimax_claude
  qc: deepseek_claude
  coordinator: glm_claude

limits:
  local_cli_global_concurrency: 8
```

**Required fields:** `runtime`, `model`, `base_url`, `api_key_env`  
**Optional fields:** all others (null = use default / not applicable)

**Removed fields:** `provider`, `provider_flavor`, `cli_kind`, `cli_binary`, `reasoning_capable`

Runtime values:
- `claude_cli` — exec `claude` binary, inject `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`
- `codex_cli` — exec `codex` binary, use isolated CODEX_HOME with `OPENAI_BASE_URL` + `OPENAI_API_KEY`

Binary name is derived from `runtime` (`claude_cli` → `claude`, `codex_cli` → `codex`). No `cli_binary` override — use a shell alias or symlink if a non-default path is needed.

## Changes by Layer

### `annotation_pipeline_skill/llm/profiles.py`

- Remove: `provider`, `provider_flavor`, `cli_kind`, `cli_binary`, `reasoning_capable` from `LLMProfile` dataclass
- Add: `runtime: Literal["claude_cli", "codex_cli"]`
- Remove: `ProviderName`, `ProviderFlavor`, `CliKind` type aliases
- `_validate_profile()`: require `runtime` + `model` + `base_url` + `api_key_env`; all other fields optional
- `_parse_profile()`: parse `runtime` instead of `provider`/`cli_kind`

### `annotation_pipeline_skill/llm/local_cli.py`

- `LocalCLIClient.generate()`: dispatch on `profile.runtime` instead of `profile.cli_kind`
- Binary name derived: `"claude" if profile.runtime == "claude_cli" else "codex"`
- Remove references to `profile.cli_kind`, `profile.cli_binary`

### `annotation_pipeline_skill/services/provider_config_service.py`

- `_profile_to_dict()`: emit `runtime` instead of `provider`/`provider_flavor`/`cli_kind`/`cli_binary`
- `_profile_diagnostics()`: branch on `runtime` instead of `provider`
  - `claude_cli`: check `claude` on PATH, `base_url` set, `api_key_env` resolves
  - `codex_cli`: check `codex` on PATH, `base_url` set, `api_key_env` resolves

### `scripts/migrate_llm_profiles.py` (new)

One-time migration from old schema to new flat schema. Mapping rules:

| Old | New `runtime` | Note |
|-----|--------------|-------|
| `local_cli` + `cli_kind: claude` | `claude_cli` | direct |
| `local_cli` + `cli_kind: codex` | `codex_cli` | direct |
| `openai_compatible` + any flavor | `claude_cli` | **warn**: `base_url` must be changed to Anthropic endpoint (e.g. `https://api.deepseek.com` → `https://api.deepseek.com/anthropic`) |

Script emits a warning for each `openai_compatible` profile requiring manual `base_url` review before writing output.

### `web/src/types.ts`

- `ProviderProfileConfig`: remove `provider`, `provider_flavor`, `cli_kind`, `cli_binary`, `reasoning_capable`; add `runtime: "claude_cli" | "codex_cli"`

### `web/src/providers.ts`

- `createProviderProfile()`: default `runtime: "claude_cli"`, remove provider-type branching logic

### `web/src/components/ProvidersPanel.tsx`

Remove all conditional field rendering based on `provider` type. Every profile shows the same fields:

```
┌─ Profile Editor ─────────────────────────────────────────┐
│  Runtime      [claude_cli ▼]    Model   [deepseek-v4-flash] │
│  Base URL     [https://api.deepseek.com/anthropic         ] │
│  API Key Env  [DEEPSEEK_API_KEY                           ] │
├─ Tuning (collapsible) ────────────────────────────────────┤
│  Timeout [120]  Max Retries [—]  Concurrency [—]          │
│  No-Progress Timeout [—]  Reasoning Effort [—]            │
│  Permission Mode [—]  Disable Continuity [□]              │
└───────────────────────────────────────────────────────────┘
```

Diagnostics section branches on `runtime` instead of `provider`.

## Migration Path

1. Run `scripts/migrate_llm_profiles.py --input llm_profiles.yaml --output llm_profiles.yaml` (dry-run by default, `--apply` to write)
2. Review any warnings about `base_url` requiring manual update
3. Verify profiles load with new `load_llm_registry()`
4. Old schema format is no longer supported after migration

## Out of Scope

- `http_api` runtime (direct REST API without CLI) — not added; all providers use CLI
- Per-profile MCP config or tool allowlists — future work
- Multi-key rotation or secret management — `api_key_env` list already supports fallback chain

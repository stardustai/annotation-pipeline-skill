# Flat LLM Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the layered `provider`/`provider_flavor`/`cli_kind` schema with a flat `runtime` field, updating all backend, service, frontend, and test layers.

**Architecture:** `LLMProfile` gains a single `runtime: Literal["claude_cli", "codex_cli"]` field. All other fields (`base_url`, `api_key_env`, `permission_mode`, etc.) are uniform across all profiles. `LocalCLIClient` dispatches on `runtime`; the `openai_compatible`/`openai_responses` factories are removed. A one-time migration script converts existing YAML files.

**Tech Stack:** Python 3.13, dataclasses, PyYAML, React/TypeScript, pytest

---

## File Map

| File | Change |
|------|--------|
| `annotation_pipeline_skill/llm/profiles.py` | Replace `provider`/`provider_flavor`/`cli_kind`/`cli_binary`/`reasoning_capable` with `runtime` |
| `annotation_pipeline_skill/llm/local_cli.py` | Dispatch on `profile.runtime`; derive binary from runtime |
| `annotation_pipeline_skill/interfaces/cli.py` | `_build_llm_client()` dispatches on `runtime` instead of `provider` |
| `annotation_pipeline_skill/services/provider_config_service.py` | `_profile_to_dict()` + `_profile_diagnostics()` use `runtime` |
| `scripts/migrate_llm_profiles.py` | New — one-time old→new YAML migration |
| `projects/v3_initial_deployment/.annotation-pipeline/llm_profiles.yaml` | Migrated to flat schema |
| `web/src/types.ts` | `ProviderProfileConfig` gets `runtime`, loses old fields |
| `web/src/providers.ts` | `createProviderProfile()` and `profileTitle()` simplified |
| `web/src/components/ProvidersPanel.tsx` | Flat editor — no conditional field rendering |
| `tests/test_llm_profiles.py` | Updated for new schema |
| `tests/test_local_cli_client.py` | Remove `cli_kind` assertions |

---

### Task 1: Update `profiles.py` — flat `LLMProfile` dataclass

**Files:**
- Modify: `annotation_pipeline_skill/llm/profiles.py`
- Test: `tests/test_llm_profiles.py`

- [ ] **Step 1: Write failing tests for new schema**

Replace the content of `tests/test_llm_profiles.py` with:

```python
from pathlib import Path
import pytest
from annotation_pipeline_skill.llm.profiles import (
    LLMProfile,
    LLMRegistry,
    ProfileValidationError,
    load_llm_registry,
)


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "llm_profiles.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_flat_registry_resolves_targets(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  ds:
    runtime: claude_cli
    model: deepseek-v4-flash
    base_url: https://api.deepseek.com/anthropic
    api_key_env: DEEPSEEK_API_KEY
    timeout_seconds: 120
  glm:
    runtime: claude_cli
    model: glm-5
    base_url: https://open.bigmodel.cn/api/anthropic
    api_key_env: GLM_API_KEY
targets:
  qc: ds
  coordinator: glm
limits:
  local_cli_global_concurrency: 4
""")
    registry = load_llm_registry(p)
    assert registry.resolve("qc").runtime == "claude_cli"
    assert registry.resolve("qc").model == "deepseek-v4-flash"
    assert registry.resolve("coordinator").runtime == "claude_cli"
    assert registry.local_cli_global_concurrency == 4


def test_codex_runtime_parsed(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  cx:
    runtime: codex_cli
    model: gpt-5.5
    base_url: https://api.openai.com
    api_key_env: OPENAI_API_KEY
targets:
  annotation: cx
""")
    registry = load_llm_registry(p)
    assert registry.resolve("annotation").runtime == "codex_cli"


def test_missing_runtime_raises(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  bad:
    model: gpt-4
    base_url: https://api.openai.com
    api_key_env: OPENAI_API_KEY
targets:
  annotation: bad
""")
    with pytest.raises(ProfileValidationError, match="runtime"):
        load_llm_registry(p)


def test_missing_base_url_raises(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  bad:
    runtime: claude_cli
    model: gpt-4
    api_key_env: OPENAI_API_KEY
targets:
  annotation: bad
""")
    with pytest.raises(ProfileValidationError, match="base_url"):
        load_llm_registry(p)


def test_missing_api_key_env_raises(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  bad:
    runtime: claude_cli
    model: gpt-4
    base_url: https://api.example.com
targets:
  annotation: bad
""")
    with pytest.raises(ProfileValidationError, match="api_key_env"):
        load_llm_registry(p)


def test_invalid_runtime_raises(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  bad:
    runtime: http_api
    model: gpt-4
    base_url: https://api.example.com
    api_key_env: KEY
targets:
  annotation: bad
""")
    with pytest.raises(ProfileValidationError, match="runtime"):
        load_llm_registry(p)


def test_optional_fields_default_to_none(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  minimal:
    runtime: claude_cli
    model: deepseek-v4-flash
    base_url: https://api.deepseek.com/anthropic
    api_key_env: DEEPSEEK_API_KEY
targets:
  annotation: minimal
""")
    profile = load_llm_registry(p).resolve("annotation")
    assert profile.reasoning_effort is None
    assert profile.permission_mode is None
    assert profile.timeout_seconds is None
    assert profile.max_retries is None
    assert profile.concurrency_limit is None
    assert profile.no_progress_timeout_seconds is None
    assert profile.disable_continuity is None


def test_api_key_env_accepts_list(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  glm:
    runtime: claude_cli
    model: glm-5
    base_url: https://open.bigmodel.cn/api/anthropic
    api_key_env:
      - GLM_API_KEY
      - BIGMODEL_MCP_API_KEY
targets:
  coordinator: glm
""")
    profile = load_llm_registry(p).resolve("coordinator")
    import os
    env = {"BIGMODEL_MCP_API_KEY": "secret"}
    assert profile.resolve_api_key(env) == "secret"


def test_resolve_api_key_returns_first_non_empty(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  p:
    runtime: claude_cli
    model: m
    base_url: https://example.com
    api_key_env:
      - MISSING_KEY
      - PRESENT_KEY
targets:
  annotation: p
""")
    profile = load_llm_registry(p).resolve("annotation")
    assert profile.resolve_api_key({"PRESENT_KEY": "val"}) == "val"
    assert profile.resolve_api_key({}) == ""
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_llm_profiles.py -v 2>&1 | tail -20
```

Expected: multiple failures — `runtime` not a known field, `ProviderName` still required, etc.

- [ ] **Step 3: Rewrite `profiles.py`**

Replace the entire file with:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import yaml


Runtime = Literal["claude_cli", "codex_cli"]

LLM_PROFILES_FILENAME = "llm_profiles.yaml"


def resolve_llm_profiles_path(
    *,
    workspace_root: Path | None = None,
    project_config_root: Path | None = None,
) -> Path | None:
    if workspace_root is not None:
        candidate = Path(workspace_root) / LLM_PROFILES_FILENAME
        if candidate.exists():
            return candidate
    if project_config_root is not None:
        candidate = Path(project_config_root) / LLM_PROFILES_FILENAME
        if candidate.exists():
            return candidate
    return None


class ProfileValidationError(ValueError):
    pass


@dataclass(frozen=True)
class LLMProfile:
    name: str
    runtime: Runtime
    model: str
    base_url: str
    api_key_env: str | list[str]
    api_key: str | None = None
    reasoning_effort: str | None = None
    permission_mode: str | None = None
    timeout_seconds: int | None = None
    max_retries: int | None = None
    concurrency_limit: int | None = None
    no_progress_timeout_seconds: int | None = None
    disable_continuity: bool | None = None

    def resolve_api_key(self, env: Mapping[str, str] = os.environ) -> str:
        if self.api_key:
            return self.api_key
        candidates = [self.api_key_env] if isinstance(self.api_key_env, str) else list(self.api_key_env)
        for name in candidates:
            value = env.get(name, "")
            if value:
                return value
        return ""


@dataclass(frozen=True)
class LLMRegistry:
    profiles: dict[str, LLMProfile]
    targets: dict[str, str]
    local_cli_global_concurrency: int | None = None

    def resolve(self, target: str) -> LLMProfile:
        profile_name = self.targets.get(target)
        if not profile_name:
            raise ProfileValidationError(f"LLM target is not configured: {target}")
        profile = self.profiles.get(profile_name)
        if profile is None:
            raise ProfileValidationError(f"LLM target {target} references missing profile {profile_name}")
        return profile


def load_llm_registry(path: Path | str) -> LLMRegistry:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProfileValidationError("LLM profile registry must be a mapping")
    raw_profiles = payload.get("profiles")
    raw_targets = payload.get("targets")
    if not isinstance(raw_profiles, dict):
        raise ProfileValidationError("LLM profile registry missing profiles")
    if not isinstance(raw_targets, dict):
        raise ProfileValidationError("LLM profile registry missing targets")
    profiles = {
        str(name): _parse_profile(str(name), values)
        for name, values in raw_profiles.items()
    }
    targets = {str(t): str(pn) for t, pn in raw_targets.items()}
    limits = payload.get("limits") or {}
    if not isinstance(limits, dict):
        raise ProfileValidationError("LLM profile limits must be a mapping")
    global_limit = _optional_positive_int(limits.get("local_cli_global_concurrency"), "limits.local_cli_global_concurrency")
    registry = LLMRegistry(profiles=profiles, targets=targets, local_cli_global_concurrency=global_limit)
    for target in targets:
        registry.resolve(target)
    return registry


def _parse_profile(name: str, raw: object) -> LLMProfile:
    if not isinstance(raw, dict):
        raise ProfileValidationError(f"LLM profile must be a mapping: {name}")
    runtime = raw.get("runtime")
    if runtime not in {"claude_cli", "codex_cli"}:
        raise ProfileValidationError(f"profile {name} runtime must be 'claude_cli' or 'codex_cli', got: {runtime!r}")
    model = _required_string(raw.get("model"), f"profile {name} model")
    base_url = _required_string(raw.get("base_url"), f"profile {name} base_url")
    api_key_env = _required_api_key_env(raw.get("api_key_env"), f"profile {name} api_key_env")
    return LLMProfile(
        name=name,
        runtime=runtime,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=_optional_string(raw.get("api_key"), f"profile {name} api_key"),
        reasoning_effort=_optional_string(raw.get("reasoning_effort"), f"profile {name} reasoning_effort"),
        permission_mode=_optional_string(raw.get("permission_mode"), f"profile {name} permission_mode"),
        timeout_seconds=_optional_positive_int(raw.get("timeout_seconds"), f"profile {name} timeout_seconds"),
        max_retries=_optional_non_negative_int(raw.get("max_retries"), f"profile {name} max_retries"),
        concurrency_limit=_optional_positive_int(raw.get("concurrency_limit"), f"profile {name} concurrency_limit"),
        no_progress_timeout_seconds=_optional_positive_int(raw.get("no_progress_timeout_seconds"), f"profile {name} no_progress_timeout_seconds"),
        disable_continuity=_optional_bool(raw.get("disable_continuity"), f"profile {name} disable_continuity"),
    )


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"invalid or missing {label}")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"invalid {label}")
    return value


def _required_api_key_env(value: object, label: str) -> str | list[str]:
    if value is None:
        raise ProfileValidationError(f"missing {label}")
    if isinstance(value, str):
        if not value.strip():
            raise ProfileValidationError(f"invalid {label}")
        return value
    if isinstance(value, list):
        if not value:
            raise ProfileValidationError(f"invalid {label}: must not be empty")
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ProfileValidationError(f"invalid {label}: each entry must be a non-empty string")
        return list(value)
    raise ProfileValidationError(f"invalid {label}: must be a string or list of strings")


def _optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ProfileValidationError(f"invalid {label}: must be true or false")


def _optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as exc:
        raise ProfileValidationError(f"invalid {label}") from exc
    if parsed <= 0:
        raise ProfileValidationError(f"invalid {label}")
    return parsed


def _optional_non_negative_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as exc:
        raise ProfileValidationError(f"invalid {label}") from exc
    if parsed < 0:
        raise ProfileValidationError(f"invalid {label}")
    return parsed
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_llm_profiles.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/llm/profiles.py tests/test_llm_profiles.py
git commit -m "refactor(profiles): flat LLMProfile — runtime replaces provider/cli_kind hierarchy"
```

---

### Task 2: Update `local_cli.py` — dispatch on `runtime`

**Files:**
- Modify: `annotation_pipeline_skill/llm/local_cli.py`
- Test: `tests/test_local_cli_client.py`

- [ ] **Step 1: Update dispatch and binary derivation**

In `LocalCLIClient.generate()` (around line 278), replace:

```python
    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        if self.profile.cli_kind == "codex":
            return await self._generate_codex(request)
        if self.profile.cli_kind == "claude":
            return await self._generate_claude(request)
        raise ValueError(f"unsupported local cli kind: {self.profile.cli_kind}")
```

with:

```python
    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        if self.profile.runtime == "codex_cli":
            return await self._generate_codex(request)
        if self.profile.runtime == "claude_cli":
            return await self._generate_claude(request)
        raise ValueError(f"unsupported runtime: {self.profile.runtime}")
```

In `_generate_codex()` (around line 288), replace:

```python
            binary=self.profile.cli_binary or "codex",
```

with:

```python
            binary="codex",
```

In `_generate_claude()` (around line 355), replace:

```python
        command = build_claude_command(
            binary=self.profile.cli_binary or "claude",
```

with:

```python
        command = build_claude_command(
            binary="claude",
```

- [ ] **Step 2: Update `test_local_cli_profile_import_contract` in `tests/test_local_cli_client.py`**

Replace:

```python
def test_local_cli_profile_import_contract():
    profile = LLMProfile(
        name="codex",
        provider="local_cli",
        model="gpt-5.4-mini",
        cli_kind="codex",
        cli_binary="codex",
    )

    assert profile.cli_kind == "codex"
```

with:

```python
def test_local_cli_profile_import_contract():
    profile = LLMProfile(
        name="codex",
        runtime="codex_cli",
        model="gpt-5.4-mini",
        base_url="https://api.openai.com",
        api_key_env="OPENAI_API_KEY",
    )

    assert profile.runtime == "codex_cli"
```

Also update the `LLMProfile(...)` construction in `test_local_codex_client_propagates_continuity_handle`:

```python
    client = LocalCLIClient(
        LLMProfile(
            name="local_codex",
            runtime="codex_cli",
            model="gpt-5.4-mini",
            base_url="https://api.openai.com",
            api_key_env="OPENAI_API_KEY",
        )
    )
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_local_cli_client.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add annotation_pipeline_skill/llm/local_cli.py tests/test_local_cli_client.py
git commit -m "refactor(local_cli): dispatch on profile.runtime, derive binary from runtime"
```

---

### Task 3: Update `cli.py` — client factory

**Files:**
- Modify: `annotation_pipeline_skill/interfaces/cli.py`

- [ ] **Step 1: Replace `_build_llm_client`**

Find `_build_llm_client` (around line 1687) and replace:

```python
def _build_llm_client(profile):
    if profile.provider == "openai_responses":
        return OpenAIResponsesClient(profile)
    if profile.provider == "openai_compatible":
        return OpenAICompatibleClient(profile)
    if profile.provider == "local_cli":
        return LocalCLIClient(profile)
    raise ProfileValidationError(f"unsupported provider: {profile.provider}")
```

with:

```python
def _build_llm_client(profile):
    return LocalCLIClient(profile)
```

- [ ] **Step 2: Remove unused imports at top of `cli.py`**

Remove these two import lines:

```python
from annotation_pipeline_skill.llm.openai_compatible import OpenAICompatibleClient
from annotation_pipeline_skill.llm.openai_responses import OpenAIResponsesClient
```

- [ ] **Step 3: Run the full test suite to check nothing breaks**

```bash
python -m pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: same pass count as before (minus the pre-existing distribution test failure).

- [ ] **Step 4: Commit**

```bash
git add annotation_pipeline_skill/interfaces/cli.py
git commit -m "refactor(cli): _build_llm_client dispatches via LocalCLIClient only"
```

---

### Task 4: Update `provider_config_service.py`

**Files:**
- Modify: `annotation_pipeline_skill/services/provider_config_service.py`

- [ ] **Step 1: Update `_profile_to_dict()`**

Find `_profile_to_dict` and replace its return dict:

```python
def _profile_to_dict(profile: LLMProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "runtime": profile.runtime,
        "model": profile.model,
        "base_url": profile.base_url,
        "api_key_env": profile.api_key_env,
        "api_key_set": bool(profile.api_key),
        "reasoning_effort": profile.reasoning_effort,
        "permission_mode": profile.permission_mode,
        "timeout_seconds": profile.timeout_seconds,
        "max_retries": profile.max_retries,
        "concurrency_limit": profile.concurrency_limit,
        "no_progress_timeout_seconds": profile.no_progress_timeout_seconds,
        "disable_continuity": profile.disable_continuity,
    }
```

- [ ] **Step 2: Update `_profile_diagnostics()`**

Replace the entire `_profile_diagnostics` function:

```python
def _profile_diagnostics(profile: LLMProfile, *, env: Mapping[str, str]) -> dict[str, Any]:
    binary = "claude" if profile.runtime == "claude_cli" else "codex"
    found = _cli_binary_found(binary)
    checks = [
        {
            "id": "cli_binary_found",
            "status": "ok" if found else "error",
            "message": f"{binary} is available" if found else f"{binary} was not found on PATH",
        },
        {
            "id": "base_url_configured",
            "status": "ok",
            "message": f"{profile.base_url} configured",
        },
    ]
    key_present = bool(profile.api_key) or bool(profile.resolve_api_key(env))
    env_label = (
        profile.api_key_env
        if isinstance(profile.api_key_env, str)
        else ", ".join(profile.api_key_env)
    )
    checks.append(
        {
            "id": "api_key_env_present",
            "status": "ok" if key_present else "error",
            "message": (
                f"{env_label} is set"
                if key_present
                else f"{env_label} is not set"
            ),
        }
    )
    status = "ok" if all(c["status"] == "ok" for c in checks) else "error"
    return {"status": status, "checks": checks}
```

- [ ] **Step 3: Update `_parse_profile_payload()` inside `save_provider_config`**

Find the section that reads incoming profile dict fields (around line 145) and remove references to `provider`, `provider_flavor`, `cli_kind`, `cli_binary`, `reasoning_capable`. The save path reads YAML fields by name so it will pass them through to `_parse_profile` in `profiles.py` — since `profiles.py` now ignores unknown fields implicitly via `raw.get(...)`, this is handled. Verify the list of fields kept in `_PRESERVE_API_KEY_FIELDS` at the top of the file still makes sense:

```python
_PRESERVE_API_KEY_FIELDS = {
    "provider",
    "provider_flavor",
    "cli_kind",
    "cli_binary",
}
```

Replace with:

```python
_PRESERVE_API_KEY_FIELDS: frozenset[str] = frozenset()
```

(API key merging logic uses `api_key` field directly — the set was only used to suppress old fields during merge; with flat schema it's no longer needed.)

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -q --tb=short -k "provider" 2>&1 | tail -20
```

Expected: provider-related tests pass.

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/services/provider_config_service.py
git commit -m "refactor(provider_service): flat schema — runtime-based diagnostics, drop old fields"
```

---

### Task 5: Write migration script

**Files:**
- Create: `scripts/migrate_llm_profiles.py`

- [ ] **Step 1: Write the script**

```python
"""Migrate llm_profiles.yaml from old layered schema to flat runtime schema.

Old fields removed: provider, provider_flavor, cli_kind, cli_binary, reasoning_capable
New field added: runtime (claude_cli | codex_cli)

Mapping:
  local_cli + cli_kind: claude   -> runtime: claude_cli
  local_cli + cli_kind: codex    -> runtime: codex_cli
  openai_compatible + any flavor -> runtime: claude_cli  (WARN: base_url may need updating)
  openai_responses               -> runtime: claude_cli  (WARN: base_url may need updating)

Usage:
  python scripts/migrate_llm_profiles.py --input path/to/llm_profiles.yaml
  python scripts/migrate_llm_profiles.py --input path/to/llm_profiles.yaml --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


_KNOWN_ANTHROPIC_SUFFIXES = ("/anthropic",)


def _needs_base_url_warning(base_url: str | None) -> bool:
    if not base_url:
        return True
    return not any(base_url.endswith(s) for s in _KNOWN_ANTHROPIC_SUFFIXES)


def migrate_profile(name: str, raw: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    out = dict(raw)

    provider = out.pop("provider", None)
    cli_kind = out.pop("cli_kind", None)
    out.pop("provider_flavor", None)
    out.pop("cli_binary", None)
    out.pop("reasoning_capable", None)

    if provider == "local_cli":
        runtime = "claude_cli" if cli_kind == "claude" else "codex_cli"
    elif provider in {"openai_compatible", "openai_responses"}:
        runtime = "claude_cli"
        base_url = out.get("base_url")
        if _needs_base_url_warning(base_url):
            warnings.append(
                f"  [{name}] base_url={base_url!r} may need updating to an Anthropic endpoint "
                f"(e.g. https://api.deepseek.com -> https://api.deepseek.com/anthropic)"
            )
    else:
        warnings.append(f"  [{name}] unknown provider={provider!r} — defaulting to claude_cli")
        runtime = "claude_cli"

    out["runtime"] = runtime
    # Ensure required fields present (warn if missing rather than crash)
    for field in ("model", "base_url", "api_key_env"):
        if not out.get(field):
            warnings.append(f"  [{name}] missing required field: {field}")

    # Reorder keys for readability
    ordered = {}
    for key in ("runtime", "model", "base_url", "api_key_env"):
        if key in out:
            ordered[key] = out.pop(key)
    ordered.update(out)
    return ordered, warnings


def migrate(input_path: Path) -> tuple[dict, list[str]]:
    payload = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    all_warnings: list[str] = []
    new_profiles: dict[str, dict] = {}
    for name, raw in (payload.get("profiles") or {}).items():
        migrated, warns = migrate_profile(str(name), dict(raw or {}))
        new_profiles[str(name)] = migrated
        all_warnings.extend(warns)
    payload["profiles"] = new_profiles
    return payload, all_warnings


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to llm_profiles.yaml")
    ap.add_argument("--output", help="Output path (default: print to stdout)")
    ap.add_argument("--apply", action="store_true", help="Write output to --input path")
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    payload, warnings = migrate(input_path)

    if warnings:
        print("WARNINGS — manual review required:", file=sys.stderr)
        for w in warnings:
            print(w, file=sys.stderr)

    out_yaml = yaml.dump(payload, allow_unicode=True, default_flow_style=False, sort_keys=False)

    if args.apply:
        output_path = Path(args.output) if args.output else input_path
        output_path.write_text(out_yaml, encoding="utf-8")
        print(f"Written to {output_path}", file=sys.stderr)
    elif args.output:
        Path(args.output).write_text(out_yaml, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(out_yaml)

    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: Smoke test on existing file (dry run)**

```bash
python scripts/migrate_llm_profiles.py \
  --input projects/v3_initial_deployment/.annotation-pipeline/llm_profiles.yaml \
  2>&1 | head -20
```

Expected: output shows new flat YAML, warnings for any profiles needing base_url review.

- [ ] **Step 3: Commit**

```bash
git add scripts/migrate_llm_profiles.py
git commit -m "feat(scripts): migrate_llm_profiles — old schema to flat runtime schema"
```

---

### Task 6: Migrate the live `llm_profiles.yaml`

**Files:**
- Modify: `projects/v3_initial_deployment/.annotation-pipeline/llm_profiles.yaml`

- [ ] **Step 1: Run migration with --apply**

```bash
python scripts/migrate_llm_profiles.py \
  --input projects/v3_initial_deployment/.annotation-pipeline/llm_profiles.yaml \
  --apply 2>&1
```

Review any warnings printed to stderr.

- [ ] **Step 2: Manually verify the output**

Open `projects/v3_initial_deployment/.annotation-pipeline/llm_profiles.yaml` and confirm:
- All profiles have `runtime:` field
- All `provider:`, `provider_flavor:`, `cli_kind:`, `cli_binary:` fields are gone
- `base_url` values for previously-openai_compatible profiles end with `/anthropic` (they should already since we updated them earlier)
- `targets` block unchanged

- [ ] **Step 3: Verify it loads**

```bash
python -c "
from annotation_pipeline_skill.llm.profiles import load_llm_registry
r = load_llm_registry('projects/v3_initial_deployment/.annotation-pipeline/llm_profiles.yaml')
for t in r.targets:
    p = r.resolve(t)
    print(f'{t}: runtime={p.runtime} model={p.model}')
"
```

Expected: all targets print cleanly with `runtime=claude_cli` or `runtime=codex_cli`.

- [ ] **Step 4: Commit**

```bash
git add projects/v3_initial_deployment/.annotation-pipeline/llm_profiles.yaml
git commit -m "chore: migrate llm_profiles.yaml to flat runtime schema"
```

---

### Task 7: Update frontend — `types.ts` and `providers.ts`

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/providers.ts`

- [ ] **Step 1: Update `types.ts`**

Find `ProviderName`, `ProviderFlavor`, `CliKind` type exports and the `ProviderProfileConfig` interface. Replace with:

```typescript
export type Runtime = "claude_cli" | "codex_cli";

export interface ProviderProfileConfig {
  name: string;
  runtime: Runtime;
  model: string;
  base_url: string;
  api_key_env: string | string[] | null;
  api_key?: string | null;       // write-only
  api_key_set?: boolean;         // read-only echo
  reasoning_effort: string | null;
  permission_mode: string | null;
  timeout_seconds: number | null;
  max_retries: number | null;
  concurrency_limit: number | null;
  no_progress_timeout_seconds: number | null;
  disable_continuity: boolean | null;
}
```

Remove the `ProviderName`, `ProviderFlavor`, `CliKind` type aliases entirely.

- [ ] **Step 2: Update `providers.ts`**

Replace the full file content with:

```typescript
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
```

- [ ] **Step 3: Check TypeScript compiles**

```bash
cd web && npx tsc --noEmit 2>&1 | head -30
```

Expected: errors only in `ProvidersPanel.tsx` (not yet updated).

- [ ] **Step 4: Commit**

```bash
git add web/src/types.ts web/src/providers.ts
git commit -m "refactor(frontend): flat ProviderProfileConfig — runtime replaces provider/cli_kind"
```

---

### Task 8: Update `ProvidersPanel.tsx`

**Files:**
- Modify: `web/src/components/ProvidersPanel.tsx`

- [ ] **Step 1: Update imports**

At the top of `ProvidersPanel.tsx`, change:

```typescript
import { createProviderProfile, profileStatusLabel, profileTitle, providerConfigPayload } from "../providers";
import type { ProviderConfigSnapshot, ProviderName, ProviderProfileConfig } from "../types";
```

to:

```typescript
import { createProviderProfile, profileStatusLabel, profileTitle, providerConfigPayload } from "../providers";
import type { ProviderConfigSnapshot, Runtime, ProviderProfileConfig } from "../types";
```

- [ ] **Step 2: Update state and add-profile handler**

Find `useState<ProviderName>("local_cli")` and replace with:

```typescript
const [newRuntime, setNewRuntime] = useState<Runtime>("claude_cli");
```

Find the add-profile `onClick` handler that calls `createProviderProfile(newProviderKind, ...)` and replace with:

```typescript
const profile = createProviderProfile(newRuntime, snapshot.profiles.length + 1);
```

- [ ] **Step 3: Update the add-profile `<select>` in the sidebar**

Find the `<select value={newProviderKind} ...>` element and replace:

```tsx
<select value={newRuntime} onChange={(event) => setNewRuntime(event.target.value as Runtime)}>
  <option value="claude_cli">claude_cli</option>
  <option value="codex_cli">codex_cli</option>
</select>
```

- [ ] **Step 4: Replace the entire profile editor form**

Find the `<div className="provider-form-grid">` block and replace it with this flat, condition-free form:

```tsx
<div className="provider-form-grid">
  <TextField label="Name" value={selected.name} onChange={(value) => updateSelected({ name: value })} />
  <SelectField
    label="Runtime"
    value={selected.runtime}
    options={["claude_cli", "codex_cli"]}
    onChange={(value) => updateSelected({ runtime: value as Runtime })}
  />
  <TextField label="Model" value={selected.model} onChange={(value) => updateSelected({ model: value })} />
  <TextField label="Base URL" value={selected.base_url ?? ""} onChange={(value) => updateSelected({ base_url: value })} />
  <TextField label="API Key Env" value={typeof selected.api_key_env === "string" ? selected.api_key_env : (selected.api_key_env ?? []).join(", ")} onChange={(value) => updateSelected({ api_key_env: value })} />
  <PasswordField
    label="API Key (inline)"
    value={selected.api_key ?? ""}
    placeholder={selected.api_key_set ? "set" : "not set"}
    hint={selected.api_key_set ? "Leave blank to keep current key" : undefined}
    onChange={(value) => updateSelected({ api_key: value })}
  />
  <TextField label="Reasoning Effort" value={selected.reasoning_effort ?? ""} onChange={(value) => updateSelected({ reasoning_effort: value || null })} />
  <TextField label="Permission Mode" value={selected.permission_mode ?? ""} onChange={(value) => updateSelected({ permission_mode: value || null })} />
  <NumberField label="Timeout Seconds" value={selected.timeout_seconds} onChange={(value) => updateSelected({ timeout_seconds: value })} />
  <NumberField label="Max Retries" value={selected.max_retries} onChange={(value) => updateSelected({ max_retries: value })} />
  <NumberField label="Concurrency Limit" value={selected.concurrency_limit} onChange={(value) => updateSelected({ concurrency_limit: value })} />
  <NumberField label="No Progress Timeout" value={selected.no_progress_timeout_seconds} onChange={(value) => updateSelected({ no_progress_timeout_seconds: value })} />
</div>
```

- [ ] **Step 5: Remove `providerDefaultsFor` function**

Delete the `providerDefaultsFor` function at the bottom of the file (it referenced old field names and is no longer called).

- [ ] **Step 6: Check TypeScript compiles cleanly**

```bash
cd web && npx tsc --noEmit 2>&1 | head -20
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add web/src/components/ProvidersPanel.tsx
git commit -m "refactor(ProvidersPanel): flat editor — remove conditional provider-type rendering"
```

---

### Task 9: Final integration check

- [ ] **Step 1: Run full backend test suite**

```bash
python -m pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: same pass count as before Task 1 (the pre-existing `test_dashboard_api_distribution` failure is unrelated).

- [ ] **Step 2: Verify profile loading end-to-end**

```bash
python -c "
from annotation_pipeline_skill.llm.profiles import load_llm_registry
r = load_llm_registry('projects/v3_initial_deployment/.annotation-pipeline/llm_profiles.yaml')
for t in ['annotation', 'qc', 'coordinator']:
    p = r.resolve(t)
    print(f'{t}: {p.runtime} {p.model} {p.base_url}')
"
```

Expected:
```
annotation: claude_cli MiniMax-M2.7 https://api.minimaxi.com/anthropic
qc: claude_cli deepseek-v4-flash https://api.deepseek.com/anthropic
coordinator: claude_cli glm-5 https://open.bigmodel.cn/api/anthropic
```

- [ ] **Step 3: Build frontend**

```bash
cd web && npm run build 2>&1 | tail -10
```

Expected: build succeeds with no errors.

- [ ] **Step 4: Final commit if any stragglers**

```bash
git status
```

If clean, done. If any modified files remain, stage and commit them:

```bash
git add -p
git commit -m "chore: cleanup after flat llm profiles migration"
```

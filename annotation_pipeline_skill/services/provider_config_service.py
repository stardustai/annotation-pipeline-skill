from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from annotation_pipeline_skill.llm.profiles import (
    LLM_PROFILES_FILENAME,
    LLMProfile,
    ProfileValidationError,
    load_llm_registry,
    resolve_llm_profiles_path,
)


PROFILE_FIELDS = (
    "runtime",
    "model",
    "api_key_env",
    "api_key",
    "base_url",
    "reasoning_effort",
    "permission_mode",
    "timeout_seconds",
    "max_retries",
    "concurrency_limit",
    "no_progress_timeout_seconds",
    "disable_continuity",
)

_PRESERVE_API_KEY_FIELDS: frozenset[str] = frozenset()


def build_provider_config_snapshot(
    config_root: Path,
    *,
    workspace_root: Path | None = None,
    env: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    profiles_path = resolve_llm_profiles_path(
        workspace_root=workspace_root,
        project_config_root=config_root,
    )
    if profiles_path is None:
        raise FileNotFoundError(
            f"no {LLM_PROFILES_FILENAME} found under workspace_root={workspace_root} "
            f"or project_config_root={config_root}"
        )
    registry = load_llm_registry(profiles_path)
    profiles = [_profile_to_dict(profile) for profile in registry.profiles.values()]
    diagnostics = {
        profile.name: _profile_diagnostics(profile, env=env)
        for profile in registry.profiles.values()
    }
    return {
        "config_valid": True,
        "profiles": profiles,
        "targets": registry.targets,
        "limits": {"local_cli_global_concurrency": registry.local_cli_global_concurrency},
        "diagnostics": diagnostics,
    }


def save_provider_config(
    config_root: Path,
    payload: Mapping[str, Any],
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    if workspace_root is None:
        raise ValueError("workspace_root is required to save provider configuration")
    data = _payload_to_yaml_data(payload)
    # Merge api_key secrets the UI couldn't see (it only ever receives
    # api_key_set: bool, never the raw value). Source of merge:
    #   1. <workspace>/llm_profiles.yaml if it exists, else
    #   2. <project_config>/llm_profiles.yaml (first-save migration path).
    # This avoids losing inline keys that lived in the project-local file
    # before workspace-global was introduced.
    existing_path = resolve_llm_profiles_path(
        workspace_root=workspace_root,
        project_config_root=config_root,
    )
    if existing_path is not None:
        try:
            existing_raw = yaml.safe_load(existing_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            existing_raw = {}
        _merge_preserved_api_keys(data, existing_raw)

    # Validate the fully-merged data (so inline-keyed profiles satisfy
    # _validate_profile's api_key OR api_key_env requirement).
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".yaml") as handle:
        temp_path = Path(handle.name)
        yaml.safe_dump(data, handle, sort_keys=False)
    try:
        load_llm_registry(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    workspace_path = Path(workspace_root)
    workspace_path.mkdir(parents=True, exist_ok=True)
    target_path = workspace_path / LLM_PROFILES_FILENAME
    target_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return build_provider_config_snapshot(config_root, workspace_root=workspace_root)


def _merge_preserved_api_keys(new_data: dict[str, Any], existing_raw: Mapping[str, Any]) -> None:
    """Restore previously-stored api_key values for profiles whose payload
    omitted (or sent empty for) api_key. The UI never receives the raw key,
    so an empty/absent value means 'keep the existing one'."""
    existing_profiles = existing_raw.get("profiles") if isinstance(existing_raw, Mapping) else None
    if not isinstance(existing_profiles, Mapping):
        return
    profiles = new_data.get("profiles")
    if not isinstance(profiles, dict):
        return
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        if profile.get("api_key"):
            continue  # user explicitly set a new value
        prior = existing_profiles.get(name)
        if isinstance(prior, Mapping) and prior.get("api_key"):
            profile["api_key"] = prior["api_key"]


def _payload_to_yaml_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_profiles = payload.get("profiles")
    raw_targets = payload.get("targets")
    raw_limits = payload.get("limits") or {}
    if not isinstance(raw_profiles, list):
        raise ProfileValidationError("provider config payload missing profiles list")
    if not isinstance(raw_targets, dict):
        raise ProfileValidationError("provider config payload missing targets mapping")
    if not isinstance(raw_limits, dict):
        raise ProfileValidationError("provider config payload limits must be a mapping")

    profiles: dict[str, dict[str, Any]] = {}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise ProfileValidationError("provider profile must be a mapping")
        name = raw_profile.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProfileValidationError("provider profile missing name")
        profiles[name] = {
            field: raw_profile[field]
            for field in PROFILE_FIELDS
            if raw_profile.get(field) not in (None, "")
        }

    return {
        "profiles": profiles,
        "targets": {str(target): str(profile_name) for target, profile_name in raw_targets.items()},
        "limits": {
            "local_cli_global_concurrency": raw_limits.get("local_cli_global_concurrency"),
        },
    }


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


def _cli_binary_found(binary: str | None) -> bool:
    if not binary:
        return False
    path = Path(binary)
    if path.is_absolute():
        return path.exists()
    return shutil.which(binary) is not None

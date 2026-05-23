from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import yaml


Runtime = Literal["codex_cli", "anthropic_sdk", "openai_sdk"]

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
    api_key_env: str | list[str] | None = None
    api_key: str | None = None
    reasoning_effort: str | None = None
    timeout_seconds: int | None = None
    max_retries: int | None = None
    concurrency_limit: int | None = None
    no_progress_timeout_seconds: int | None = None
    disable_continuity: bool | None = None
    structured_output: bool | None = None
    tools: list[dict] | None = None

    def resolve_api_key(self, env: Mapping[str, str] = os.environ) -> str:
        if self.api_key:
            return self.api_key
        if not self.api_key_env:
            return ""
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
    max_concurrent_tasks: int | None = None

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
    max_concurrent_tasks = _optional_positive_int(limits.get("max_concurrent_tasks"), "limits.max_concurrent_tasks")
    system_tools = _optional_tool_groups(payload.get("tools"), "tools") or []
    if system_tools:
        # SDK runtimes consume tool declarations via the in-process
        # tool_registry. Workspace-level `tools:` is unioned into every
        # per-profile `tools` list so SDK profiles inherit access to the
        # KB + validator groups without YAML duplication. codex_cli does
        # not have in-process tool dispatch yet, so this propagation
        # skips it.
        profiles = {
            name: dataclasses.replace(
                profile,
                tools=system_tools + (profile.tools or []),
            ) if profile.runtime in {"anthropic_sdk", "openai_sdk"} else profile
            for name, profile in profiles.items()
        }
    registry = LLMRegistry(
        profiles=profiles,
        targets=targets,
        max_concurrent_tasks=max_concurrent_tasks,
    )
    for target in targets:
        registry.resolve(target)
    return registry


def _parse_profile(name: str, raw: object) -> LLMProfile:
    if not isinstance(raw, dict):
        raise ProfileValidationError(f"LLM profile must be a mapping: {name}")
    runtime = raw.get("runtime")
    if runtime not in {"codex_cli", "anthropic_sdk", "openai_sdk"}:
        raise ProfileValidationError(
            f"profile {name} runtime must be 'codex_cli', 'anthropic_sdk', "
            f"or 'openai_sdk', got: {runtime!r}"
        )
    model = _required_string(raw.get("model"), f"profile {name} model")
    base_url = _required_string(raw.get("base_url"), f"profile {name} base_url")
    api_key_env = _optional_api_key_env(raw.get("api_key_env"), f"profile {name} api_key_env")
    return LLMProfile(
        name=name,
        runtime=runtime,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=_optional_string(raw.get("api_key"), f"profile {name} api_key"),
        reasoning_effort=_optional_string(raw.get("reasoning_effort"), f"profile {name} reasoning_effort"),
        timeout_seconds=_optional_positive_int(raw.get("timeout_seconds"), f"profile {name} timeout_seconds"),
        max_retries=_optional_non_negative_int(raw.get("max_retries"), f"profile {name} max_retries"),
        concurrency_limit=_optional_positive_int(raw.get("concurrency_limit"), f"profile {name} concurrency_limit"),
        no_progress_timeout_seconds=_optional_positive_int(raw.get("no_progress_timeout_seconds"), f"profile {name} no_progress_timeout_seconds"),
        disable_continuity=_optional_bool(raw.get("disable_continuity"), f"profile {name} disable_continuity"),
        structured_output=_optional_bool(raw.get("structured_output"), f"profile {name} structured_output"),
        tools=_optional_tool_groups(raw.get("tools"), f"profile {name} tools"),
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


def _optional_api_key_env(value: object, label: str) -> str | list[str] | None:
    if value is None:
        return None
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


def _optional_tool_groups(value: object, label: str) -> list[dict] | None:
    """Parse the YAML `tools:` field — a list of tool-group names.

    A group declaration is shaped ``{name: str}``. SDK runtimes look the
    name up in ``annotation_pipeline_skill.llm.tool_registry`` to find
    the schema + dispatch wiring.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ProfileValidationError(f"{label} must be a list")
    out: list[dict] = []
    for i, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ProfileValidationError(f"{label}[{i}] must be a mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProfileValidationError(f"{label}[{i}].name must be a non-empty string")
        out.append({"name": name})
    return out

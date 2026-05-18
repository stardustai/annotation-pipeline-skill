from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import yaml


ProviderName = Literal["openai_responses", "openai_compatible", "local_cli"]
ProviderFlavor = Literal["deepseek", "glm", "minimax"]
CliKind = Literal["codex", "claude"]


LLM_PROFILES_FILENAME = "llm_profiles.yaml"


def resolve_llm_profiles_path(
    *,
    workspace_root: Path | None = None,
    project_config_root: Path | None = None,
) -> Path | None:
    """Return the first existing llm_profiles.yaml from the resolution order.

    Order: workspace-global > project-local. Returns None if neither exists.
    Workspace-global: <workspace_root>/llm_profiles.yaml
    Project-local:    <project_config_root>/llm_profiles.yaml
    """
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
    provider: ProviderName
    model: str
    provider_flavor: ProviderFlavor | None = None
    api_key: str | None = None
    api_key_env: str | list[str] | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None
    cli_kind: CliKind | None = None
    cli_binary: str | None = None
    permission_mode: str | None = None
    timeout_seconds: int | None = None
    max_retries: int | None = None
    concurrency_limit: int | None = None
    no_progress_timeout_seconds: int | None = None
    reasoning_capable: bool | None = None
    # When True, the Responses API client will NOT forward
    # ``previous_response_id`` even if the task has a stored handle from a
    # prior turn. Required for stateless gateways (e.g. LiteLLM's
    # /v1/responses translation) that mint response ids per call but
    # don't persist them — passing the id back yields 404.
    disable_continuity: bool | None = None

    def resolve_api_key(self, env: Mapping[str, str] = os.environ) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env is None:
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
    targets = {str(target): str(profile_name) for target, profile_name in raw_targets.items()}
    limits = payload.get("limits") or {}
    if not isinstance(limits, dict):
        raise ProfileValidationError("LLM profile limits must be a mapping")
    global_limit = _optional_positive_int(limits.get("local_cli_global_concurrency"), "limits.local_cli_global_concurrency")
    registry = LLMRegistry(profiles=profiles, targets=targets, local_cli_global_concurrency=global_limit)
    for target in targets:
        registry.resolve(target)
    return registry


def reasoning_kwargs(model: str | None, effort: str | None, *, reasoning_capable: bool | None = None) -> dict:
    normalized_effort = str(effort or "").strip().lower()
    if normalized_effort in {"", "none", "default"}:
        return {}
    capable = reasoning_capable if reasoning_capable is not None else _is_reasoning_model(model)
    if not capable:
        return {}
    return {"reasoning": {"effort": normalized_effort}}


def _parse_profile(name: str, raw: object) -> LLMProfile:
    if not isinstance(raw, dict):
        raise ProfileValidationError(f"LLM profile must be a mapping: {name}")
    provider = raw.get("provider")
    if provider not in {"openai_responses", "openai_compatible", "local_cli"}:
        raise ProfileValidationError(f"LLM profile {name} has invalid provider")
    model = _required_string(raw.get("model"), f"profile {name} model")
    profile = LLMProfile(
        name=name,
        provider=provider,
        model=model,
        provider_flavor=_optional_provider_flavor(raw.get("provider_flavor"), f"profile {name} provider_flavor"),
        api_key=_optional_string(raw.get("api_key"), f"profile {name} api_key"),
        api_key_env=_optional_api_key_env(raw.get("api_key_env"), f"profile {name} api_key_env"),
        base_url=_optional_string(raw.get("base_url"), f"profile {name} base_url"),
        reasoning_effort=_optional_string(raw.get("reasoning_effort"), f"profile {name} reasoning_effort"),
        cli_kind=_optional_cli_kind(raw.get("cli_kind"), f"profile {name} cli_kind"),
        cli_binary=_optional_string(raw.get("cli_binary"), f"profile {name} cli_binary"),
        permission_mode=_optional_string(raw.get("permission_mode"), f"profile {name} permission_mode"),
        timeout_seconds=_optional_positive_int(raw.get("timeout_seconds"), f"profile {name} timeout_seconds"),
        max_retries=_optional_non_negative_int(raw.get("max_retries"), f"profile {name} max_retries"),
        concurrency_limit=_optional_positive_int(raw.get("concurrency_limit"), f"profile {name} concurrency_limit"),
        no_progress_timeout_seconds=_optional_positive_int(
            raw.get("no_progress_timeout_seconds"),
            f"profile {name} no_progress_timeout_seconds",
        ),
        reasoning_capable=_optional_bool(raw.get("reasoning_capable"), f"profile {name} reasoning_capable"),
        disable_continuity=_optional_bool(raw.get("disable_continuity"), f"profile {name} disable_continuity"),
    )
    _validate_profile(profile)
    return profile


def _validate_profile(profile: LLMProfile) -> None:
    if profile.provider == "openai_responses":
        if not profile.base_url:
            raise ProfileValidationError(f"LLM profile {profile.name} missing base_url")
        if not (profile.api_key or profile.api_key_env):
            raise ProfileValidationError(f"LLM profile {profile.name} missing api_key or api_key_env")
        return
    if profile.provider == "openai_compatible":
        if not profile.provider_flavor:
            raise ProfileValidationError(f"LLM profile {profile.name} missing provider_flavor")
        if not profile.base_url:
            raise ProfileValidationError(f"LLM profile {profile.name} missing base_url")
        if not (profile.api_key or profile.api_key_env):
            raise ProfileValidationError(f"LLM profile {profile.name} missing api_key or api_key_env")
        return
    if profile.provider == "local_cli":
        if not profile.cli_kind:
            raise ProfileValidationError(f"LLM profile {profile.name} missing cli_kind")
        if not profile.cli_binary:
            raise ProfileValidationError(f"LLM profile {profile.name} missing cli_binary")
        return


def _optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ProfileValidationError(f"invalid {label}: must be true or false")


def _is_reasoning_model(model: str | None) -> bool:
    normalized = str(model or "")
    return normalized.startswith("gpt-5") or normalized.startswith("o1") or normalized.startswith("o3")


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"invalid {label}")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"invalid {label}")
    return value


def _optional_cli_kind(value: object, label: str) -> CliKind | None:
    if value is None:
        return None
    if value not in {"codex", "claude"}:
        raise ProfileValidationError(f"invalid {label}")
    return value


def _optional_provider_flavor(value: object, label: str) -> ProviderFlavor | None:
    if value is None:
        return None
    if value not in {"deepseek", "glm", "minimax"}:
        raise ProfileValidationError(f"invalid {label}")
    return value


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

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


def test_missing_api_key_env_is_allowed(tmp_path: Path):
    p = _write_yaml(tmp_path, """
profiles:
  minimal:
    runtime: claude_cli
    model: claude-sonnet-4-5
    base_url: https://api.anthropic.com
targets:
  annotation: minimal
""")
    profile = load_llm_registry(p).resolve("annotation")
    assert profile.api_key_env is None
    assert profile.resolve_api_key({}) == ""


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

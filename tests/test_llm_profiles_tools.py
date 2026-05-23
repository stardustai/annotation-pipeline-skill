from pathlib import Path

import pytest

from annotation_pipeline_skill.llm.profiles import (
    load_llm_registry,
    ProfileValidationError,
)


def _write_yaml(tmp: Path, body: str) -> Path:
    path = tmp / "llm_profiles.yaml"
    path.write_text(body)
    return path


def test_profile_parses_tools(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  annotator_kb:
    runtime: anthropic_sdk
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    tools:
      - name: annotation-kb
      - name: annotation-validator
targets:
  annotator: annotator_kb
""")
    reg = load_llm_registry(path)
    profile = reg.profiles["annotator_kb"]
    assert profile.tools == [
        {"name": "annotation-kb"},
        {"name": "annotation-validator"},
    ]


def test_profile_tool_fields_optional(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  classic:
    runtime: anthropic_sdk
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
targets:
  annotator: classic
""")
    reg = load_llm_registry(path)
    p = reg.profiles["classic"]
    assert p.tools is None


def test_profile_rejects_malformed_tools(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  bad:
    runtime: anthropic_sdk
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    tools: "not a list"
targets:
  annotator: bad
""")
    with pytest.raises(ProfileValidationError):
        load_llm_registry(path)


def test_profile_rejects_tool_group_missing_required_keys(tmp_path):
    """Each tool group entry must carry a `name` key."""
    path = _write_yaml(tmp_path, """
profiles:
  bad:
    runtime: anthropic_sdk
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    tools:
      - description: "name missing"
targets:
  annotator: bad
""")
    with pytest.raises(ProfileValidationError):
        load_llm_registry(path)

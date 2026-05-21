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


def test_profile_parses_mcp_servers(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  annotator_claude_kb:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    mcp_servers:
      - name: annotation-kb
        command: python
        args: ["-m", "annotation_pipeline_skill.mcp.kb_server"]
    strict_mcp_config: true
    disallowed_tools: ["Bash", "Edit", "Write"]
targets:
  annotator: annotator_claude_kb
""")
    reg = load_llm_registry(path)
    profile = reg.profiles["annotator_claude_kb"]
    assert profile.mcp_servers == [
        {"name": "annotation-kb", "command": "python",
         "args": ["-m", "annotation_pipeline_skill.mcp.kb_server"]}
    ]
    assert profile.strict_mcp_config is True
    assert profile.disallowed_tools == ["Bash", "Edit", "Write"]


def test_profile_mcp_fields_optional(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  classic:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
targets:
  annotator: classic
""")
    reg = load_llm_registry(path)
    p = reg.profiles["classic"]
    assert p.mcp_servers is None
    assert p.strict_mcp_config is None
    assert p.disallowed_tools is None


def test_profile_rejects_malformed_mcp_servers(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  bad:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    mcp_servers: "not a list"
targets:
  annotator: bad
""")
    with pytest.raises(ProfileValidationError):
        load_llm_registry(path)


def test_profile_rejects_mcp_server_missing_required_keys(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  bad:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    mcp_servers:
      - command: python
        args: ["-m", "annotation_pipeline_skill.mcp.kb_server"]
targets:
  annotator: bad
""")
    with pytest.raises(ProfileValidationError):
        load_llm_registry(path)


def test_profile_rejects_disallowed_tools_non_list(tmp_path):
    path = _write_yaml(tmp_path, """
profiles:
  bad:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    disallowed_tools: "Bash,Edit"
targets:
  annotator: bad
""")
    with pytest.raises(ProfileValidationError):
        load_llm_registry(path)

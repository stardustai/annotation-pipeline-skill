from pathlib import Path

from annotation_pipeline_skill.llm.local_cli import build_claude_command


def test_build_claude_command_without_mcp_unchanged():
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
    )
    assert "--mcp-config" not in cmd
    assert "--strict-mcp-config" not in cmd
    assert "--disallowedTools" not in cmd
    # Existing flags must still be there.
    assert "--bare" in cmd
    assert "--no-session-persistence" in cmd


def test_build_claude_command_includes_mcp_config_path(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text("{}")
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        mcp_config_path=cfg_path, strict_mcp_config=True,
        disallowed_tools=["Bash", "Edit"],
    )
    assert "--mcp-config" in cmd
    assert str(cfg_path) in cmd
    assert "--strict-mcp-config" in cmd
    assert "--disallowedTools" in cmd
    assert "Bash,Edit" in cmd


def test_build_claude_command_disallowed_tools_only():
    """disallowed_tools without mcp_config is also valid (lock down tools)."""
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        disallowed_tools=["Bash"],
    )
    assert "--disallowedTools" in cmd
    assert "Bash" in cmd
    # strict-mcp-config should NOT be added when no mcp config.
    assert "--strict-mcp-config" not in cmd


def test_build_claude_command_strict_without_config_is_noop():
    """strict_mcp_config alone (without mcp_config_path) is meaningless and ignored."""
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        strict_mcp_config=True,
    )
    assert "--strict-mcp-config" not in cmd
    assert "--mcp-config" not in cmd

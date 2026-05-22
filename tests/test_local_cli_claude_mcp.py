from pathlib import Path

from annotation_pipeline_skill.llm.local_cli import build_claude_command


def test_build_claude_command_without_mcp_unchanged():
    # Explicit persist_session=False (continuity disabled, first turn) — this
    # matches the disable_continuity=True profile case, where no session file
    # should be written to disk.
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        persist_session=False,
    )
    assert "--mcp-config" not in cmd
    assert "--strict-mcp-config" not in cmd
    assert "--disallowedTools" not in cmd
    # Existing flags must still be there.
    assert "--bare" in cmd
    assert "--no-session-persistence" in cmd


def test_first_turn_with_continuity_omits_no_session_persistence():
    """Continuity-enabled profile, turn 1 (no session_id yet). MUST NOT pass
    --no-session-persistence — that would prevent claude from writing the
    session file, breaking the next turn's --resume and silently zeroing
    out the entire multi-turn continuity chain (vLLM prefix-cache regression)."""
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        session_id=None, persist_session=True,
    )
    assert "--no-session-persistence" not in cmd
    assert "--resume" not in cmd


def test_resume_path_omits_no_session_persistence():
    """When a session_id is provided we --resume it; --no-session-persistence
    must not also appear (mutually exclusive — claude would refuse)."""
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        session_id="sess-abc", persist_session=True,
    )
    assert "--resume" in cmd
    assert "sess-abc" in cmd
    assert "--no-session-persistence" not in cmd


def test_no_session_with_disable_continuity_still_persists_flag():
    """First turn under disable_continuity=True must keep
    --no-session-persistence — otherwise the disk fills with one-shot
    session files for profiles that explicitly opt out of resume."""
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        session_id=None, persist_session=False,
    )
    assert "--no-session-persistence" in cmd


def test_build_claude_command_includes_mcp_config_path(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text("{}")
    cmd = build_claude_command(
        binary="claude", model="sonnet", permission_mode=None,
        mcp_config_path=cfg_path, strict_mcp_config=True,
        disallowed_tools=["Bash", "Edit"],
    )
    # --mcp-config uses equals-form so the trailing `-` stdin marker can't be
    # swallowed by claude's multi-path mcp-config parser.
    assert f"--mcp-config={cfg_path}" in cmd
    assert "--strict-mcp-config" in cmd
    assert "--disallowedTools" in cmd
    assert "Bash,Edit" in cmd
    # The stdin marker must stay distinct (not collapsed into the mcp-config arg).
    assert cmd[-1] == "-"


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

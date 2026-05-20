import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from annotation_pipeline_skill.llm.client import LLMGenerateRequest
from annotation_pipeline_skill.llm.local_cli import (
    LocalCLIClient,
    build_claude_command,
    build_codex_command,
    codex_shell_environment,
    isolated_codex_home,
    parse_claude_stream_events,
    parse_codex_json_events,
)
from annotation_pipeline_skill.llm.profiles import LLMProfile


def test_codex_shell_environment_allows_only_safe_keys():
    env = codex_shell_environment(
        {
            "PATH": "/usr/bin",
            "HOME": "/tmp/home",
            "SHELL": "/bin/bash",
            "OPENAI_API_KEY": "do-not-pass",
            "SECRET_TOKEN": "do-not-pass",
            "CONNECTOR_API_KEY": "connector-key",
        }
    )

    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/tmp/home",
        "SHELL": "/bin/bash",
        "CONNECTOR_API_KEY": "connector-key",
    }


def test_build_codex_command_includes_json_resume_and_model():
    command, prompt_file = build_codex_command(
        binary="codex",
        prompt="Annotate this",
        developer_instructions="Return JSON",
        thread_id="thread-1",
        model="gpt-5.4-mini",
        reasoning_effort="none",
    )

    assert command[:3] == ["codex", "exec", "resume"]
    assert "--json" in command
    assert "--ignore-user-config" in command
    assert "--ephemeral" not in command
    assert command[command.index("--disable") + 1] == "apps"
    assert command[command.index("--disable", command.index("--disable") + 1) + 1] == "plugins"
    assert "--model" in command
    assert "gpt-5.4-mini" in command
    assert "--developer-message" not in command
    assert command[-2:] == ["thread-1", prompt_file.read_text(encoding="utf-8")]
    assert "Return JSON" in prompt_file.read_text(encoding="utf-8")
    assert "Annotate this" in prompt_file.read_text(encoding="utf-8")
    prompt_file.unlink()


def test_isolated_codex_home_strips_desktop_context_and_preserves_auth(tmp_path: Path):
    source_home = tmp_path / "source"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"token":"demo"}', encoding="utf-8")
    (source_home / "config.toml").write_text('model = "gpt-5.4"\n[plugins."gmail"]\nenabled = true\n', encoding="utf-8")

    with isolated_codex_home(
        {
            "CODEX_HOME": str(source_home),
            "CODEX_THREAD_ID": "desktop-thread",
            "OPENAI_API_KEY": "strip-me",
            "PATH": os.environ.get("PATH", ""),
        },
        model="gpt-5.4-mini",
        reasoning_effort="none",
        home_id=None,
        thread_id=None,
    ) as (isolated_env, isolated_home, _home_id):
        assert isolated_env["CODEX_HOME"] == str(isolated_home)
        assert isolated_env["HOME"] == str(isolated_home)
        assert "CODEX_THREAD_ID" not in isolated_env
        assert "OPENAI_API_KEY" not in isolated_env
        assert (isolated_home / "auth.json").exists()
        config = (isolated_home / "config.toml").read_text(encoding="utf-8")
        assert 'model = "gpt-5.4-mini"' in config
        assert 'model_reasoning_effort = "none"' in config
        assert "[plugins." not in config


def test_isolated_codex_home_does_not_copy_user_tui_state(tmp_path: Path):
    source_home = tmp_path / "source"
    source_home.mkdir()
    (source_home / "config.toml").write_text(
        'model = "gpt-5.4"\n[tui]\nmodel_availability_nux = "gpt-5.4-mini"\n',
        encoding="utf-8",
    )

    with isolated_codex_home(
        {"CODEX_HOME": str(source_home), "ANNOTATION_CODEX_HOME_ROOT": str(tmp_path / "runtime")},
        model="gpt-5.4-mini",
        reasoning_effort="none",
        home_id=None,
        thread_id=None,
    ) as (_isolated_env, isolated_home, _home_id):
        config = (isolated_home / "config.toml").read_text(encoding="utf-8")
        assert "[tui]" not in config
        assert "model_availability_nux" not in config


def test_isolated_codex_home_can_use_non_tmp_runtime_root(tmp_path: Path):
    source_home = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    source_home.mkdir()

    with isolated_codex_home(
        {"CODEX_HOME": str(source_home), "ANNOTATION_CODEX_HOME_ROOT": str(runtime_root)},
        model="gpt-5.4-mini",
        reasoning_effort="none",
        home_id=None,
        thread_id=None,
    ) as (isolated_env, isolated_home, home_id):
        assert isolated_home.is_relative_to(runtime_root)
        assert isolated_env["CODEX_HOME"] == str(isolated_home)
        assert home_id is not None and len(home_id) > 0


def test_parse_codex_json_events_extracts_thread_and_final_text():
    result = parse_codex_json_events(
        [
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"final answer"}}',
            '{"type":"turn.completed","usage":{"input_tokens":11,"output_tokens":2}}',
        ],
        provider="local_cli",
        model="gpt-5.4-mini",
    )

    assert result.continuity_handle == "thread-1"
    assert result.final_text == "final answer"
    assert result.usage == {"input_tokens": 11, "output_tokens": 2}


def test_build_claude_command_uses_stream_json_and_stdin_prompt():
    command = build_claude_command(
        binary="claude",
        model="claude-sonnet-4-5",
        permission_mode="dontAsk",
    )

    assert command[:2] == ["claude", "-p"]
    assert "--no-session-persistence" in command
    assert "--resume" not in command
    assert "--output-format" in command
    assert "stream-json" in command
    assert command[command.index("--model") + 1] == "claude-sonnet-4-5"
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert command[-1] == "-"


def test_build_claude_command_uses_resume_when_session_id_provided():
    command = build_claude_command(
        binary="claude",
        model="claude-sonnet-4-5",
        permission_mode=None,
        session_id="abcd-1234",
    )

    assert "--no-session-persistence" not in command
    assert command[command.index("--resume") + 1] == "abcd-1234"
    assert command[-1] == "-"


def test_parse_claude_stream_events_extracts_text_and_usage():
    result = parse_claude_stream_events(
        [
            '{"type":"system","session_id":"session-1"}',
            '{"type":"assistant","message":{"content":[{"type":"text","text":"final answer"}]}}',
            '{"type":"result","usage":{"input_tokens":5,"output_tokens":2}}',
        ],
        provider="local_cli",
        model="claude-sonnet-4-5",
    )

    assert result.continuity_handle == "session-1"
    assert result.final_text == "final answer"
    assert result.usage == {"input_tokens": 5, "output_tokens": 2}
    assert result.raw_response[1]["type"] == "assistant"


def test_local_cli_profile_import_contract():
    profile = LLMProfile(
        name="codex",
        runtime="codex_cli",
        model="gpt-5.4-mini",
        base_url="https://api.openai.com",
        api_key_env="OPENAI_API_KEY",
    )

    assert profile.runtime == "codex_cli"


@pytest.mark.asyncio
async def test_local_codex_client_propagates_continuity_handle(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt", encoding="utf-8")

    def fake_build_codex_command(**kwargs):
        captured["thread_id"] = kwargs["thread_id"]
        return ["codex", "exec", "--json", "prompt"], prompt_file

    @contextmanager
    def fake_isolated_codex_home(env, *, model, reasoning_effort, home_id, thread_id, provider_api_key=None, provider_base_url=None):
        captured["home_id"] = home_id
        captured["iso_thread_id"] = thread_id
        resolved = home_id or "fake-home-id"
        yield {"PATH": env.get("PATH", "")}, tmp_path, resolved

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return (
                b'{"type":"thread.started","thread_id":"thread-new"}\n'
                b'{"type":"item.completed","item":{"type":"agent_message","text":"{}"}}\n',
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    import annotation_pipeline_skill.llm.local_cli as local_cli

    monkeypatch.setattr(local_cli, "build_codex_command", fake_build_codex_command)
    monkeypatch.setattr(local_cli, "isolated_codex_home", fake_isolated_codex_home)
    monkeypatch.setattr(local_cli.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = LocalCLIClient(
        LLMProfile(
            name="local_codex",
            runtime="codex_cli",
            model="gpt-5.4-mini",
            base_url="https://api.openai.com",
            api_key_env="OPENAI_API_KEY",
        )
    )

    # First call: no prior handle → thread_id=None passed down, result gets home::thread
    result = await client.generate(LLMGenerateRequest(prompt="prompt"))
    assert result.continuity_handle == "fake-home-id::thread-new"
    assert captured["thread_id"] is None
    assert captured["home_id"] is None

    # Resume call: handle parsed into home_id + thread_id
    result2 = await client.generate(
        LLMGenerateRequest(prompt="prompt", continuity_handle="my-home::thread-old")
    )
    assert captured["home_id"] == "my-home"
    assert captured["thread_id"] == "thread-old"
    assert captured["iso_thread_id"] == "thread-old"
    assert result2.continuity_handle == "my-home::thread-new"

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
    isolated_claude_home,
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
    # persist_session=False mirrors the disable_continuity=True profile case;
    # without that, the new default omits --no-session-persistence (so
    # continuity-enabled profiles' turn-1 calls actually write the session
    # file for turn-2 to resume).
    command = build_claude_command(
        binary="claude",
        model="claude-sonnet-4-5",
        permission_mode="dontAsk",
        persist_session=False,
    )

    assert command[:3] == ["claude", "--bare", "-p"]
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

    assert "--bare" in command
    assert "--no-session-persistence" not in command
    assert command[command.index("--resume") + 1] == "abcd-1234"
    assert command[-1] == "-"


def test_isolated_claude_home_creates_home_and_forces_HOME(tmp_path: Path):
    runtime_root = tmp_path / "runtime"

    with isolated_claude_home(
        {
            "ANNOTATION_CLAUDE_HOME_ROOT": str(runtime_root),
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "should-be-stripped",
            "ANTHROPIC_API_KEY": "should-be-overridden",
        },
        home_id=None,
        provider_api_key="sk-provider",
        provider_base_url="https://api.example.com/anthropic",
    ) as (env, isolated_home, home_id):
        assert isolated_home.is_dir()
        assert isolated_home.is_relative_to(runtime_root)
        assert env["HOME"] == str(isolated_home)
        assert env["ANTHROPIC_API_KEY"] == "sk-provider"
        assert env["ANTHROPIC_BASE_URL"] == "https://api.example.com/anthropic"
        # Unrelated keys are stripped — only the safe-key whitelist + injected auth
        assert "OPENAI_API_KEY" not in env
        # Crucially, no credential file or .claude subtree is seeded — auth lives
        # only in env vars, so the subprocess has nothing to read or clobber.
        assert not (isolated_home / ".credentials.json").exists()
        assert not (isolated_home / ".claude").exists()
        assert home_id and len(home_id) > 0


def test_isolated_claude_home_reuses_home_id(tmp_path: Path):
    runtime_root = tmp_path / "runtime"

    with isolated_claude_home(
        {"ANNOTATION_CLAUDE_HOME_ROOT": str(runtime_root)},
        home_id="stable-id",
    ) as (_env, isolated_home, home_id):
        assert home_id == "stable-id"
        assert isolated_home == runtime_root / "stable-id"
        (isolated_home / "marker").write_text("hi", encoding="utf-8")

    with isolated_claude_home(
        {"ANNOTATION_CLAUDE_HOME_ROOT": str(runtime_root)},
        home_id="stable-id",
    ) as (_env, isolated_home, _home_id):
        assert (isolated_home / "marker").read_text(encoding="utf-8") == "hi"


def test_isolated_claude_home_writes_task_id_into_userID(tmp_path: Path):
    """user_id_override (= the pipeline task_id) must be written to
    <HOME>/.claude.json:userID. claude packs that field into
    body.metadata.user_id as the device_id component, which is the
    sticky-routing key gateways like LiteLLM hash on. Without the override,
    every isolated home gets a stable random-looking userID that doesn't
    correlate with task identity → cross-task requests collide on a small
    number of routing buckets and prefix-cache locality collapses."""
    import json
    runtime_root = tmp_path / "runtime"

    with isolated_claude_home(
        {"ANNOTATION_CLAUDE_HOME_ROOT": str(runtime_root)},
        home_id="task-home-1",
        user_id_override="v3_initial_deployment-000342",
    ) as (_env, isolated_home, _home_id):
        data = json.loads((isolated_home / ".claude.json").read_text(encoding="utf-8"))
        assert data["userID"] == "v3_initial_deployment-000342"


def test_isolated_claude_home_preserves_existing_claude_json_fields(tmp_path: Path):
    """Writing the userID override must merge — claude caches feature flags,
    migration markers, etc. in .claude.json across runs. Wiping those
    fields would force claude to re-initialise on every call (extra startup
    cost, lost growth-book state)."""
    import json
    runtime_root = tmp_path / "runtime"
    home_dir = runtime_root / "task-home-2"
    home_dir.mkdir(parents=True)
    (home_dir / ".claude.json").write_text(
        json.dumps({
            "userID": "old-device-hash",
            "cachedGrowthBookFeatures": {"flag_a": True},
            "firstStartTime": "2026-05-21T12:00:00Z",
        }),
        encoding="utf-8",
    )

    with isolated_claude_home(
        {"ANNOTATION_CLAUDE_HOME_ROOT": str(runtime_root)},
        home_id="task-home-2",
        user_id_override="v3_initial_deployment-000999",
    ) as (_env, isolated_home, _home_id):
        data = json.loads((isolated_home / ".claude.json").read_text(encoding="utf-8"))
        assert data["userID"] == "v3_initial_deployment-000999"
        # Non-userID fields must survive the rewrite.
        assert data["cachedGrowthBookFeatures"] == {"flag_a": True}
        assert data["firstStartTime"] == "2026-05-21T12:00:00Z"


def test_isolated_claude_home_skip_user_id_when_none(tmp_path: Path):
    """No override → don't touch .claude.json. Profiles without task_id wired
    through (anything other than the annotation pipeline) keep the old
    behaviour of letting claude generate its stable per-home userID."""
    runtime_root = tmp_path / "runtime"

    with isolated_claude_home(
        {"ANNOTATION_CLAUDE_HOME_ROOT": str(runtime_root)},
        home_id="task-home-3",
        user_id_override=None,
    ) as (_env, isolated_home, _home_id):
        # No file touched on entry; claude will write it on first run.
        assert not (isolated_home / ".claude.json").exists()


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
        captured["preexec_fn"] = kwargs.get("preexec_fn")
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
    assert captured["preexec_fn"] is local_cli._die_with_parent

    # Resume call: handle parsed into home_id + thread_id
    result2 = await client.generate(
        LLMGenerateRequest(prompt="prompt", continuity_handle="my-home::thread-old")
    )
    assert captured["home_id"] == "my-home"
    assert captured["thread_id"] == "thread-old"
    assert captured["iso_thread_id"] == "thread-old"
    assert result2.continuity_handle == "my-home::thread-new"


@pytest.mark.asyncio
async def test_local_claude_client_propagates_continuity_handle(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_build_claude_command(**kwargs):
        captured["session_id"] = kwargs["session_id"]
        return ["claude", "--bare", "-p", "-"]

    @contextmanager
    def fake_isolated_claude_home(env, *, home_id, provider_api_key=None, provider_base_url=None, user_id_override=None):
        captured["home_id"] = home_id
        captured["provider_api_key"] = provider_api_key
        captured["provider_base_url"] = provider_base_url
        captured["user_id_override"] = user_id_override
        resolved = home_id or "fake-home-id"
        yield {"PATH": env.get("PATH", "")}, tmp_path, resolved

    class FakeProcess:
        returncode = 0

        async def communicate(self, stdin=None):
            return (
                b'{"type":"system","session_id":"session-new"}\n'
                b'{"type":"assistant","message":{"content":[{"type":"text","text":"{}"}]}}\n',
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["preexec_fn"] = kwargs.get("preexec_fn")
        return FakeProcess()

    import annotation_pipeline_skill.llm.local_cli as local_cli

    monkeypatch.setattr(local_cli, "build_claude_command", fake_build_claude_command)
    monkeypatch.setattr(local_cli, "isolated_claude_home", fake_isolated_claude_home)
    monkeypatch.setattr(local_cli.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = LocalCLIClient(
        LLMProfile(
            name="claude_provider",
            runtime="claude_cli",
            model="claude-sonnet-4-5",
            base_url="https://api.example.com/anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            api_key="sk-test",
        )
    )

    # First call: no prior handle → session_id=None, result gets home::session
    result = await client.generate(LLMGenerateRequest(prompt="prompt"))
    assert result.continuity_handle == "fake-home-id::session-new"
    assert captured["session_id"] is None
    assert captured["home_id"] is None
    assert captured["provider_api_key"] == "sk-test"
    assert captured["provider_base_url"] == "https://api.example.com/anthropic"
    # PR_SET_PDEATHSIG wired up so a SIGKILLed runtime doesn't leave orphan
    # claude children writing to ~/.claude/.credentials.json (the orphan
    # leak that caused this whole bug class in the first place).
    assert captured["preexec_fn"] is local_cli._die_with_parent

    # Resume call: handle parsed into home_id + session_id
    result2 = await client.generate(
        LLMGenerateRequest(prompt="prompt", continuity_handle="my-home::session-old")
    )
    assert captured["home_id"] == "my-home"
    assert captured["session_id"] == "session-old"
    assert result2.continuity_handle == "my-home::session-new"

    # Legacy bare session_id → cold start (cannot reuse orphaned session file)
    captured.clear()
    captured["session_id"] = "sentinel"
    captured["home_id"] = "sentinel"
    await client.generate(
        LLMGenerateRequest(prompt="prompt", continuity_handle="legacy-bare-id")
    )
    assert captured["home_id"] is None
    assert captured["session_id"] is None


@pytest.mark.asyncio
async def test_generate_claude_does_not_touch_real_credentials(tmp_path: Path, monkeypatch):
    """Regression test for the bug where _generate_claude inherited HOME from
    the user's shell and let `claude` rewrite ~/.claude/.credentials.json
    while reusing third-party provider tokens. The fix routes the subprocess
    through isolated_claude_home with a fresh HOME; this test pins that
    invariant by asserting the subprocess env never points at the real home."""
    real_home = tmp_path / "real_home"
    (real_home / ".claude").mkdir(parents=True)
    creds = real_home / ".claude" / ".credentials.json"
    creds.write_text('{"do":"not-touch"}', encoding="utf-8")
    original_mtime_ns = creds.stat().st_mtime_ns

    class FakeProcess:
        returncode = 0

        async def communicate(self, stdin=None):
            return (b'{"type":"system","session_id":"s"}\n', b"")

    captured_env: dict[str, str] = {}

    async def fake_create_subprocess_exec(*args, env, **kwargs):
        captured_env.update(env)
        return FakeProcess()

    import annotation_pipeline_skill.llm.local_cli as local_cli

    monkeypatch.setattr(local_cli.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("ANNOTATION_CLAUDE_HOME_ROOT", str(tmp_path / "iso_root"))

    client = LocalCLIClient(
        LLMProfile(
            name="p",
            runtime="claude_cli",
            model="m",
            base_url="https://api.example.com/anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            api_key="sk-test",
        )
    )
    await client.generate(LLMGenerateRequest(prompt="hi"))

    # The subprocess MUST NOT inherit the user's real HOME — that's what
    # let the previous code corrupt ~/.claude/.credentials.json.
    assert captured_env["HOME"] != str(real_home)
    assert Path(captured_env["HOME"]).is_relative_to(tmp_path / "iso_root")
    # Auth flows through env vars only, never via a credentials file in the
    # isolated home (so the subprocess has no file to read OR write back to).
    assert captured_env["ANTHROPIC_API_KEY"] == "sk-test"
    assert captured_env["ANTHROPIC_BASE_URL"] == "https://api.example.com/anthropic"
    assert not (Path(captured_env["HOME"]) / ".credentials.json").exists()
    assert not (Path(captured_env["HOME"]) / ".claude").exists()
    # And the real credentials file is untouched.
    assert creds.stat().st_mtime_ns == original_mtime_ns
    assert creds.read_text(encoding="utf-8") == '{"do":"not-touch"}'

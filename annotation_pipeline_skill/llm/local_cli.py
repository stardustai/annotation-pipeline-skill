from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from annotation_pipeline_skill.llm.client import LLMGenerateRequest, LLMGenerateResult
from annotation_pipeline_skill.llm.profiles import LLMProfile

_SAFE_ENV_KEYS = {
    "PATH",
    "HOME",
    "SHELL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "CODEX_HOME",
    "ANNOTATION_CODEX_HOME_ROOT",
    "ANNOTATION_CLAUDE_HOME_ROOT",
}


class LocalCLIExecutionError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _die_with_parent() -> None:
    """preexec_fn that asks the kernel to SIGKILL this child when its parent
    dies (Linux PR_SET_PDEATHSIG). Without this, a SIGKILLed runtime leaves
    its claude/codex subprocesses as PPID=1 orphans that keep talking to the
    provider — that's how the OAuth credentials got corrupted in the first
    place. Silent no-op on non-Linux."""
    try:
        import ctypes
        import signal as _signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # PR_SET_PDEATHSIG = 1; arg = signal to deliver on parent death
        libc.prctl(1, _signal.SIGKILL, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass


def codex_shell_environment(env: Mapping[str, str] = os.environ) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in env.items():
        if key in _SAFE_ENV_KEYS or key.endswith("_CONNECTOR_API_KEY") or key == "CONNECTOR_API_KEY":
            safe[key] = value
    return safe


def _parse_codex_handle(handle: str | None) -> tuple[str | None, str | None]:
    """Split 'home_id::thread_id' into components. Returns (None, None) if no handle."""
    if not handle:
        return None, None
    if "::" in handle:
        home_id, thread_id = handle.split("::", 1)
        return home_id, thread_id
    return None, handle  # legacy bare thread_id


def _parse_claude_handle(handle: str | None) -> tuple[str | None, str | None]:
    """Split 'home_id::session_id' into components. Legacy bare session_id is
    treated as no handle: the session file lived in the user's real ~/.claude,
    which the isolated runtime no longer touches, so resume would orphan-fail."""
    if not handle or "::" not in handle:
        return None, None
    home_id, session_id = handle.split("::", 1)
    return home_id, session_id


def build_codex_command(
    *,
    binary: str,
    prompt: str,
    developer_instructions: str | None,
    thread_id: str | None,
    model: str,
    reasoning_effort: str | None,
) -> tuple[list[str], Path]:
    import tempfile
    prompt_file = Path(tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False).name)
    full_prompt = prompt
    if developer_instructions:
        full_prompt = f"{developer_instructions}\n\n{prompt}"
    prompt_file.write_text(full_prompt, encoding="utf-8")

    command = [binary, "exec"]
    if thread_id:
        command.append("resume")
    command.extend(
        [
            "--ignore-user-config",
            "--ignore-rules",
            "--disable",
            "apps",
            "--disable",
            "plugins",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            "--model",
            model,
            "--config",
            "enabled_tools=[]",
        ]
    )
    if reasoning_effort:
        command.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
    if thread_id:
        command.append(thread_id)
    command.append(prompt_file.read_text(encoding="utf-8"))
    return command, prompt_file


def build_claude_command(
    *,
    binary: str,
    model: str,
    permission_mode: str | None,
    session_id: str | None = None,
    mcp_config_path: Path | None = None,
    strict_mcp_config: bool = False,
    disallowed_tools: list[str] | None = None,
    persist_session: bool = True,
) -> list[str]:
    # --bare: never read OAuth / keychain / ~/.claude credentials. Auth is
    # strictly ANTHROPIC_API_KEY (no token writeback can clobber real creds).
    # Also skips hooks, auto-memory, CLAUDE.md auto-discovery, background
    # prefetches — exactly the surface we don't want in a worker.
    command = [binary, "--bare", "-p"]
    if session_id:
        command.extend(["--resume", session_id])
    elif not persist_session:
        # Only suppress session persistence when the profile has explicitly
        # disabled continuity. Previously we ALWAYS added this flag whenever
        # session_id was None — which meant turn 1 of a continuity-enabled
        # task generated a session_id but never wrote the session file, so
        # turn 2's `--resume <id>` silently started a fresh session instead
        # of replaying history. Net effect: claimed continuity was broken
        # end-to-end and vLLM prefix-cache hit rate stayed at 0%.
        # Documented behaviour (claude --help): "Disable session persistence
        # - sessions will not be saved to disk and cannot be resumed."
        command.append("--no-session-persistence")
    command.extend([
        "--verbose",
        "--output-format",
        "stream-json",
        "--model",
        model,
    ])
    # Default to bypassPermissions when the profile doesn't specify a mode:
    # --bare -p auto-approves built-in tools (Bash/Read/etc.) but MCP tools
    # still go through the default "prompt for each call" gate, which blocks
    # every tool_use the agent makes with "Claude requested permissions ...
    # but you haven't granted it yet." Workers can't answer prompts. The
    # historical workers all ran under bypassPermissions implicitly; setting
    # it explicitly when None preserves that behaviour AND unblocks the new
    # MCP-validator tool path. Profiles that want stricter scoping can set
    # permission_mode explicitly.
    command.extend(["--permission-mode", permission_mode or "bypassPermissions"])
    if mcp_config_path is not None:
        # `--mcp-config=PATH` (single token, equals syntax) — the space form
        # `--mcp-config PATH` causes claude to greedily consume the trailing
        # `-` (our stdin-prompt marker, appended below) as a second config
        # path, producing "MCP config file not found: <cwd>/-". The equals
        # form binds the path to the flag unambiguously.
        command.append(f"--mcp-config={mcp_config_path}")
        if strict_mcp_config:
            command.append("--strict-mcp-config")
    if disallowed_tools:
        command.extend(["--disallowedTools", ",".join(disallowed_tools)])
    command.append("-")
    return command


@contextmanager
def isolated_codex_home(
    env: Mapping[str, str],
    *,
    model: str,
    reasoning_effort: str | None,
    home_id: str | None,
    thread_id: str | None,
    provider_api_key: str | None = None,
    provider_base_url: str | None = None,
) -> Iterator[tuple[dict[str, str], Path, str]]:
    source_home = Path(env.get("CODEX_HOME") or Path(env.get("HOME", "~")).expanduser() / ".codex")
    runtime_root = Path(env.get("ANNOTATION_CODEX_HOME_ROOT") or Path.cwd() / ".annotation-pipeline-codex-homes")
    runtime_root.mkdir(parents=True, exist_ok=True)

    if home_id:
        isolated_home = runtime_root / home_id
        isolated_home.mkdir(parents=True, exist_ok=True)
    else:
        home_id = str(uuid.uuid4())
        isolated_home = runtime_root / home_id
        isolated_home.mkdir(parents=True, exist_ok=True)
        if provider_api_key:
            # Provider-specific home: write minimal auth with provider key only
            (isolated_home / "auth.json").write_text(
                json.dumps({"OPENAI_API_KEY": provider_api_key}),
                encoding="utf-8",
            )
        else:
            # Default path: copy user's codex auth + OAuth state.
            # Files to preserve for OAuth (ChatGPT-mode) authentication:
            #   - auth.json: {auth_mode, OPENAI_API_KEY?, tokens{id,access,refresh}, last_refresh}
            #   - config.toml: gets overwritten below by _write_isolated_codex_config, but
            #     copy first so any non-overwritten fields (mcp configs, etc.) survive
            #   - credentials.json: legacy key-mode credentials (rare)
            #   - .credentials.json: hidden file some codex versions use (current ~/.codex
            #     observed at 1071 bytes)
            #   - installation_id: OAuth client identifier; some refresh flows fail
            #     silently without it
            # NOT copied (would leak user state / are transient):
            #   - history.jsonl, log/, logs_2.sqlite*, cache/, .tmp/ — user history & runtime state
            #   - app-server-control/, app-server-daemon/ — IPC socket dirs
            #   - memories/ — user-curated memory store
            for filename in (
                "auth.json",
                "config.toml",
                "credentials.json",
                ".credentials.json",
                "installation_id",
            ):
                source_file = source_home / filename
                if source_file.exists():
                    shutil.copy2(source_file, isolated_home / filename)

    _write_isolated_codex_config(
        isolated_home / "config.toml",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    isolated_env = codex_shell_environment(env)
    isolated_env["CODEX_HOME"] = str(isolated_home)
    isolated_env["HOME"] = str(isolated_home)
    isolated_env.pop("CODEX_THREAD_ID", None)
    if thread_id:
        isolated_env["CODEX_RESUME_THREAD_ID"] = thread_id
    if provider_api_key:
        isolated_env["OPENAI_API_KEY"] = provider_api_key
    if provider_base_url:
        isolated_env["OPENAI_BASE_URL"] = provider_base_url

    yield isolated_env, isolated_home, home_id


@contextmanager
def isolated_claude_home(
    env: Mapping[str, str],
    *,
    home_id: str | None,
    provider_api_key: str | None = None,
    provider_base_url: str | None = None,
    user_id_override: str | None = None,
) -> Iterator[tuple[dict[str, str], Path, str]]:
    """Per-task isolated HOME for `claude --bare`. Never copies real ~/.claude;
    auth comes from ANTHROPIC_API_KEY in the env. Defense-in-depth on top of
    --bare so any path that bypasses --bare (skill plugin dirs, settings
    lookup, history files) still cannot reach the user's real .claude tree.

    OAuth fallback: when no ``provider_api_key`` is supplied AND the user has
    a Claude Code Pro / Max subscription (``~/.claude/.credentials.json``
    contains a ``claudeAiOauth.accessToken``), reuse that OAuth access token
    as ``ANTHROPIC_API_KEY``. This mirrors what bare-shell ``claude`` does
    when launched directly: the binary looks up the OAuth credentials and
    sends them as the bearer token. With ``--bare`` the binary itself won't
    read the credentials file (verified via strace — it opens it but still
    reports ``Not logged in``), so we inject the token into env explicitly.
    Lets users run Anthropic claude_sonnet / claude_haiku targets without
    needing a separate sk-ant-... API key.

    ``user_id_override`` (typically the pipeline ``task_id``) is written into
    ``<HOME>/.claude.json``'s ``userID`` field BEFORE the claude subprocess
    starts. The claude binary reads that field and packs it into
    ``body.metadata.user_id`` as the ``device_id`` component (verified by
    capturing live POST /v1/messages traffic). That body.user/device_id is
    what gateways like LiteLLM use for sticky routing — without the override
    every isolated home's userID is a stable hash that doesn't change per
    task, so cross-task requests hash to a small number of buckets and
    prefix-cache locality collapses. Passing task_id here makes the
    device_id per-task, so a sticky-on-body.user routing config naturally
    pins all turns of one task to one upstream instance.
    """
    runtime_root = Path(env.get("ANNOTATION_CLAUDE_HOME_ROOT") or Path.cwd() / ".annotation-pipeline-claude-homes")
    runtime_root.mkdir(parents=True, exist_ok=True)

    if home_id:
        isolated_home = runtime_root / home_id
        isolated_home.mkdir(parents=True, exist_ok=True)
    else:
        home_id = str(uuid.uuid4())
        isolated_home = runtime_root / home_id
        isolated_home.mkdir(parents=True, exist_ok=True)

    if user_id_override:
        _write_claude_user_id(isolated_home, user_id_override)

    isolated_env = codex_shell_environment(env)
    isolated_env["HOME"] = str(isolated_home)

    # OAuth fallback BEFORE applying explicit provider_api_key so a real
    # provider key always wins.
    resolved_api_key = provider_api_key
    if not resolved_api_key:
        oauth_token = _read_claude_oauth_token(env)
        if oauth_token:
            resolved_api_key = oauth_token

    if resolved_api_key:
        isolated_env["ANTHROPIC_API_KEY"] = resolved_api_key
    if provider_base_url:
        isolated_env["ANTHROPIC_BASE_URL"] = provider_base_url

    yield isolated_env, isolated_home, home_id


def _write_claude_user_id(home: Path, user_id: str) -> None:
    """Overwrite ``<home>/.claude.json`` 's ``userID`` field with the given
    value, preserving any other fields claude has cached there (growth-book
    feature flags, migration markers, etc.). claude reads this field on
    startup and uses it as the ``device_id`` component of the body.metadata
    .user_id JSON object it sends with every /v1/messages request — so by
    writing task_id here we get per-task sticky-routing for free, without
    a proxy.

    Best-effort: any read/write/parse failure leaves the file as-is and the
    next claude call falls back to its default behaviour (a stable hashed
    device_id derived once per home). We don't want this to crash a worker
    over a non-essential routing optimisation.
    """
    path = home / ".claude.json"
    data: dict
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}
    if data.get("userID") == user_id:
        return
    data["userID"] = user_id
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def _read_claude_oauth_token(env: Mapping[str, str]) -> str | None:
    """Return the Claude Code Pro / Max OAuth access token from the user's
    ``~/.claude/.credentials.json`` if present, else None.

    Resolves the user HOME from the input ``env`` mapping (not ``os.environ``)
    so tests and callers passing a synthetic HOME hit a predictable path.
    Returns None on any read/parse error — caller falls back to the no-key
    code path and the subsequent claude call will surface its own auth
    failure with the proper error.
    """
    home_str = env.get("HOME")
    if not home_str:
        return None
    creds_path = Path(home_str) / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return None
    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    oauth = data.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    token = oauth.get("accessToken")
    return token if isinstance(token, str) and token else None


def parse_codex_json_events(
    lines: list[str],
    *,
    provider: str,
    model: str,
) -> LLMGenerateResult:
    thread_id: str | None = None
    final_text_parts: list[str] = []
    raw_events: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    error_event: dict[str, Any] | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            final_text_parts.append(stripped)
            continue
        if not isinstance(event, dict):
            continue
        raw_events.append(event)
        event_type = event.get("type")
        if event_type in {"thread.started", "thread.resumed"} and isinstance(event.get("thread_id"), str):
            thread_id = event["thread_id"]
        item = event.get("item")
        if event_type == "item.completed" and isinstance(item, dict):
            text = item.get("text") or item.get("content")
            if item.get("type") in {"agent_message", "message"} and isinstance(text, str):
                final_text_parts.append(text)
        message = event.get("message")
        if event_type in {"agent_message", "message"} and isinstance(message, str):
            final_text_parts.append(message)
        event_usage = event.get("usage")
        if event_type == "turn.completed" and isinstance(event_usage, dict):
            usage = event_usage
        # Provider-side errors (402 Insufficient Balance, 401 auth, 429
        # rate limit, 5xx, etc.) come back as a terminal `result` event
        # with `is_error: true`. The HTTP status is in `api_error_status`
        # and the human message in `result`. Without this branch the
        # whole event is silently included in raw_events but the upstream
        # only sees `final_text=""` + `returncode!=0`, losing the cause.
        if event_type == "result" and event.get("is_error"):
            error_event = {
                "api_error_status": event.get("api_error_status"),
                "result_text": event.get("result"),
                "duration_ms": event.get("duration_ms"),
                "stop_reason": event.get("stop_reason"),
            }

    diagnostics: dict[str, Any] = {"line_count": len(lines), "event_count": len(raw_events)}
    if error_event is not None:
        diagnostics["error_event"] = error_event
    return LLMGenerateResult(
        runtime="codex_cli",
        provider=provider,
        model=model,
        continuity_handle=thread_id,
        final_text="\n".join(final_text_parts),
        usage=usage,
        raw_response=raw_events,
        diagnostics=diagnostics,
    )


def parse_claude_stream_events(
    lines: list[str],
    *,
    provider: str,
    model: str,
) -> LLMGenerateResult:
    session_id: str | None = None
    final_text_parts: list[str] = []
    raw_events: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    error_event: dict[str, Any] | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            final_text_parts.append(stripped)
            continue
        if not isinstance(event, dict):
            continue
        raw_events.append(event)
        if isinstance(event.get("session_id"), str):
            session_id = event["session_id"]
        event_type = event.get("type")
        if event_type == "assistant":
            text = _claude_event_text(event)
            if text:
                final_text_parts.append(text)
        event_usage = event.get("usage")
        if event_type == "result" and isinstance(event_usage, dict):
            usage = event_usage
        # Provider-side errors (402 Insufficient Balance, 401 auth, 429
        # rate limit, 5xx, etc.) come back as a terminal `result` event
        # with `is_error: true`. The HTTP status is in `api_error_status`
        # and the human message in `result`. Without this branch the
        # whole event is silently dropped and upstream only sees
        # `final_text=""` + `returncode != 0`, losing the cause.
        if event_type == "result" and event.get("is_error"):
            error_event = {
                "api_error_status": event.get("api_error_status"),
                "result_text": event.get("result"),
                "duration_ms": event.get("duration_ms"),
                "stop_reason": event.get("stop_reason"),
            }

    diagnostics: dict[str, Any] = {"line_count": len(lines), "event_count": len(raw_events)}
    if error_event is not None:
        diagnostics["error_event"] = error_event
    return LLMGenerateResult(
        runtime="local_cli",
        provider=provider,
        model=model,
        continuity_handle=session_id,
        final_text="\n".join(final_text_parts),
        usage=usage,
        raw_response=raw_events,
        diagnostics=diagnostics,
    )


class LocalCLIClient:
    def __init__(self, profile: LLMProfile):
        self.profile = profile

    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        if self.profile.runtime == "codex_cli":
            return await self._generate_codex(request)
        if self.profile.runtime == "claude_cli":
            return await self._generate_claude(request)
        raise ValueError(f"unsupported runtime: {self.profile.runtime}")

    async def _generate_codex(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        handle = None if self.profile.disable_continuity else request.continuity_handle
        home_id, thread_id = _parse_codex_handle(handle)
        api_key = self.profile.resolve_api_key({**os.environ, **request.env}) or None
        command, prompt_file = build_codex_command(
            binary="codex",
            prompt=request.prompt or _messages_to_prompt(request.input_items),
            developer_instructions=request.instructions,
            thread_id=thread_id,
            model=self.profile.model,
            reasoning_effort=self.profile.reasoning_effort,
        )
        try:
            with isolated_codex_home(
                {**os.environ, **request.env},
                model=self.profile.model,
                reasoning_effort=self.profile.reasoning_effort,
                home_id=home_id,
                thread_id=thread_id,
                provider_api_key=api_key,
                provider_base_url=self.profile.base_url,
            ) as (env, _home, resolved_home_id):
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(request.cwd) if request.cwd else None,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    preexec_fn=_die_with_parent,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.profile.timeout_seconds,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # Awaiter is gone but the child subprocess is still
                    # running. Without an explicit kill the codex/claude CLI
                    # keeps occupying a slot under the global concurrency
                    # cap AND ties up file descriptors; the worker comes back
                    # to claim a new task with the previous subprocess still
                    # alive in the background. Kill + reap before re-raising.
                    process.kill()
                    try:
                        await process.wait()
                    except Exception:  # noqa: BLE001
                        pass
                    raise
            lines = stdout.decode("utf-8", errors="replace").splitlines()
            result = parse_codex_json_events(lines, provider=self.profile.name, model=self.profile.model)
            diagnostics = dict(result.diagnostics or {})
            diagnostics["returncode"] = process.returncode
            if stderr:
                diagnostics["stderr"] = stderr.decode("utf-8", errors="replace")[-4000:]
            if process.returncode != 0:
                raise LocalCLIExecutionError("local CLI provider failed", diagnostics)
            new_thread_id = result.continuity_handle
            new_handle = f"{resolved_home_id}::{new_thread_id}" if new_thread_id else None
            return LLMGenerateResult(
                runtime=result.runtime,
                provider=result.provider,
                model=result.model,
                continuity_handle=None if self.profile.disable_continuity else new_handle,
                final_text=result.final_text,
                usage=result.usage,
                raw_response=result.raw_response,
                diagnostics=diagnostics,
            )
        finally:
            prompt_file.unlink(missing_ok=True)

    async def _generate_claude(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        handle = None if self.profile.disable_continuity else request.continuity_handle
        home_id, session_id = _parse_claude_handle(handle)
        api_key = self.profile.resolve_api_key({**os.environ, **request.env}) or None
        # persist_session=True for continuity-enabled profiles so turn 1 of a
        # task actually writes its session file to disk; turn 2's --resume
        # would otherwise miss and silently start a fresh session, breaking
        # the entire multi-turn continuity chain.
        persist_session = not self.profile.disable_continuity
        command = build_claude_command(
            binary="claude",
            model=self.profile.model,
            permission_mode=self.profile.permission_mode,
            session_id=session_id,
            mcp_config_path=None,  # set inside the isolated_claude_home block below
            strict_mcp_config=bool(self.profile.strict_mcp_config),
            disallowed_tools=self.profile.disallowed_tools,
            persist_session=persist_session,
        )
        prompt = request.prompt or _messages_to_prompt(request.input_items)
        if request.instructions:
            prompt = f"{request.instructions}\n\n{prompt}"
        with isolated_claude_home(
            {**os.environ, **request.env},
            home_id=home_id,
            provider_api_key=api_key,
            provider_base_url=self.profile.base_url,
            user_id_override=request.task_id,
        ) as (env, _home, resolved_home_id):
            # Sticky-routing hint: forward task_id to the gateway so a
            # LiteLLM router can pin every turn/retry of one task to the
            # same vLLM instance and let prefix-cache hits accumulate.
            # Merge with any preexisting ANTHROPIC_CUSTOM_HEADERS so we
            # don't clobber user-supplied headers.
            if request.task_id:
                header_line = f"x-task-id: {request.task_id}"
                existing = env.get("ANTHROPIC_CUSTOM_HEADERS")
                env["ANTHROPIC_CUSTOM_HEADERS"] = (
                    f"{existing}\n{header_line}" if existing else header_line
                )
            # Materialize the per-invocation mcp-config.json inside the
            # isolated home. The home is persistent across invocations
            # (needed for session resume); this file is simply overwritten
            # each call. No try/finally needed — the file has no secrets
            # and the overwrite is idempotent.
            mcp_servers = self.profile.mcp_servers or []
            if mcp_servers:
                mcp_payload = {
                    "mcpServers": {
                        s["name"]: {"command": s["command"], "args": s["args"]}
                        for s in mcp_servers
                    }
                }
                mcp_config_path = _home / "mcp-config.json"
                mcp_config_path.write_text(json.dumps(mcp_payload), encoding="utf-8")
                # Rebuild the command now that we have a real path.
                command = build_claude_command(
                    binary="claude",
                    model=self.profile.model,
                    permission_mode=self.profile.permission_mode,
                    session_id=session_id,
                    mcp_config_path=mcp_config_path,
                    strict_mcp_config=bool(self.profile.strict_mcp_config),
                    persist_session=persist_session,
                    disallowed_tools=self.profile.disallowed_tools,
                )
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(request.cwd) if request.cwd else None,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_die_with_parent,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode("utf-8")),
                    timeout=self.profile.timeout_seconds,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                process.kill()
                try:
                    await process.wait()
                except Exception:  # noqa: BLE001
                    pass
                raise
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        result = parse_claude_stream_events(lines, provider=self.profile.name, model=self.profile.model)
        diagnostics = dict(result.diagnostics or {})
        diagnostics["returncode"] = process.returncode
        if stderr:
            diagnostics["stderr"] = stderr.decode("utf-8", errors="replace")[-4000:]
        if process.returncode != 0:
            raise LocalCLIExecutionError("local CLI provider failed", diagnostics)
        new_session_id = result.continuity_handle
        new_handle = f"{resolved_home_id}::{new_session_id}" if new_session_id else None
        return LLMGenerateResult(
            runtime=result.runtime,
            provider=result.provider,
            model=result.model,
            continuity_handle=None if self.profile.disable_continuity else new_handle,
            final_text=result.final_text,
            usage=result.usage,
            raw_response=result.raw_response,
            diagnostics=diagnostics,
        )


def _write_isolated_codex_config(path: Path, *, model: str, reasoning_effort: str | None) -> None:
    lines = []
    lines.append(f'model = "{model}"')
    if reasoning_effort:
        lines.append(f'model_reasoning_effort = "{reasoning_effort}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _messages_to_prompt(input_items: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("content", item)) for item in input_items)


def _claude_event_text(event: dict[str, Any]) -> str:
    message = event.get("message")
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = event.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text") or part.get("content")
        else:
            text = getattr(part, "text", None) or getattr(part, "content", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)

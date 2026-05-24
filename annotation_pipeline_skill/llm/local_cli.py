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

class LocalCLIClient:
    def __init__(
        self,
        profile: LLMProfile,
        *,
        store: "SqliteStore | None" = None,
        project_id: str | None = None,
    ) -> None:
        self.profile = profile
        # SDK runtimes need the store + project_id at construction (the
        # in-process tool dispatcher binds to them). codex_cli ignores
        # them — it has no in-process tool layer.
        self._store = store
        self._project_id = project_id
        self._anthropic_impl: object | None = None
        self._openai_impl: object | None = None
        if profile.runtime == "anthropic_sdk":
            # Lazy import keeps the anthropic package optional for
            # workers that only run codex_cli profiles.
            from annotation_pipeline_skill.llm.anthropic_sdk import (
                AnthropicSDKClient,
            )
            self._anthropic_impl = AnthropicSDKClient(
                profile, store=store, project_id=project_id,
            )
        elif profile.runtime == "openai_sdk":
            from annotation_pipeline_skill.llm.openai_sdk import OpenAISDKClient
            self._openai_impl = OpenAISDKClient(
                profile, store=store, project_id=project_id,
            )

    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        if self.profile.runtime == "anthropic_sdk":
            assert self._anthropic_impl is not None  # set in __init__
            return await self._anthropic_impl.generate(request)
        if self.profile.runtime == "openai_sdk":
            assert self._openai_impl is not None  # set in __init__
            return await self._openai_impl.generate(request)
        if self.profile.runtime == "codex_cli":
            return await self._generate_codex(request)
        raise ValueError(f"unsupported runtime: {self.profile.runtime}")

    async def _generate_codex(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        handle = None if self.profile.disable_continuity else request.continuity_handle
        home_id, thread_id = _parse_codex_handle(handle)
        api_key = self.profile.resolve_api_key({**os.environ, **request.env}) or None
        developer_instructions = _inject_schema_into_instructions(
            request.instructions, request.response_format
        )
        command, prompt_file = build_codex_command(
            binary="codex",
            prompt=request.prompt or _messages_to_prompt(request.input_items),
            developer_instructions=developer_instructions,
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



def _write_isolated_codex_config(path: Path, *, model: str, reasoning_effort: str | None) -> None:
    lines = []
    lines.append(f'model = "{model}"')
    if reasoning_effort:
        lines.append(f'model_reasoning_effort = "{reasoning_effort}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _messages_to_prompt(input_items: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("content", item)) for item in input_items)


def _inject_schema_into_instructions(
    instructions: str | None,
    response_format: dict[str, Any] | None,
) -> str | None:
    """Append the strict JSON schema to developer_instructions for codex_cli.

    openai_sdk enforces response_format at the API level; codex_cli ignores
    the parameter entirely. This function extracts the schema from a
    json_schema response_format and appends it as a plain-text block so the
    model sees the exact required output shape.

    Returns the original instructions unchanged when response_format is absent,
    is not type=json_schema, or has no schema payload.
    """
    if not response_format:
        return instructions
    if response_format.get("type") != "json_schema":
        return instructions
    js = response_format.get("json_schema") or {}
    schema = js.get("schema")
    if not schema:
        return instructions
    name = js.get("name", "output")
    schema_text = (
        f"\n\nREQUIRED OUTPUT FORMAT — respond with a single JSON object matching "
        f"this schema exactly (name: {name}):\n"
        + json.dumps(schema, indent=2)
    )
    return (instructions or "") + schema_text



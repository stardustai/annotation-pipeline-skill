"""Direct Anthropic Messages API client.

Replaces ``claude --bare -p`` subprocess invocations with in-process
HTTP calls so we have full control over the outgoing request body. The
motivation traces to live production data: even after sticky routing,
session-file persistence, and prompt-builder key stability, vLLM's
prefix cache stayed at 0% hit rate because claude CLI prepends an
unconditional ``x-anthropic-billing-header: ...; cch=<hex>;`` text
block as ``system[0]``. The ``cch=`` counter is per-call, so vLLM's
rolling block hash misses from byte 1 even when everything downstream
of it is byte-identical. The CLI offers no flag to disable the
header — see ``claude --help`` and the dumped binary strings — so the
only path to a byte-stable prefix is owning the request body
ourselves.

Surface: ``AnthropicSDKClient(profile, store=..., project_id=...)``
exposes ``async def generate(LLMGenerateRequest) -> LLMGenerateResult``
mirroring ``LocalCLIClient``'s contract so the call sites in
``runtime/subagent_cycle.py`` need no changes when the profile's
``runtime`` flips from ``claude_cli`` to ``anthropic_sdk``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from annotation_pipeline_skill.llm.client import LLMGenerateRequest, LLMGenerateResult
from annotation_pipeline_skill.llm.profiles import LLMProfile
from annotation_pipeline_skill.llm.tool_registry import build_tool_registry
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


logger = logging.getLogger("annotation_pipeline_skill.llm.anthropic_sdk")


# Hard cap on agent-loop iterations per generate() call. Claude CLI itself
# doesn't cap, but workers can't supervise themselves — a hostile / buggy
# tool loop must terminate. 10 covers the validator self-check pattern
# (typically 2-3 iters: draft → check → fix → check → submit).
_MAX_AGENT_ITERATIONS = 10

# Hard cap on consecutive failures of the SAME (tool_name, exception_class)
# combo within one generate() call. Defends against a malformed task_id
# making a tool raise forever — model would otherwise keep retrying and
# burn gateway budget unbounded. After this many repeats, raise with
# diagnostics so the worker-bail layer can take over.
_TOOL_FAILURE_BREAKER = 3


class LocalCLIExecutionError(Exception):
    """Raised on any unrecoverable provider error. Mirrors the name used
    by ``local_cli.LocalCLIClient`` so upstream retry/alert/fallback
    logic in ``runtime/subagent_cycle.py`` and ``runtime/local_scheduler.py``
    works uniformly across runtimes."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class AnthropicSDKClient:
    """Generates LLM completions via the Anthropic Python SDK.

    Construction is per-profile, so workers can reuse one client across
    many tasks (sticky-routing fields are set per request, not on the
    client). When the profile declares ``mcp_servers``, a ``store`` (and,
    for the KB server, a ``project_id``) is required so the in-process
    tool registry can dispatch tool calls without an MCP subprocess.
    """

    def __init__(
        self,
        profile: LLMProfile,
        *,
        store: SqliteStore | None = None,
        project_id: str | None = None,
    ) -> None:
        self.profile = profile
        self._store = store
        self._project_id = project_id
        api_key = profile.resolve_api_key() or "sk-no-key-configured"
        # max_retries=0: the scheduler's worker-bail layer (see
        # local_scheduler.py:451 `_is_provider_transient_error`) owns
        # retry policy. Letting the SDK retry on 5xx duplicates work
        # and obscures real failure rates.
        #
        # timeout: the SDK refuses non-streaming requests whose worst-
        # case duration (estimated from max_tokens) exceeds 10min unless
        # an explicit timeout is set. Our profile.timeout_seconds caps
        # the whole generate() via asyncio.wait_for, but the SDK's own
        # check happens before our wait_for engages — so we must also
        # tell the SDK directly. Match the profile's timeout, defaulting
        # to 900s if unset.
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=profile.base_url,
            max_retries=0,
            timeout=float(profile.timeout_seconds or 900),
        )
        # Build the tool registry once at construction. The set of MCP
        # servers a profile declares is static; tool callability is
        # bound to (store, project_id) which are also worker-stable.
        mcp_names = {
            s.get("name", "")
            for s in (profile.mcp_servers or [])
            if isinstance(s, dict)
        }
        self._tools = build_tool_registry(
            store=store,
            project_id=project_id,
            mcp_server_names=mcp_names,
        )
        self._tool_schemas = [entry.schema for entry in self._tools.values()]

    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        """Single entry point — match LocalCLIClient.generate()."""
        timeout = self.profile.timeout_seconds or 900
        try:
            return await asyncio.wait_for(self._generate(request), timeout=timeout)
        except asyncio.TimeoutError:
            raise LocalCLIExecutionError(
                f"anthropic_sdk timeout after {timeout}s",
                {"timeout_seconds": timeout, "runtime": "anthropic_sdk"},
            ) from None

    async def _generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        session_uuid = (
            None
            if self.profile.disable_continuity
            else request.continuity_handle
        )
        messages, session_uuid = self._load_or_init_session(session_uuid)

        # Append the new user turn.
        user_text = request.prompt or _messages_to_text(request.input_items)
        messages.append({"role": "user", "content": user_text})

        loop_result = await self._run_agent_loop(
            request=request,
            messages=messages,
        )

        if loop_result.persist:
            self._save_session(session_uuid, messages)

        # When disable_continuity, surface no handle so the runtime's
        # _read_pinned_handle returns None on the next turn — matches the
        # existing claude_cli semantics at local_cli.py:545.
        out_handle = None if self.profile.disable_continuity else session_uuid

        return LLMGenerateResult(
            runtime="anthropic_sdk",
            provider=self.profile.name,
            model=self.profile.model,
            continuity_handle=out_handle,
            final_text=loop_result.final_text,
            usage=loop_result.usage,
            raw_response=loop_result.raw_response,
            diagnostics=loop_result.diagnostics,
        )

    # ---- Agent loop ------------------------------------------------------

    async def _run_agent_loop(
        self,
        *,
        request: LLMGenerateRequest,
        messages: list[dict[str, Any]],
    ) -> "_AgentLoopResult":
        """Multi-iter tool-use loop. Mutates ``messages`` in place
        (appends assistant + tool_result turns) so the caller can
        persist the full conversation. Returns the extracted final text
        plus aggregated diagnostics."""
        started_at = time.monotonic()
        tool_failure_counts: dict[tuple[str, str], int] = {}
        usage_acc: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        last_response: Any = None

        # Sticky-routing hint headers, set per-call so different tasks
        # reuse the same client. metadata.user_id is what LiteLLM's
        # `body.user` routing hashes on; x-task-id is the header path
        # for routers that prefer headers.
        extra_headers = {"x-task-id": request.task_id} if request.task_id else {}
        metadata = (
            {"user_id": request.task_id}
            if request.task_id
            else None
        )

        for iteration in range(_MAX_AGENT_ITERATIONS):
            try:
                response = await self._client.messages.create(
                    model=self.profile.model,
                    # System MUST be a plain str — passing a list-of-blocks
                    # changes the wire format and reintroduces per-call
                    # variability via block boundaries. The point of this
                    # entire rewrite is byte stability.
                    system=request.instructions or "",
                    messages=messages,
                    tools=self._tool_schemas or anthropic.NOT_GIVEN,
                    metadata=metadata or anthropic.NOT_GIVEN,
                    max_tokens=request.max_output_tokens or 32000,
                    extra_headers=extra_headers or None,
                )
            except anthropic.APIError as exc:
                diagnostics = {
                    "runtime": "anthropic_sdk",
                    "iteration": iteration,
                    "api_error_status": getattr(exc, "status_code", None),
                    "api_error_type": type(exc).__name__,
                }
                raise LocalCLIExecutionError(str(exc), diagnostics) from exc

            last_response = response
            _accumulate_usage(usage_acc, response.usage)

            # Persist the assistant turn even on early termination so
            # the next --resume sees the full history.
            messages.append({"role": "assistant", "content": _to_serializable(response.content)})

            stop_reason = response.stop_reason
            if stop_reason in {"end_turn", "stop_sequence"}:
                return _AgentLoopResult(
                    final_text=_extract_text(response.content),
                    usage=usage_acc,
                    raw_response=_to_serializable(last_response.model_dump()),
                    diagnostics={
                        "runtime": "anthropic_sdk",
                        "iterations": iteration + 1,
                        "stop_reason": stop_reason,
                        "duration_ms": int((time.monotonic() - started_at) * 1000),
                    },
                    persist=True,
                )
            if stop_reason == "max_tokens":
                # Return partial. Upstream callers inspect
                # diagnostics["truncated"] and either retry with a
                # larger budget or escalate to HR. Do NOT silently
                # continue — annotation JSON gets truncated and the
                # downstream parser will fail anyway.
                return _AgentLoopResult(
                    final_text=_extract_text(response.content),
                    usage=usage_acc,
                    raw_response=_to_serializable(last_response.model_dump()),
                    diagnostics={
                        "runtime": "anthropic_sdk",
                        "iterations": iteration + 1,
                        "stop_reason": "max_tokens",
                        "truncated": True,
                        "duration_ms": int((time.monotonic() - started_at) * 1000),
                    },
                    persist=True,
                )
            if stop_reason == "refusal":
                raise LocalCLIExecutionError(
                    "model refusal",
                    {
                        "runtime": "anthropic_sdk",
                        "iterations": iteration + 1,
                        "stop_reason": "refusal",
                    },
                )
            if stop_reason == "pause_turn":
                # Extended-thinking control flow. The model has more
                # work; loop again with the same messages plus the
                # pause-turn assistant content already appended.
                continue
            if stop_reason == "tool_use":
                tool_result_blocks = await self._dispatch_tool_uses(
                    response.content, tool_failure_counts
                )
                # Append all tool_results as ONE user turn (Anthropic
                # API expects multiple tool_use → multiple tool_result
                # in a single user message).
                messages.append({"role": "user", "content": tool_result_blocks})
                continue

            raise LocalCLIExecutionError(
                f"unknown stop_reason: {stop_reason!r}",
                {
                    "runtime": "anthropic_sdk",
                    "iterations": iteration + 1,
                    "stop_reason": stop_reason,
                },
            )

        # Iter cap exhausted — model never reached end_turn.
        raise LocalCLIExecutionError(
            f"agent loop exceeded {_MAX_AGENT_ITERATIONS} iterations",
            {
                "runtime": "anthropic_sdk",
                "iterations": _MAX_AGENT_ITERATIONS,
                "stop_reason": "iteration_cap",
            },
        )

    async def _dispatch_tool_uses(
        self,
        content: list[Any],
        failure_counts: dict[tuple[str, str], int],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for block in content:
            block_type = getattr(block, "type", None) or block.get("type")
            if block_type != "tool_use":
                continue
            name = getattr(block, "name", None) or block.get("name", "")
            tool_id = getattr(block, "id", None) or block.get("id", "")
            args = getattr(block, "input", None)
            if args is None:
                args = block.get("input", {})
            entry = self._tools.get(name)
            if entry is None:
                results.append(_tool_result_block(
                    tool_id,
                    {"error": f"unknown tool: {name}"},
                    is_error=True,
                ))
                continue
            try:
                result = await entry.dispatch(args)
                results.append(_tool_result_block(tool_id, result, is_error=False))
            except Exception as exc:  # noqa: BLE001 — see breaker below
                key = (name, type(exc).__name__)
                failure_counts[key] = failure_counts.get(key, 0) + 1
                if failure_counts[key] >= _TOOL_FAILURE_BREAKER:
                    raise LocalCLIExecutionError(
                        f"tool stuck in failure loop: {name} kept raising {type(exc).__name__}",
                        {
                            "runtime": "anthropic_sdk",
                            "tool_name": name,
                            "exception_class": type(exc).__name__,
                            "exception_message": str(exc)[:500],
                            "failure_count": failure_counts[key],
                        },
                    ) from exc
                results.append(_tool_result_block(
                    tool_id,
                    {"error": f"{type(exc).__name__}: {exc!s}"},
                    is_error=True,
                ))
        return results

    # ---- Session persistence --------------------------------------------

    def _conversations_dir(self) -> Path:
        # The store's root is the canonical anchor. When a client is
        # constructed without a store (tests), fall back to a temp-ish
        # path under CWD so the round-trip persistence test still works.
        root = self._store.root if self._store is not None else Path.cwd() / ".anthropic_sdk_conversations"
        d = root / "conversations"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _session_path(self, session_uuid: str) -> Path:
        return self._conversations_dir() / f"{session_uuid}.jsonl"

    def _load_or_init_session(
        self, session_uuid: str | None
    ) -> tuple[list[dict[str, Any]], str]:
        if session_uuid:
            path = self._session_path(session_uuid)
            if path.exists():
                try:
                    messages = [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    return messages, session_uuid
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "anthropic_sdk: session %s exists but failed to parse (%s); "
                        "starting fresh",
                        session_uuid, exc,
                    )
        # New session.
        return [], str(uuid.uuid4())

    def _save_session(self, session_uuid: str, messages: list[dict[str, Any]]) -> None:
        path = self._session_path(session_uuid)
        tmp = path.with_suffix(".jsonl.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for m in messages:
                    f.write(json.dumps(m, ensure_ascii=False, default=str))
                    f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning(
                "anthropic_sdk: failed to persist session %s: %s",
                session_uuid, exc,
            )
            try:
                tmp.unlink()
            except OSError:
                pass


# ---- Helpers / dataclasses ------------------------------------------------


class _AgentLoopResult:
    """Lightweight bag of agent-loop outputs. Not frozen so the loop
    can construct it incrementally if needed."""

    def __init__(
        self,
        *,
        final_text: str,
        usage: dict[str, int],
        raw_response: Any,
        diagnostics: dict[str, Any],
        persist: bool,
    ) -> None:
        self.final_text = final_text
        self.usage = usage
        self.raw_response = raw_response
        self.diagnostics = diagnostics
        self.persist = persist


def _extract_text(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None) or block.get("type")
        if block_type == "text":
            text = getattr(block, "text", None) or block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _to_serializable(value: Any) -> Any:
    """Convert SDK Pydantic objects to plain dicts/lists for storage. The
    caller stuffs the result into ``LLMGenerateResult.raw_response`` and
    persists it under ``task.metadata`` — needs to be json-serializable."""
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


def _accumulate_usage(acc: dict[str, int], usage: Any) -> None:
    if usage is None:
        return
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    for f in fields:
        v = getattr(usage, f, None)
        if v is None and isinstance(usage, dict):
            v = usage.get(f)
        if isinstance(v, int):
            acc[f] = acc.get(f, 0) + v


def _tool_result_block(
    tool_use_id: str, content: Any, *, is_error: bool
) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(content, ensure_ascii=False, default=str),
        "is_error": is_error,
    }


def _messages_to_text(input_items: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("content", item)) for item in input_items)

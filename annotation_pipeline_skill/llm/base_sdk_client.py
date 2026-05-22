"""Shared agent loop, JSONL session management, and abstract interface
for SDK-based LLM runtimes (openai_sdk and anthropic_sdk).

Internal message format: OpenAI Chat Completions
(role: system/user/assistant/tool). Both subclasses store and load
in this format; AnthropicSDKClient converts to/from Anthropic wire
format inside _call_api().
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from annotation_pipeline_skill.llm.client import LLMGenerateRequest, LLMGenerateResult
from annotation_pipeline_skill.llm.profiles import LLMProfile
from annotation_pipeline_skill.llm.tool_registry import build_tool_registry
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


logger = logging.getLogger("annotation_pipeline_skill.llm.base_sdk_client")

_MAX_AGENT_ITERATIONS = 10
_TOOL_FAILURE_BREAKER = 3


class LocalCLIExecutionError(Exception):
    """Raised on any unrecoverable provider error, uniform across runtimes."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics: dict[str, Any] = diagnostics or {}


@dataclass
class _ApiCallResult:
    """Canonical result from one API call, in OpenAI format."""
    stop_reason: Literal["end_turn", "tool_calls", "max_tokens", "refusal", "unknown"]
    text: str
    tool_calls: list[dict]       # [{"id": str, "name": str, "args": dict}]
    assistant_message: dict      # OpenAI-format, ready to append to messages
    usage: dict[str, int]


class BaseSdkClient(ABC):
    """Abstract base for SDK-based LLM clients.

    Subclasses implement _call_api() only. Everything else — session
    loading/saving, agent loop, tool dispatch — lives here.
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

    # ---- public API --------------------------------------------------------

    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        timeout = self.profile.timeout_seconds or 900
        try:
            return await asyncio.wait_for(self._generate(request), timeout=float(timeout))
        except asyncio.TimeoutError:
            raise LocalCLIExecutionError(
                f"sdk timeout after {timeout}s",
                {"timeout_seconds": timeout, "runtime": self.profile.runtime},
            ) from None

    # ---- internal ----------------------------------------------------------

    async def _generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        session_uuid = (
            None if self.profile.disable_continuity else request.continuity_handle
        )
        messages, session_uuid = self._load_or_init_session(session_uuid)
        messages.append({"role": "user", "content": request.prompt or _items_to_text(request.input_items)})

        started_at = time.monotonic()
        tool_failure_counts: dict[tuple[str, str], int] = {}
        usage_acc: dict[str, int] = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }

        final_text = ""
        diagnostics: dict[str, Any] = {"runtime": self.profile.runtime}
        persist = False

        for iteration in range(_MAX_AGENT_ITERATIONS):
            result = await self._call_api(
                system=request.instructions or "",
                messages=messages,
                tools=self._tool_schemas,
                task_id=request.task_id,
            )
            _add_usage(usage_acc, result.usage)
            messages.append(result.assistant_message)

            if result.stop_reason == "end_turn":
                final_text = result.text
                diagnostics.update({
                    "stop_reason": result.stop_reason,
                    "iterations": iteration + 1,
                    "duration_ms": int((time.monotonic() - started_at) * 1000),
                })
                persist = True
                break

            if result.stop_reason == "max_tokens":
                final_text = result.text
                diagnostics.update({
                    "stop_reason": "max_tokens",
                    "truncated": True,
                    "iterations": iteration + 1,
                    "duration_ms": int((time.monotonic() - started_at) * 1000),
                })
                persist = True
                break

            if result.stop_reason == "refusal":
                raise LocalCLIExecutionError(
                    "model refusal",
                    {"runtime": self.profile.runtime, "iterations": iteration + 1, "stop_reason": "refusal"},
                )

            if result.stop_reason == "tool_calls":
                tool_msgs = await self._dispatch_tools(result.tool_calls, tool_failure_counts)
                messages.extend(tool_msgs)
                continue

            raise LocalCLIExecutionError(
                f"unknown stop_reason: {result.stop_reason!r}",
                {"runtime": self.profile.runtime, "iterations": iteration + 1, "stop_reason": result.stop_reason},
            )
        else:
            raise LocalCLIExecutionError(
                f"agent loop exceeded {_MAX_AGENT_ITERATIONS} iterations",
                {"runtime": self.profile.runtime, "iterations": _MAX_AGENT_ITERATIONS, "stop_reason": "iteration_cap"},
            )

        if persist and not self.profile.disable_continuity:
            self._save_session(session_uuid, messages)

        out_handle = None if self.profile.disable_continuity else session_uuid

        return LLMGenerateResult(
            runtime=self.profile.runtime,
            provider=self.profile.name,
            model=self.profile.model,
            continuity_handle=out_handle,
            final_text=final_text,
            usage=usage_acc,
            raw_response=[],
            diagnostics=diagnostics,
        )

    async def _dispatch_tools(
        self,
        tool_calls: list[dict],
        failure_counts: dict[tuple[str, str], int],
    ) -> list[dict]:
        """Dispatch tool calls; return OpenAI-format role:tool messages."""
        results: list[dict] = []
        for tc in tool_calls:
            name, tool_id, args = tc["name"], tc["id"], tc["args"]
            entry = self._tools.get(name)
            if entry is None:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps({"error": f"unknown tool: {name}"}),
                })
                continue
            try:
                value = await entry.dispatch(args)
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(value, ensure_ascii=False, default=str),
                })
            except Exception as exc:  # noqa: BLE001
                key = (name, type(exc).__name__)
                failure_counts[key] = failure_counts.get(key, 0) + 1
                if failure_counts[key] >= _TOOL_FAILURE_BREAKER:
                    raise LocalCLIExecutionError(
                        f"tool stuck in failure loop: {name} kept raising {type(exc).__name__}",
                        {
                            "runtime": self.profile.runtime,
                            "tool_name": name,
                            "exception_class": type(exc).__name__,
                            "exception_message": str(exc)[:500],
                            "failure_count": failure_counts[key],
                        },
                    ) from exc
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps({"error": f"{type(exc).__name__}: {exc!s}"}),
                })
        return results

    # ---- JSONL session -----------------------------------------------------

    def _conversations_dir(self) -> Path:
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
                    msgs = [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    return msgs, session_uuid
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("base_sdk: session %s unreadable (%s); starting fresh", session_uuid, exc)
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
            logger.warning("base_sdk: failed to persist session %s: %s", session_uuid, exc)
            try:
                tmp.unlink()
            except OSError:
                pass

    # ---- abstract ----------------------------------------------------------

    @abstractmethod
    async def _call_api(
        self,
        system: str,
        messages: list[dict[str, Any]],  # OpenAI Chat Completions format
        tools: list[dict[str, Any]],     # Anthropic registry format (input_schema)
        task_id: str | None = None,
    ) -> _ApiCallResult: ...


# ---- helpers ---------------------------------------------------------------

def _items_to_text(items: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("content", item)) for item in items)


def _add_usage(acc: dict[str, int], usage: dict[str, int]) -> None:
    for k, v in usage.items():
        if isinstance(v, int):
            acc[k] = acc.get(k, 0) + v

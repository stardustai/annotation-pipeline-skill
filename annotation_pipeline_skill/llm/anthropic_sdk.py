"""Anthropic Messages API adapter for BaseSdkClient.

Converts OpenAI-format messages (canonical storage format) to Anthropic
wire format before each API call, and converts the response back to
OpenAI format for storage. Tool schemas are used as-is (already in
Anthropic input_schema format).

Re-exports LocalCLIExecutionError for backward compatibility with
callers that import it from this module.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import anthropic
from anthropic import AsyncAnthropic

from annotation_pipeline_skill.llm.base_sdk_client import (
    _ApiCallResult,
    BaseSdkClient,
    LocalCLIExecutionError,  # noqa: F401  re-exported for callers
)
from annotation_pipeline_skill.llm.profiles import LLMProfile
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


logger = logging.getLogger("annotation_pipeline_skill.llm.anthropic_sdk")


class AnthropicSDKClient(BaseSdkClient):

    def __init__(
        self,
        profile: LLMProfile,
        *,
        store: SqliteStore | None = None,
        project_id: str | None = None,
    ) -> None:
        super().__init__(profile, store=store, project_id=project_id)
        api_key = profile.resolve_api_key() or None
        auth_token: str | None = None
        if not api_key:
            auth_token = _read_oauth_access_token(os.environ)
        if not api_key and not auth_token:
            api_key = "sk-no-key-configured"
        self._client = AsyncAnthropic(
            **({"auth_token": auth_token} if auth_token else {"api_key": api_key}),
            base_url=profile.base_url,
            max_retries=0,
            timeout=float(profile.timeout_seconds or 900),
        )

    async def _call_api(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        task_id: str | None = None,
    ) -> _ApiCallResult:
        _, anthropic_messages = _openai_to_anthropic_messages(messages)

        # Sticky-routing hint: metadata.user_id is what LiteLLM's body.user
        # routing hashes on; x-task-id is the header-routing path.
        metadata = {"user_id": task_id} if task_id else None
        extra_headers = {"x-task-id": task_id} if task_id else {}

        # pause_turn is Anthropic extended-thinking: loop internally so
        # the outer agent loop iteration count is not inflated.
        for _pause_iter in range(10):
            try:
                response = await self._client.messages.create(
                    model=self.profile.model,
                    system=system or "",
                    messages=anthropic_messages,
                    tools=tools or anthropic.NOT_GIVEN,
                    metadata=metadata or anthropic.NOT_GIVEN,
                    max_tokens=32000,
                    extra_headers=extra_headers or None,
                )
            except anthropic.APIError as exc:
                raise LocalCLIExecutionError(
                    str(exc),
                    {
                        "runtime": "anthropic_sdk",
                        "error_event": {
                            "api_error_status": getattr(exc, "status_code", None),
                            "result_text": str(exc),
                        },
                    },
                ) from exc

            if response.stop_reason != "pause_turn":
                break
            # Append the pause_turn assistant content and loop.
            anthropic_messages.append({
                "role": "assistant",
                "content": _to_serializable(response.content),
            })

        stop_reason = _map_anthropic_stop_reason(response.stop_reason)
        text = _extract_text(response.content)
        tool_calls = [
            {"id": b.id, "name": b.name, "args": b.input}
            for b in response.content
            if getattr(b, "type", None) == "tool_use"
        ]
        assistant_message = _anthropic_content_to_openai_assistant(response.content)
        usage = _extract_usage(response.usage)

        return _ApiCallResult(
            stop_reason=stop_reason,
            text=text,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            usage=usage,
        )


# ---- message conversion ----------------------------------------------------

def _openai_to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_string, anthropic_messages). The system message is
    extracted and removed from the list; consecutive role:tool messages
    are merged into one role:user message with type:tool_result content.
    """
    system = ""
    anthropic_msgs: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def _flush_tools() -> None:
        if pending_tool_results:
            anthropic_msgs.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            system = msg.get("content") or ""
            continue

        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            })
            continue

        # Any non-tool message flushes pending tool results first.
        _flush_tools()

        if role == "user":
            anthropic_msgs.append({"role": "user", "content": msg.get("content", "")})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                content: list[dict] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                content.extend([
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    }
                    for tc in tool_calls
                ])
                anthropic_msgs.append({"role": "assistant", "content": content})
            else:
                text = msg.get("content") or ""
                anthropic_msgs.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}] if text else [],
                })

    _flush_tools()
    return system, anthropic_msgs


def _anthropic_content_to_openai_assistant(content: list[Any]) -> dict[str, Any]:
    """Convert Anthropic response content blocks to an OpenAI-format assistant message."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "")
            if text:
                text_parts.append(text)
        elif btype == "tool_use":
            tool_calls.append({
                "id": getattr(block, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(block, "name", ""),
                    "arguments": json.dumps(getattr(block, "input", {})),
                },
            })

    if tool_calls:
        return {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
            "tool_calls": tool_calls,
        }
    return {"role": "assistant", "content": "\n".join(text_parts)}


# ---- helpers ---------------------------------------------------------------

def _map_anthropic_stop_reason(reason: str | None) -> str:
    return {
        "end_turn": "end_turn",
        "stop_sequence": "end_turn",
        "tool_use": "tool_calls",
        "max_tokens": "max_tokens",
        "refusal": "refusal",
    }.get(reason or "", "unknown")


def _extract_text(content: list[Any]) -> str:
    parts = []
    for block in content:
        if getattr(block, "type", None) == "text":
            t = getattr(block, "text", "")
            if isinstance(t, str):
                parts.append(t)
    return "\n".join(parts)


def _extract_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    result: dict[str, int] = {}
    for f in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        v = getattr(usage, f, None)
        if isinstance(v, int):
            result[f] = v
    return result


def _to_serializable(value: Any) -> Any:
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _read_oauth_access_token(env: Mapping[str, str]) -> str | None:
    home_str = env.get("HOME")
    if not home_str:
        return None
    creds_path = Path(home_str) / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return None
    try:
        import json as _json
        data = _json.loads(creds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    oauth = data.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    token = oauth.get("accessToken")
    return token if isinstance(token, str) and token else None

"""OpenAI Chat Completions adapter for BaseSdkClient.

Converts Anthropic-format tool schemas (input_schema) to OpenAI format
(parameters), calls chat.completions.create(), and converts the response
back to _ApiCallResult in the canonical OpenAI format.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import openai
from openai import AsyncOpenAI

from annotation_pipeline_skill.llm.base_sdk_client import (
    _ApiCallResult,
    BaseSdkClient,
    LocalCLIExecutionError,
)
from annotation_pipeline_skill.llm.profiles import LLMProfile
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


logger = logging.getLogger("annotation_pipeline_skill.llm.openai_sdk")


class OpenAISDKClient(BaseSdkClient):

    def __init__(
        self,
        profile: LLMProfile,
        *,
        store: SqliteStore | None = None,
        project_id: str | None = None,
    ) -> None:
        super().__init__(profile, store=store, project_id=project_id)
        api_key = profile.resolve_api_key() or "sk-no-key-configured"
        self._client = AsyncOpenAI(
            api_key=api_key,
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
        response_format: dict[str, Any] | None = None,
    ) -> _ApiCallResult:
        openai_tools = [_to_openai_tool(t) for t in tools] if tools else None
        full_messages = ([{"role": "system", "content": system}] if system else []) + messages

        extra_headers: dict[str, str] = {}
        if task_id:
            extra_headers["x-task-id"] = task_id

        try:
            kwargs: dict[str, Any] = dict(
                model=self.profile.model,
                messages=full_messages,
            )
            if openai_tools:
                kwargs["tools"] = openai_tools
            if self.profile.reasoning_effort:
                kwargs["reasoning_effort"] = self.profile.reasoning_effort
            if response_format:
                kwargs["response_format"] = response_format
            if extra_headers:
                kwargs["extra_headers"] = extra_headers

            response = await self._client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise LocalCLIExecutionError(
                str(exc),
                {
                    "runtime": "openai_sdk",
                    "error_event": {
                        "api_error_status": getattr(exc, "status_code", None),
                        "result_text": str(exc),
                    },
                },
            ) from exc

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        msg = choice.message

        stop_reason = _map_finish_reason(finish_reason)
        text = msg.content or ""
        tool_calls: list[dict[str, Any]] = []
        assistant_message: dict[str, Any]

        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]
            assistant_message = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        else:
            assistant_message = {"role": "assistant", "content": text}

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            }
            extras = getattr(response.usage, "model_extra", {}) or {}
            usage.update({k: v for k, v in extras.items() if isinstance(v, int)})

        return _ApiCallResult(
            stop_reason=stop_reason,
            text=text,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            usage=usage,
        )


# ---- helpers ---------------------------------------------------------------

def _to_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic tool schema to OpenAI function tool format."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get("input_schema", {"type": "object"}),
        },
    }


def _map_finish_reason(reason: str | None) -> str:
    return {
        "stop": "end_turn",
        "tool_calls": "tool_calls",
        "length": "max_tokens",
        "content_filter": "refusal",
    }.get(reason or "", "unknown")

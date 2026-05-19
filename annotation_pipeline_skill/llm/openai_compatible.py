from __future__ import annotations

import inspect
from typing import Any

from openai import AsyncOpenAI

from annotation_pipeline_skill.llm.client import LLMGenerateRequest, LLMGenerateResult
from annotation_pipeline_skill.llm.profiles import LLMProfile


class OpenAICompatibleClient:
    def __init__(self, profile: LLMProfile, client: Any | None = None):
        self.profile = profile
        self.client = client or AsyncOpenAI(
            api_key=profile.resolve_api_key(),
            base_url=profile.base_url,
            max_retries=profile.max_retries or 2,
            timeout=profile.timeout_seconds,
        )

    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "messages": _request_messages(request),
        }
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        response = await self.client.chat.completions.create(**kwargs)
        return LLMGenerateResult(
            runtime="openai_compatible",
            provider=self.profile.name,
            model=self.profile.model,
            continuity_handle=_response_id(response),
            final_text=_assistant_text(response),
            usage=_usage(response),
            raw_response=_dump_response(response),
            diagnostics={"provider_flavor": self.profile.provider_flavor},
        )

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if close is None:
            close = getattr(self.client, "aclose", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _request_messages(request: LLMGenerateRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})
    if request.input_items:
        messages.extend(_normalize_input_items(request.input_items))
    else:
        messages.append({"role": "user", "content": request.prompt or ""})
    return messages


def _normalize_input_items(input_items: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in input_items:
        role = item.get("role")
        content = item.get("content")
        messages.append(
            {
                "role": str(role) if role else "user",
                "content": content if isinstance(content, str) else str(item),
            }
        )
    return messages


def _response_id(response: Any) -> str | None:
    if isinstance(response, dict):
        value = response.get("id")
    else:
        value = getattr(response, "id", None)
    return str(value) if value is not None else None


def _assistant_text(response: Any) -> str:
    choice = _first_choice(response)
    if not choice:
        return ""
    message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
            else:
                text = getattr(part, "text", None) or getattr(part, "content", None)
            if isinstance(text, str):
                texts.append(text)
        return "\n".join(texts)
    return ""


def _first_choice(response: Any) -> Any | None:
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not choices:
        return None
    return choices[0]


def _usage(response: Any) -> dict[str, Any] | None:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    dumped = _dump_response(usage)
    return dumped if isinstance(dumped, dict) else None


def _dump_response(response: Any) -> dict[str, Any] | list[dict[str, Any]]:
    if response is None:
        return {}
    if isinstance(response, (dict, list)):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", warnings="none")
    return {"repr": repr(response)}

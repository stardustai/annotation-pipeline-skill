from __future__ import annotations

import inspect
from typing import Any

from openai import AsyncOpenAI

from annotation_pipeline_skill.llm.client import (
    LLMGenerateRequest,
    LLMGenerateResult,
    LLMStructuredRequest,
    LLMStructuredResult,
)
from annotation_pipeline_skill.llm.profiles import LLMProfile
from annotation_pipeline_skill.llm.structured import extract_parsed_output


class OpenAIResponsesClient:
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
            "input": request.input_items or request.prompt or "",
        }
        if request.instructions:
            kwargs["instructions"] = request.instructions
        # Skip previous_response_id for stateless gateways (e.g. LiteLLM
        # /v1/responses translation) — they mint per-call IDs but don't
        # persist them, so forwarding the handle 404s.
        if request.continuity_handle and not self.profile.disable_continuity:
            kwargs["previous_response_id"] = request.continuity_handle
        if request.reasoning:
            kwargs["reasoning"] = request.reasoning
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens

        response = await self.client.responses.create(**kwargs)
        raw_response = _dump_response(response)
        return LLMGenerateResult(
            runtime="openai_responses",
            provider=self.profile.name,
            model=self.profile.model,
            continuity_handle=_response_id(response),
            final_text=_output_text(response),
            usage=_usage(response),
            raw_response=raw_response,
            diagnostics=None,
        )

    async def parse_structured(self, request: LLMStructuredRequest) -> LLMStructuredResult:
        kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "input": request.messages,
            "text_format": request.text_format,
        }
        # Skip previous_response_id for stateless gateways (e.g. LiteLLM
        # /v1/responses translation) — they mint per-call IDs but don't
        # persist them, so forwarding the handle 404s.
        if request.continuity_handle and not self.profile.disable_continuity:
            kwargs["previous_response_id"] = request.continuity_handle
        if request.reasoning:
            kwargs["reasoning"] = request.reasoning

        response = await self.client.responses.parse(**kwargs)
        return LLMStructuredResult(
            id=_response_id(response),
            output_parsed=extract_parsed_output(response),
            raw_response=_dump_response(response),
            diagnostics=None,
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


def _response_id(response: Any) -> str | None:
    if isinstance(response, dict):
        value = response.get("id")
    else:
        value = getattr(response, "id", None)
    return str(value) if value is not None else None


def _output_text(response: Any) -> str:
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text
        return _text_from_output(response.get("output", []))
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    return _text_from_output(getattr(response, "output", []) or [])


def _text_from_output(output: Any) -> str:
    texts: list[str] = []
    for item in output or []:
        content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        for part in content or []:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def _usage(response: Any) -> dict[str, Any] | None:
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
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

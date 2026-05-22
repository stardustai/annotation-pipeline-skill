from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class LLMGenerateRequest:
    instructions: str | None = None
    input_items: list[dict[str, Any]] = field(default_factory=list)
    prompt: str | None = None
    reasoning: dict[str, Any] = field(default_factory=dict)
    continuity_handle: str | None = None
    max_output_tokens: int | None = None
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    # OpenAI-compatible chat-completions ``response_format`` payload — e.g.
    # ``{"type": "json_object"}`` for forced-JSON or
    # ``{"type": "json_schema", "json_schema": {...}}`` for strict schema
    # enforcement. Currently ignored by both codex_cli and claude_cli runtimes,
    # but preserved on the request for callers that already populate it and
    # for future runtimes that may honor it.
    response_format: dict[str, Any] | None = None
    # Stable identifier (e.g. pipeline task_id) that runtimes may forward to
    # the gateway as a sticky-routing hint. claude_cli surfaces it as the
    # ``x-task-id`` HTTP header via ``ANTHROPIC_CUSTOM_HEADERS`` so a
    # LiteLLM router can pin every retry/turn for one task to the same
    # vLLM instance, letting that instance's prefix-cache fill up across
    # the multi-turn annotation conversation.
    task_id: str | None = None


@dataclass(frozen=True)
class LLMGenerateResult:
    runtime: str
    provider: str
    model: str
    continuity_handle: str | None
    final_text: str
    usage: dict[str, Any] | None
    raw_response: dict[str, Any] | list[dict[str, Any]]
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMStructuredRequest:
    messages: list[dict[str, Any]]
    text_format: type[BaseModel]
    reasoning: dict[str, Any] = field(default_factory=dict)
    continuity_handle: str | None = None
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMStructuredResult:
    id: str | None
    output_parsed: BaseModel
    raw_response: dict[str, Any] | list[dict[str, Any]]
    diagnostics: dict[str, Any] | None = None


class LLMClient(Protocol):
    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        ...

    async def parse_structured(self, request: LLMStructuredRequest) -> LLMStructuredResult:
        ...

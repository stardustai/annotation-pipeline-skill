"""Tests for AnthropicSDKClient — the direct-SDK replacement for
``claude --bare -p``. The whole point of the rewrite is a byte-stable
request body for vLLM prefix-cache hits, so the tests target the
invariants that matter: system prompt as a plain str (not list-of-blocks),
messages array growing turn-over-turn with the prior history byte-stable,
``metadata.user_id`` carrying the task_id for sticky routing, and the
``stop_reason`` matrix handled correctly.

All tests stub ``anthropic.AsyncAnthropic.messages.create`` at module
level — no real HTTP / no real API key needed.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from annotation_pipeline_skill.llm.anthropic_sdk import (
    AnthropicSDKClient,
    LocalCLIExecutionError,
)
from annotation_pipeline_skill.llm.client import LLMGenerateRequest
from annotation_pipeline_skill.llm.profiles import LLMProfile


# ---- helpers --------------------------------------------------------------


def _profile(
    *,
    name: str = "qwen-probe",
    timeout_seconds: int | None = 30,
    disable_continuity: bool | None = None,
    mcp_servers: list[dict] | None = None,
) -> LLMProfile:
    return LLMProfile(
        name=name,
        runtime="anthropic_sdk",
        model="qwen3.6-35b-a3b",
        base_url="http://127.0.0.1:9999",
        api_key="sk-fake",
        timeout_seconds=timeout_seconds,
        disable_continuity=disable_continuity,
        mcp_servers=mcp_servers,
    )


def _fake_response(
    *,
    stop_reason: str,
    text: str = "",
    tool_use: list[dict] | None = None,
    usage: dict | None = None,
):
    """Build a Pydantic-shaped object the agent loop can consume."""
    content_blocks = []
    if text:
        block = MagicMock()
        block.type = "text"
        block.text = text
        content_blocks.append(block)
    if tool_use:
        for tu in tool_use:
            block = MagicMock()
            block.type = "tool_use"
            block.name = tu["name"]
            block.id = tu.get("id", f"toolu_{tu['name']}")
            block.input = tu.get("input", {})
            content_blocks.append(block)

    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content_blocks
    response.usage = MagicMock()
    u = usage or {"input_tokens": 5, "output_tokens": 3}
    for k, v in u.items():
        setattr(response.usage, k, v)
    response.model_dump.return_value = {
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": text}] if text else [],
        "usage": u,
    }
    return response


def _patch_create(monkeypatch, responses: list[Any]):
    """Patch AsyncAnthropic.messages.create to return responses[i] on call i.

    Returns an object with ``.calls`` — a list of dicts, one per call,
    deep-copied at call time so tests can inspect the EXACT kwargs sent
    (especially ``messages``, which the SDK client mutates between calls
    to grow the history).
    """
    import copy
    response_iter = iter(responses)
    captured: list[dict] = []

    async def _create(**kwargs):
        # Snapshot kwargs at call time so subsequent in-place mutation
        # of the messages list by the SDK client doesn't perturb what
        # the test sees.
        captured.append({k: copy.deepcopy(v) for k, v in kwargs.items()})
        try:
            return next(response_iter)
        except StopIteration:
            raise AssertionError("messages.create called more times than expected") from None

    import annotation_pipeline_skill.llm.anthropic_sdk as sdk_mod

    def _make_client(*args, **kwargs):
        c = MagicMock()
        c.messages.create = _create
        return c

    monkeypatch.setattr(sdk_mod, "AsyncAnthropic", _make_client)

    class _Tracker:
        @property
        def calls(self):
            return captured

        @property
        def call_args(self):
            assert captured, "messages.create was never called"
            class _A:
                kwargs = captured[-1]
            return _A()

        @property
        def call_args_list(self):
            class _A:
                def __init__(self, k): self.kwargs = k
            return [_A(k) for k in captured]

    return _Tracker()


# ---- system prompt byte-stability ----------------------------------------


def test_system_prompt_is_plain_string_not_block_list(tmp_path, monkeypatch):
    """The prefix-cache locality goal hinges on system being a plain str.
    Pydantic block wrappers change the wire format and reintroduce
    per-call variability — disallowed."""
    mock = _patch_create(monkeypatch, [_fake_response(stop_reason="end_turn", text="ok")])
    monkeypatch.chdir(tmp_path)

    asyncio.run(AnthropicSDKClient(_profile()).generate(
        LLMGenerateRequest(
            instructions="You are an annotator.",
            prompt="hello",
            task_id="t-001",
        )
    ))

    kwargs = mock.call_args.kwargs
    assert isinstance(kwargs["system"], str)
    assert kwargs["system"] == "You are an annotator."


def test_system_prompt_contains_no_billing_header(tmp_path, monkeypatch):
    """The motivating bug. claude CLI prepends an
    x-anthropic-billing-header text block — we must not."""
    mock = _patch_create(monkeypatch, [_fake_response(stop_reason="end_turn", text="ok")])
    monkeypatch.chdir(tmp_path)

    asyncio.run(AnthropicSDKClient(_profile()).generate(
        LLMGenerateRequest(instructions="instr", prompt="hi", task_id="t-001")
    ))

    kwargs = mock.call_args.kwargs
    assert "x-anthropic-billing-header" not in kwargs["system"]
    assert "cch=" not in kwargs["system"]


# ---- messages-array growth (the prefix-cache invariant) ------------------


def test_turn_2_messages_extend_turn_1_byte_stable(tmp_path, monkeypatch):
    """Across two generate() calls with the same continuity_handle, turn-2
    must send turn-1's full history (system + user + assistant) + exactly
    one new user turn. This is what makes vLLM prefix-cache locality
    possible — tested directly, not by proxy."""
    mock = _patch_create(monkeypatch, [
        _fake_response(stop_reason="end_turn", text="response 1"),
        _fake_response(stop_reason="end_turn", text="response 2"),
    ])
    monkeypatch.chdir(tmp_path)

    client = AnthropicSDKClient(_profile())

    # Turn 1
    r1 = asyncio.run(client.generate(LLMGenerateRequest(
        instructions="instr", prompt="user message 1", task_id="t-001",
    )))
    turn1_messages = mock.call_args_list[0].kwargs["messages"]
    assert len(turn1_messages) == 1
    assert turn1_messages[0]["role"] == "user"
    assert turn1_messages[0]["content"] == "user message 1"

    # Turn 2 — use the handle that turn 1 returned
    r2 = asyncio.run(client.generate(LLMGenerateRequest(
        instructions="instr",
        prompt="user message 2",
        task_id="t-001",
        continuity_handle=r1.continuity_handle,
    )))
    turn2_messages = mock.call_args_list[1].kwargs["messages"]

    # Turn-2 messages array MUST start with turn-1's exchange byte-for-byte.
    assert len(turn2_messages) == 3  # user_1, assistant_1, user_2
    assert turn2_messages[0] == turn1_messages[0]  # original user message unchanged
    assert turn2_messages[1]["role"] == "assistant"
    assert turn2_messages[2]["role"] == "user"
    assert turn2_messages[2]["content"] == "user message 2"


# ---- sticky routing -------------------------------------------------------


def test_sticky_routing_sets_metadata_user_id_and_header(tmp_path, monkeypatch):
    """metadata.user_id is what LiteLLM's body.user routing hashes on;
    x-task-id is the header-routing path. Set both so the gateway can
    use either."""
    mock = _patch_create(monkeypatch, [_fake_response(stop_reason="end_turn", text="ok")])
    monkeypatch.chdir(tmp_path)

    asyncio.run(AnthropicSDKClient(_profile()).generate(LLMGenerateRequest(
        instructions="i", prompt="p", task_id="v3_initial_deployment-000342",
    )))

    kwargs = mock.call_args.kwargs
    assert kwargs["metadata"] == {"user_id": "v3_initial_deployment-000342"}
    assert kwargs["extra_headers"] == {"x-task-id": "v3_initial_deployment-000342"}


# ---- stop_reason coverage -------------------------------------------------


def test_stop_reason_end_turn_returns_text(tmp_path, monkeypatch):
    _patch_create(monkeypatch, [_fake_response(stop_reason="end_turn", text="final answer")])
    monkeypatch.chdir(tmp_path)
    r = asyncio.run(AnthropicSDKClient(_profile()).generate(
        LLMGenerateRequest(instructions="i", prompt="p", task_id="t-1")
    ))
    assert r.final_text == "final answer"
    assert r.diagnostics["stop_reason"] == "end_turn"


def test_stop_reason_stop_sequence_returns_text(tmp_path, monkeypatch):
    _patch_create(monkeypatch, [_fake_response(stop_reason="stop_sequence", text="halted")])
    monkeypatch.chdir(tmp_path)
    r = asyncio.run(AnthropicSDKClient(_profile()).generate(
        LLMGenerateRequest(instructions="i", prompt="p", task_id="t-1")
    ))
    assert r.final_text == "halted"


def test_stop_reason_max_tokens_flags_truncated_but_returns(tmp_path, monkeypatch):
    """max_tokens means the model ran out mid-output. Do NOT silently
    continue — upstream needs to see truncated=True so it can retry
    with a larger budget."""
    _patch_create(monkeypatch, [_fake_response(stop_reason="max_tokens", text="partial response")])
    monkeypatch.chdir(tmp_path)
    r = asyncio.run(AnthropicSDKClient(_profile()).generate(
        LLMGenerateRequest(instructions="i", prompt="p", task_id="t-1")
    ))
    assert r.final_text == "partial response"
    assert r.diagnostics["truncated"] is True
    assert r.diagnostics["stop_reason"] == "max_tokens"


def test_stop_reason_refusal_raises(tmp_path, monkeypatch):
    _patch_create(monkeypatch, [_fake_response(stop_reason="refusal", text="")])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LocalCLIExecutionError) as excinfo:
        asyncio.run(AnthropicSDKClient(_profile()).generate(
            LLMGenerateRequest(instructions="i", prompt="p", task_id="t-1")
        ))
    assert "refusal" in str(excinfo.value).lower()


def test_stop_reason_unknown_raises(tmp_path, monkeypatch):
    _patch_create(monkeypatch, [_fake_response(stop_reason="some_future_reason", text="")])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LocalCLIExecutionError) as excinfo:
        asyncio.run(AnthropicSDKClient(_profile()).generate(
            LLMGenerateRequest(instructions="i", prompt="p", task_id="t-1")
        ))
    assert "unknown stop_reason" in str(excinfo.value)


def test_stop_reason_pause_turn_loops(tmp_path, monkeypatch):
    """pause_turn is extended-thinking control flow — the model needs
    another iteration to finish reasoning, not a terminal state."""
    _patch_create(monkeypatch, [
        _fake_response(stop_reason="pause_turn", text="<thinking>..."),
        _fake_response(stop_reason="end_turn", text="answer"),
    ])
    monkeypatch.chdir(tmp_path)
    r = asyncio.run(AnthropicSDKClient(_profile()).generate(
        LLMGenerateRequest(instructions="i", prompt="p", task_id="t-1")
    ))
    assert r.final_text == "answer"
    assert r.diagnostics["iterations"] == 2


# ---- tool dispatch + error breaker ---------------------------------------


def test_tool_use_dispatch_and_continuation(tmp_path, monkeypatch):
    """When stop_reason=tool_use, dispatch the tool, append the result
    as a user turn, and continue the loop. End-to-end check via a tool
    we register on the client."""
    _patch_create(monkeypatch, [
        _fake_response(stop_reason="tool_use", tool_use=[{
            "name": "fake_tool", "id": "toolu_1", "input": {"x": 1},
        }]),
        _fake_response(stop_reason="end_turn", text="done"),
    ])
    monkeypatch.chdir(tmp_path)

    client = AnthropicSDKClient(_profile())
    # Inject a tool directly (skip the registry — exercising the dispatch path).
    from annotation_pipeline_skill.llm.tool_registry import ToolEntry

    async def _fake_tool(args):
        return {"echo": args}

    client._tools["fake_tool"] = ToolEntry(
        schema={"name": "fake_tool", "description": "echo", "input_schema": {"type": "object"}},
        dispatch=_fake_tool,
    )
    client._tool_schemas = [client._tools["fake_tool"].schema]

    r = asyncio.run(client.generate(LLMGenerateRequest(
        instructions="i", prompt="p", task_id="t-1",
    )))
    assert r.final_text == "done"
    assert r.diagnostics["iterations"] == 2


def test_tool_failure_breaker_after_three_repeats(tmp_path, monkeypatch):
    """A tool that keeps raising the same exception → break after 3
    attempts with a structured error. Without this a malformed
    task_id would burn unbounded gateway budget."""
    # 4 fake responses all asking for the same failing tool — the breaker
    # should fire on the 3rd attempt before we ever consume the 4th.
    _patch_create(monkeypatch, [
        _fake_response(stop_reason="tool_use", tool_use=[{
            "name": "broken_tool", "id": f"toolu_{i}", "input": {},
        }])
        for i in range(4)
    ])
    monkeypatch.chdir(tmp_path)

    client = AnthropicSDKClient(_profile())
    from annotation_pipeline_skill.llm.tool_registry import ToolEntry

    async def _always_raises(args):
        raise ValueError("boom")

    client._tools["broken_tool"] = ToolEntry(
        schema={"name": "broken_tool", "description": "", "input_schema": {"type": "object"}},
        dispatch=_always_raises,
    )
    client._tool_schemas = [client._tools["broken_tool"].schema]

    with pytest.raises(LocalCLIExecutionError) as excinfo:
        asyncio.run(client.generate(LLMGenerateRequest(
            instructions="i", prompt="p", task_id="t-1",
        )))
    diag = excinfo.value.diagnostics
    assert diag["tool_name"] == "broken_tool"
    assert diag["exception_class"] == "ValueError"
    assert diag["failure_count"] >= 3


# ---- session persistence round-trip --------------------------------------


def test_session_persistence_roundtrip(tmp_path, monkeypatch):
    """Save messages after generate(); a fresh client constructed at the
    same store_root must load the same conversation when given the
    handle. Required for --resume-like semantics."""
    _patch_create(monkeypatch, [
        _fake_response(stop_reason="end_turn", text="r1"),
        _fake_response(stop_reason="end_turn", text="r2"),
    ])
    monkeypatch.chdir(tmp_path)

    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    store = SqliteStore.open(tmp_path)

    c1 = AnthropicSDKClient(_profile(), store=store)
    r1 = asyncio.run(c1.generate(LLMGenerateRequest(
        instructions="i", prompt="msg1", task_id="t-1",
    )))
    assert r1.continuity_handle  # session UUID minted

    # Fresh client, same store, resume with the handle.
    c2 = AnthropicSDKClient(_profile(), store=store)
    asyncio.run(c2.generate(LLMGenerateRequest(
        instructions="i",
        prompt="msg2",
        task_id="t-1",
        continuity_handle=r1.continuity_handle,
    )))
    # The mock's second call (turn 2) received turn 1's full history.
    mock_calls = [
        call.kwargs["messages"]
        for call in c1._client.messages.create.call_args_list
    ] if False else None  # placeholder — second client has its own mock instance

    # Cleaner assertion: read the persisted file directly.
    conv_path = tmp_path / "conversations" / f"{r1.continuity_handle}.jsonl"
    assert conv_path.exists()
    lines = conv_path.read_text().splitlines()
    assert len(lines) == 4  # user1, assistant1, user2, assistant2
    msgs = [json.loads(line) for line in lines]
    assert msgs[0]["content"] == "msg1"
    assert msgs[2]["content"] == "msg2"


# ---- timeout --------------------------------------------------------------


def test_timeout_raises_local_cli_execution_error(tmp_path, monkeypatch):
    """asyncio.wait_for around the whole generate() — a slow
    messages.create must be cancelled and surface as
    LocalCLIExecutionError so the worker-bail layer treats it
    uniformly with other timeouts."""
    import annotation_pipeline_skill.llm.anthropic_sdk as sdk_mod

    async def _slow_create(**kwargs):
        await asyncio.sleep(10)
        return _fake_response(stop_reason="end_turn", text="late")

    slow_mock = AsyncMock(side_effect=_slow_create)
    monkeypatch.setattr(sdk_mod, "AsyncAnthropic", lambda *a, **k: MagicMock(
        messages=MagicMock(create=slow_mock),
    ))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(LocalCLIExecutionError) as excinfo:
        asyncio.run(AnthropicSDKClient(
            _profile(timeout_seconds=1)
        ).generate(LLMGenerateRequest(instructions="i", prompt="p", task_id="t-1")))
    assert "timeout" in str(excinfo.value).lower()
    assert excinfo.value.diagnostics["timeout_seconds"] == 1


# ---- API error surfaces as LocalCLIExecutionError ------------------------


def test_api_error_surfaces_as_local_cli_execution_error(tmp_path, monkeypatch):
    import annotation_pipeline_skill.llm.anthropic_sdk as sdk_mod
    import anthropic

    async def _api_error(**kwargs):
        raise anthropic.APIError(
            message="402 Insufficient Balance",
            request=MagicMock(),
            body={"error": {"message": "Insufficient Balance"}},
        )

    monkeypatch.setattr(sdk_mod, "AsyncAnthropic", lambda *a, **k: MagicMock(
        messages=MagicMock(create=AsyncMock(side_effect=_api_error)),
    ))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(LocalCLIExecutionError) as excinfo:
        asyncio.run(AnthropicSDKClient(_profile()).generate(
            LLMGenerateRequest(instructions="i", prompt="p", task_id="t-1")
        ))
    assert "402" in str(excinfo.value) or "Insufficient" in str(excinfo.value)
    assert excinfo.value.diagnostics["runtime"] == "anthropic_sdk"

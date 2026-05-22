"""Tests for OpenAISDKClient.

Patches AsyncOpenAI.chat.completions.create — no real HTTP needed.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from annotation_pipeline_skill.llm.base_sdk_client import LocalCLIExecutionError
from annotation_pipeline_skill.llm.client import LLMGenerateRequest
from annotation_pipeline_skill.llm.openai_sdk import OpenAISDKClient
from annotation_pipeline_skill.llm.profiles import LLMProfile


def _profile(**kw) -> LLMProfile:
    defaults = dict(
        name="test-openai",
        runtime="openai_sdk",
        model="gpt-5.4-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        timeout_seconds=30,
    )
    defaults.update(kw)
    return LLMProfile(**defaults)


def _fake_chat_response(
    *,
    finish_reason: str = "stop",
    content: str = "answer",
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
):
    """Build a mock matching openai.types.chat.ChatCompletion shape."""
    msg = MagicMock()
    msg.content = content if not tool_calls else None
    if tool_calls:
        tcs = []
        for tc in tool_calls:
            m = MagicMock()
            m.id = tc["id"]
            m.type = "function"
            m.function = MagicMock()
            m.function.name = tc["name"]
            m.function.arguments = json.dumps(tc.get("args", {}))
            tcs.append(m)
        msg.tool_calls = tcs
    else:
        msg.tool_calls = None

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = msg

    u = usage or {"prompt_tokens": 5, "completion_tokens": 3}
    usage_obj = MagicMock()
    usage_obj.prompt_tokens = u.get("prompt_tokens", 0)
    usage_obj.completion_tokens = u.get("completion_tokens", 0)
    usage_obj.model_extra = {}

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage_obj
    resp.model_dump.return_value = {"choices": [], "usage": u}
    return resp


def _patch_create(monkeypatch, responses: list[Any]):
    """Patch AsyncOpenAI.chat.completions.create. Returns captured calls."""
    import copy
    captured: list[dict] = []
    response_iter = iter(responses)

    async def _create(**kwargs):
        captured.append({k: copy.deepcopy(v) for k, v in kwargs.items()})
        try:
            return next(response_iter)
        except StopIteration:
            raise AssertionError("create called more times than expected") from None

    import annotation_pipeline_skill.llm.openai_sdk as mod

    def _make_client(*args, **kwargs):
        c = MagicMock()
        c.chat.completions.create = _create
        return c

    monkeypatch.setattr(mod, "AsyncOpenAI", _make_client)

    class _T:
        @property
        def calls(self): return captured
        @property
        def call_args(self):
            class _A: kwargs = captured[-1]
            return _A()
    return _T()


# --- tool schema conversion -------------------------------------------------

def test_anthropic_tool_schema_converted_to_openai(tmp_path, monkeypatch):
    """input_schema → parameters, wrapped in {"type": "function", "function": {...}}"""
    mock = _patch_create(monkeypatch, [_fake_chat_response()])
    monkeypatch.chdir(tmp_path)

    from annotation_pipeline_skill.llm.tool_registry import ToolEntry

    client = OpenAISDKClient(_profile())
    client._tools["my_tool"] = ToolEntry(
        schema={
            "name": "my_tool",
            "description": "does stuff",
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        },
        dispatch=AsyncMock(return_value={}),
    )
    client._tool_schemas = [client._tools["my_tool"].schema]

    asyncio.run(client.generate(LLMGenerateRequest(
        instructions="sys", prompt="go", task_id="t-1",
    )))

    tools_sent = mock.call_args.kwargs["tools"]
    assert len(tools_sent) == 1
    t = tools_sent[0]
    assert t["type"] == "function"
    assert t["function"]["name"] == "my_tool"
    assert t["function"]["description"] == "does stuff"
    assert "parameters" in t["function"]
    assert t["function"]["parameters"]["properties"]["x"]["type"] == "integer"
    assert "input_schema" not in t["function"]


# --- system message in messages array ---------------------------------------

def test_system_message_prepended_to_messages_array(tmp_path, monkeypatch):
    mock = _patch_create(monkeypatch, [_fake_chat_response(content="hi")])
    monkeypatch.chdir(tmp_path)

    asyncio.run(OpenAISDKClient(_profile()).generate(LLMGenerateRequest(
        instructions="Be concise.", prompt="hello", task_id="t-1",
    )))

    messages = mock.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "Be concise."}
    assert messages[1]["role"] == "user"


# --- stop reason mapping ----------------------------------------------------

def test_finish_reason_stop_returns_text(tmp_path, monkeypatch):
    _patch_create(monkeypatch, [_fake_chat_response(finish_reason="stop", content="result")])
    monkeypatch.chdir(tmp_path)
    r = asyncio.run(OpenAISDKClient(_profile()).generate(
        LLMGenerateRequest(instructions="s", prompt="p", task_id="t-1"),
    ))
    assert r.final_text == "result"
    assert r.diagnostics["stop_reason"] == "end_turn"


def test_finish_reason_length_flags_truncated(tmp_path, monkeypatch):
    _patch_create(monkeypatch, [_fake_chat_response(finish_reason="length", content="cut")])
    monkeypatch.chdir(tmp_path)
    r = asyncio.run(OpenAISDKClient(_profile()).generate(
        LLMGenerateRequest(instructions="s", prompt="p", task_id="t-1"),
    ))
    assert r.diagnostics["truncated"] is True
    assert r.diagnostics["stop_reason"] == "max_tokens"


def test_finish_reason_content_filter_raises_refusal(tmp_path, monkeypatch):
    _patch_create(monkeypatch, [_fake_chat_response(finish_reason="content_filter")])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LocalCLIExecutionError) as exc:
        asyncio.run(OpenAISDKClient(_profile()).generate(
            LLMGenerateRequest(instructions="s", prompt="p", task_id="t-1"),
        ))
    assert "refusal" in str(exc.value).lower()


# --- tool call round-trip ---------------------------------------------------

def test_tool_call_dispatched_result_appended(tmp_path, monkeypatch):
    mock = _patch_create(monkeypatch, [
        _fake_chat_response(
            finish_reason="tool_calls",
            tool_calls=[{"id": "call_1", "name": "echo", "args": {"v": 42}}],
        ),
        _fake_chat_response(finish_reason="stop", content="done"),
    ])
    monkeypatch.chdir(tmp_path)

    from annotation_pipeline_skill.llm.tool_registry import ToolEntry

    client = OpenAISDKClient(_profile())

    async def _echo(args):
        return {"echoed": args["v"]}

    client._tools["echo"] = ToolEntry(
        schema={"name": "echo", "description": "", "input_schema": {"type": "object"}},
        dispatch=_echo,
    )
    client._tool_schemas = [client._tools["echo"].schema]

    r = asyncio.run(client.generate(LLMGenerateRequest(
        instructions="s", prompt="p", task_id="t-1",
    )))
    assert r.final_text == "done"
    # Second call messages include role:tool
    second_msgs = mock.calls[1]["messages"]
    tool_msg = next(m for m in second_msgs if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert json.loads(tool_msg["content"]) == {"echoed": 42}


# --- reasoning_effort -------------------------------------------------------

def test_reasoning_effort_forwarded(tmp_path, monkeypatch):
    mock = _patch_create(monkeypatch, [_fake_chat_response()])
    monkeypatch.chdir(tmp_path)
    asyncio.run(OpenAISDKClient(_profile(reasoning_effort="low")).generate(
        LLMGenerateRequest(instructions="s", prompt="p", task_id="t-1"),
    ))
    assert mock.call_args.kwargs.get("reasoning_effort") == "low"


def test_no_reasoning_effort_when_not_set(tmp_path, monkeypatch):
    mock = _patch_create(monkeypatch, [_fake_chat_response()])
    monkeypatch.chdir(tmp_path)
    asyncio.run(OpenAISDKClient(_profile()).generate(
        LLMGenerateRequest(instructions="s", prompt="p", task_id="t-1"),
    ))
    # NOT_GIVEN or absent — openai SDK handles NOT_GIVEN by omitting the param
    kwargs = mock.call_args.kwargs
    assert kwargs.get("reasoning_effort") is None or str(kwargs.get("reasoning_effort")) == "NOT_GIVEN"


# --- multi-turn history -----------------------------------------------------

def test_multi_turn_messages_accumulate(tmp_path, monkeypatch):
    mock = _patch_create(monkeypatch, [
        _fake_chat_response(content="r1"),
        _fake_chat_response(content="r2"),
    ])
    monkeypatch.chdir(tmp_path)
    client = OpenAISDKClient(_profile())

    r1 = asyncio.run(client.generate(LLMGenerateRequest(
        instructions="sys", prompt="turn1", task_id="t-1",
    )))
    asyncio.run(client.generate(LLMGenerateRequest(
        instructions="sys", prompt="turn2", task_id="t-1",
        continuity_handle=r1.continuity_handle,
    )))

    second_msgs = mock.calls[1]["messages"]
    # system + user1 + assistant1 + user2
    assert second_msgs[0]["role"] == "system"
    assert second_msgs[1]["content"] == "turn1"
    assert second_msgs[2]["content"] == "r1"
    assert second_msgs[3]["content"] == "turn2"


# --- api error --------------------------------------------------------------

def test_api_error_raises_local_cli_execution_error(tmp_path, monkeypatch):
    import annotation_pipeline_skill.llm.openai_sdk as mod
    from openai import APIError

    async def _fail(**kwargs):
        raise APIError("401 Unauthorized", request=MagicMock(), body={})

    monkeypatch.setattr(mod, "AsyncOpenAI", lambda *a, **k: MagicMock(
        chat=MagicMock(completions=MagicMock(create=AsyncMock(side_effect=_fail))),
    ))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LocalCLIExecutionError) as exc:
        asyncio.run(OpenAISDKClient(_profile()).generate(
            LLMGenerateRequest(instructions="s", prompt="p", task_id="t-1"),
        ))
    assert "401" in str(exc.value) or "Unauthorized" in str(exc.value)
    assert exc.value.diagnostics["runtime"] == "openai_sdk"

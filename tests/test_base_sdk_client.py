"""Tests for BaseSdkClient via a minimal concrete stub."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from annotation_pipeline_skill.llm.base_sdk_client import (
    _ApiCallResult,
    BaseSdkClient,
    LocalCLIExecutionError,
)
from annotation_pipeline_skill.llm.client import LLMGenerateRequest
from annotation_pipeline_skill.llm.profiles import LLMProfile


def _profile(**kw) -> LLMProfile:
    defaults = dict(
        name="stub",
        runtime="openai_sdk",
        model="stub-model",
        base_url="http://localhost",
        api_key="sk-test",
        timeout_seconds=30,
    )
    defaults.update(kw)
    return LLMProfile(**defaults)


def _result(
    stop_reason: str = "end_turn",
    text: str = "ok",
    tool_calls: list | None = None,
) -> _ApiCallResult:
    tc = tool_calls or []
    if tc:
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": t["id"],
                    "type": "function",
                    "function": {"name": t["name"], "arguments": json.dumps(t["args"])},
                }
                for t in tc
            ],
        }
    else:
        assistant_msg = {"role": "assistant", "content": text}
    return _ApiCallResult(
        stop_reason=stop_reason,
        text=text,
        tool_calls=tc,
        assistant_message=assistant_msg,
        usage={"input_tokens": 1, "output_tokens": 1},
    )


class _StubClient(BaseSdkClient):
    """Concrete stub that returns pre-canned _ApiCallResults."""

    def __init__(
        self,
        profile: LLMProfile,
        responses: list[_ApiCallResult],
        *,
        store=None,
    ):
        super().__init__(profile, store=store)
        self._responses = iter(responses)
        self._calls: list[dict] = []

    async def _call_api(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        task_id: str | None = None,
    ) -> _ApiCallResult:
        self._calls.append({"system": system, "messages": list(messages), "tools": tools, "task_id": task_id})
        return next(self._responses)


# --- LocalCLIExecutionError -------------------------------------------------

def test_local_cli_execution_error_carries_diagnostics():
    err = LocalCLIExecutionError("boom", {"code": 42})
    assert str(err) == "boom"
    assert err.diagnostics == {"code": 42}


def test_local_cli_execution_error_empty_diagnostics():
    err = LocalCLIExecutionError("oops")
    assert err.diagnostics == {}


# --- _ApiCallResult ---------------------------------------------------------

def test_api_call_result_fields():
    r = _ApiCallResult(
        stop_reason="end_turn",
        text="hi",
        tool_calls=[],
        assistant_message={"role": "assistant", "content": "hi"},
        usage={"input_tokens": 3, "output_tokens": 2},
    )
    assert r.stop_reason == "end_turn"
    assert r.text == "hi"
    assert r.usage["input_tokens"] == 3


# --- JSONL session ----------------------------------------------------------

def test_session_persisted_and_reloaded(tmp_path: Path):
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    profile = _profile()

    c1 = _StubClient(profile, [_result(text="r1")], store=store)
    r1 = asyncio.run(c1.generate(LLMGenerateRequest(
        instructions="sys", prompt="hello", task_id="t-1",
    )))
    assert r1.continuity_handle is not None

    conv_path = tmp_path / "conversations" / f"{r1.continuity_handle}.jsonl"
    assert conv_path.exists()
    msgs = [json.loads(l) for l in conv_path.read_text().splitlines() if l.strip()]
    # user + assistant
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1] == {"role": "assistant", "content": "r1"}


def test_session_loaded_on_second_turn(tmp_path: Path):
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    profile = _profile()

    c = _StubClient(profile, [_result(text="r1"), _result(text="r2")], store=store)
    r1 = asyncio.run(c.generate(LLMGenerateRequest(
        instructions="sys", prompt="msg1", task_id="t-1",
    )))
    asyncio.run(c.generate(LLMGenerateRequest(
        instructions="sys", prompt="msg2", task_id="t-1",
        continuity_handle=r1.continuity_handle,
    )))

    # Second call received 3 messages: user1, assistant1, user2
    assert len(c._calls[1]["messages"]) == 3
    assert c._calls[1]["messages"][0]["content"] == "msg1"
    assert c._calls[1]["messages"][1]["content"] == "r1"
    assert c._calls[1]["messages"][2]["content"] == "msg2"


def test_disable_continuity_no_handle(tmp_path: Path):
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    store = SqliteStore.open(tmp_path)
    profile = _profile(disable_continuity=True)
    c = _StubClient(profile, [_result()], store=store)
    r = asyncio.run(c.generate(LLMGenerateRequest(
        instructions="s", prompt="p", task_id="t-1",
    )))
    assert r.continuity_handle is None
    conv_dir = tmp_path / "conversations"
    assert not conv_dir.exists() or not list(conv_dir.glob("*.jsonl"))


# --- agent loop: stop reasons -----------------------------------------------

def test_end_turn_returns_text(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _StubClient(_profile(), [_result(stop_reason="end_turn", text="final")])
    r = asyncio.run(c.generate(LLMGenerateRequest(
        instructions="s", prompt="p", task_id="t-1",
    )))
    assert r.final_text == "final"
    assert r.diagnostics["stop_reason"] == "end_turn"


def test_max_tokens_returns_partial_with_truncated_flag(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _StubClient(_profile(), [_result(stop_reason="max_tokens", text="partial")])
    r = asyncio.run(c.generate(LLMGenerateRequest(
        instructions="s", prompt="p", task_id="t-1",
    )))
    assert r.final_text == "partial"
    assert r.diagnostics["truncated"] is True


def test_refusal_raises(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _StubClient(_profile(), [_result(stop_reason="refusal", text="")])
    with pytest.raises(LocalCLIExecutionError) as exc:
        asyncio.run(c.generate(LLMGenerateRequest(
            instructions="s", prompt="p", task_id="t-1",
        )))
    assert "refusal" in str(exc.value).lower()


def test_unknown_stop_reason_raises(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _StubClient(_profile(), [_result(stop_reason="future_reason")])
    with pytest.raises(LocalCLIExecutionError) as exc:
        asyncio.run(c.generate(LLMGenerateRequest(
            instructions="s", prompt="p", task_id="t-1",
        )))
    assert "unknown stop_reason" in str(exc.value)


# --- agent loop: tool dispatch ----------------------------------------------

def test_tool_dispatch_continuation(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from annotation_pipeline_skill.llm.tool_registry import ToolEntry

    c = _StubClient(_profile(), [
        _result(stop_reason="tool_calls", tool_calls=[
            {"id": "call_1", "name": "echo_tool", "args": {"x": 7}},
        ]),
        _result(stop_reason="end_turn", text="done"),
    ])

    async def _echo(args):
        return {"echoed": args["x"]}

    c._tools["echo_tool"] = ToolEntry(
        schema={"name": "echo_tool", "description": "", "input_schema": {"type": "object"}},
        dispatch=_echo,
    )

    r = asyncio.run(c.generate(LLMGenerateRequest(
        instructions="s", prompt="p", task_id="t-1",
    )))
    assert r.final_text == "done"
    # Second _call_api call received the tool result message
    second_messages = c._calls[1]["messages"]
    tool_result = next(m for m in second_messages if m.get("role") == "tool")
    assert tool_result["tool_call_id"] == "call_1"
    assert json.loads(tool_result["content"]) == {"echoed": 7}


def test_tool_failure_breaker(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from annotation_pipeline_skill.llm.tool_registry import ToolEntry

    c = _StubClient(_profile(), [
        _result(stop_reason="tool_calls", tool_calls=[
            {"id": f"call_{i}", "name": "broken", "args": {}},
        ])
        for i in range(4)
    ])

    async def _boom(args):
        raise ValueError("always fails")

    c._tools["broken"] = ToolEntry(
        schema={"name": "broken", "description": "", "input_schema": {"type": "object"}},
        dispatch=_boom,
    )

    with pytest.raises(LocalCLIExecutionError) as exc:
        asyncio.run(c.generate(LLMGenerateRequest(
            instructions="s", prompt="p", task_id="t-1",
        )))
    assert exc.value.diagnostics["tool_name"] == "broken"
    assert exc.value.diagnostics["failure_count"] >= 3


# --- task_id threading -------------------------------------------------------

def test_task_id_threaded_to_call_api(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _StubClient(_profile(), [_result()])
    asyncio.run(c.generate(LLMGenerateRequest(
        instructions="s", prompt="p", task_id="my-task-123",
    )))
    assert c._calls[0]["task_id"] == "my-task-123"


# --- timeout ----------------------------------------------------------------

def test_timeout_raises(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _SlowClient(_StubClient):
        async def _call_api(self, system, messages, tools, task_id=None):
            await asyncio.sleep(10)
            return _result()

    c = _SlowClient(_profile(timeout_seconds=1), [])
    with pytest.raises(LocalCLIExecutionError) as exc:
        asyncio.run(c.generate(LLMGenerateRequest(
            instructions="s", prompt="p", task_id="t-1",
        )))
    assert "timeout" in str(exc.value).lower()

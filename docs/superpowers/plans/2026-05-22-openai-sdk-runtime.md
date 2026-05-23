# OpenAI SDK Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `openai_sdk` runtime that routes DeepSeek/GLM/MiniMax/qwen profiles through the OpenAI Chat Completions API, while refactoring `anthropic_sdk` into a thin adapter over a shared base.

**Architecture:** Extract agent loop + JSONL session management into `BaseSdkClient`. `OpenAISDKClient` is a thin adapter (tool schema conversion + response parsing). `AnthropicSDKClient` is refactored to convert OpenAI-format messages ↔ Anthropic wire format on each call, storing history in the canonical OpenAI format. `LocalCLIClient` gains a fourth dispatch branch.

**Tech Stack:** `openai` Python SDK (`AsyncOpenAI`), `anthropic` Python SDK (`AsyncAnthropic`), `pytest`, `unittest.mock`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `annotation_pipeline_skill/llm/base_sdk_client.py` | **Create** | `_ApiCallResult`, `LocalCLIExecutionError`, `BaseSdkClient` abstract class: JSONL session, agent loop, tool dispatch |
| `annotation_pipeline_skill/llm/openai_sdk.py` | **Create** | `OpenAISDKClient(BaseSdkClient)`: tool schema conversion, `chat.completions.create()`, response→`_ApiCallResult` |
| `annotation_pipeline_skill/llm/anthropic_sdk.py` | **Rewrite** | `AnthropicSDKClient(BaseSdkClient)`: OpenAI↔Anthropic message conversion, `messages.create()`, re-export `LocalCLIExecutionError` |
| `annotation_pipeline_skill/llm/profiles.py` | **Modify** | Add `"openai_sdk"` to `Runtime` literal |
| `annotation_pipeline_skill/llm/local_cli.py` | **Modify** | Add `openai_sdk` branch in `LocalCLIClient.__init__` and `generate()` |
| `projects/llm_profiles.yaml` | **Modify** | Switch `deepseek_*`, `glm_*`, `minimax_*`, `qwen*` to `runtime: openai_sdk` + correct `base_url` |
| `tests/test_base_sdk_client.py` | **Create** | Tests for session round-trip, agent loop, tool dispatch, failure breaker via concrete stub |
| `tests/test_openai_sdk_client.py` | **Create** | Tests for tool schema conversion, stop reason mapping, multi-turn, `reasoning_effort` |
| `tests/test_anthropic_sdk_client.py` | **Modify** | Fix imports after `LocalCLIExecutionError` moves; update message-format assertions |

---

## Task 1: `base_sdk_client.py` — shared types, session, agent loop

**Files:**
- Create: `annotation_pipeline_skill/llm/base_sdk_client.py`
- Create: `tests/test_base_sdk_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_base_sdk_client.py
"""Tests for BaseSdkClient via a minimal concrete stub."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
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

    def __init__(self, profile: LLMProfile, responses: list[_ApiCallResult]):
        super().__init__(profile)
        self._responses = iter(responses)
        self._calls: list[dict] = []

    async def _call_api(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> _ApiCallResult:
        self._calls.append({"system": system, "messages": list(messages), "tools": tools})
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

    c1 = _StubClient(profile, [_result(text="r1")])
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

    c = _StubClient(profile, [_result(text="r1"), _result(text="r2")])
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
    profile = _profile(disable_continuity=True)
    c = _StubClient(profile, [_result()])
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


# --- timeout ----------------------------------------------------------------

def test_timeout_raises(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _SlowClient(_StubClient):
        async def _call_api(self, system, messages, tools):
            await asyncio.sleep(10)
            return _result()

    c = _SlowClient(_profile(timeout_seconds=1), [])
    with pytest.raises(LocalCLIExecutionError) as exc:
        asyncio.run(c.generate(LLMGenerateRequest(
            instructions="s", prompt="p", task_id="t-1",
        )))
    assert "timeout" in str(exc.value).lower()
```

- [ ] **Step 2: Run tests — expect ImportError (module doesn't exist yet)**

```bash
python -m pytest tests/test_base_sdk_client.py -x -q 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'annotation_pipeline_skill.llm.base_sdk_client'`

- [ ] **Step 3: Implement `base_sdk_client.py`**

```python
# annotation_pipeline_skill/llm/base_sdk_client.py
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
from dataclasses import dataclass, field
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
            )
            _add_usage(usage_acc, result.usage)
            messages.append(result.assistant_message)

            if result.stop_reason in ("end_turn",):
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
    ) -> _ApiCallResult: ...


# ---- helpers ---------------------------------------------------------------

def _items_to_text(items: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("content", item)) for item in items)


def _add_usage(acc: dict[str, int], usage: dict[str, int]) -> None:
    for k, v in usage.items():
        if isinstance(v, int):
            acc[k] = acc.get(k, 0) + v
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/test_base_sdk_client.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/llm/base_sdk_client.py tests/test_base_sdk_client.py
git commit -m "feat(llm): add BaseSdkClient with shared agent loop and JSONL session management"
```

---

## Task 2: `openai_sdk.py` — OpenAI Chat Completions adapter

**Files:**
- Create: `annotation_pipeline_skill/llm/openai_sdk.py`
- Create: `tests/test_openai_sdk_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_openai_sdk_client.py
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
python -m pytest tests/test_openai_sdk_client.py -x -q 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'annotation_pipeline_skill.llm.openai_sdk'`

- [ ] **Step 3: Implement `openai_sdk.py`**

```python
# annotation_pipeline_skill/llm/openai_sdk.py
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
    _add_usage,
)
from annotation_pipeline_skill.llm.client import LLMGenerateRequest, LLMGenerateResult
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
    ) -> _ApiCallResult:
        openai_tools = [_to_openai_tool(t) for t in tools] if tools else None
        full_messages = ([{"role": "system", "content": system}] if system else []) + messages

        extra_headers: dict[str, str] = {}

        try:
            kwargs: dict[str, Any] = dict(
                model=self.profile.model,
                messages=full_messages,
                max_tokens=32000,
            )
            if openai_tools:
                kwargs["tools"] = openai_tools
            if self.profile.reasoning_effort:
                kwargs["reasoning_effort"] = self.profile.reasoning_effort
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_openai_sdk_client.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/llm/openai_sdk.py tests/test_openai_sdk_client.py
git commit -m "feat(llm): add OpenAISDKClient — OpenAI Chat Completions adapter over BaseSdkClient"
```

---

## Task 3: Refactor `anthropic_sdk.py` as thin Anthropic adapter

**Files:**
- Rewrite: `annotation_pipeline_skill/llm/anthropic_sdk.py`
- Modify: `tests/test_anthropic_sdk_client.py`

- [ ] **Step 1: Add message-conversion tests to the existing test file**

Add these tests to the END of `tests/test_anthropic_sdk_client.py`:

```python
# --- message format conversion (new after base_sdk_client refactor) ---------

from annotation_pipeline_skill.llm.anthropic_sdk import (
    _openai_to_anthropic_messages,
    _anthropic_content_to_openai_assistant,
)


def test_openai_to_anthropic_extracts_system():
    system, msgs = _openai_to_anthropic_messages([
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "hello"},
    ])
    assert system == "Be helpful."
    assert len(msgs) == 1
    assert msgs[0] == {"role": "user", "content": "hello"}


def test_openai_to_anthropic_converts_tool_calls():
    _, msgs = _openai_to_anthropic_messages([
        {"role": "user", "content": "annotate"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "check", "arguments": '{"task_id": "t-1"}'},
            }],
        },
    ])
    assert msgs[1]["role"] == "assistant"
    content = msgs[1]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "tool_use"
    assert content[0]["id"] == "call_1"
    assert content[0]["name"] == "check"
    assert content[0]["input"] == {"task_id": "t-1"}


def test_openai_to_anthropic_merges_tool_results():
    """Consecutive role:tool messages → one role:user with list of tool_result blocks."""
    _, msgs = _openai_to_anthropic_messages([
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": 1}'},
        {"role": "tool", "tool_call_id": "c2", "content": '{"ok": 2}'},
    ])
    # user + assistant(tool_use) + user(tool_result x2)
    assert len(msgs) == 3
    result_msg = msgs[2]
    assert result_msg["role"] == "user"
    assert len(result_msg["content"]) == 2
    assert result_msg["content"][0]["type"] == "tool_result"
    assert result_msg["content"][0]["tool_use_id"] == "c1"
    assert result_msg["content"][1]["tool_use_id"] == "c2"


def test_anthropic_text_content_to_openai():
    block = MagicMock()
    block.type = "text"
    block.text = "hello"
    msg = _anthropic_content_to_openai_assistant([block])
    assert msg == {"role": "assistant", "content": "hello"}


def test_anthropic_tool_use_content_to_openai():
    block = MagicMock()
    block.type = "tool_use"
    block.id = "toolu_1"
    block.name = "check"
    block.input = {"x": 1}
    msg = _anthropic_content_to_openai_assistant([block])
    assert msg["role"] == "assistant"
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "toolu_1"
    assert tc["function"]["name"] == "check"
    assert json.loads(tc["function"]["arguments"]) == {"x": 1}
```

Also update the top import block of `test_anthropic_sdk_client.py` — change:
```python
from annotation_pipeline_skill.llm.anthropic_sdk import (
    AnthropicSDKClient,
    LocalCLIExecutionError,
)
```
to:
```python
from annotation_pipeline_skill.llm.anthropic_sdk import (
    AnthropicSDKClient,
    LocalCLIExecutionError,
    _openai_to_anthropic_messages,
    _anthropic_content_to_openai_assistant,
)
```

Also update the `test_session_persistence_roundtrip` assertion at line ~421 — the persisted messages are now in OpenAI format, so `msgs[1]` is `{"role": "assistant", "content": "r1"}` not a list of blocks:

```python
    # After refactor: JSONL stores OpenAI-format messages.
    msgs = [json.loads(line) for line in lines]
    assert msgs[0]["content"] == "msg1"        # user turn 1
    assert msgs[1]["role"] == "assistant"       # assistant turn 1 (OpenAI format)
    assert msgs[1]["content"] == "r1"
    assert msgs[2]["content"] == "msg2"        # user turn 2
```

Also update `_patch_create` — the mock now needs to return responses matching what `AnthropicSDKClient._call_api()` calls (still `messages.create`), so the mock stays the same. But `call_args_list` in `test_turn_2_messages_extend_turn_1_byte_stable` now receives Anthropic-format messages (because `_call_api` converts before calling). Update those assertions:

```python
    # turn1_messages sent to Anthropic API — plain user message unchanged
    turn1_messages = mock.call_args_list[0].kwargs["messages"]
    assert len(turn1_messages) == 1
    assert turn1_messages[0]["role"] == "user"
    assert turn1_messages[0]["content"] == "user message 1"
    # (Anthropic format: user content is plain string)

    turn2_messages = mock.call_args_list[1].kwargs["messages"]
    assert len(turn2_messages) == 3  # user_1, assistant_1, user_2
    assert turn2_messages[0]["content"] == "user message 1"
    assert turn2_messages[1]["role"] == "assistant"
    # assistant content in Anthropic format is a list
    assert isinstance(turn2_messages[1]["content"], list)
    assert turn2_messages[2]["role"] == "user"
    assert turn2_messages[2]["content"] == "user message 2"
```

- [ ] **Step 2: Run the new conversion tests — expect ImportError**

```bash
python -m pytest tests/test_anthropic_sdk_client.py -k "openai_to_anthropic or anthropic_text or anthropic_tool" -x -q 2>&1 | head -10
```

Expected: ImportError on `_openai_to_anthropic_messages`.

- [ ] **Step 3: Rewrite `anthropic_sdk.py`**

```python
# annotation_pipeline_skill/llm/anthropic_sdk.py
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
    ) -> _ApiCallResult:
        _, anthropic_messages = _openai_to_anthropic_messages(messages)

        # pause_turn is Anthropic extended-thinking: loop internally.
        for _pause_iter in range(10):
            try:
                response = await self._client.messages.create(
                    model=self.profile.model,
                    system=system or "",
                    messages=anthropic_messages,
                    tools=tools or anthropic.NOT_GIVEN,
                    max_tokens=32000,
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
    for field in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        v = getattr(usage, field, None)
        if isinstance(v, int):
            result[field] = v
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
```

- [ ] **Step 4: Run all anthropic SDK tests**

```bash
python -m pytest tests/test_anthropic_sdk_client.py -v
```

Expected: all green. Fix any assertion mismatches from message format changes before proceeding.

- [ ] **Step 5: Run base SDK tests to confirm nothing regressed**

```bash
python -m pytest tests/test_base_sdk_client.py tests/test_openai_sdk_client.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add annotation_pipeline_skill/llm/anthropic_sdk.py tests/test_anthropic_sdk_client.py
git commit -m "refactor(llm): AnthropicSDKClient as thin adapter over BaseSdkClient; store history in OpenAI format"
```

---

## Task 4: Wire up `openai_sdk` runtime in `profiles.py` and `local_cli.py`

**Files:**
- Modify: `annotation_pipeline_skill/llm/profiles.py:12`
- Modify: `annotation_pipeline_skill/llm/local_cli.py`
- Modify: `tests/test_llm_profiles.py` (add `openai_sdk` profile test)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_llm_profiles.py`:

```python
def test_openai_sdk_profile_is_valid_runtime():
    from annotation_pipeline_skill.llm.profiles import _parse_profile
    profile = _parse_profile("qwen-test", {
        "runtime": "openai_sdk",
        "model": "qwen3.6-35b-a3b",
        "base_url": "http://127.0.0.1:8900",
        "api_key": "sk-local",
    })
    assert profile.runtime == "openai_sdk"
    assert profile.model == "qwen3.6-35b-a3b"
```

Add to `tests/test_local_cli_client.py`:

```python
def test_local_cli_client_routes_openai_sdk(tmp_path, monkeypatch):
    """LocalCLIClient with runtime=openai_sdk dispatches to OpenAISDKClient."""
    from annotation_pipeline_skill.llm.local_cli import LocalCLIClient
    from annotation_pipeline_skill.llm.profiles import LLMProfile
    from annotation_pipeline_skill.llm.client import LLMGenerateRequest

    profile = LLMProfile(
        name="qwen-test",
        runtime="openai_sdk",
        model="qwen3.6-35b-a3b",
        base_url="http://127.0.0.1:9999",
        api_key="sk-test",
        timeout_seconds=30,
    )
    client = LocalCLIClient(profile)
    assert client._openai_impl is not None

    import annotation_pipeline_skill.llm.openai_sdk as mod
    from unittest.mock import AsyncMock, MagicMock
    import json

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(
        finish_reason="stop",
        message=MagicMock(content="routed ok", tool_calls=None),
    )]
    fake_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, model_extra={})

    monkeypatch.setattr(mod, "AsyncOpenAI", lambda *a, **k: MagicMock(
        chat=MagicMock(completions=MagicMock(create=AsyncMock(return_value=fake_resp))),
    ))
    monkeypatch.chdir(tmp_path)

    import asyncio
    r = asyncio.run(client.generate(LLMGenerateRequest(
        instructions="s", prompt="p", task_id="t-1",
    )))
    assert r.final_text == "routed ok"
    assert r.runtime == "openai_sdk"
```

- [ ] **Step 2: Run new tests — expect failures**

```bash
python -m pytest tests/test_llm_profiles.py::test_openai_sdk_profile_is_valid_runtime tests/test_local_cli_client.py::test_local_cli_client_routes_openai_sdk -v 2>&1 | tail -15
```

Expected: `ProfileValidationError: profile qwen-test runtime must be 'claude_cli', 'codex_cli', or 'anthropic_sdk'`

- [ ] **Step 3: Update `profiles.py` — add `"openai_sdk"` to Runtime**

In `annotation_pipeline_skill/llm/profiles.py`, change line 12:

```python
# Before:
Runtime = Literal["claude_cli", "codex_cli", "anthropic_sdk"]

# After:
Runtime = Literal["claude_cli", "codex_cli", "anthropic_sdk", "openai_sdk"]
```

And in `_parse_profile` at line 133, update the validation check:

```python
# Before:
if runtime not in {"claude_cli", "codex_cli", "anthropic_sdk"}:
    raise ProfileValidationError(
        f"profile {name} runtime must be 'claude_cli', 'codex_cli', or 'anthropic_sdk', "
        f"got: {runtime!r}"
    )

# After:
if runtime not in {"claude_cli", "codex_cli", "anthropic_sdk", "openai_sdk"}:
    raise ProfileValidationError(
        f"profile {name} runtime must be 'claude_cli', 'codex_cli', 'anthropic_sdk', "
        f"or 'openai_sdk', got: {runtime!r}"
    )
```

Also in `load_llm_registry`, the `system_mcp_servers` block propagates to `claude_cli` and `anthropic_sdk` profiles. Add `openai_sdk` to the set:

```python
# Before:
if profile.runtime in {"claude_cli", "anthropic_sdk"} else profile

# After:
if profile.runtime in {"claude_cli", "anthropic_sdk", "openai_sdk"} else profile
```

- [ ] **Step 4: Update `local_cli.py` — add `openai_sdk` dispatch**

In `LocalCLIClient.__init__` in `annotation_pipeline_skill/llm/local_cli.py`, after the existing `anthropic_sdk` block:

```python
        if profile.runtime == "anthropic_sdk":
            from annotation_pipeline_skill.llm.anthropic_sdk import (
                AnthropicSDKClient,
            )
            self._anthropic_impl = AnthropicSDKClient(
                profile, store=store, project_id=project_id,
            )
        # ADD AFTER:
        elif profile.runtime == "openai_sdk":
            from annotation_pipeline_skill.llm.openai_sdk import OpenAISDKClient
            self._openai_impl: object | None = OpenAISDKClient(
                profile, store=store, project_id=project_id,
            )
```

Also add `self._openai_impl: object | None = None` in the `__init__` preamble alongside `self._anthropic_impl`.

In `generate()`, add the dispatch branch:

```python
    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        if self.profile.runtime == "anthropic_sdk":
            assert self._anthropic_impl is not None
            return await self._anthropic_impl.generate(request)
        # ADD:
        if self.profile.runtime == "openai_sdk":
            assert self._openai_impl is not None
            return await self._openai_impl.generate(request)
        if self.profile.runtime == "codex_cli":
            ...
```

- [ ] **Step 5: Run new tests**

```bash
python -m pytest tests/test_llm_profiles.py::test_openai_sdk_profile_is_valid_runtime tests/test_local_cli_client.py::test_local_cli_client_routes_openai_sdk -v
```

Expected: both pass.

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
python -m pytest tests/test_llm_profiles.py tests/test_llm_profiles_mcp.py tests/test_llm_profiles_resolution.py tests/test_local_cli_client.py tests/test_anthropic_sdk_client.py tests/test_base_sdk_client.py tests/test_openai_sdk_client.py -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add annotation_pipeline_skill/llm/profiles.py annotation_pipeline_skill/llm/local_cli.py tests/test_llm_profiles.py tests/test_local_cli_client.py
git commit -m "feat(llm): register openai_sdk runtime in profiles + LocalCLIClient dispatch"
```

---

## Task 5: Update `llm_profiles.yaml` — switch third-party profiles to `openai_sdk`

**Files:**
- Modify: `projects/llm_profiles.yaml`

- [ ] **Step 1: Edit `projects/llm_profiles.yaml`**

Change the following profiles (leave `claude_*` and `codex_*` unchanged):

```yaml
  deepseek_flash:
    runtime: openai_sdk          # was: anthropic_sdk
    model: deepseek-v4-flash
    api_key_env: DEEPSEEK_API_KEY
    base_url: https://api.deepseek.com   # was: https://api.deepseek.com/anthropic
    reasoning_effort: low
    timeout_seconds: 900
    disable_continuity: true

  deepseek_pro:
    runtime: openai_sdk          # was: anthropic_sdk
    model: deepseek-v4-pro
    api_key_env: DEEPSEEK_API_KEY
    base_url: https://api.deepseek.com   # was: https://api.deepseek.com/anthropic
    reasoning_effort: low
    timeout_seconds: 900
    disable_continuity: true
    api_key: sk-be84988a7682453aaf4f96331a77ea3a

  glm_46:
    runtime: openai_sdk          # was: anthropic_sdk
    model: glm-4.6
    api_key_env:
    - GLM_CODING_API_KEY
    - BIGMODEL_MCP_API_KEY
    base_url: https://open.bigmodel.cn/api/paas/v4   # was: https://open.bigmodel.cn/api/anthropic
    timeout_seconds: 900
    disable_continuity: true
    api_key: 6add81a3696dbff37b3ebf80dc216d3a.VoTFesrSd1co27Qh

  glm_51:
    runtime: openai_sdk          # was: anthropic_sdk
    model: glm-5.1
    api_key_env:
    - GLM_CODING_API_KEY
    - BIGMODEL_MCP_API_KEY
    base_url: https://open.bigmodel.cn/api/paas/v4   # was: https://open.bigmodel.cn/api/anthropic
    timeout_seconds: 900
    disable_continuity: true
    api_key: 6add81a3696dbff37b3ebf80dc216d3a.VoTFesrSd1co27Qh

  minimax_2.7:
    runtime: openai_sdk          # was: anthropic_sdk
    model: MiniMax-M2.7
    api_key_env: MINIMAX_API_KEY
    base_url: https://api.minimax.io/v1   # was: https://api.minimaxi.com/anthropic
    timeout_seconds: 900
    disable_continuity: true
    api_key: sk-cp-veUCgN4Y_3TB9gEKgdnhRIkbGv8BC_2Fgs_CohKWVqdo6DiN8rp_OtOBwefXpYqxPl3lvBh2gW7LXPlvbRzOk_3J6sE262alMr2VfoGb5LqHb-Q7GwisaM0

  qwen3.6-35b-a3b:
    runtime: openai_sdk          # was: anthropic_sdk
    model: qwen3.6-35b-a3b
    api_key_env: LOCAL_GATEWAY_API_KEY
    base_url: http://127.0.0.1:8900   # unchanged
    reasoning_effort: low
    timeout_seconds: 900
    no_progress_timeout_seconds: 180
    api_key: sk-local-gateway

  qwen3.6-27b:
    runtime: openai_sdk          # was: anthropic_sdk
    model: qwen3.6-27b
    api_key_env: LOCAL_GATEWAY_API_KEY
    base_url: http://127.0.0.1:8900   # unchanged
    reasoning_effort: low
    timeout_seconds: 900
    no_progress_timeout_seconds: 180
    api_key: sk-local-gateway
```

- [ ] **Step 2: Verify registry loads without errors**

```bash
python -c "
from annotation_pipeline_skill.llm.profiles import load_llm_registry
r = load_llm_registry('projects/llm_profiles.yaml')
for t, p in r.targets.items():
    prof = r.resolve(t)
    print(f'{t:20s} -> {p:20s} runtime={prof.runtime}')
"
```

Expected output (no exceptions):
```
annotation            -> qwen3.6-35b-a3b     runtime=openai_sdk
arbiter               -> codex_5.5           runtime=codex_cli
arbiter_secondary     -> claude_sonnet        runtime=anthropic_sdk
coordinator           -> glm_46              runtime=openai_sdk
fallback              -> codex_5.4_mini       runtime=codex_cli
qc                    -> minimax_2.7          runtime=openai_sdk
```

- [ ] **Step 3: Run profile tests**

```bash
python -m pytest tests/test_llm_profiles.py tests/test_llm_profiles_mcp.py tests/test_llm_profiles_resolution.py -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add projects/llm_profiles.yaml
git commit -m "chore(profiles): switch deepseek/glm/minimax/qwen to openai_sdk runtime"
```

---

## Final verification

- [ ] **Run the complete LLM test suite**

```bash
python -m pytest tests/test_base_sdk_client.py tests/test_openai_sdk_client.py tests/test_anthropic_sdk_client.py tests/test_local_cli_client.py tests/test_llm_profiles.py tests/test_llm_profiles_mcp.py tests/test_llm_profiles_resolution.py -v
```

Expected: all green, no regressions in existing tests.

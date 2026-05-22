# OpenAI SDK Runtime Design

**Date**: 2026-05-22  
**Status**: Approved

## Background

The pipeline currently has three LLM runtimes:

| Runtime | Mechanism | Profiles |
|---------|-----------|---------|
| `codex_cli` | `codex exec` subprocess, ChatGPT OAuth | `codex_5.4_mini`, `codex_5.5` |
| `anthropic_sdk` | Anthropic Python SDK, in-process | `claude_*`, `deepseek_*`, `glm_*`, `minimax_*`, `qwen*` |
| `claude_cli` | `claude --bare -p` subprocess (unused) | — |

The non-Claude profiles currently use the Anthropic-compatible endpoints exposed by DeepSeek, GLM, and MiniMax. Switching these to the native OpenAI Chat Completions format (which all three providers also expose) is preferable because:

1. OpenAI Chat Completions is the canonical format: both `codex_cli` and the new `openai_sdk` runtime use it natively, making it the majority format.
2. The Anthropic-compatible endpoints of third-party providers are secondary/less-tested surfaces.
3. Unifies the internal message storage format across all SDK-based runtimes.

## Research Findings

### Provider API support

| Provider | Chat Completions | Responses API (`/v1/responses`) | Anthropic compat |
|----------|:---:|:---:|:---:|
| DeepSeek | ✅ `api.deepseek.com` | ❌ | ✅ |
| MiniMax | ✅ `api.minimax.io/v1` | ❌ | ✅ |
| GLM (ZhipuAI) | ✅ `open.bigmodel.cn/api/paas/v4` | ❌ | ✅ |
| Local qwen gateway | ✅ `127.0.0.1:8900` | implementation-specific | — |

None of the third-party providers support the OpenAI Responses API (`previous_response_id`). LiteLLM proxy does support it as a translation layer, but as of 2026-04-21 there is an open bug (issue #26167) causing multi-turn tool calls to fail on bridged backends with no fix ETA. Since the annotation pipeline relies heavily on tool calls (`check_annotation_draft`, `lookup_row_text`, `check_past_experience`), this path is not viable.

### Codex CLI auth

`codex` CLI in `auth_mode: "chatgpt"` uses a proprietary ChatGPT backend API with `AgentAssertion` tokens. These tokens lack the `model.request` scope and cannot be used with the OpenAI Python SDK. The `codex_cli` runtime must remain.

### Anthropic subscription auth

Claude Max OAuth tokens (`sk-ant-oat01-...`) with `rateLimitTier: default_claude_max_20x` work with the Anthropic Python SDK via the `auth_token` parameter. The `anthropic_sdk` runtime is correct for Claude profiles. Observed 429 errors are rate limiting under concurrent load, not an auth failure.

## Decision

**Strategy**: Canonical internal message format is OpenAI Chat Completions. Introduce a new `openai_sdk` runtime for third-party providers. Refactor `anthropic_sdk` into a thin adapter. Three runtimes remain post-implementation.

| Runtime | Post-implementation status | Profiles |
|---------|---------------------------|---------|
| `codex_cli` | Unchanged | `codex_5.4_mini`, `codex_5.5` |
| `anthropic_sdk` | Refactored as thin adapter | `claude_sonnet`, `claude_haiku` |
| `openai_sdk` | New | `deepseek_*`, `glm_*`, `minimax_2.7`, `qwen3.6-*` |
| `claude_cli` | Remains unused | — |

## Architecture

```
llm/
  base_sdk_client.py    ← new: shared agent loop + JSONL session
  openai_sdk.py         ← new: OpenAI Chat Completions adapter
  anthropic_sdk.py      ← refactored: Anthropic Messages adapter
  local_cli.py          ← minor: add openai_sdk dispatch branch
  profiles.py           ← minor: add "openai_sdk" to Runtime literal
  tool_registry.py      ← unchanged
  client.py             ← unchanged
```

## `base_sdk_client.py`

Contains all SDK-agnostic logic:

**JSONL session management**  
`_load_or_init_session(session_uuid)` → `(messages, uuid)`  
`_save_session(session_uuid, messages)` — atomic write via tmp+rename with `fsync`  
Storage: `<store_root>/conversations/<uuid>.jsonl`, one JSON object per line.  
Format: **OpenAI Chat Completions messages** (`role: user/assistant/tool/system`).

**Agent loop** (`_run_agent_loop`)  
Iterates up to `_MAX_AGENT_ITERATIONS = 10`. Per iteration:
1. Calls `_call_api(system, messages, tools)` → `_ApiCallResult`
2. Appends `result.assistant_message` to `messages`
3. On `stop_reason == "tool_calls"`: dispatches tools, appends tool result messages, continues
4. On `stop_reason == "end_turn"`: returns final text
5. On `stop_reason == "max_tokens"`: returns partial with `truncated=True` in diagnostics
6. On `stop_reason == "refusal"`: raises `LocalCLIExecutionError`
7. Consecutive same-tool failure breaker: raises after `_TOOL_FAILURE_BREAKER = 3` repeats

**Tool dispatch** — unchanged from current `anthropic_sdk.py`, format-agnostic.

**Abstract interface**:

```python
@dataclass
class _ApiCallResult:
    stop_reason: Literal["end_turn", "tool_calls", "max_tokens", "refusal", "unknown"]
    text: str
    tool_calls: list[dict]       # [{"id": "...", "name": "...", "args": {...}}]
    assistant_message: dict      # OpenAI-format, ready to append to messages list
    usage: dict

class BaseSdkClient(ABC):
    @abstractmethod
    async def _call_api(
        self,
        system: str,
        messages: list[dict],    # OpenAI Chat Completions format
        tools: list[dict],       # Anthropic registry format (input_schema)
    ) -> _ApiCallResult: ...
```

## `openai_sdk.py` — OpenAI Adapter

`_call_api()` performs three steps:

**1. Tool schema conversion** (Anthropic registry format → OpenAI):
```
{"name": N, "description": D, "input_schema": S}
→ {"type": "function", "function": {"name": N, "description": D, "parameters": S}}
```

**2. API call**:
```python
await self._client.chat.completions.create(
    model=profile.model,
    messages=[{"role": "system", "content": system}, *messages],
    tools=openai_tools or NOT_GIVEN,
    reasoning_effort=profile.reasoning_effort or NOT_GIVEN,
    max_tokens=request.max_output_tokens or 32000,
    extra_headers={"x-task-id": task_id} if task_id else None,
)
```

**3. Response → `_ApiCallResult`**:

| OpenAI `finish_reason` | canonical `stop_reason` |
|------------------------|------------------------|
| `stop` | `end_turn` |
| `tool_calls` | `tool_calls` |
| `length` | `max_tokens` |
| `content_filter` | `refusal` |

`tool_calls`: extracted from `message.tool_calls[]`, `arguments` JSON-decoded.  
`assistant_message`: OpenAI format stored verbatim.

## `anthropic_sdk.py` — Anthropic Adapter (Refactored)

`_call_api()` performs three steps:

**1. Messages conversion** (OpenAI → Anthropic):
- `role: system` extracted as standalone `system` parameter, removed from list
- `role: assistant` with `tool_calls` → `content: [{"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": json.loads(tc.function.arguments)}]`
- Consecutive `role: tool` messages → merged into one `role: user` message: `content: [{"type": "tool_result", "tool_use_id": tc.tool_call_id, "content": tc.content}, ...]`
- Plain `role: assistant` text → `content: [{"type": "text", "text": "..."}]`

Tool schemas are passed unchanged (already in Anthropic format).

**2. API call**: `messages.create()` with converted messages and system.

**3. Response → `_ApiCallResult`**:

| Anthropic `stop_reason` | canonical `stop_reason` |
|------------------------|------------------------|
| `end_turn` | `end_turn` |
| `tool_use` | `tool_calls` |
| `max_tokens` | `max_tokens` |
| `refusal` | `refusal` |
| `pause_turn` | loop continues (no result returned) |

`tool_calls`: extracted from `content[]` where `type == "tool_use"`.  
`assistant_message`: converted back to OpenAI format for JSONL storage:
- Text blocks → `{"role": "assistant", "content": "..."}`
- Tool use blocks → `{"role": "assistant", "content": null, "tool_calls": [...]}`

## `local_cli.py` Changes

```python
# __init__:
elif profile.runtime == "openai_sdk":
    from annotation_pipeline_skill.llm.openai_sdk import OpenAISDKClient
    self._openai_impl = OpenAISDKClient(profile, store=store, project_id=project_id)

# generate():
if self.profile.runtime == "openai_sdk":
    return await self._openai_impl.generate(request)
```

`LocalCLIClient` remains the unified entry point and public API.

## Profile Changes (`llm_profiles.yaml`)

| Profile | `runtime` | `base_url` |
|---------|-----------|------------|
| `deepseek_flash` | `openai_sdk` | `https://api.deepseek.com` |
| `deepseek_pro` | `openai_sdk` | `https://api.deepseek.com` |
| `glm_46` | `openai_sdk` | `https://open.bigmodel.cn/api/paas/v4` |
| `glm_51` | `openai_sdk` | `https://open.bigmodel.cn/api/paas/v4` |
| `minimax_2.7` | `openai_sdk` | `https://api.minimax.io/v1` |
| `qwen3.6-35b-a3b` | `openai_sdk` | `http://127.0.0.1:8900` (unchanged) |
| `qwen3.6-27b` | `openai_sdk` | `http://127.0.0.1:8900` (unchanged) |
| `claude_sonnet` | `anthropic_sdk` (unchanged) | unchanged |
| `claude_haiku` | `anthropic_sdk` (unchanged) | unchanged |
| `codex_5.4_mini` | `codex_cli` (unchanged) | unchanged |
| `codex_5.5` | `codex_cli` (unchanged) | unchanged |

## Session Storage

Canonical format: OpenAI Chat Completions messages in JSONL.  
Path: `<store_root>/conversations/<uuid>.jsonl`  
`disable_continuity: true` profiles write no session file and pass no `continuity_handle`.  
Both `openai_sdk` and `anthropic_sdk` share this format; `anthropic_sdk` converts on call, not on storage.

## Error Handling

`LocalCLIExecutionError` remains the unified exception type across all runtimes. Both adapters raise it with `diagnostics` dict including `runtime`, `stop_reason`, and `error_event`.

## Out of Scope

- LiteLLM `previous_response_id` path (blocked by issue #26167, no fix ETA)
- `claude_cli` runtime removal (code stays, no profiles use it)
- Codex CLI → OpenAI SDK migration (ChatGPT subscription OAuth is incompatible with developer API)
- 429 rate-limit handling for `claude_*` profiles (separate concern)

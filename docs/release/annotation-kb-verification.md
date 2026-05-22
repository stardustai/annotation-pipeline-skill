# Annotation Knowledge Base — End-to-End Verification

This doc walks through verifying the `check_past_experience` MCP tool wiring end-to-end with a real Claude CLI subagent. Requires an Anthropic API key (or equivalent third-party Anthropic-compatible endpoint).

> **No environment-variable export required.** All `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` injection happens inside `isolated_claude_home`, scoped to the subagent subprocess. Your shell state is never modified.

## Step 1: Initialize a fixture project

```bash
cd /tmp
rm -rf annotation-kb-fixture
annotation-pipeline init --project-root ./annotation-kb-fixture
```

## Step 2: Seed conventions with row trace data

Run from the annotation-pipeline-skill repo:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python -c "
from pathlib import Path
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.services.entity_convention_service import EntityConventionService

store = SqliteStore.open(Path('/tmp/annotation-kb-fixture/.annotation-pipeline'))
svc = EntityConventionService(store)
for i in range(5):
    svc.record_decision(
        project_id='memory-ner-v2', span='Android', entity_type='technology',
        source=f'qc_consensus:seed{i}', task_id=f'task_{i}', row_id=f'row_{i}',
        row_content=f'Crashes on Android 10 in scenario {i}.',
    )
print('seeded')
"
```

## Step 3: Add the annotator profile

Bootstrap `llm_profiles.yaml` from the example template:

```bash
cp /tmp/annotation-kb-fixture/.annotation-pipeline/llm_profiles.example.yaml \
   /tmp/annotation-kb-fixture/.annotation-pipeline/llm_profiles.yaml
```

Edit the new `llm_profiles.yaml` and append a profile:

```yaml
profiles:
  # ... existing profiles ...
  annotator_claude_kb:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    # bypassPermissions is REQUIRED for the agent to actually invoke MCP
    # tools in non-interactive (`--print`) mode. Other modes (`default`,
    # `dontAsk`, `acceptEdits`) deny MCP tool calls because they're not on
    # claude's built-in allow list, and `--print` has no interactive
    # approval channel. The MCP server itself is sandboxed (read-only SQL
    # against the project's own DB; no shell, no file writes), so giving
    # it permission is safe.
    permission_mode: bypassPermissions
    mcp_servers:
      - name: annotation-kb
        command: python
        args:
          - -m
          - annotation_pipeline_skill.mcp.kb_server
          - --project-root
          - /tmp/annotation-kb-fixture/.annotation-pipeline
          - --project-id
          - memory-ner-v2
    strict_mcp_config: true
    disallowed_tools: ["Bash", "Edit", "Write"]
```

(Adjust `api_key_env` to whatever variable name holds your API key — or use a third-party Anthropic-compatible endpoint with its own `base_url` + `api_key_env`.)

## Step 4: Bind the profile to the annotation stage and run a single cycle

Bind `annotator_claude_kb` to the `annotation` stage target in the project's runtime configuration (via the dashboard or by editing the runtime config files directly). Then:

```bash
annotation-pipeline runtime once --project-root /tmp/annotation-kb-fixture
```

The runtime composes the claude invocation with `--bare`, `--mcp-config <path-inside-isolated-home>`, `--strict-mcp-config`, `--disallowedTools=Bash,Edit,Write`. `isolated_claude_home` injects `base_url` + `api_key` into the subprocess environment in isolation — the parent shell's env is untouched.

## Step 5: Inspect the transcript

Find the subagent's stream-json transcript under the project's artifacts directory and verify:

1. The agent advertised the `check_past_experience` tool (via the MCP `tools/list` response).
2. The agent invoked it with an argument like `{"entry": "Android"}`.
3. The response payload included `convention.type == "technology"`, `convention.evidence_count == 5`, and at least one example string starting with `[task_*/row_*]`.

If all three appear, the wiring is correct end-to-end.

## Step 6 (optional): Try a third-party Anthropic-compatible endpoint

Change `base_url` to e.g. `https://api.deepseek.com/anthropic` and `api_key_env` to a corresponding env var. Re-run Step 4. The agent should still invoke the tool; the actual completion text will come from the third-party model.

# Annotation Knowledge Base — Design Spec

**Date:** 2026-05-19 (revised 2026-05-21)
**Status:** Shipped

---

## Problem

Annotators (LLM subagents) currently receive `annotation_rules.yaml` and the entity-convention auto-injection (`EntityConventionService.find_matches_in_text`) baked into the prompt. This has three shortcomings:

1. **Passive injection wastes tokens.** All matching conventions are dumped into every prompt regardless of whether the agent actually needs them on a given row. Long contexts crowd out the row text itself.
2. **Statistical summaries lack instructive value.** A figure like `type_entropy: 0.85` or `top_share: 0.6` tells an agent the span is contested, but not *when* to choose each type. Agents need sentence-level cases to mode-match against.
3. **No way to consult history mid-decision.** When an agent encounters an unfamiliar span, it has no mechanism to ask "what did past tasks decide here?" — the system either pre-injected the answer or it didn't.

The `memory-ner` project accumulates merge-derived guidance in `ANNOTATION_GUIDE.merge_updates.md`, a 1349-line file of human-curated rules with evidence. We want the same effect — agents informed by past decisions — but **on-demand, retrieval-driven, and trace-grounded** rather than written manually.

---

## Goals

- Provide annotator / QC / arbiter subagents an MCP tool, `check_past_experience(entry)`, that returns past decisions for a candidate span.
- The tool returns: current convention, type distribution, and **per-type sentence-level examples** so the agent learns from concrete cases, not statistics.
- Examples are selected for **maximum diversity** (MinHash + farthest-first traversal) so 2–3 snippets cover the breadth of past contexts.
- The tool is registered via the MCP protocol layer (`--mcp-config`), not prompt injection.
- LLM provider switching (Claude API, DeepSeek, etc.) works via the `ANTHROPIC_BASE_URL` environment variable, configured per profile in `llm_profiles.yaml`.

---

## Non-Goals

- **No new domain table.** No `SpanKnowledge` / `Evidence` / `MergeRule` tables. All data is sourced from the existing `entity_conventions` (with one schema extension) and `posterior_audit` (live computation).
- **No row-level full-text BM25 index.** The added cost (jieba indexing, rebuild scheduling, freshness invariants) is not justified until we have evidence the per-span examples are insufficient.
- **No automated pattern rules.** We do *not* synthesize `annotation_rules.yaml` entries from merge events. Pattern learning is example-driven, left to the agent.
- **No Codex CLI MCP integration in this phase.** First pass targets Claude CLI; Codex parity is a follow-up.
- **No replacement of existing `find_matches_in_text` auto-injection.** It continues to operate; the new tool is additive.

---

## Design Rationale

These were the load-bearing decisions from the brainstorming session, in the order they were resolved.

### 1. Agent-pull (MCP tool) instead of prompt-push (auto-injection)

The starting point was `EntityConventionService.find_matches_in_text` — it scans every annotator prompt for convention-matching spans and prepends them. Two problems compounded the longer the project ran: the prompt grew with the convention table (regardless of relevance to *this* row), and the agent had no signal that a convention even existed for a span until the runtime had already paid for shipping the full block.

The reverse — exposing the conventions as a **tool** the agent calls when it wants to — flips both:

- **Token cost scales with curiosity, not history.** A row whose spans are all obvious skips the lookup entirely. A row with one ambiguous span pays for one tool round-trip.
- **The agent decides what's ambiguous.** It already does this internally to produce annotations; surfacing the decision as "do I need to ask?" is a more honest interface than "here's everything you might need."

MCP was the natural plumbing because Claude CLI already supports it via `--mcp-config`, and the tool/result round-trip is part of the assistant's normal protocol — no prompt-engineered "if you want X, output Y" hack.

### 2. No new domain table — compose at query time

The first design draft included a `SpanKnowledge` table with denormalized fields (`convention_type`, `type_distribution`, etc.) populated by writes from `EntityConventionService` and the posterior audit job. The user pushed back: these values are derived from data we *already store* (`entity_conventions.proposals`, `posterior_audit` cache, `wordfreq`), and a denormalized table introduces two failure modes — write paths that forget to update, and staleness windows after backfill operations.

The shipped design composes the response at query time inside `mcp.check_past_experience`:

- `convention` ← read `entity_conventions`
- `distribution` ← aggregate `entity_conventions.proposals[*].type`
- `examples_by_type` ← group `proposals[*].context_snippet` by type, then diversity-sample
- `meta.wordfreq_zipf` ← live call to `wordfreq.zipf_frequency`

The only schema change is **extending `proposals_json`** with two optional fields (`row_id`, `context_snippet`). Since `proposals_json` was already a free-form JSON blob, this is migration-free — legacy proposals lack the keys and the tool gracefully omits them from `examples_by_type`.

### 3. Sentence-level per-type examples over statistical summaries

An earlier draft returned `type_entropy`, `top_share`, `runner_up_share` for contested spans. The user flagged that these are *describing* the problem, not *helping* solve it. An agent told "this span has entropy 0.85" still has no idea *when* to pick which type.

The shipped design returns up to 3 verbatim context snippets per type, e.g.:

```
"Apple": {
  organization: [
    "[task_019/row_18452] ...Apple's customer support helped me...",
    "[task_022/row_22310] ...Apple announced a new privacy policy..."
  ],
  product: [
    "[task_021/row_21100] ...My Apple iPad keeps crashing..."
  ]
}
```

This is the mode-matching pattern LLMs are good at: "which of these example contexts looks most like the row I'm annotating?" Statistics get pruned out of the response entirely (except the bare `evidence_count`, which is useful as a confidence signal).

### 4. Why MinHash + farthest-first for example selection

The naive options for picking 3 examples out of N proposals:

| Approach | Problem |
|---|---|
| First N (chronological) | Recent rows tend to share context (same batch, same source). Returned examples are near-duplicates. |
| Random N | No guarantee of coverage; can return three near-duplicates by chance on small N. |
| Most-recent N | Same chronological-clustering problem as above. |
| Embedding-based clustering | Adds a heavyweight dependency (sentence encoder) and a vector store. Overkill for ≤50-snippet buckets. |

MinHash + farthest-first traversal gives us coverage of the snippet space with $O(N \cdot k)$ pairwise Jaccard comparisons — cheap at this scale — and `datasketch` was already a project dependency for `row_dedup`. The seed is the lex-smallest snippet (deterministic on repeated calls) and the tie-break is also lex-smallest, so the output is reproducible.

### 5. Why no row-level BM25 index (yet)

The brainstorming considered a parallel feature: an inverted index over `context_snippet` text so the agent could query "show me rows similar to *this row's text*" rather than "show me rows for *this span*". We rejected it for v1:

- The per-span pull (`check_past_experience(entry)`) already covers the dominant case where the agent has a specific candidate span in mind and wants prior context.
- BM25 adds a rebuild cadence, freshness invariants, and a jieba-tokenized index for CJK — meaningful operational cost.
- We have no evidence yet that per-span examples are *insufficient*. Adding BM25 before that signal would be speculative complexity.

This is explicit in Non-Goals so future readers don't reopen the discussion without new data.

### 6. CJK gate in `shingle()`, not a unified jieba path

The MinHash diversity sampler needs meaningful tokens on CJK text. Unconditionally jieba-segmenting *all* input was rejected after testing: jieba splits English contractions (`Apple's → apple / ' / s`) and emits apostrophe-bearing 3-grams that don't appear in the whitespace-tokenized version. The empirical measurement on a typical app-review row was 4 split-based 3-grams vs 6 jieba-based 3-grams with only 3 shared — enough to shift `row_dedup_service`'s Jaccard scores and break every project's already-calibrated `jaccard_threshold`.

The CJK gate (`_CJK_RE.search(text)` → jieba path; else → `text.split()` path) leaves pure-ASCII inputs untouched (so `row_dedup` is unaffected) and upgrades CJK inputs from "degenerate single shingle" to "meaningful n-grams" (so the KB sampler and CJK-row dedup both improve). Pre-existing thresholds in projects with mixed CJK + ASCII may want re-verification because CJK rows used to be effectively unmatched — this is a known one-time recalibration, called out in Risks.

### 7. `base_url` profile field → subprocess env, never operator shell

To run annotators against non-Anthropic endpoints (DeepSeek, MiniMax, GLM), the runtime must put `ANTHROPIC_BASE_URL` somewhere Claude CLI reads it. Three options were on the table:

1. **`export ANTHROPIC_BASE_URL=...`** in the operator's shell. **Rejected** because it pollutes the parent process — any subsequent Claude Code session the operator launches in the same shell would also hit the third-party endpoint. Verified empirically: there is no shell-variable scope that protects a long-running parent session.
2. **A `settings.json` field** loaded via Claude CLI's `--settings`. **Rejected** after testing six candidate field names (`baseURL`, `baseUrl`, `base_url`, `anthropicBaseUrl`, `apiBaseURL`, `endpoint`) — none are honored. The `env` block inside `settings.json` is also not applied to Claude's own API config. `apiKeyHelper` works but only for the API key, not the endpoint.
3. **`isolated_claude_home` injects into `subprocess.Popen(env=...)`**. **Shipped.** The runtime constructs a fresh env dict, sets `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL` from the profile, and passes it only to the spawned claude subprocess. `os.environ` of the parent process is never modified — Unix process model guarantees subprocess env is a copy. Operator's shell sees nothing.

The operator's profile carries `base_url` and `api_key_env` as plain YAML fields; the env-var name is an implementation detail of the runtime.

### 8. `bypassPermissions` is required for MCP profiles

Claude CLI's `--permission-mode` controls how it handles tool calls that aren't on its built-in allow list. We tested all relevant modes in non-interactive (`--print`) annotation runs:

| Mode | MCP tool call |
|---|---|
| `default` | Asks interactively → no TTY → denied |
| `dontAsk` | Denies non-allowed tools by default |
| `acceptEdits` | Auto-accepts edits but asks for others → denied |
| `bypassPermissions` | Allows |

`bypassPermissions` is the only mode where the agent can actually call `mcp__annotation-kb__check_past_experience` in batch mode. The mode is safe in this context: the MCP server's only capability is read-only SQL against the project's own DB (no shell, no file writes, no network). Profiles using KB-aware annotation should set `permission_mode: bypassPermissions` — the spec's profile YAML examples and the verification guide both do.

### 9. System-level prompt for the KB tool, not per-project rules

The agent needs to know *when* to call the tool. Two places to put that instruction:

- **`annotation_rules.yaml`** (per-project) — would force every project that wants KB-aware annotation to copy in the same boilerplate. Easy to drift, easy to forget.
- **`_annotation_instructions()` in `subagent_cycle.py`** (runtime-level) — every annotator subagent gets the instruction unconditionally. The block is conditional in content (`"when the mcp__annotation-kb__check_past_experience tool appears in your tools list…"`), so profiles without an MCP server attached read the same paragraph but no-op on it.

The runtime-level placement matches where the tool itself is wired — both come from the framework, not from per-project configuration. A parallel block in `_build_qc_instructions()` handles the QC verifier path.

---

## Architecture

```
                  ┌──────────────────────────────────┐
                  │  annotator / qc / arbiter subagent│
                  │  (claude CLI, ANTHROPIC_BASE_URL  │
                  │   may point at any LLM)          │
                  └────────────────┬─────────────────┘
                       MCP stdio (via --mcp-config)
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │  annotation-kb MCP server         │
                  │  (Python, stdio)                  │
                  │  Exposes: check_past_experience   │
                  └────────────────┬─────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
       ┌────────────┐      ┌────────────┐      ┌────────────┐
       │  entity_   │      │ posterior_ │      │  wordfreq  │
       │ conventions│      │   audit    │      │  (library) │
       │  (SQLite)  │      │ (computed) │      │            │
       └────────────┘      └────────────┘      └────────────┘
```

Three components are added:

1. **`entity_conventions.proposals` schema extension** — each proposal carries `row_id` and `context_snippet` for trace and example surfacing.
2. **`annotation_pipeline_skill.mcp.kb_server`** — a stdio MCP server exposing one tool.
3. **`llm_profiles.yaml` `mcp_servers` + `env` fields** — the runtime composes the `claude` invocation from the profile.

A fourth, smaller change extends `annotation_pipeline_skill.similarity.minhash.shingle()` with a CJK fallback so the diversity scoring works on Chinese text.

---

## Tool Contract

### `check_past_experience(entry: str) -> dict`

**Input.**
- `entry`: a candidate entity / span text (case-insensitive lookup; original case preserved in returned snippets).

**Output.**

```json
{
  "entry": "Apple",
  "convention": {
    "status": "disputed",
    "type": null,
    "evidence_count": 8
  },
  "distribution": {
    "organization": 5,
    "product": 3
  },
  "examples_by_type": {
    "organization": [
      "[task_019/row_18452] ...Apple's customer support helped me yesterday with my refund...",
      "[task_022/row_22310] ...Apple announced a new privacy policy for developers..."
    ],
    "product": [
      "[task_021/row_21100] ...My Apple iPad keeps crashing on the latest update..."
    ]
  },
  "meta": {
    "wordfreq_zipf": 4.1,
    "generic_word": false
  }
}
```

**Field semantics.**

| Field | Source | Notes |
|---|---|---|
| `convention.status` | `entity_conventions.status` | `"active"`, `"disputed"`, or `"none"` (if span not in conventions table). |
| `convention.type` | `entity_conventions.entity_type` | `null` when `status` is `disputed` or `none`. |
| `convention.evidence_count` | `entity_conventions.evidence_count` | 0 when `status` is `"none"`. |
| `distribution` | Aggregated from `entity_conventions.proposals[*].type` | Counts of each type ever proposed for this span. Empty `{}` when no proposals. |
| `examples_by_type` | `entity_conventions.proposals[*].context_snippet`, grouped by `type`, diversity-selected | Up to 3 snippets per type; format `[<task_id>/<row_id>] <snippet>`. Empty `{}` when no snippets available (legacy proposals predating the schema extension). |
| `meta.wordfreq_zipf` | Reuse `_wordfreq_score()` (currently in `interfaces/api.py`) — see "Shared utility move" below | Zipf scale 0–7. |
| `meta.generic_word` | `wordfreq_zipf >= 5.0 and evidence_count < 5` | Conservative flag; agent makes final call. |

**Shared utility move.** The existing `_wordfreq_score()` private function in `interfaces/api.py` does exactly what we need (jieba-free tokenization via `wordfreq.tokenize`, language auto-detection by CJK range). We promote it to a new module `annotation_pipeline_skill/text/wordfreq_utils.py` so both `api.py` (existing low_info computation) and `kb_server.py` (new tool) import the same implementation.

**Error / empty behavior.**

- Unknown span (not in `entity_conventions`): return `convention.status = "none"`, `evidence_count = 0`, empty `distribution`, empty `examples_by_type`. `meta.wordfreq_zipf` is still computed.
- Empty string entry: return `{"error": "entry is required"}` (the MCP layer surfaces this as a tool error).
- Database unavailable: tool raises; MCP server returns a structured error so the agent can fall back to non-tool reasoning.

---

## Schema Extension

`entity_conventions.proposals` is a JSON array column. Each proposal is currently:

```python
{"type": str, "source": str, "task_id": str | None, "notes": str | None, "at": str}
```

After this change:

```python
{
  "type": str,
  "source": str,
  "task_id": str | None,
  "row_id": str | None,              # NEW
  "context_snippet": str | None,     # NEW — see below
  "notes": str | None,
  "at": str,
}
```

**`context_snippet` construction.** When `record_decision()` is called with a `row_content` argument, the snippet is `row_content[max(0, hit-80) : hit+len(span)+80]` where `hit` is the case-insensitive first occurrence of `span` in `row_content`. The snippet is clamped to 200 characters and surrounded by `…` on each end when truncated. If `row_content` is not provided (e.g., operator declarations from the dashboard), `context_snippet` is `None`.

**Migration.** No schema migration is required — `proposals_json` is a free-form JSON blob. Legacy rows simply lack the two new keys; the tool treats missing `context_snippet` as "no example available" and gracefully omits them from `examples_by_type`.

**`record_decision()` signature change.**

```python
def record_decision(
    self,
    *,
    project_id: str,
    span: str,
    entity_type: str,
    source: str,
    task_id: str | None = None,
    row_id: str | None = None,         # NEW
    row_content: str | None = None,    # NEW
    notes: str | None = None,
) -> EntityConvention:
    ...
```

All call sites are updated to pass `row_id` and `row_content` where available. Sites that don't know (e.g., operator declarations from a UI button) pass neither and produce no snippet.

---

## Diversity Sampling

For each `(span, type)` pair, the tool collects all proposals' `context_snippet` values, then selects up to 3 that maximize mutual dissimilarity.

### MinHash shingle update (CJK fallback)

`annotation_pipeline_skill/similarity/minhash.py::shingle()` currently word-splits on whitespace. For CJK text this degenerates to a single token, making Jaccard binary (0 or 1) and useless for diversity ranking.

```python
import re

_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")

def shingle(text: str, n: int = 5) -> set[str]:
    if not text:
        return set()
    normalized = _WHITESPACE_RE.sub(" ", text.lower()).strip()

    if _CJK_RE.search(normalized):
        # CJK path: lazy-import jieba so ASCII-only projects don't pay the load cost.
        import jieba
        tokens = [t for t in jieba.cut(normalized) if t.strip()]
    else:
        tokens = normalized.split(" ")

    if len(tokens) < n:
        return {normalized} if normalized else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
```

The existing `MinHashLSHFinder` and `row_dedup_service` callers see no behavioral change for ASCII inputs — the CJK branch is dormant unless CJK characters are present.

### Farthest-first selection

```python
def select_diverse_examples(snippets: list[str], k: int = 3) -> list[str]:
    if len(snippets) <= k:
        return snippets
    minhashes = [build_minhash(s) for s in snippets]
    # Seed with the lexicographically smallest snippet for determinism.
    seed_idx = min(range(len(snippets)), key=lambda i: snippets[i])
    selected = [seed_idx]
    while len(selected) < k:
        best_i, best_dist = -1, -1.0
        for i in range(len(snippets)):
            if i in selected:
                continue
            d = 1.0 - max(minhashes[i].jaccard(minhashes[j]) for j in selected)
            if d > best_dist:
                best_dist, best_i = d, i
        selected.append(best_i)
    return [snippets[i] for i in selected]
```

Determinism note: snippets are deduplicated before sampling, and the seed is chosen deterministically so identical input always yields identical output.

---

## MCP Server

**Location:** `annotation_pipeline_skill/mcp/kb_server.py`

**Protocol:** stdio JSON-RPC, the MCP standard.

**Launch:** invoked as a subprocess by Claude CLI when configured via `--mcp-config`. Receives the project root via CLI flag.

```bash
python -m annotation_pipeline_skill.mcp.kb_server --project-root <path>
```

**Tools exposed:** `check_past_experience` only. (Single-tool scope avoids over-promising; additional tools are explicit future work.)

**Dependencies (new to pyproject.toml):**

- `mcp` — official MCP Python SDK for the stdio server.
- `jieba` — only loaded when CJK text is detected by `shingle()`; lazy import keeps cold-start cost off ASCII-only projects.

`wordfreq` is already a project dependency. Server holds a read-only SQLite connection to the project's `db.sqlite` and instantiates `EntityConventionService` for queries — it does not perform writes.

**Performance budget:** each `check_past_experience` call should return in <100 ms for typical inputs (single span, <50 proposals). The hot path is one indexed SELECT on `entity_conventions` plus MinHash over <50 snippets.

---

## Profile Integration

`llm_profiles.yaml` profiles gain three optional top-level fields. LLM provider switching keeps using the existing `base_url` + `api_key_env` fields — `isolated_claude_home` already translates those to `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` inside the subprocess (operator's shell is never modified).

```yaml
profiles:
  annotator_claude:
    runtime: claude_cli
    model: sonnet
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    # bypassPermissions is required to actually invoke MCP tools in
    # non-interactive (`--print`) mode — every other mode denies them.
    # The MCP server is sandboxed (read-only SQL on the project DB), so
    # granting it permission is safe.
    permission_mode: bypassPermissions
    mcp_servers:
      - name: annotation-kb
        command: python
        args:
          - -m
          - annotation_pipeline_skill.mcp.kb_server
          - --project-root
          - /path/to/project/.annotation-pipeline
          - --project-id
          - my-project
    strict_mcp_config: true
    disallowed_tools: ["Bash", "Edit", "Write"]

  annotator_deepseek:
    runtime: claude_cli            # reuse Claude CLI binary
    model: deepseek-v4-flash
    base_url: https://api.deepseek.com/anthropic
    api_key_env: DEEPSEEK_API_KEY
    permission_mode: bypassPermissions
    mcp_servers:
      - name: annotation-kb
        command: python
        args:
          - -m
          - annotation_pipeline_skill.mcp.kb_server
          - --project-root
          - /path/to/project/.annotation-pipeline
          - --project-id
          - my-project
    strict_mcp_config: true
    disallowed_tools: ["Bash", "Edit", "Write"]
```

The runtime (`llm/local_cli.py::_generate_claude`) materializes a per-invocation `mcp-config.json` **inside the isolated claude home** (`<isolated_home>/mcp-config.json`), then invokes claude approximately as:

```bash
claude --bare -p --no-session-persistence \
       --verbose --output-format stream-json \
       --model <model> \
       --permission-mode <mode> \
       --mcp-config <isolated_home>/mcp-config.json \
       --strict-mcp-config \
       --disallowedTools "Bash,Edit,Write"
```

with `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` set only in the subprocess `env` (`isolated_claude_home` constructs the dict from `profile.base_url` and `profile.resolve_api_key()`). The operator's shell state is never modified.

`--bare` keeps the subagent free of host-side hooks, plugins, and auto-memory, ensuring the only tools available are the MCP-provided ones and built-ins not in `disallowed_tools`.

---

## Lifecycle Example

A condensed end-to-end trace, focusing on the new behavior.

1. **Task A imported.** No effect on this system (no BM25 to rebuild).
2. **Row 18452 reaches annotator.** Agent reads `"Flynx is much better than the old browser. Crashes on Android 10."`. It calls `check_past_experience("Flynx")` and `check_past_experience("Android")` via MCP. Both return `convention.status = "none"`. Agent falls back to its prompt rules, tags `Android` only — misses `Flynx`.
3. **QC + arbiter accept golden** with both `Flynx → technology` and `Android → technology`. `record_decision()` runs twice with `row_id="row_18452"` and `row_content="Flynx is much better…"`. Each proposal carries a fresh `context_snippet`.
4. **Row 18453 reaches annotator** (`"Telegram is faster than Facebook on my Redmi 3S."`). Agent calls `check_past_experience("Android")` — gets back `{convention: technology (active, 1 evidence), examples_by_type: {technology: ["[task_019/row_18452] …Crashes on Android 10…"]}}`. Agent generalizes "named tech terms → technology" from the single example and tags Telegram, Facebook, Redmi 3S correctly.
5. **Task B imported, weeks later, with span `"Apple"`.** Apple's history now contains 5 `organization` proposals and 3 `product` proposals across multiple tasks. Tool returns `convention.status = "disputed"`, both type buckets in `examples_by_type` with diversity-selected snippets. Agent matches current row context ("Apple's customer support helped…") to the `organization` examples and chooses correctly.

---

## Testing

**Unit.**

- `EntityConventionService.record_decision` with new args persists `row_id` and `context_snippet` in the proposal JSON; without `row_content`, the snippet is `None`.
- `shingle()` produces meaningful n-grams for a Chinese sentence (jieba path) and unchanged output for an English sentence (whitespace path).
- `select_diverse_examples` is deterministic, returns ≤ k items, and prefers low-Jaccard pairs over a hand-crafted near-duplicate set.

**Integration.**

- End-to-end: insert a synthetic project with 8 proposals for the span `"Apple"` (mix of organization / product, with context snippets), call `check_past_experience("Apple")`, assert the returned `examples_by_type` has both buckets, snippet counts ≤ 3, and that the chosen snippets are not pairwise near-duplicates.
- Empty / unknown span returns the documented "none" shape with no exceptions.
- MCP server smoke test: spawn the server via stdio, send a `tools/list` request, then a `tools/call` for `check_past_experience`, validate the response shape.

**Profile integration.**

- `subagent_cycle.py` composes a valid `mcp-config.json` from a profile carrying `mcp_servers`; the resulting `claude` invocation includes `--mcp-config` and `--strict-mcp-config`.
- Live Claude CLI test (manual, behind a flag): launch a subagent against a fixture project and verify the agent can call the tool. Skipped in CI to avoid network and authentication requirements.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Agents over-call the tool (e.g., for every token) and inflate latency. | Tool returns in <100 ms; system prompt advises "consult only on candidate entity spans, not every token." |
| `context_snippet` leaks PII from past rows into a different task's agent. | Snippets are limited to 200 chars around the span; project-scoped query enforces same-project boundary; no cross-project leakage. |
| Agents copy snippet content verbatim into their answer instead of generalizing. | Snippets are not gold output for the *current* row — they're from *other* rows. The system prompt explicitly directs the agent to use them as analogies, not as answers. |
| jieba initialization adds ~0.2 s to MCP server startup. | One-time cost on server spawn; the server is long-lived per annotation run. Acceptable. |
| `shingle()` behavior change leaks into `row_dedup_service` for CJK rows. | Intentional: prior word-5-gram path degenerated CJK rows to a single shingle (Jaccard 0/1, effectively only catching exact duplicates). The CJK gate restores meaningful near-duplicate detection. **Pure-ASCII rows are unaffected** — the gate skips the jieba branch entirely. CJK-heavy projects should re-verify their `jaccard_threshold` after this change; the verification belongs in this work's test phase. |
| Rejected alternative: unified jieba path for all text. | Empirically broke ASCII gram sets: `"Apple's customer support helped me yesterday"` produces 4 split-based 3-grams vs 6 jieba-based 3-grams (only 3 shared) because jieba splits `Apple's → apple / ' / s` and emits apostrophe-bearing grams. Every project's `row_dedup` threshold would need recalibration. Not worth the simpler code path. |
| MCP SDK version drift breaks the stdio protocol. | Pin the `mcp` package version in `pyproject.toml`; integration smoke test catches breakage. |
| Profile authors forget to set `ANTHROPIC_BASE_URL` and silently hit Anthropic. | Profile validation in `provider_config_service` warns when `mcp_servers` is set but `ANTHROPIC_BASE_URL` is unset or matches the default. |

---

## Open Questions

- Should `meta.generic_word` thresholds (`zipf >= 5.0`, `evidence_count < 5`) be project-configurable? Defer until we have a project that needs to tune them.
- Codex CLI MCP support: out of scope here, but worth confirming Codex's MCP integration story before promising parity.

---

## Phasing

This spec is intended for a **single implementation plan**. The work decomposes into roughly the following slices, each of which should be a separate commit / PR in the implementation plan:

1. Schema extension and `record_decision()` signature change, with all call sites updated.
2. `shingle()` CJK fallback and `select_diverse_examples()` utility.
3. MCP server (`kb_server.py`) and the `check_past_experience` tool implementation.
4. Profile integration (`llm_profiles.yaml` schema, `subagent_cycle.py` invocation update).
5. Tests across all of the above, including a Claude CLI smoke test gated by a fixture flag.

Estimated total effort: **5–6 person-days**.

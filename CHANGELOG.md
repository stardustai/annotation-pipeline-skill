# Changelog

## Unreleased

### Added

- **Low-Info Filter & Naming Alignment** (`docs/superpowers/specs/2026-05-19-low-info-filter-design.md`, plan `2026-05-19-low-info-filter.md`).

  - **`divergent_entries` rename.** `EntityStatisticsService.contested_spans()` renamed to `divergent_entries()` project-wide (service, API, scripts, TypeScript types). Old name removed.

  - **`type_entropy` field on divergent entries.** Each entry in the `/posterior-audit` response's `divergent_entries` array now includes `type_entropy: float` — Shannon entropy H = −Σ(c/T)·log₂(c/T) over the span's prior distribution. Surfaces in the **Divergent Entries** tab as an **Entropy** column.

  - **`low_info_entries` in posterior-audit response.** New top-level key containing all spans in `entity_statistics` whose average Zipf token frequency (`wordfreq_score`) ≥ `LOW_INFO_THRESHOLD` (4.0). Scans the full `entity_statistics` table — not limited to divergent spans. Each entry carries `span`, `prior_total`, `prior_distribution`, `wordfreq`.

  - **Low Info Entries UI tab.** New third sub-tab in the Posterior Audit panel:
    - Adjustable wordfreq threshold input (warns if set below backend floor 4.0).
    - Per-row "Set not_an_entity" button and row checkbox.
    - Always-visible **"Batch set 'Not an entity'"** button (disabled when nothing selected); shows selected count when active.
    - 30-row pagination.

  - **`scripts/bulk_set_lowinfo_not_an_entity.py`** — one-shot operator script to mark all spans with wordfreq ≥ threshold (default 5.0) as `not_an_entity` efficiently: scans each ACCEPTED task once, strips all target spans in a single annotation write, purges `entity_statistics` rows iteratively (handles historical/non-ACCEPTED sources), clears `posterior_audit_cache`. Applied to `v3_initial_deployment`: 3,555 tasks patched, 93,382 span occurrences removed, ~115K `entity_statistics` rows deleted.

  - **`annotation_pipeline_skill.text.wordfreq_utils.wordfreq_score`** shared helper, replacing a duplicate inline function previously in `interfaces/api.py`.
  - **`annotation_pipeline_skill.similarity.diverse.select_diverse_examples`** — MinHash-based farthest-first traversal sampler used to pick representative context snippets per (span, type) bucket.

### Changed

  - **`build_posterior_audit` response shape** updated: `contested_spans` → `divergent_entries` (with added `type_entropy`); new `low_info_entries` array. Cache surgery sites (`retroactive-fix`, `convention-set`) updated to use new keys. `low_info_entries` is not affected by convention changes (scans raw statistics, not convention-filtered view).

- **Annotation knowledge base MCP tool (`check_past_experience`).** Annotator / QC / arbiter subagents launched via Claude CLI can now query the project's accumulated convention history on demand. The tool returns: current convention status + type, type distribution from past proposals, up to 3 diversity-selected sentence-level examples per type (via MinHash farthest-first sampling), and Zipf wordfreq metadata. Wired via `--mcp-config` and exposed by a stdio MCP server (`annotation_pipeline_skill.mcp.kb_server`).
- **Annotator + QC system prompt updates.** The runtime-level `_annotation_instructions()` and `_build_qc_instructions()` now include a conditional `KNOWLEDGE BASE TOOL:` paragraph that fires when `mcp__annotation-kb__check_past_experience` is present in the agent's tools list — directs the agent to consult it for ambiguous named-entity spans, prefer the active convention, use per-type examples as analogies for `disputed` spans, and skip the tool for obvious tokens. Lives at the framework layer, not in per-project `annotation_rules.yaml`.
- **`LLMProfile` schema additions:** `mcp_servers`, `strict_mcp_config`, `disallowed_tools` for declaring per-profile MCP server configurations and locking down built-in tools. Profiles attaching an MCP server should set `permission_mode: bypassPermissions` so the agent can actually invoke MCP tools in non-interactive (`--print`) mode (every other mode denies them; the MCP server itself is sandboxed to read-only SQL on the project DB). LLM provider switching continues to use the existing `base_url` field — `isolated_claude_home` injects it into each subagent subprocess in isolation, leaving the operator's shell untouched.
- **CJK fallback in `similarity.shingle()`.** Rows containing CJK Unified Ideographs (U+4E00–U+9FFF) or Extension A (U+3400–U+4DBF) are now segmented with jieba instead of degenerating to a single shingle. Improves both the knowledge-base diversity sampling and existing row_dedup precision on Chinese text.
- **`annotation_pipeline_skill.text.wordfreq_utils.wordfreq_score`** shared helper, replacing a duplicate inline function previously in `interfaces/api.py`.
- **`annotation_pipeline_skill.similarity.diverse.select_diverse_examples`** — MinHash-based farthest-first traversal sampler used to pick representative context snippets per (span, type) bucket.

### Changed

- **`EntityConventionService.record_decision()`** accepts two new optional kwargs: `row_id: str | None` and `row_content: str | None`. Proposals now carry a `context_snippet` window (±80 chars around the span, with ellipsis markers) and the originating `row_id`. The QC consensus path (`runtime/subagent_cycle.py`) and HR correction path (`services/human_review_service.py`) thread the trace data through automatically; operator-declared sites leave `row_id` and `row_content` unset.

### Fixed

- **`SubagentRuntime._profile_name_for_target` no longer constructs a throwaway client per call.** The old implementation called `client_factory(target)` purely to read `.profile.name` and discarded the client. Cheap in production but in finite-list test stubs each probe consumed one client per call, breaking retry flows (one test would assert `failed=1` on the rerun, another would loop forever in QC because `local_scheduler.py`'s bail-counter only resets ANNOTATING tasks). Replaced with a cache populated as a side-effect of `_call_client`, keyed by `result.provider` so the cache and pinned-handle profile column agree.

### Dependencies

- Added `jieba` (CJK word segmentation; lazy-imported only when CJK chars are detected).
- Added `mcp` (official MCP Python SDK; required by the new MCP server).

## 2026-05-17

- V1.2 prior-driven verifier feature SHIPPED. Implementation complete
  per the 13-task plan; bootstrap script populated 155k (project, span,
  type) counters from 4133 historical accepted tasks; runtime restarted
  with the verifier active.
  - Spec: `docs/superpowers/specs/2026-05-17-prior-driven-verifier-design.md`
  - Plan: `docs/superpowers/plans/2026-05-17-prior-driven-verifier.md`
  - PRODUCT_DESIGN §8.5 + TECHNICAL_ARCHITECTURE §11.9 cover the design.

- **Operator setup note**: the verifier's second-arbiter path needs an
  `arbiter_secondary` target in the workspace's `llm_profiles.yaml`
  (gitignored because it carries API keys). For a different-family
  cross-check, add a profile like:

  ```yaml
  profiles:
    claude_sonnet_arbiter:
      provider: local_cli
      cli_kind: claude
      cli_binary: claude
      model: sonnet
      permission_mode: dontAsk
      timeout_seconds: 900
  targets:
    arbiter_secondary: claude_sonnet_arbiter
  ```

  Without this target the runtime falls back to first-arbiter acceptance
  on prior-divergent tasks (recorded as `prior_verifier_action=
  second_arbiter_unavailable` in the audit log).

- New table `entity_statistics` (additive migration). New service
  `EntityStatisticsService` with `increment` / `distribution` / `check`
  / `contested_spans` and a module-level `iter_span_decisions` helper.

- Wired into all four surfaces:
  - QC-pass (annotator+QC consensus): divergent → ARBITRATING + BLOCKING
    `prior_disagreement` feedback; agree → ACCEPTED + stats++ +
    conventions++; cold_start → ACCEPTED + stats++ (no conventions++)
  - First-arbiter accept: stats++ then re-check against prior; on
    divergence, mark task metadata for second-arbiter dispatch
  - Scheduler claim loop: tasks with the divergence flag get routed to
    `_resolve_first_arbiter_divergence_async` instead of `run_task_async`
  - Second-arbiter resolution: matches first → accept first; matches
    prior → flip annotation to prior's type; third option → HR
  - HumanReviewService.submit_correction: verifier check + `force=True`
    override + stats++ with `HR_WEIGHT=5`
  - HumanReviewService.decide(accept): verifier check (no force here;
    operator must use submit_correction to override) + stats++

- New `GET /api/posterior-audit` endpoint + Posterior Audit tab in the
  dashboard. Operator-triggered "Check" lists task-level deviations
  (Send to HR) + project-level contested spans (Declare canonical
  type).

- Bootstrap script `scripts/bootstrap_entity_statistics.py` seeds
  stats from existing ACCEPTED tasks. HR_WEIGHT=5 for HR-authored
  artifacts, 1x for everything else. The historical sample is "clean"
  because all current ACCEPTED tasks predate the dictionary-injection
  feature.

## 2026-05-16

- BREAKING (HR routing): the runtime no longer escalates to HUMAN_REVIEW on
  any "couldn't resolve" outcome. HR is now reserved strictly for genuine
  arbiter uncertainty — verdicts with `confidence` label `tentative` or
  `unsure` (i.e. `arb["unresolved"] > 0`). All other non-terminal arbiter
  outcomes (codex subprocess error, missing `corrected_annotation`, JSON
  parse fail, non-verbatim correction, unknown verdict value) are
  classified as **mechanical failures**: the task stays in `arbitrating`
  status for re-pickup, and the next worker cycle re-runs the arbiter on
  the same annotation (the annotation didn't change — there's no point
  re-running the annotator).
- New `arb["mechanical_fail"]` counter in `_arbitrate_and_apply` outcome.
  The "qc-wins-but-no-fix" case used to bump `unresolved` (forcing HR);
  now bumps `mechanical_fail` (allowing retry).
- `SubagentRuntime.ARBITER_MECHANICAL_RETRY_CAP = 3`. The runtime tracks
  consecutive mechanical retries per task in `task.metadata.arbiter_mechanical_retries`.
  When the counter reaches 3, the task is forced to HR with a clear reason.
  Counter is persistent across restarts.
- Provider fallback: `_generate_async` now wraps the target client and
  retries via the `fallback` target on rate-limit errors
  (`openai.RateLimitError`, `status_code == 429`, or message strings
  matching "rate limit"/"429"/"too many requests"). No circuit breaker;
  try-first semantics.
- Codex CLI invocation gains `--ignore-rules` and `--config enabled_tools=[]`
  to suppress user-installed rule files and tool-use, keeping arbiter calls
  pure-JSON and faster.
- Worker bail reset: when an `annotating` worker raises (rate-limit etc.),
  the `finally` block resets the task to `pending` instead of leaving
  `annotating` for the smart-resume path. Eliminates a tight loop that
  produced ~700 spurious audit events/min during MiniMax 429 storms.
- Audit script `scripts/audit_verbatim_accepted.py` to scan ACCEPTED tasks
  for verbatim violations (5% audit found ~11% violations) and route
  them back to ARBITRATING under the new arbiter / verbatim guard.
- HR drawer banner: drop redundant "Routed to human review." fallback line.
- `docs/RUNTIME_DESIGN.md` removed; content merged into
  `TECHNICAL_ARCHITECTURE.md` §6 (state machine), §10 (runtime), §11
  (execution model), §12 (error model), §13 (config) — these sections
  now describe the actual implementation rather than the original
  aspirational design.

## 2026-05-11

- Auto-escalate tasks to HUMAN_REVIEW after `RuntimeConfig.max_qc_rounds` (default 3) QC rejections, replacing the silent infinite-loop hazard. Triggered by counting `FeedbackRecord(source_stage=QC)` per task; configurable via `runtime.max_qc_rounds` in `workflow.yaml`.
- JSON Schema gate on all writes that produce annotation ground truth:
  - Annotator subagent output is parsed and validated against `task.source_ref.payload.annotation_guidance.output_schema`. Failures record a BLOCKING `FeedbackRecord(category="schema_invalid", source_stage=VALIDATION)` and return the task to PENDING. Tasks without an `output_schema` are passed through unchanged.
  - Human review correction (new endpoint `POST /api/tasks/<id>/human_review_correction` and CLI `apl human-review correct ...`) validates the submitted answer against the same schema. Failures return 400 with structured error list. Human-side writes require an `output_schema` and fail loudly with `missing_schema` if absent.
- New `human_review_answer` artifact kind. Export service prefers it over `annotation_result` when both exist; exported training rows include `human_authored: bool`.
- New dependency: `jsonschema>=4.0`.

## 2026-05-10

- BREAKING: replaced JSON/JSONL `FileStore` with `SqliteStore` (single
  `db.sqlite` per workspace, WAL mode, per-thread connections). Indexed
  queries on `(pipeline_id, status, created_at)` replace full-directory
  scans for hot paths in `coordinator_service`, `readiness_service`,
  `export_service`, `outbox_dispatch_service`, and `subagent_cycle`.
- New CLI: `db init`, `db status`, `db backup`, `db dump-json`.
- Migration: run
  `PYTHONPATH=. python scripts/migrate_filestore_to_sqlite.py
  --src <old-root> --dst <new-root>` once; the script archives the
  source tree to `backups/genesis-YYYYMMDD/` for recovery.
- Atomic runtime lease acquisition via `UNIQUE(task_id, stage)` constraint
  (replaces filesystem `open("x")` trick).
- `RuntimeLease`, `OutboxRecord` dispatcher, and task scheduler now use
  indexed SQL queries instead of in-memory filtering.
- Runtime monitoring (`heartbeat.json`, `cycle_stats.jsonl`,
  `runtime_snapshot.json`) remains file-based.
- `FileStore` retained at `store/file_store.py` solely for the migration
  script; will be removed in a future release.

## v0.1.0 - 2026-05-05

Initial local-first release for an agent-operated annotation pipeline skill.

### Added

- Installable `SKILL.md` for algorithm-engineer annotation projects.
- Python package and `annotation-pipeline` CLI.
- File-backed task store with tasks, attempts, artifacts, audit events, feedback, feedback discussions, outbox records, exports, runtime snapshots, provider config, and Coordinator records.
- JSONL task ingestion, external HTTP task pull, status/submit outbox, readiness reports, and training-data export.
- Configurable provider profiles for OpenAI Responses API, OpenAI-compatible APIs, Codex CLI, and Claude CLI.
- Monitored local runtime for annotation, deterministic validation, QC, retry/heartbeat/capacity reporting, and feedback-driven reruns.
- Optional Human Review after QC with `accept`, `reject`, and `request_changes`.
- Consensus-based annotator/QC feedback discussions.
- React/Vite dashboard with Kanban, Runtime, Readiness, Outbox, Providers, Coordinator, Configuration, Event Log, task details, and image bounding-box preview support.
- Clean agent handoff verification through `scripts/verify_agent_handoff.sh`.
- Real provider smoke scripts for Codex and DeepSeek.
- Memory-ner truth evaluation through `scripts/verify_memory_ner_truth_eval.sh`.
- Memory-ner accepted-state E2E through `scripts/verify_memory_ner_accepted_e2e.sh`.
- Memory-ner dashboard UI acceptance verification through `scripts/verify_memory_ner_ui_acceptance.sh`.
- Active learning/RL workflow design document for the next implementation phase.
- Runtime QC parsing for model responses wrapped in JSON markdown fences.
- Per-task QC sampling policy with `--qc-sample-count`, `--qc-sample-ratio`, and external source QC settings.
- Dashboard editing for task QC policies with audit events.
- File-backed runtime leases, missing snapshot failure reporting, operator-stage Kanban read model, strict QC parse-error handling, provider failure taxonomy, and indexed dashboard summaries.
- Read-only annotation manager v2 import that creates new QC-stage review tasks from old accepted/merged `.annotated.jsonl` outputs without mutating the source project.

### Known Limits

- The core is local-first and file-backed; it does not include a distributed scheduler.
- Real multimodal rendering is limited to image bounding-box preview artifacts.
- Active learning/RL workflow support is designed but not implemented in v0.1.0.
- GitHub repository metadata must be configured outside the codebase when GitHub CLI authentication is unavailable.

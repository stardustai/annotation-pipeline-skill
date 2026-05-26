from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from robust_json import loads as _robust_json_loads

from annotation_pipeline_skill.core.models import ArtifactRef, Attempt, FeedbackDiscussionEntry, FeedbackRecord, Task, utc_now
from annotation_pipeline_skill.core.runtime import RuntimeConfig
from annotation_pipeline_skill.core.schema_validation import (
    SchemaValidationError,
    resolve_output_schema,
    validate_payload_against_task_schema,
)
from annotation_pipeline_skill.core.states import AttemptStatus, FeedbackSeverity, FeedbackSource, TaskStatus
from annotation_pipeline_skill.core.transitions import transition_task
from annotation_pipeline_skill.llm.client import LLMClient, LLMGenerateRequest, LLMGenerateResult
from annotation_pipeline_skill.llm.structured_output import (
    build_annotation_strict_schema,
    build_arbiter_strict_schema,
    build_qc_strict_schema,
    make_json_schema_response_format,
)
from annotation_pipeline_skill.services.feedback_service import build_feedback_bundle, build_feedback_consensus_summary
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@dataclass(frozen=True)
class SubagentRuntimeResult:
    started: int
    accepted: int
    failed: int


class _ArbiterClientUnavailable(Exception):
    """Raised by `_run_arbiter_llm` when the target client can't be built
    (target missing from llm_profiles.yaml, factory raised, etc.)."""


class _ArbiterCallFailed(Exception):
    """Raised by `_run_arbiter_llm` when the LLM call or its response
    parsing failed (network error, malformed JSON, no verdicts list)."""


class _ArbiterRateLimited(Exception):
    """Raised by `_run_arbiter_llm` when the call failed due to provider
    rate-limiting (429 / too-many-requests). Distinct from _ArbiterCallFailed
    so callers can retry with backoff instead of counting toward the
    mechanical-failure cap that routes tasks to human review."""


class QCParseError(ValueError):
    def __init__(self, message: str, *, raw_text: str):
        super().__init__(message)
        self.diagnostics = {"error_kind": "parse_error", "raw_text": raw_text}


# Mirror ``schema_validation._TRAILING_SENTENCE_PUNCT`` so the in-runtime
# auto-align can prefer the punct-trimmed form when the source helper would
# otherwise flag it at apply time.
_TRAILING_SENTENCE_PUNCT_RT = ".,;:!?。，；：！？"


def _strip_trailing_sentence_punct(span: str) -> str:
    return span.rstrip(_TRAILING_SENTENCE_PUNCT_RT)


def _is_rate_limited(exc: BaseException) -> bool:
    """Detect provider rate-limit / quota errors across SDKs and local-CLI clients.

    Covers openai.RateLimitError (status 429), generic APIStatusError with
    .status_code==429, and CLI-style errors that just carry a message — we
    inspect both the type name and the string representation.
    Also covers ProviderCallError from AnthropicSDKClient which buries
    the HTTP status in diagnostics["error_event"]["api_error_status"].
    """
    name = type(exc).__name__
    if "RateLimit" in name:
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    # AnthropicSDKClient wraps anthropic.APIError as ProviderCallError
    # with the HTTP status in diagnostics["error_event"]["api_error_status"].
    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, dict):
        err_ev = diagnostics.get("error_event")
        if isinstance(err_ev, dict) and err_ev.get("api_error_status") == 429:
            return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "too many requests" in text


def _is_provider_transient_error(exc: BaseException) -> bool:
    """Detect provider-side errors worth retrying through the fallback target:
    rate-limiting (429) AND server-side errors (5xx). Without 5xx handling
    a single bad upstream wedges the worker pool in a tight worker_bail
    loop (lease released, exception swallowed, task re-claimed, same 500,
    repeat — observed ~1 req/sec, 4000+ events/min).

    Explicitly EXCLUDES 4xx errors other than 429 (404 wrong endpoint, 401
    bad key, 400 schema mismatch, etc.). These are configuration problems
    that retrying won't fix; ``_is_provider_permanent_error`` short-circuits
    them straight to HR instead of consuming the 5-bail cap.
    """
    if _is_rate_limited(exc):
        return True
    name = type(exc).__name__
    if "InternalServerError" in name or "ServiceUnavailable" in name or "BadGateway" in name:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 500 <= status < 600:
        return True
    # AnthropicSDKClient wraps HTTP status in diagnostics (same pattern as _is_rate_limited).
    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, dict):
        err_ev = diagnostics.get("error_event")
        if isinstance(err_ev, dict):
            api_status = err_ev.get("api_error_status")
            if isinstance(api_status, int) and 500 <= api_status < 600:
                return True
    text = str(exc).lower()
    return any(s in text for s in (" 500 ", " 502 ", " 503 ", " 504 ",
                                    "internal server error", "service unavailable",
                                    "bad gateway", "gateway timeout"))


def _is_provider_permanent_error(exc: BaseException) -> bool:
    """Detect provider 4xx errors that won't change on retry: 404 (wrong
    endpoint/model), 401 (bad api_key), 400 (malformed request). 429 is
    intentionally NOT covered — it's transient. These errors are config
    bugs; retrying or falling-back to the same broken config is pure
    waste, so the worker should skip the 5-bail dance and route straight
    to HR with the operator-actionable error message.
    """
    if _is_rate_limited(exc):
        return False
    name = type(exc).__name__
    if name in {"NotFoundError", "AuthenticationError", "PermissionDeniedError",
                "BadRequestError", "UnprocessableEntityError"}:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in {400, 401, 403, 404, 422}:
        return True
    text = str(exc).lower()
    if any(s in text for s in ("not found", "unauthorized", "forbidden",
                                    " 400 ", " 401 ", " 403 ", " 404 ", " 422 ")):
        return True
    # ProviderCallError surfaces CLI exit failures with diagnostics
    # (stderr, returncode). Subprocess providers don't raise SDK-style
    # NotFoundError/AuthenticationError — they exit non-zero and bury the
    # cause in stderr. Without this branch, OAuth-broken / bad-api-key /
    # wrong-endpoint failures all classify as transient and back off
    # forever instead of escalating.
    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, dict):
        # claude/codex stream-json `result.is_error=true` event carries
        # the HTTP status in api_error_status. Operator-actionable codes
        # (401/402/403/404/422) classify as permanent so the worker
        # short-circuits to the alert + fallback path instead of bailing
        # in a tight 1-second-per-request retry loop (~3600 wasted calls/h).
        err_ev = diagnostics.get("error_event")
        if isinstance(err_ev, dict):
            status = err_ev.get("api_error_status")
            if isinstance(status, int) and status in {400, 401, 402, 403, 404, 422}:
                return True
        stderr_text = str(diagnostics.get("stderr") or "").lower()
        if any(s in stderr_text for s in (
            "unauthorized", "invalid api key", "invalid_api_key",
            "authentication failed", "auth failed", "auth error",
            "permission denied", "forbidden", "not authenticated",
            "missing api key", " 401 ", " 402 ", " 403 ", " 404 ", " 422 ",
            "insufficient balance", "insufficient_quota", "payment required",
        )):
            return True
    return False


class SubagentRuntime:
    def __init__(
        self,
        store: SqliteStore,
        client_factory: Callable[[str], LLMClient],
        *,
        max_qc_rounds: int | None = None,
        config: RuntimeConfig | None = None,
        structured_output_targets: frozenset[str] = frozenset(),
    ):
        self.store = store
        self.client_factory = client_factory
        # Profile-name cache populated as a side-effect of ``_call_client``
        # and consumed by ``_profile_name_for_target``. Without this, probing
        # for a profile name to validate a pinned continuity handle would
        # call the factory an extra time — cheap in production but in finite-
        # list test stubs it consumes a client and breaks retry flows.
        self._profile_name_cache: dict[str, str | None] = {}
        self._structured_output_targets = structured_output_targets
        # ``config`` carries the project-level QC sampling policy and the
        # max-rounds setting. When omitted (callers that predate the lift, or
        # tests that only care about the per-task flow), fall back to defaults.
        self.config = config or RuntimeConfig()
        # Explicit ``max_qc_rounds`` still wins for backward compat with the
        # local scheduler kwarg that already passed it directly.
        self.max_qc_rounds = (
            max_qc_rounds if max_qc_rounds is not None else self.config.max_qc_rounds
        )
        # Rolling per-role confidence history used to normalize raw model
        # output. LLMs are systematically miscalibrated (QC tends to output
        # 0.85-0.99; annotator the same), so the literal numbers don't
        # compare. Tracking each role's recent min/max and re-scaling lets us
        # treat 0.85 as "low for this role" or "high for this role" depending
        # on the speaker's habits.
        self._confidence_history: dict[str, list[float]] = {"qc": [], "annotator": []}
        self._confidence_window = 200
        self._confidence_min_samples = 10
        # Extracted validator — used in parallel with the inline
        # _check_annotation_validation for now. Will fully replace the inline
        # body once the extraction is verified stable.
        from annotation_pipeline_skill.runtime.annotation_validator import AnnotationValidator
        self._annotation_validator = AnnotationValidator(
            output_schema=None,
            store=self.store,
        )
        # Extracted prompt builder (Task 8). Old method bodies in this class
        # remain intact for now; this wires the new module in for future use.
        from annotation_pipeline_skill.runtime.prompt_builder import AnnotationPromptBuilder
        # SubagentRuntime is store-scoped, not pipeline-scoped; tasks carry
        # their own pipeline_id which AnnotationPromptBuilder resolves at call
        # time (e.g. task.pipeline_id in build_conventions_block).  Pass "" as
        # the instance-level project_id — the hasattr guard that existed here
        # previously always evaluated to "" anyway (no _project_id attribute is
        # set on this class), but the guard was misleading.
        self._prompt_builder = AnnotationPromptBuilder(
            store=self.store,
            project_id="",
            config=self.config,
        )

    def run_once(self, stage_target: str = "annotation", limit: int | None = None) -> SubagentRuntimeResult:
        pending_tasks = self.store.list_tasks_by_status({TaskStatus.PENDING})
        if limit is not None:
            pending_tasks = pending_tasks[:limit]

        accepted = 0
        failed = 0
        for task in pending_tasks:
            try:
                self.run_task(task, stage_target)
            except Exception:
                failed += 1
                continue
            if task.status is TaskStatus.ACCEPTED:
                accepted += 1
        return SubagentRuntimeResult(started=len(pending_tasks), accepted=accepted, failed=failed)

    def run_task(self, task: Task, stage_target: str = "annotation") -> None:
        """Synchronous entry point. Wraps the async core for tests and CLI use."""
        asyncio.run(self.run_task_async(task, stage_target))

    async def run_task_async(self, task: Task, stage_target: str = "annotation") -> None:
        """Async entry point used by the scheduler to run tasks concurrently."""
        await self._run_task(task, stage_target)

    def _load_guideline(self, task: Task) -> str | None:
        # Preferred: task is bound to a versioned AnnotationDocument.
        if task.document_version_id:
            try:
                ver = self.store.load_document_version(task.document_version_id)
                return f"Annotation guideline ({ver.version}):\n{ver.content}"
            except FileNotFoundError:
                pass  # fall through to project-level fallback
        # Secondary: latest version of the singleton "Annotation Rules"
        # document maintained by the dashboard's Annotation Rules tab.
        try:
            for doc in self.store.list_documents():
                if doc.metadata.get("role") == "annotation_rules":
                    versions = self.store.list_document_versions(doc.document_id)
                    if versions:
                        latest = max(versions, key=lambda v: v.created_at)
                        return f"Annotation guideline ({latest.version}):\n{latest.content}"
                    break
        except Exception:
            pass
        return None

    async def _run_task(self, task: Task, stage_target: str) -> None:
        if task.status is TaskStatus.ARBITRATING:
            # Manual rearbitrate path: human dragged a REJECTED/HR card into the
            # Arbitration column. Re-run the arbiter over the full feedback
            # history (including consensus-closed entries from a prior arbiter
            # pass) and dispatch the outcome.
            await self._run_rearbitration(task)
            return

        if task.status is TaskStatus.QC and task.metadata.get("runtime_next_stage") == "qc":
            await self._run_qc_only(task)
            return

        if (
            task.status is TaskStatus.PENDING
            and task.current_attempt == 0
            and task.metadata.get("prelabeled")
        ):
            # Use the pre-label annotation as-is, skipping the LLM call.
            # current_attempt == 0 is the guard: after any real annotation
            # attempt (pass or fail) current_attempt > 0, so we fall through to
            # a fresh LLM call rather than reusing a stale or invalid result.
            prelabeled = [
                artifact for artifact in self.store.list_artifacts(task.task_id)
                if artifact.kind == "annotation_result"
                and artifact.metadata.get("runtime") == "import"
            ]
            if prelabeled:
                annotation_artifact = prelabeled[-1]
                attempts = self.store.list_attempts(task.task_id)
                annotation_attempt_id = (
                    attempts[-1].attempt_id if attempts else f"prelabeled-{task.task_id}"
                )
                task.current_attempt = 1
                payload = self._read_artifact_payload(annotation_artifact)
                if isinstance(payload, dict):
                    final_text = payload.get("text", json.dumps(payload, sort_keys=True))
                else:
                    final_text = json.dumps(payload, sort_keys=True)
                self._transition(
                    task,
                    TaskStatus.ANNOTATING,
                    reason="prelabeled annotation reused; skipping LLM annotation",
                    stage="annotation",
                    attempt_id=annotation_attempt_id,
                    metadata={"prelabeled": True},
                )
                await self._run_validation_and_qc(
                    task,
                    annotation_artifact,
                    annotation_attempt_id,
                    final_text,
                )
                return

        guideline = self._load_guideline(task)
        annotation_attempt_id = self._next_attempt_id(task)
        self._transition(
            task,
            TaskStatus.ANNOTATING,
            reason="subagent runtime started annotation",
            stage="annotation",
            attempt_id=annotation_attempt_id,
        )

        annotation_started_at = utc_now()
        conventions_block = self._build_conventions_block(task)
        continuation_handle = self._read_pinned_handle(task, "continuity_handle", stage_target)
        # Prefix-cache layout — keep the system prompt bytestable across
        # tasks of the same project. Per-task content (conventions block,
        # task source rows, feedback) goes in the user message; project-
        # wide content (schema, validator workflow, span rules) stays in
        # system.
        from annotation_pipeline_skill.core.schema_validation import resolve_output_schema
        annotation_user_prompt = self._annotation_prompt(
            task, continuation_handle=continuation_handle,
        )
        if conventions_block:
            # Conventions are per-task — prepend as a header to the user
            # message rather than appending to the system prompt.
            annotation_user_prompt = (
                conventions_block + "\n\n" + annotation_user_prompt
            )
        _ann_output_schema = resolve_output_schema(task, self.store)
        annotation_result = await self._generate_async(
            stage_target,
            LLMGenerateRequest(
                instructions=_annotation_instructions(
                    task,
                    guideline=guideline,
                    output_schema=_ann_output_schema,
                ),
                prompt=annotation_user_prompt,
                continuity_handle=continuation_handle,
                response_format=self._build_response_format(
                    stage_target, stage="annotation", output_schema=_ann_output_schema
                ),
                task_id=task.task_id,
            ),
        )
        annotation_finished_at = utc_now()
        task.current_attempt += 1
        cleaned_annotation_text = _serialize_llm_json(annotation_result.final_text, task=task)
        annotation_artifact = self._write_stage_artifact(
            task,
            annotation_result,
            kind="annotation_result",
            attempt_id=annotation_attempt_id,
            payload={"text": cleaned_annotation_text},
        )
        self._append_attempt(
            Attempt(
                attempt_id=annotation_attempt_id,
                task_id=task.task_id,
                index=task.current_attempt,
                stage="annotation",
                status=AttemptStatus.SUCCEEDED,
                started_at=annotation_started_at,
                finished_at=annotation_finished_at,
                provider_id=annotation_result.provider,
                model=annotation_result.model,
                effort=None,
                route_role=stage_target,
                summary=annotation_result.final_text[:500],
                artifacts=[annotation_artifact],
            ),
            annotation_artifact,
        )
        self._record_annotator_replies(task, annotation_attempt_id, annotation_result.final_text)

        # Confidence-based early escalation: both sides uncertain on at least
        # one open feedback → bounce to human reviewer instead of burning more
        # rounds. _record_annotator_replies sets the flag.
        if task.metadata.pop("needs_early_hr_low_confidence", False):
            low_ids = task.metadata.get("low_confidence_feedback_ids", [])
            reason_key = task.metadata.get("early_hr_reason", "low_confidence")
            reason_msg = {
                "low_confidence": "escalated: QC and annotator both have low confidence (<0.5) on disputed feedback",
                "high_confidence_stalemate": "escalated: QC and annotator both highly confident (>=0.85) and disagreeing — semantic stalemate",
            }.get(reason_key, "escalated: confidence-based dispute resolution selected human review")
            arb = await self._arbitrate_and_apply(task, annotation_attempt_id, stage="annotation")
            terminal = self._terminal_from_arbiter(task, annotation_attempt_id, "annotation", arb)
            if terminal is not None:
                # Arbiter made an authoritative call — ACCEPTED or REJECTED.
                return
            if arb["closed"] > 0 and self._retry_round_count(task.task_id) == 0:
                # All open disputes closed in annotator's favor; resume normal loop.
                task.metadata.pop("needs_early_hr_low_confidence", None)
                task.metadata.pop("early_hr_reason", None)
                task.metadata.pop("low_confidence_feedback_ids", None)
                task.metadata.pop("early_hr_confidence", None)
            else:
                self._transition(
                    task,
                    TaskStatus.HUMAN_REVIEW,
                    reason=reason_msg,
                    stage="annotation",
                    attempt_id=annotation_attempt_id,
                    metadata={
                        "low_confidence_feedback_ids": low_ids,
                        "early_hr_reason": reason_key,
                        "early_hr_confidence": task.metadata.get("early_hr_confidence", {}),
                        "arbiter_ran": arb["ran"],
                        "arbiter_unresolved": arb["unresolved"],
                    },
                )
                return

        self._write_pinned_handle(
            task, "continuity_handle",
            annotation_result.continuity_handle, annotation_result.provider,
        )
        self._snapshot_sent_feedback(task)
        # Validation runs against the CLEANED text so any auto-fix done in
        # _serialize_llm_json (boundary trims, near-verbatim alignments) is
        # what the validators see. Without this, validation parsed the raw
        # LLM output and could reject a span the artifact had already cleaned
        # up — costing a free retry on a non-issue.
        await self._run_validation_and_qc(
            task,
            annotation_artifact,
            annotation_attempt_id,
            cleaned_annotation_text,
        )

    # Hard cap on consecutive arbiter mechanical retries. After this many
    # arbiter pickups produce no actionable verdict (codex error / no fix /
    # bad correction), give up and route to HR. Prevents a stuck task from
    # looping forever when the LLM consistently fails on it.
    ARBITER_MECHANICAL_RETRY_CAP = 3
    # Separate, much larger budget for verbatim-only failures. Reason:
    # validation-layer auto-fix (auto_fix_safe_spans_in_place) absorbs
    # boundary-only mismatches at write time, so verbatim-exhausted on the
    # arbiter side typically means a genuine model hallucination — but it's
    # also the LLM-noise-prone failure mode that tends to clear on the
    # next pickup. Give it ~2× the mechanical budget before escalating; if
    # the arbiter is consistently hallucinating on this task, HR is still
    # the right destination, just not after 3 pickups.
    ARBITER_VERBATIM_BAIL_CAP = 6

    def _handle_arbiter_mechanical_fail(
        self,
        task: Task,
        attempt_id: str,
        arb: dict,
        stage: str,
        hr_extra_metadata: dict,
    ) -> None:
        """Bump the per-task arbiter-retry counter for the right failure
        mode (mechanical shape/parse vs verbatim-exhausted). If the matching
        cap is reached, transition to HR. Otherwise leave the task in
        ARBITRATING for re-pickup — the next worker takes a fresh shot.
        """
        if arb.get("rate_limited"):
            # Transient error (429 / 5xx). Don't count toward any failure cap.
            # Stamp next_retry_at with exponential backoff (30s × bail#, cap 300s)
            # so the task isn't immediately re-claimed against the same overloaded
            # provider.
            bail_n = int(task.metadata.get("arbiter_transient_bail_count", 0)) + 1
            task.metadata["arbiter_transient_bail_count"] = bail_n
            backoff = min(30 * bail_n, 300)
            task.next_retry_at = utc_now() + timedelta(seconds=backoff)
            return

        verbatim_exhausted = bool(arb.get("verbatim_retry_exhausted"))
        if verbatim_exhausted:
            count = int(task.metadata.get("arbiter_verbatim_bail_count", 0)) + 1
            task.metadata["arbiter_verbatim_bail_count"] = count
            cap = self.ARBITER_VERBATIM_BAIL_CAP
            if count >= cap:
                metadata = {
                    **hr_extra_metadata,
                    "arbiter_mechanical_retries": int(task.metadata.get("arbiter_mechanical_retries", 0)),
                    "arbiter_verbatim_bail_count": count,
                    "arbiter_ran": arb["ran"],
                    "arbiter_unresolved": arb["unresolved"],
                    "arbiter_mechanical_fail": arb["mechanical_fail"],
                    "arbiter_verbatim_retry_exhausted": True,
                    "arbiter_failed_correction": arb.get("failed_verbatim_correction"),
                    "arbiter_failed_verbatim_target": arb.get("failed_verbatim_target"),
                }
                for k in ("arbiter_last_exception_class", "arbiter_last_exception_message"):
                    if task.metadata.get(k):
                        metadata.setdefault(k, task.metadata[k])
                target = arb.get("failed_verbatim_target") or {}
                failed_span = target.get("span")
                if failed_span:
                    self.store.append_feedback(
                        FeedbackRecord.new(
                            task_id=task.task_id,
                            attempt_id=attempt_id,
                            source_stage=FeedbackSource.HUMAN_REVIEW,
                            severity=FeedbackSeverity.ERROR,
                            category="arbiter_correction_failed",
                            message=(
                                f"Arbiter could not produce a verbatim-compliant correction "
                                f"for span {failed_span!r} at {target.get('field')!r} "
                                f"row {target.get('row_index')} after {count} attempt(s). "
                                f"The span does not appear verbatim in the source text."
                            ),
                            target={
                                "span": failed_span,
                                "field": target.get("field"),
                                "row_index": target.get("row_index"),
                            },
                            suggested_action="request_changes",
                            created_by="arbiter",
                        )
                    )
                self._transition(
                    task,
                    TaskStatus.HUMAN_REVIEW,
                    reason=(
                        f"Arbiter ruled qc/neither but could not produce a verbatim-compliant "
                        f"correction after {count} pickup(s); routing to human review "
                        f"(failed span: {target.get('span')!r} at "
                        f"{target.get('field')!r} row {target.get('row_index')})"
                    ),
                    stage=stage,
                    attempt_id=attempt_id,
                    metadata=metadata,
                )
        else:
            # Mechanical failure (JSON parse / shape errors / LLM exception).
            # These are transient quality glitches — replaying arbitration usually
            # succeeds. Stay in ARBITRATING with exponential backoff instead of
            # escalating to HR; HR cannot fix a parse error and the task would
            # just be rewound anyway.
            count = int(task.metadata.get("arbiter_mechanical_retries", 0)) + 1
            task.metadata["arbiter_mechanical_retries"] = count
            for k in ("exception_class", "exception_message"):
                if arb.get(k):
                    task.metadata[f"arbiter_last_{k}"] = arb[k]
            if count >= self.ARBITER_MECHANICAL_RETRY_CAP:
                self._transition(
                    task,
                    TaskStatus.HUMAN_REVIEW,
                    reason=(
                        f"Arbiter retried {count} times but kept failing to return a usable "
                        f"answer (JSON parse / shape errors); routing to human review"
                    ),
                    stage=stage,
                    attempt_id=attempt_id,
                    metadata={
                        **hr_extra_metadata,
                        "arbiter_mechanical_retries": count,
                        "arbiter_ran": arb["ran"],
                        "arbiter_mechanical_fail": arb["mechanical_fail"],
                    },
                )
            else:
                backoff = min(30 * count, 300)
                task.next_retry_at = utc_now() + timedelta(seconds=backoff)

    def _retry_round_count(self, task_id: str) -> int:
        """Count how many *open* retry rounds have happened for this task.

        A round is a QC/VALIDATION rejection attempt. Multiple feedback records
        sharing the same attempt_id belong to the same round. A round is closed
        only when ALL its feedback records have reached consensus; otherwise it
        still counts toward the escalation threshold.
        """
        discussions = self.store.list_feedback_discussions(task_id)
        consensus_ids = {d.feedback_id for d in discussions if d.consensus}
        open_attempt_ids: set[str] = {
            f.attempt_id
            for f in self.store.list_feedback(task_id)
            if (f.source_stage is FeedbackSource.QC or f.source_stage is FeedbackSource.VALIDATION)
            and f.feedback_id not in consensus_ids
        }
        return len(open_attempt_ids)

    async def _run_validation_and_qc(
        self,
        task: Task,
        annotation_artifact: ArtifactRef,
        annotation_attempt_id: str,
        annotation_final_text: str,
    ) -> None:
        validation_failure = self._check_annotation_validation(task, annotation_final_text)
        if validation_failure is not None:
            # For verbatim failures specifically: emit ONE feedback per
            # violation instead of just the first. Otherwise the arbiter
            # only sees one bad span, fixes that one, and the merge step
            # in _apply_arbiter_correction brings the OTHER untouched
            # violations back from the annotator's output — causing the
            # corrected annotation to fail validation on rows the arbiter
            # was never asked about. Burns a mechanical retry per missed
            # violation.
            extra_violations: list[dict] = []
            if validation_failure["category"] == "non_verbatim_span":
                try:
                    payload = _parse_llm_json(annotation_final_text)
                    from annotation_pipeline_skill.core.schema_validation import (
                        find_verbatim_violations,
                    )
                    all_violations = find_verbatim_violations(task, payload)
                    extra_violations = all_violations[1:]  # first is already in validation_failure
                except (json.JSONDecodeError, ValueError):
                    pass
            self._record_validation_feedback(
                task,
                annotation_attempt_id,
                category=validation_failure["category"],
                message=validation_failure["message"],
                target=validation_failure.get("target", {}),
            )
            for v in extra_violations:
                self._record_validation_feedback(
                    task,
                    annotation_attempt_id,
                    category="non_verbatim_span",
                    message=(
                        f"Row {v['row_index']} {v['field']}: span {v['span']!r} "
                        f"is not a verbatim substring of the input text."
                    ),
                    target=v,
                )
            round_count = self._retry_round_count(task.task_id)
            if round_count >= self.max_qc_rounds:
                # Last shot before HR: invoke the arbiter even if the
                # annotator never produced a discussion rebuttal. Without
                # this, silent annotators (models that don't emit
                # discussion_replies) bypass arbitration entirely and
                # always fall through to HR — see audit metadata where
                # arbiter_ran=False and arbiter_unresolved=0.
                arb = await self._arbitrate_and_apply(
                    task, annotation_attempt_id, stage="validation",
                    require_rebuttal=False,
                )
                terminal = self._terminal_from_arbiter(task, annotation_attempt_id, "validation", arb)
                if terminal is not None:
                    self.store.save_task(task)
                    return
                # HR only when arbiter said tentative/unsure on at least one
                # verdict. Mechanical failures (codex error, missing fix,
                # bad correction) keep the task in ARBITRATING so the next
                # worker pickup re-runs the arbiter — no point sending back
                # to the annotator, the annotation didn't change.
                if arb["unresolved"] > 0:
                    # First arbiter uncertain — defer to a second arbiter rather
                    # than escalating immediately. Scheduler detects the flag and
                    # calls _resolve_uncertain_arbiter_async.
                    task.metadata["arbiter_uncertain_needs_second"] = True
                    self.store.save_task(task)
                else:
                    self._handle_arbiter_mechanical_fail(
                        task, annotation_attempt_id, arb, stage="validation",
                        hr_extra_metadata={"round_count": round_count, "max_qc_rounds": self.max_qc_rounds},
                    )
            else:
                self._transition(
                    task,
                    TaskStatus.PENDING,
                    reason=validation_failure["reason"],
                    stage="validation",
                    attempt_id=annotation_attempt_id,
                )
            self.store.save_task(task)
            return

        # Non-blocking quality warnings — record before QC so the next
        # round's feedback bundle includes them, but don't bounce the task.
        # Currently: duplicate same-type spans (auto-deduped at serialize,
        # but worth flagging so the annotator learns).
        self._record_duplicate_warning_feedback(task, annotation_attempt_id, annotation_final_text)
        # Annotation succeeded — reset the scheduler's worker_bail_count so
        # a streak of past failures doesn't trip the bail-cap if the task
        # re-enters annotation later via a normal QC rerun.
        task.metadata.pop("worker_bail_count", None)
        self._transition(
            task,
            TaskStatus.QC,
            reason="deterministic validation passed",
            stage="qc",
            attempt_id=annotation_attempt_id,
        )
        await self._run_qc_stage(task, annotation_artifact)
        self.store.save_task(task)

    def _record_duplicate_warning_feedback(
        self, task: Task, attempt_id: str, annotation_text: str
    ) -> None:
        from annotation_pipeline_skill.core.schema_validation import find_duplicate_spans
        try:
            payload = _parse_llm_json(annotation_text)
        except (json.JSONDecodeError, ValueError):
            return
        dups = find_duplicate_spans(payload)
        if not dups:
            return
        sample = dups[0]
        self.store.append_feedback(
            FeedbackRecord.new(
                task_id=task.task_id,
                attempt_id=attempt_id,
                source_stage=FeedbackSource.VALIDATION,
                severity=FeedbackSeverity.WARNING,
                category="duplicate_span",
                message=(
                    f"Found {len(dups)} duplicate span(s) within entity/json_structures types. "
                    f"First: row {sample['row_index']} {sample['field']} repeats {sample['span']!r}. "
                    f"Each (type, span) pair should appear at most once per row. "
                    f"Auto-deduped at write time; eliminate the duplicate in the next emission."
                ),
                target={"duplicates": dups},
                suggested_action="annotator_dedupe",
                created_by="validation",
            )
        )

    async def _run_qc_only(self, task: Task) -> None:
        annotation_artifact = self._latest_annotation_artifact(task.task_id)
        await self._run_qc_stage(task, annotation_artifact)
        self.store.save_task(task)

    async def _run_qc_stage(self, task: Task, annotation_artifact: ArtifactRef) -> None:
        guideline = self._load_guideline(task)
        qc_attempt_id = self._next_attempt_id(task)
        qc_started_at = utc_now()
        qc_user_prompt = self._qc_prompt(task, annotation_artifact)
        conventions_block = self._build_conventions_block(task)
        if conventions_block:
            qc_user_prompt = conventions_block + "\n\n" + qc_user_prompt
        qc_result = await self._generate_async(
            "qc",
            LLMGenerateRequest(
                instructions=self._qc_instructions(task, guideline=guideline),
                prompt=qc_user_prompt,
                continuity_handle=self._read_pinned_handle(task, "qc_continuity_handle", "qc"),
                response_format=self._build_response_format("qc", stage="qc"),
                task_id=task.task_id,
            ),
        )
        qc_finished_at = utc_now()
        try:
            qc_decision = _parse_qc_decision(qc_result.final_text)
        except QCParseError as exc:
            self._record_qc_parse_error(task, qc_attempt_id, qc_result, exc, started_at=qc_started_at)
            raise
        task.current_attempt += 1
        qc_artifact = self._write_stage_artifact(
            task,
            qc_result,
            kind="qc_result",
            attempt_id=qc_attempt_id,
            payload={"decision": qc_decision},
        )
        self._append_attempt(
            Attempt(
                attempt_id=qc_attempt_id,
                task_id=task.task_id,
                index=task.current_attempt,
                stage="qc",
                status=AttemptStatus.SUCCEEDED,
                started_at=qc_started_at,
                finished_at=qc_finished_at,
                provider_id=qc_result.provider,
                model=qc_result.model,
                effort=None,
                route_role="qc",
                summary=qc_result.final_text[:500],
                artifacts=[qc_artifact],
            ),
            qc_artifact,
        )

        self._write_pinned_handle(
            task, "qc_continuity_handle",
            qc_result.continuity_handle, qc_result.provider,
        )
        task.metadata.pop("runtime_next_stage", None)
        # Honor explicit consensus from QC (e.g. accepted annotator rebuttal)
        # even when overall QC verdict is fail — those specific feedbacks are
        # closed by consensus and won't count toward future retry rounds.
        self._record_explicit_consensus(task, qc_attempt_id, qc_artifact, qc_decision)
        if qc_decision["passed"]:
            self._record_feedback_resolution(task, qc_attempt_id, qc_artifact, qc_decision)
            # Prior verifier: compare each (span, type) against project history
            # BEFORE recording any convention. Spec §3.2 only allows conventions
            # to grow from "annotator+QC consensus + verifier agree" — divergent
            # and cold_start paths must not contribute to the dictionary.
            verifier_failure = self._check_prior_verifier_on_annotation(
                task, annotation_artifact
            )
            if verifier_failure is not None:
                # Divergent — route to ARBITRATING for first-arbiter resolution.
                # No convention update (the verifier just flagged the decision).
                self.store.append_feedback_many([vf["feedback"] for vf in verifier_failure])
                self._transition(
                    task,
                    TaskStatus.ARBITRATING,
                    reason="prior verifier flagged divergence at QC pass",
                    stage="prior_verifier",
                    attempt_id=qc_attempt_id,
                    metadata={
                        "qc_artifact_id": qc_artifact.artifact_id,
                        "prior_verifier_action": "qc_pass_divergent",
                        "verifier_payload": verifier_failure[0]["payload"],
                    },
                )
                self.store.save_task(task)
                return
            # Agree or cold_start. Only the agree path contributes to
            # conventions (spec §3.2); cold_start has no prior to confirm.
            # Stats++ on both paths (broad verifier-source signal).
            if self._verifier_confirmed_all_spans(task, annotation_artifact):
                self._record_conventions_from_qc_consensus(task, annotation_artifact)
            self._increment_entity_statistics_for_task(task, annotation_artifact, weight=1)
            self._transition(
                task,
                TaskStatus.ACCEPTED,
                reason="subagent qc accepted result",
                stage="qc",
                attempt_id=qc_attempt_id,
                metadata={"qc_artifact_id": qc_artifact.artifact_id},
            )
        else:
            feedbacks = _feedback_from_qc_decision(task, qc_attempt_id, qc_decision)
            self.store.append_feedback_many(feedbacks)
            qc_conf = _clamp_confidence(feedbacks[0].metadata.get("confidence"))
            if qc_conf is not None:
                self._record_confidence_sample("qc", qc_conf)
            round_count = self._retry_round_count(task.task_id)
            if round_count >= self.max_qc_rounds:
                # Last shot before HR: same rationale as the validation path.
                arb = await self._arbitrate_and_apply(
                    task, qc_attempt_id, stage="qc",
                    require_rebuttal=False,
                )
                terminal = self._terminal_from_arbiter(task, qc_attempt_id, "qc", arb)
                if terminal is not None:
                    self.store.save_task(task)
                    return
                # HR only on genuine arbiter uncertainty. Mechanical failures
                # leave the task in ARBITRATING for re-pickup; the arbiter
                # gets another shot on the same annotation.
                if arb["unresolved"] > 0:
                    # First arbiter uncertain — defer to second arbiter.
                    task.metadata["arbiter_uncertain_needs_second"] = True
                    self.store.save_task(task)
                else:
                    self._handle_arbiter_mechanical_fail(
                        task, qc_attempt_id, arb, stage="qc",
                        hr_extra_metadata={
                            "round_count": round_count,
                            "max_qc_rounds": self.max_qc_rounds,
                            "feedback_id": feedbacks[0].feedback_id,
                            "qc_artifact_id": qc_artifact.artifact_id,
                        },
                    )
            else:
                self._transition(
                    task,
                    TaskStatus.PENDING,
                    reason="subagent qc requested annotator rerun",
                    stage="qc",
                    attempt_id=qc_attempt_id,
                    metadata={"feedback_id": feedbacks[0].feedback_id, "qc_artifact_id": qc_artifact.artifact_id},
                )
        self.store.save_task(task)

    def _record_conventions_from_qc_consensus(
        self,
        task: Task,
        annotation_artifact: ArtifactRef,
    ) -> None:
        """Capture entity-type decisions from an annotator+QC consensus into
        the project's convention dictionary.

        Trigger: QC passed without arbiter intervention. The (span, type)
        decisions in the current annotation reflect joint annotator+QC
        agreement, suitable for guiding future tasks. Decisions made by
        arbiter (the closed/fixed acceptance paths) are intentionally NOT
        recorded — arbiter is another LLM, not human-level authority.
        """
        from annotation_pipeline_skill.services.entity_convention_service import (
            EntityConventionService,
            extract_entity_type_decisions_with_row,
        )
        # Read the latest annotation payload (cleaned canonical JSON written
        # by _serialize_llm_json) and the prelabel baseline to diff against.
        try:
            current = self._read_artifact_payload(annotation_artifact)
            if not isinstance(current, dict):
                return
            text = current.get("text")
            if isinstance(text, str):
                try:
                    current = _parse_llm_json(text)
                except (json.JSONDecodeError, ValueError):
                    return
            prelabel = None
            for art in self.store.list_artifacts(task.task_id):
                if art.kind == "annotation_result" and art.metadata.get("provider") == "prelabel":
                    prelabel_outer = self._read_artifact_payload(art)
                    if isinstance(prelabel_outer, dict):
                        pre_text = prelabel_outer.get("text")
                        if isinstance(pre_text, str):
                            try:
                                prelabel = _parse_llm_json(pre_text)
                            except (json.JSONDecodeError, ValueError):
                                prelabel = None
                        else:
                            prelabel = prelabel_outer
                    break
            # Pull source rows from the task payload so the extractor can
            # attach row_content for each decision (used by the KB MCP tool).
            source_rows: list[dict] | None = None
            try:
                payload = task.source_ref["payload"]
                if isinstance(payload, dict):
                    candidate = payload.get("rows")
                    if isinstance(candidate, list):
                        source_rows = candidate
            except (KeyError, TypeError):
                source_rows = None

            decisions = extract_entity_type_decisions_with_row(
                prelabel or {}, current, source_rows=source_rows,
            )
            if not decisions:
                return
            svc = EntityConventionService(self.store)
            for span, entity_type, row_id, row_content in decisions:
                try:
                    svc.record_decision(
                        project_id=task.pipeline_id,
                        span=span,
                        entity_type=entity_type,
                        source="qc_consensus",
                        task_id=task.task_id,
                        row_id=row_id,
                        row_content=row_content,
                    )
                except Exception:  # noqa: BLE001 — convention recording is best-effort
                    continue
        except Exception:  # noqa: BLE001
            return

    def _load_annotation_payload(self, annotation_artifact: ArtifactRef) -> dict | None:
        """Read the canonical JSON annotation payload from an artifact.

        Mirrors how _record_conventions_from_qc_consensus already reads it,
        kept as a single helper so the QC-pass / arbiter / HR sites all
        share the same parsing semantics.
        """
        try:
            outer = self._read_artifact_payload(annotation_artifact)
            if not isinstance(outer, dict):
                return None
            text = outer.get("text")
            if isinstance(text, str):
                try:
                    return _parse_llm_json(text)
                except (json.JSONDecodeError, ValueError):
                    return None
            return outer
        except Exception:  # noqa: BLE001
            return None

    def _check_prior_verifier_on_annotation(
        self,
        task: Task,
        annotation_artifact: ArtifactRef,
    ) -> list[dict] | None:
        """Return list of {feedback, payload} for ALL divergent (span, type) pairs,
        or None when every span is agree/cold_start.
        """
        from annotation_pipeline_skill.services.entity_statistics_service import (
            EntityStatisticsService,
            iter_span_decisions,
        )
        payload = self._load_annotation_payload(annotation_artifact)
        if payload is None:
            return None
        svc = EntityStatisticsService(self.store)
        attempts = self.store.list_attempts(task.task_id)
        attempt_id = attempts[-1].attempt_id if attempts else f"{task.task_id}-attempt-0"
        divergents: list[dict] = []
        for span, entity_type in iter_span_decisions(payload):
            result = svc.check(
                project_id=task.pipeline_id,
                span=span,
                proposed_type=entity_type,
            )
            if result.status != "divergent":
                continue
            verifier_payload = {
                "span": result.span,
                "proposed_type": result.proposed_type,
                "dominant_type": result.dominant_type,
                "dominant_count": result.dominant_count,
                "total": result.total,
                "distribution": result.distribution,
            }
            divergents.append({
                "payload": verifier_payload,
                "feedback": FeedbackRecord.new(
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    source_stage=FeedbackSource.VALIDATION,
                    severity=FeedbackSeverity.BLOCKING,
                    category="prior_disagreement",
                    message=(
                        f"Span {result.span!r} was classified as {result.proposed_type!r} "
                        f"but project history (N={result.total}) puts "
                        f"{result.dominant_count}/{result.total} "
                        f"({result.dominant_count * 100 // result.total}%) under "
                        f"{result.dominant_type!r}. Re-evaluate via arbiter."
                    ),
                    target=verifier_payload,
                    suggested_action="arbiter_rerun",
                    created_by="prior_verifier",
                ),
            })
        return divergents if divergents else None

    def _mark_first_arbiter_divergence_if_any(
        self,
        task: Task,
        annotation_artifact: ArtifactRef,
    ) -> None:
        """If the just-accepted annotation has any (span, type) that diverges
        from prior, stash the verifier payload on task.metadata so the
        second-arbiter trigger (next task) can detect and invoke.

        The caller is expected to invoke ``_transition`` immediately after
        this returns, which persists ``task`` (and therefore the metadata)
        via ``store.save_task``. Calling this twice is a no-op overwrite —
        the helper is idempotent.
        """
        divergence = self._check_prior_verifier_on_annotation(task, annotation_artifact)
        if divergence is None:
            return
        task.metadata["prior_verifier_first_arbiter_divergent"] = True
        task.metadata["prior_verifier_payload"] = divergence[0]["payload"]

    def _verifier_confirmed_all_spans(
        self,
        task: Task,
        annotation_artifact: ArtifactRef,
    ) -> bool:
        """True only when every (span, type) in the annotation received an
        ``agree`` verdict from the verifier — i.e. each span had a prior
        with ≥ MIN_PRIOR_SAMPLES total observations and the dominant type
        matched the annotator+QC consensus (or no dominant existed).

        cold_start spans (insufficient prior) do NOT count as confirmed.
        Per spec §3.2, conventions only grow from confirmed consensus so
        the dictionary stays a high-trust subset of the broader stats.
        """
        from annotation_pipeline_skill.services.entity_statistics_service import (
            EntityStatisticsService,
            iter_span_decisions,
        )
        payload = self._load_annotation_payload(annotation_artifact)
        if payload is None:
            return False
        svc = EntityStatisticsService(self.store)
        any_agree = False
        for span, entity_type in iter_span_decisions(payload):
            r = svc.check(
                project_id=task.pipeline_id,
                span=span,
                proposed_type=entity_type,
            )
            if r.status == "divergent":
                return False
            if r.status == "agree":
                any_agree = True
        return any_agree

    def _increment_entity_statistics_for_task(
        self,
        task: Task,
        annotation_artifact: ArtifactRef,
        *,
        weight: int,
    ) -> None:
        """Increment entity_statistics for every (span, type) in the task's
        final annotation. Best-effort — never raise to the caller.
        """
        from annotation_pipeline_skill.services.entity_statistics_service import (
            EntityStatisticsService,
            iter_span_decisions,
        )
        payload = self._load_annotation_payload(annotation_artifact)
        if payload is None:
            return
        svc = EntityStatisticsService(self.store)
        for span, entity_type in iter_span_decisions(payload):
            try:
                svc.increment(
                    project_id=task.pipeline_id,
                    span=span,
                    entity_type=entity_type,
                    weight=weight,
                )
            except Exception:  # noqa: BLE001
                continue

    def _record_feedback_resolution(
        self,
        task: Task,
        qc_attempt_id: str,
        qc_artifact: ArtifactRef,
        qc_decision: dict[str, Any],
    ) -> None:
        open_feedback_ids = build_feedback_consensus_summary(self.store, task.task_id)["open_feedback"]
        if not open_feedback_ids:
            return

        message = str(qc_decision.get("summary") or "Resolved by a subsequent QC pass.")
        for feedback_id in open_feedback_ids:
            self.store.append_feedback_discussion(
                FeedbackDiscussionEntry.new(
                    task_id=task.task_id,
                    feedback_id=feedback_id,
                    role="qc",
                    stance="resolved",
                    message=message,
                    proposed_resolution="Subsequent annotation attempt passed QC.",
                    consensus=True,
                    created_by="qc-agent",
                    metadata={
                        "attempt_id": qc_attempt_id,
                        "qc_artifact_id": qc_artifact.artifact_id,
                        "resolution_source": "subsequent_qc_pass",
                    },
                )
            )

    def _record_explicit_consensus(
        self,
        task: Task,
        qc_attempt_id: str,
        qc_artifact: ArtifactRef,
        qc_decision: dict[str, Any],
    ) -> None:
        """Mark feedbacks as resolved by consensus when QC explicitly acks an annotator rebuttal."""
        ack_ids = qc_decision.get("consensus_acknowledgements") or []
        if not ack_ids:
            return
        known_feedback_ids = {f.feedback_id for f in self.store.list_feedback(task.task_id)}
        for feedback_id in ack_ids:
            if feedback_id not in known_feedback_ids:
                continue
            self.store.append_feedback_discussion(
                FeedbackDiscussionEntry.new(
                    task_id=task.task_id,
                    feedback_id=feedback_id,
                    role="qc",
                    stance="agree",
                    message="QC accepted annotator rebuttal; feedback closed by consensus.",
                    consensus=True,
                    created_by="qc-agent",
                    metadata={
                        "attempt_id": qc_attempt_id,
                        "qc_artifact_id": qc_artifact.artifact_id,
                        "resolution_source": "consensus_acknowledgement",
                    },
                )
            )

    def _latest_annotation_artifact(self, task_id: str) -> ArtifactRef:
        annotation_artifacts = [
            artifact for artifact in self.store.list_artifacts(task_id)
            if artifact.kind == "annotation_result"
        ]
        if not annotation_artifacts:
            raise QCParseError("QC retry requires an annotation artifact.", raw_text="")
        return annotation_artifacts[-1]

    def _record_qc_parse_error(
        self,
        task: Task,
        attempt_id: str,
        result: LLMGenerateResult,
        error: QCParseError,
        *,
        started_at: datetime,
    ) -> None:
        finished_at = utc_now()
        task.current_attempt += 1
        artifact = self._write_stage_artifact(
            task,
            result,
            kind="qc_result",
            attempt_id=attempt_id,
            payload={"parse_error": error.diagnostics},
        )
        self._append_attempt(
            Attempt(
                attempt_id=attempt_id,
                task_id=task.task_id,
                index=task.current_attempt,
                stage="qc",
                status=AttemptStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                provider_id=result.provider,
                model=result.model,
                route_role="qc",
                summary=str(error),
                error={"kind": "parse_error", "message": str(error)},
                artifacts=[artifact],
            ),
            artifact,
        )
        self._write_pinned_handle(
            task, "qc_continuity_handle",
            result.continuity_handle, result.provider,
        )
        task.metadata["runtime_next_stage"] = "qc"
        self.store.save_task(task)

    def _generate(self, target: str, request: LLMGenerateRequest) -> LLMGenerateResult:
        """Sync wrapper retained for any external callers; the runtime uses _generate_async."""
        return asyncio.run(self._generate_async(target, request))

    # Class-level cooldown for provider alerts: dedup (target, status) within
    # ALERT_COOLDOWN_SECONDS so a 1000-task 402 storm doesn't write 1000 log
    # lines. We keep one alert per (target, status) per cooldown window.
    _provider_alert_cooldown: dict[tuple[str, Any], float] = {}
    ALERT_COOLDOWN_SECONDS: float = 300.0

    def _emit_provider_alert(self, target: str, exc: BaseException) -> None:
        """Surface a user-actionable provider error (401/402/403/404/422/etc)
        to stderr + the project's alerts.jsonl file. Deduplicated by
        (target, api_error_status) within a 5-min cooldown.

        Triggered when ``_is_provider_permanent_error(exc)`` is True. The
        operator needs to act (refill balance, fix API key, swap model) —
        the runtime cannot self-heal these, so it should fall back AND
        loudly tell whoever is watching.
        """
        import sys, time
        diag = getattr(exc, "diagnostics", None) or {}
        err_ev = diag.get("error_event") if isinstance(diag, dict) else None
        status: Any = None
        message = str(exc)[:200]
        if isinstance(err_ev, dict):
            status = err_ev.get("api_error_status")
            if err_ev.get("result_text"):
                message = str(err_ev["result_text"])[:300]
        cooldown_key = (target, status if status is not None else type(exc).__name__)
        now = time.time()
        last = SubagentRuntime._provider_alert_cooldown.get(cooldown_key, 0.0)
        if now - last < SubagentRuntime.ALERT_COOLDOWN_SECONDS:
            return
        SubagentRuntime._provider_alert_cooldown[cooldown_key] = now
        banner = (
            f"\n🚨 PROVIDER ALERT  target={target}  status={status}  "
            f"class={type(exc).__name__}\n   {message}\n"
            f"   (operator action required — fallback target will be tried automatically)\n"
        )
        try:
            print(banner, file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001
            pass
        from annotation_pipeline_skill.runtime.alerts import append_alert
        append_alert(self.store.root, {
            "ts": utc_now().isoformat(),
            "kind": "provider_alert",
            "target": target,
            "api_error_status": status,
            "exception_class": type(exc).__name__,
            "message": message,
        })

    def _emit_enum_coerce_alert(
        self,
        task: Task,
        dropped: dict[str, int],
        rescued: dict[str, int] | None = None,
    ) -> None:
        """Log a warning when the arbiter put valid types in the wrong field
        (rescued) or invented non-schema types (dropped).
        """
        logger.warning(
            "arbiter_enum_coerce task=%s dropped=%s rescued=%s",
            task.task_id, dropped, rescued or {},
        )

    async def _generate_async(self, target: str, request: LLMGenerateRequest) -> LLMGenerateResult:
        try:
            return await self._call_client(target, request)
        except Exception as exc:  # noqa: BLE001 — try fallback on transient/permanent provider errors
            if target == "fallback":
                raise
            if _is_provider_transient_error(exc):
                try:
                    return await self._call_client("fallback", request)
                except Exception:  # noqa: BLE001 — fallback unavailable/failed; re-raise original
                    raise exc from None
            if _is_provider_permanent_error(exc):
                # Operator-actionable error (auth/balance/wrong model).
                # Alert + try fallback once. If fallback also fails, raise
                # the ORIGINAL exception so HR metadata pins the real cause.
                self._emit_provider_alert(target, exc)
                try:
                    return await self._call_client("fallback", request)
                except Exception:  # noqa: BLE001
                    raise exc from None
            raise

    def _profile_name_for_target(self, target: str) -> str | None:
        """Return the LLM profile name observed for ``target``, or ``None``
        if no client for that target has been constructed yet in this
        runtime instance.

        Used to invalidate cross-provider continuity handles — a
        ``previous_response_id`` minted by codex (e.g.) is meaningless to
        a Qwen-backed gateway and causes the gateway to 404. The cache is
        populated as a side effect of ``_call_client``; this avoids the
        eager-probe pattern (constructing a throwaway client just to read
        ``.profile.name``) which is cheap in production but exhausts
        finite-list test stubs.

        On a true cache miss (first call ever for a target) the caller
        gracefully degrades — returning ``None`` here makes
        ``_read_pinned_handle`` accept the handle as-is. If the handle is
        actually stale, the upstream 404 will be observed and the runtime
        will retry without it.
        """
        return self._profile_name_cache.get(target)

    def _build_response_format(
        self,
        target: str,
        *,
        stage: str,
        output_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the appropriate response_format for a generate call.

        Structured output is project-driven: when output_schema is present
        (the project has defined an annotation schema), strict json_schema
        enforcement is used for all three stages. QC uses a fixed schema
        independent of the project schema. Falls back to json_object only
        when no schema is available.
        """
        if stage == "annotation" and output_schema:
            return make_json_schema_response_format(
                build_annotation_strict_schema(output_schema), name="annotation"
            )
        if stage == "qc":
            return make_json_schema_response_format(
                build_qc_strict_schema(), name="qc_decision"
            )
        if stage == "arbiter" and output_schema:
            return make_json_schema_response_format(
                build_arbiter_strict_schema(output_schema), name="arbiter_decision"
            )
        return {"type": "json_object"}

    def _read_pinned_handle(self, task: Task, key: str, target: str) -> str | None:
        """Return ``task.metadata[key]`` only if it was minted by the SAME
        profile currently configured for ``target``. Otherwise return None
        (the upstream won't know the handle, so passing it would cause a
        404 / invalid_request_error and crash the worker).
        """
        handle = task.metadata.get(key)
        if not handle:
            return None
        minted_by = task.metadata.get(f"{key}_profile")
        # Defer the factory probe until we actually need to compare profiles.
        # When minted_by is None there's nothing to compare against — return
        # the handle as-is. Probing here would have an unwanted side effect:
        # ``_profile_name_for_target`` calls ``client_factory(target)`` which
        # in production is cheap but in finite-list test stubs consumes one
        # client per probe, exhausting the list and breaking retry flows.
        if minted_by is None:
            return handle
        current = self._profile_name_for_target(target)
        if current is None or minted_by == current:
            return handle
        return None

    def _write_pinned_handle(
        self, task: Task, key: str, handle: str | None, profile_name: str | None,
    ) -> None:
        """Record the handle alongside the profile name that minted it so the
        next ``_read_pinned_handle`` can detect stale cross-provider IDs."""
        if handle:
            task.metadata[key] = handle
            if profile_name:
                task.metadata[f"{key}_profile"] = profile_name
            else:
                task.metadata.pop(f"{key}_profile", None)
        else:
            task.metadata.pop(key, None)
            task.metadata.pop(f"{key}_profile", None)

    async def _call_client(self, target: str, request: LLMGenerateRequest) -> LLMGenerateResult:
        client = self.client_factory(target)
        try:
            result = await client.generate(request)
        finally:
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()
        # Cache the profile name (== the value that ``_write_pinned_handle``
        # will record as the handle's mint origin) so future
        # ``_profile_name_for_target`` lookups don't need to call the
        # factory again. Using ``result.provider`` keeps the cache and the
        # pinned-handle profile column written in the same alphabet.
        self._profile_name_cache[target] = getattr(result, "provider", None) or self._profile_name_cache.get(target)
        return result

    def _append_attempt(self, attempt: Attempt, artifact: ArtifactRef) -> None:
        self.store.append_attempt(attempt)
        self.store.append_artifact(artifact)

    def _next_attempt_id(self, task: Task) -> str:
        # Parse the numeric suffix out of existing attempt_ids and pick
        # max+1. We can't use task.current_attempt (resettable by import
        # UPSERT) and we can't use max(idx) either: the arbiter path can
        # write a row whose attempt_id ends in `-12` while idx stays at
        # 11 (single arbiter run produces multiple attempt rows sharing
        # the same logical round). Looking at idx in that case yields
        # `attempt-12` again on the next call, blowing up on the
        # UNIQUE(attempt_id) constraint and trapping the task in a
        # silent re-pickup loop. The id-suffix is the durable identity
        # of the row, so derive next from it directly.
        import re as _re
        suffixes: list[int] = []
        for a in self.store.list_attempts(task.task_id):
            m = _re.search(r"-attempt-(\d+)(?:-|$)", a.attempt_id)
            if m:
                suffixes.append(int(m.group(1)))
        next_n = max(suffixes, default=-1) + 1
        return f"{task.task_id}-attempt-{next_n}"

    def _record_validation_feedback(
        self,
        task: Task,
        attempt_id: str,
        *,
        category: str = "empty_annotation",
        message: str = "Annotation result was empty.",
        target: dict | None = None,
    ) -> None:
        self.store.append_feedback(
            FeedbackRecord.new(
                task_id=task.task_id,
                attempt_id=attempt_id,
                source_stage=FeedbackSource.VALIDATION,
                severity=FeedbackSeverity.BLOCKING,
                category=category,
                message=message,
                target=target or {},
                suggested_action="annotator_rerun",
                created_by="validation",
            )
        )

    def _latest_annotation_is_valid_json(self, task: Task) -> bool:
        """Return True if the latest annotation_result artifact's text payload
        parses as JSON after standard wrapper stripping. Used as a sanity
        gate before accepting an annotation that the arbiter ruled in
        annotator's favor — see _terminal_from_arbiter.
        """
        artifacts = [a for a in self.store.list_artifacts(task.task_id) if a.kind == "annotation_result"]
        if not artifacts:
            return False
        path = self.store.root / artifacts[-1].path
        if not path.exists():
            return False
        try:
            outer = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(outer, dict):
            return False
        text = outer.get("text")
        if not isinstance(text, str):
            return False
        try:
            _parse_llm_json(text)
        except (json.JSONDecodeError, ValueError):
            return False
        return True

    def _check_annotation_validation(self, task: Task, final_text: str) -> dict | None:
        if not final_text.strip():
            return {
                "category": "empty_annotation",
                "message": "Annotation result was empty.",
                "reason": "deterministic validation failed",
            }
        try:
            payload = _parse_llm_json(final_text)
        except (json.JSONDecodeError, ValueError) as exc:
            return {
                "category": "schema_invalid",
                "message": f"Annotation result is not valid JSON: {exc}",
                "reason": "schema validation failed",
            }
        return self._annotation_validator.validate(task, payload)

    def _check_verbatim_spans(self, task: Task, payload: Any) -> dict | None:
        return self._annotation_validator.check_verbatim_spans(task, payload)

    def _auto_align_corrected_annotation(self, task: Task, corrected: dict) -> int:
        """Wrapper kept for callers in the arbiter retry loop. Delegates to
        the shared ``auto_fix_safe_spans_in_place`` so annotation, arbiter,
        and any future write paths share one safe-fix implementation.
        """
        from annotation_pipeline_skill.core.schema_validation import auto_fix_safe_spans_in_place
        return auto_fix_safe_spans_in_place(task, corrected)

    def _verbatim_candidate_spans(
        self, task: Task, *, row_index: int, failed_span: str, max_candidates: int = 6
    ) -> list[str]:
        """Suggest substrings from row ``row_index``'s input.text that
        overlap with ``failed_span``. Used to guide the arbiter's retry —
        instead of asking it to copy-paste blind, we hand it a short list
        of "what's actually in the text" to choose from.

        Heuristic: for each whitespace-separated word in ``failed_span``,
        find the longest sentence-bounded substring of input.text that
        contains that word, and return the top-k by length. Cheap, no
        external deps, language-agnostic enough for our mixed-EN/CN inputs.
        """
        if not failed_span:
            return []
        # If the row we'd be looking up is masked, treat it as if the row
        # doesn't exist — the operator removed it; we shouldn't surface
        # its text in retry context. apply_masks_to_task drops the row
        # entirely, so the for-loop below naturally returns no text.
        from annotation_pipeline_skill.services.row_mask_service import (
            apply_masks_to_task,
        )
        mtask = apply_masks_to_task(self.store, task)
        source_payload = mtask.source_ref.get("payload") if isinstance(mtask.source_ref, dict) else None
        if not isinstance(source_payload, dict):
            return []
        source_rows = source_payload.get("rows")
        if not isinstance(source_rows, list):
            return []
        input_text: str | None = None
        for i, r in enumerate(source_rows):
            if not isinstance(r, dict):
                continue
            idx = r.get("row_index") if isinstance(r.get("row_index"), int) else i
            if idx == row_index:
                text = r.get("input")
                if isinstance(text, str):
                    input_text = text
                break
        if not input_text:
            return []
        import re as _re
        # Coarse sentence-ish chunking; we just want phrase-length candidates
        # to surface, not a perfect segmentation.
        chunks = [c.strip() for c in _re.split(r"[.!?。！？\n]+", input_text) if c.strip()]
        words = [w for w in _re.split(r"\s+", failed_span.strip()) if len(w) >= 2]
        if not words:
            return []
        seen: set[str] = set()
        ranked: list[tuple[int, str]] = []
        for chunk in chunks:
            chunk_l = chunk.lower()
            score = sum(1 for w in words if w.lower() in chunk_l)
            if score == 0:
                continue
            if chunk in seen:
                continue
            seen.add(chunk)
            # Sort key: highest overlap first, then shorter chunk first.
            ranked.append((-score, len(chunk), chunk))
        ranked.sort()
        return [c for *_, c in ranked[:max_candidates]]

    def _record_annotator_replies(self, task: Task, attempt_id: str, final_text: str) -> int:
        try:
            payload = _parse_llm_json(final_text)
        except (json.JSONDecodeError, ValueError):
            return 0
        if not isinstance(payload, dict):
            return 0
        # Annotator may emit discussion_replies at the top level OR nested
        # inside each row (rows[i].discussion_replies). The prompt doesn't
        # mandate a location and live outputs use the per-row form.
        replies: list = []
        top_level = payload.get("discussion_replies")
        if isinstance(top_level, list):
            replies.extend(top_level)
        rows = payload.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_replies = row.get("discussion_replies")
                if isinstance(row_replies, list):
                    replies.extend(row_replies)
        if not replies:
            return 0
        feedback_index = {f.feedback_id: f for f in self.store.list_feedback(task.task_id)}
        written = 0
        for reply in replies:
            if not isinstance(reply, dict):
                continue
            fid = reply.get("feedback_id")
            if not isinstance(fid, str) or fid not in feedback_index:
                continue
            message = str(reply.get("message") or "").strip()
            if not message:
                continue
            ann_label = _resolve_confidence_label(reply.get("confidence"))
            metadata: dict[str, Any] = {"attempt_id": attempt_id}
            if ann_label is not None:
                metadata["confidence"] = ann_label
            self.store.append_feedback_discussion(
                FeedbackDiscussionEntry.new(
                    task_id=task.task_id,
                    feedback_id=fid,
                    role="annotator",
                    stance=str(reply.get("stance") or "comment"),
                    message=message,
                    agreed_points=[str(p) for p in (reply.get("agreed_points") or []) if isinstance(p, str)],
                    disputed_points=[str(p) for p in (reply.get("disputed_points") or []) if isinstance(p, str)],
                    proposed_resolution=(
                        str(reply["proposed_resolution"])
                        if isinstance(reply.get("proposed_resolution"), str)
                        else None
                    ),
                    consensus=False,
                    created_by="annotator-agent",
                    metadata=metadata,
                )
            )
            written += 1
            # Label-based resolution. Per the empirical calibration study
            # (every confidence bucket for both roles produced the same actual
            # correctness rate, so numeric comparison was noise), decisions
            # branch on the verbal label only — no thresholds.
            ann_label = _resolve_confidence_label(reply.get("confidence"))
            if ann_label is None:
                continue
            qc_feedback = feedback_index[fid]
            qc_label = _resolve_confidence_label(qc_feedback.metadata.get("confidence"))
            # QC: unsure → drop the feedback as noise. QC itself admitted it
            # wasn't sure; no point burning a retry on a guess.
            if qc_label == "unsure":
                self.store.append_feedback_discussion(
                    FeedbackDiscussionEntry.new(
                        task_id=task.task_id,
                        feedback_id=fid,
                        role="qc",
                        stance="agree",
                        message="QC was unsure when filing this; closing by consensus.",
                        consensus=True,
                        created_by="label-resolver",
                        metadata={"attempt_id": attempt_id, "resolution_source": "qc_unsure"},
                    )
                )
                continue
            # Annotator unsure (and QC isn't) → annotator concedes; the
            # natural retry loop continues with whatever fix the annotator
            # silently produced.
            if ann_label == "unsure":
                continue
            # Both sides have at least some confidence and disagree (annotator
            # filed a rebuttal). Don't auto-resolve — let the dispute reach
            # the arbiter at max_qc_rounds. Genuine disagreement is what the
            # arbiter exists for.
        return written

    def _terminal_from_arbiter(
        self,
        task: Task,
        attempt_id: str,
        stage: str,
        arb: dict[str, Any],
    ) -> TaskStatus | None:
        """If the arbiter made an authoritative call, transition the task to a
        terminal state and return it. Otherwise return None (caller continues
        with the normal HR / retry flow).

        Rules:
        - Any unresolved verdict (arbiter label tentative/unsure) → None (HR fallthrough).
        - Any fixed verdict (qc-wins or neither, label certain/confident) AND
          corrected_annotation present → write the correction as the final
          annotation and ACCEPT.
        - All open feedbacks closed in annotator's favor (label certain/confident)
          and zero unresolved → ACCEPT with the current annotation.
        - Anything else → None (HR fallthrough).
        """
        if not arb.get("ran"):
            return None
        if arb["unresolved"] > 0:
            # The arbiter wasn't sure on at least one dispute; let HR handle it.
            return None
        if arb["fixed"] > 0:
            corrected = arb.get("corrected_annotation")
            if not isinstance(corrected, dict):
                return None
            applied = self._apply_arbiter_correction(task, attempt_id, corrected, arb)
            return applied
        if arb["closed"] > 0:
            # Before accepting "annotator's annotation stands", re-validate
            # that the current annotation_result artifact actually parses as
            # JSON. Pre-2026-05-16 some accepted tasks had artifacts whose
            # text wasn't valid JSON (raw <think> block + structurally
            # broken JSON from an older schema), and the export step later
            # blocked on them. Treat parse failure here as a mechanical fail
            # so the caller leaves the task in ARBITRATING for re-pickup
            # (and eventually hits the mechanical retry cap → HR).
            if not self._latest_annotation_is_valid_json(task):
                arb["mechanical_fail"] += 1
                return None
            # ALSO re-validate verbatim / cross-type / trailing-punct on the
            # current annotation_result. The annotator-wins path was the
            # gap: a re-run annotation that introduced non-verbatim spans
            # could pass QC (no automated verbatim check there) and reach
            # arbiter, which then sides with the annotator and accepts the
            # bad data without ever re-verifying. The defect-fix path
            # (_apply_arbiter_correction) already runs these checks; mirror
            # them here so both arbiter outcomes guarantee the same minimum
            # quality before ACCEPTED.
            annotation_artifact = self._latest_annotation_artifact(task.task_id)
            try:
                _payload = self._load_annotation_payload(annotation_artifact)
            except Exception:  # noqa: BLE001
                _payload = None
            if isinstance(_payload, dict):
                from annotation_pipeline_skill.core.schema_validation import (
                    find_cross_type_collisions,
                    find_trailing_punctuation_spans,
                    find_verbatim_violations,
                )
                if find_verbatim_violations(task, _payload) \
                        or find_cross_type_collisions(_payload) \
                        or find_trailing_punctuation_spans(task, _payload):
                    arb["mechanical_fail"] += 1
                    return None
                # Row coverage check: every source row_id must appear in the
                # annotation. When the latest artifact fails this check (e.g.
                # a previous arbiter correction dropped some rows), try rolling
                # back to the most recent annotation artifact with full coverage
                # instead of looping forever on the broken correction.
                try:
                    source_rows = task.source_ref["payload"]["rows"]
                    if isinstance(source_rows, list) and source_rows:
                        source_ids = {r["row_id"] for r in source_rows if isinstance(r, dict) and "row_id" in r}
                        if source_ids:
                            ann_rows = _payload.get("rows", [])
                            ann_ids = {r["row_id"] for r in ann_rows if isinstance(r, dict) and "row_id" in r}
                            if source_ids - ann_ids:
                                prior = self._find_last_complete_annotation_artifact(
                                    task, annotation_artifact.artifact_id, source_ids
                                )
                                if prior is None:
                                    arb["mechanical_fail"] += 1
                                    return None
                                prior_payload = self._load_annotation_payload(prior)
                                if not isinstance(prior_payload, dict):
                                    arb["mechanical_fail"] += 1
                                    return None
                                if (find_verbatim_violations(task, prior_payload)
                                        or find_cross_type_collisions(prior_payload)
                                        or find_trailing_punctuation_spans(task, prior_payload)):
                                    arb["mechanical_fail"] += 1
                                    return None
                                prior_ids = {r["row_id"] for r in prior_payload.get("rows", []) if isinstance(r, dict) and "row_id" in r}
                                if source_ids - prior_ids:
                                    arb["mechanical_fail"] += 1
                                    return None
                                # Prior artifact is complete and clean — promote it
                                # so _latest_annotation_artifact returns it next time.
                                annotation_artifact = self._promote_annotation_artifact(task, prior)
                                _payload = prior_payload
                except (KeyError, TypeError):
                    pass
            self._increment_entity_statistics_for_task(
                task, annotation_artifact, weight=1
            )
            self._mark_first_arbiter_divergence_if_any(task, annotation_artifact)
            if task.metadata.get("prior_verifier_first_arbiter_divergent"):
                # Annotator-wins ruling but still diverges from the project
                # prior — leave the task in ARBITRATING so the scheduler's
                # divergent-flag dispatch picks it up and calls a second
                # arbiter (see _resolve_first_arbiter_divergence_async).
                # Don't ACCEPT here; that would strand the flag.
                self.store.save_task(task)
                return None
            self._transition(
                task,
                TaskStatus.ACCEPTED,
                reason="arbiter resolved all disputes in annotator's favor",
                stage=stage,
                attempt_id=attempt_id,
                metadata={
                    "resolution_source": "arbiter",
                    "arbiter_closed": arb["closed"],
                },
            )
            return TaskStatus.ACCEPTED
        return None

    def _find_last_complete_annotation_artifact(
        self,
        task: Task,
        exclude_artifact_id: str,
        source_ids: set[str],
    ) -> "ArtifactRef | None":
        """Walk annotation_result artifacts in reverse seq order, skipping
        ``exclude_artifact_id``, and return the most recent one whose payload
        contains every row_id in ``source_ids``. Returns None if no such
        artifact exists.
        """
        artifacts = [
            a for a in self.store.list_artifacts(task.task_id)
            if a.kind == "annotation_result" and a.artifact_id != exclude_artifact_id
        ]
        for art in reversed(artifacts):
            payload = self._load_annotation_payload(art)
            if not isinstance(payload, dict):
                continue
            ann_ids = {r["row_id"] for r in payload.get("rows", []) if isinstance(r, dict) and "row_id" in r}
            if not (source_ids - ann_ids):
                return art
        return None

    def _promote_annotation_artifact(self, task: Task, artifact: "ArtifactRef") -> "ArtifactRef":
        """Re-insert an annotation_result artifact at the highest seq so that
        _latest_annotation_artifact returns it on the next pickup. The original
        artifact_ref is left intact; a new one pointing to the same file is
        appended.
        """
        from annotation_pipeline_skill.core.models import ArtifactRef
        new_ref = ArtifactRef.new(
            task_id=task.task_id,
            kind="annotation_result",
            path=artifact.path,
            content_type=artifact.content_type,
            metadata={**artifact.metadata, "promoted_from": artifact.artifact_id},
        )
        self.store.append_artifact(new_ref)
        return new_ref

    def _apply_arbiter_correction(
        self,
        task: Task,
        attempt_id: str,
        corrected: dict[str, Any],
        arb: dict[str, Any],
    ) -> TaskStatus | None:
        """Write the arbiter's corrected_annotation as a fresh annotation_result
        artifact and accept the task. Returns ACCEPTED on success or None if the
        correction couldn't be applied (caller falls through to HR).
        """
        from annotation_pipeline_skill.core.schema_validation import (
            SchemaValidationError,
            resolve_output_schema,
            validate_payload_against_task_schema,
        )

        # Enum coerce: arbiters occasionally invent entity/structure types
        # (e.g. "attribute", "system") alongside legitimate ones. Strict
        # schema validation then rejects the WHOLE correction, including
        # the valid spans. Drop only the invented keys, keep the rest. If
        # anything was coerced, record a one-shot alert so we can see how
        # often the arbiter hallucinates types (and which ones).
        try:
            schema_resolved = resolve_output_schema(task, self.store)
        except Exception:  # noqa: BLE001
            schema_resolved = None
        dropped, rescued = _coerce_to_enum_in_place(corrected, schema_resolved)
        if dropped or rescued:
            if dropped:
                arb.setdefault("enum_coerce_dropped", {}).update(dropped)
            if rescued:
                arb.setdefault("enum_coerce_rescued", {}).update(rescued)
            self._emit_enum_coerce_alert(task, dropped, rescued)
        # Schema check the corrected annotation up front. If it fails we punt
        # back to HR rather than save a bad artifact.
        try:
            validate_payload_against_task_schema(task, corrected, store=self.store)
        except SchemaValidationError:
            return None
        # Use the masked task for verbatim and row-coverage checks. The
        # arbiter prompt was built from masked_task (masked rows excluded),
        # so the corrected_annotation won't include those rows. Checking
        # against the full unmasked task would fail both checks for tasks
        # with masked rows: masked row IDs would always be missing from
        # the coverage requirement.
        from annotation_pipeline_skill.services.row_mask_service import (
            apply_masks_to_task as _apply_masks_to_task,
        )
        _masked_task = _apply_masks_to_task(self.store, task)
        # Verbatim check — arbiter sometimes paraphrases / normalizes spans
        # (e.g., traditional→simplified Chinese, dropped articles) that pass
        # schema but break the input.text substring guarantee. Without this
        # check, hallucinated/normalized spans landed in ACCEPTED tasks
        # (5% audit found ~11% violation rate). Strip hallucinated spans and
        # retry before giving up: the arbiter's intent (type correction for
        # the disputed span) is usually correct even when it hallucinates an
        # unrelated entity alongside it. Only return None if violations remain
        # after stripping (i.e., the target span itself is non-verbatim).
        from annotation_pipeline_skill.core.schema_validation import find_verbatim_violations
        violations = find_verbatim_violations(_masked_task, corrected)
        if violations:
            _strip_non_verbatim_spans_in_place(corrected, violations)
            if find_verbatim_violations(_masked_task, corrected):
                return None
        # Cross-type collision — same span tagged as two entity types. Block
        # the correction; arbiter's internal retry loop already ran, so the
        # outer caller will hit mechanical_fail and either retry or escalate.
        from annotation_pipeline_skill.core.schema_validation import (
            find_cross_type_collisions,
            find_trailing_punctuation_spans,
        )
        if find_cross_type_collisions(corrected):
            return None
        # Trailing-punctuation boundary check — same rule as the annotator
        # validation path. Arbiter should fix this in retries.
        if find_trailing_punctuation_spans(task, corrected):
            return None
        # Row coverage — all *non-masked* source rows must appear in the
        # corrected annotation. Masked rows are excluded from the LLM prompt
        # (both annotation and arbitration), so neither the annotator nor the
        # arbiter includes them. Use _masked_task.source_ref to build
        # source_ids so masked row IDs are naturally absent from the
        # requirement.
        try:
            source_rows = _masked_task.source_ref["payload"]["rows"]
            if isinstance(source_rows, list) and source_rows:
                source_ids = {r["row_id"] for r in source_rows if isinstance(r, dict) and "row_id" in r}
                if source_ids:
                    corr_rows = corrected.get("rows", []) if isinstance(corrected, dict) else []
                    corr_ids = {r["row_id"] for r in corr_rows if isinstance(r, dict) and "row_id" in r}
                    if source_ids - corr_ids:
                        return None
        except (KeyError, TypeError):
            pass
        # Match the annotator write path (see _serialize_llm_json): dedupe
        # within-type. No character-level normalization — the downstream
        # GLiNER pipeline requires byte-for-byte verbatim spans.
        _dedupe_within_type_spans(corrected)

        cleaned_text = json.dumps(corrected, sort_keys=True, indent=2)
        relative_path = f"artifact_payloads/{task.task_id}/{attempt_id}_arbiter_correction.json"
        artifact_path = self.store.root / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "text": cleaned_text,
                    "task_id": task.task_id,
                    "source": "arbiter_correction",
                    "diagnostics": {"resolution_source": "arbiter"},
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact = ArtifactRef.new(
            task_id=task.task_id,
            kind="annotation_result",
            path=relative_path,
            content_type="application/json",
            metadata={"source": "arbiter_correction", "attempt_id": attempt_id},
        )
        self.store.append_artifact(artifact)
        # Stats + verifier post-check on the corrected annotation that was just persisted.
        self._increment_entity_statistics_for_task(task, artifact, weight=1)
        self._mark_first_arbiter_divergence_if_any(task, artifact)
        if task.metadata.get("prior_verifier_first_arbiter_divergent"):
            # Corrected annotation still diverges from the project prior.
            # Same handling as the annotator-wins branch in _terminal_from_arbiter:
            # leave the task in ARBITRATING for the second-arbiter dispatch
            # to resolve.
            self.store.save_task(task)
            return None
        self._transition(
            task,
            TaskStatus.ACCEPTED,
            reason="arbiter produced corrected annotation; task accepted",
            stage="arbitration",
            attempt_id=attempt_id,
            metadata={
                "resolution_source": "arbiter",
                "arbiter_closed": arb["closed"],
                "arbiter_fixed": arb["fixed"],
                "arbiter_correction_artifact_id": artifact.artifact_id,
            },
        )
        return TaskStatus.ACCEPTED

    async def _invoke_second_arbiter(
        self,
        task: Task,
        annotation_artifact: ArtifactRef,
    ) -> dict | None:
        """Run the SECOND arbiter against the prior-divergence dispute.

        Architectural note: the second arbiter is the same `_run_arbiter_llm`
        machinery as the first arbiter — same prompt shape (input + current
        annotation + output_schema + disputed_items + conventions), same
        verbatim retries, same artifact persistence — only the provider
        target differs. The "dispute" the second arbiter sees is a
        synthetic feedback item built from the prior_verifier_payload so
        the LLM knows exactly which (span, type) pair is contested. This
        replaces the old slimmed-down `_build_arbiter_request` whose
        prompt didn't tell the second arbiter what was actually in dispute.

        Returns the parsed arbiter response payload (verdicts +
        corrected_annotation), or None if the call failed / response was
        unparseable. The caller (`_resolve_first_arbiter_divergence_async`)
        interprets the payload and decides the terminal transition.
        """
        payload_meta = task.metadata.get("prior_verifier_payload") or {}
        span = payload_meta.get("span") or ""
        first_type = payload_meta.get("proposed_type") or ""
        prior_type = payload_meta.get("dominant_type") or ""
        prior_count = payload_meta.get("dominant_count") or 0
        prior_total = payload_meta.get("total") or 0
        distribution = payload_meta.get("distribution") or {}
        # Synthesize a single `disputed_items` entry that frames the prior
        # disagreement in the same shape `_arbitrate_and_apply` uses for
        # real QC↔annotator disputes. The second arbiter prompt then sees
        # the span + first arbiter's type + prior dominant type in the
        # standard format and can rule with full context.
        synth_feedback_id = "prior_verifier_synth"
        synth_items = [{
            "feedback_id": synth_feedback_id,
            "category": "prior_disagreement",
            "qc": {
                "message": (
                    f"Project history: span {span!r} has been labeled "
                    f"{prior_type!r} on {prior_count}/{prior_total} prior "
                    f"accepted tasks. Distribution: {distribution!r}. The "
                    f"current annotation labels it {first_type!r}, which "
                    f"diverges from the dominant prior. Should the current "
                    f"context-specific labeling stand, or should the prior win?"
                ),
                "confidence": "informational",
                "target": {
                    "span": span,
                    "proposed_type": first_type,
                    "prior_dominant_type": prior_type,
                    "distribution": distribution,
                    "dominant_count": prior_count,
                    "total": prior_total,
                },
            },
            "annotator": {
                "message": (
                    f"First arbiter (different LLM) reviewed this row's "
                    f"context and judged {span!r} → {first_type!r}. The full "
                    f"annotation is provided as current_annotation; the row "
                    f"in question is in input.text."
                ),
                "confidence": None,
                "disputed_points": [],
                "agreed_points": [],
            },
        }]
        try:
            return await self._run_arbiter_llm(
                task=task,
                items=synth_items,
                target_name="arbiter_secondary",
                attempt_metadata={
                    "target": "arbiter_secondary",
                    "synthetic_feedback_id": synth_feedback_id,
                    "disputed_span": span,
                    "first_arbiter_type": first_type,
                    "prior_dominant_type": prior_type,
                },
            )
        except _ArbiterClientUnavailable:
            return None
        except _ArbiterCallFailed:
            return None

    def _resolve_first_arbiter_divergence(self, task: Task) -> None:
        """Sync entry called by the scheduler when it sees a task with the
        ``prior_verifier_first_arbiter_divergent`` flag set. Runs the second
        arbiter and applies the resolution per spec §6.
        """
        asyncio.run(self._resolve_first_arbiter_divergence_async(task))

    async def _resolve_first_arbiter_divergence_async(self, task: Task) -> None:
        """Three-way resolution per spec §6, with EXPLICIT AFFIRMATION
        required to override the project prior:

        - second arbiter explicitly affirms first arbiter's type (corrected
          annotation contains span→first_type, OR verdict='annotator' with
          certain/confident on the synthetic feedback) → ACCEPTED, override
          prior (two LLMs from different families outvote the historical
          prior on a context-specific exception)
        - second arbiter affirms the prior (verdict='qc' high-conf or
          corrected_annotation contains span→prior_type) → flip first
          arbiter's call to prior, ACCEPTED
        - second arbiter picks a third type (corrected_annotation has
          span→other) → HUMAN_REVIEW (genuine three-way disagreement)
        - second arbiter is silent/uncertain (corrected_annotation=null
          AND no high-conf verdict, or tentative/unsure verdict) →
          HUMAN_REVIEW (don't infer agreement from silence — that's the
          bug that let COVID-19 → technology through despite an 83/55
          event-prior majority)
        - second arbiter unavailable (target missing, network error,
          unparseable response) → HUMAN_REVIEW (safer than leaving the
          task stranded in ARBITRATING or rubber-stamping first arbiter)
        """
        annotation_artifact = self._latest_annotation_artifact(task.task_id)
        if annotation_artifact is None:
            self._clear_divergence_flag(task)
            self.store.save_task(task)
            return
        payload_meta = task.metadata.get("prior_verifier_payload") or {}
        span = payload_meta.get("span")
        first_type = payload_meta.get("proposed_type")
        prior_type = payload_meta.get("dominant_type")
        if not span or not first_type or not prior_type:
            self._clear_divergence_flag(task)
            self.store.save_task(task)
            return

        second_payload = await self._invoke_second_arbiter(task, annotation_artifact)
        attempt_id = self._next_attempt_id(task)
        if not isinstance(second_payload, dict):
            # Second arbiter unavailable / failed. Don't fall back to first —
            # that's the rubber-stamp bug. Route to HR so a human can adjudicate.
            task.metadata["prior_verifier_action"] = "second_arbiter_unavailable"
            self._clear_divergence_flag(task)
            self.store.append_feedback(
                FeedbackRecord.new(
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    source_stage=FeedbackSource.HUMAN_REVIEW,
                    severity=FeedbackSeverity.ERROR,
                    category="prior_divergence",
                    message=(
                        f"Span {span!r}: arbiter selected {first_type!r} but project history "
                        f"dominant is {prior_type!r}. Second arbiter unavailable — choose the correct type."
                    ),
                    target={"span": span, "first_arbiter_type": first_type,
                            "prior_dominant_type": prior_type, "proposed_type": first_type},
                    suggested_action="human_select_type",
                    created_by="prior_verifier",
                )
            )
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason=(
                    "Prior-divergence detected (first arbiter "
                    f"picked {first_type!r}, project history dominant {prior_type!r}); "
                    "second arbiter unavailable or returned unparseable response — needs human review"
                ),
                stage="prior_verifier",
                attempt_id=attempt_id,
                metadata={"prior_verifier_action": "second_arbiter_unavailable"},
            )
            self.store.save_task(task)
            return

        # Verbatim retries exhausted: the second arbiter tried to write a
        # corrected_annotation but couldn't produce verbatim spans after
        # `arbiter_verbatim_retries+1` attempts. Don't fall through to the
        # verdict-based decision — a high-conf verdict alongside a broken
        # correction is contradictory. Route to HR with the failed
        # correction preserved so the operator can see what the arbiter
        # wanted to do.
        if second_payload.get("_verbatim_retry_exhausted"):
            failed_correction = second_payload.get("corrected_annotation")
            failed_target = second_payload.get("_verbatim_failed_target") or {}
            # Non-verbatim spans in the correction are hallucinated entities —
            # strip them and try to apply the pruned correction before giving
            # up to HR. The arbiter's type choice for the disputed span is
            # usually correct even when it adds a spurious entity alongside.
            if isinstance(failed_correction, dict):
                from annotation_pipeline_skill.core.schema_validation import find_verbatim_violations
                from annotation_pipeline_skill.services.row_mask_service import (
                    apply_masks_to_task as _apply_masks_to_task,
                )
                _masked = _apply_masks_to_task(self.store, task)
                viols = find_verbatim_violations(_masked, failed_correction)
                if viols:
                    _strip_non_verbatim_spans_in_place(failed_correction, viols)
                if not find_verbatim_violations(_masked, failed_correction):
                    applied = self._apply_arbiter_correction(
                        task, attempt_id, failed_correction, second_payload
                    )
                    if applied is not None:
                        self._clear_divergence_flag(task)
                        self.store.save_task(task)
                        return
            task.metadata["prior_verifier_action"] = "second_arbiter_verbatim_failed"
            self._clear_divergence_flag(task)
            self.store.append_feedback(
                FeedbackRecord.new(
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    source_stage=FeedbackSource.HUMAN_REVIEW,
                    severity=FeedbackSeverity.ERROR,
                    category="prior_divergence",
                    message=(
                        f"Span {span!r}: arbiter selected {first_type!r} but project history "
                        f"dominant is {prior_type!r}. Second arbiter correction failed verbatim check — choose the correct type."
                    ),
                    target={"span": span, "first_arbiter_type": first_type,
                            "prior_dominant_type": prior_type, "proposed_type": first_type},
                    suggested_action="human_select_type",
                    created_by="prior_verifier",
                )
            )
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason=(
                    f"Second arbiter tried to correct {span!r} but its "
                    f"corrected_annotation contains a non-verbatim span "
                    f"({failed_target.get('span')!r} at "
                    f"{failed_target.get('field')!r}) after retries exhausted "
                    f"and strip could not salvage a valid correction. "
                    f"Needs human review (first arbiter said {first_type!r}, "
                    f"project prior dominant {prior_type!r})"
                ),
                stage="prior_verifier",
                attempt_id=attempt_id,
                metadata={
                    "prior_verifier_action": "second_arbiter_verbatim_failed",
                    "first_arbiter_type": first_type,
                    "prior_dominant_type": prior_type,
                    "span": span,
                    "failed_verbatim_target": failed_target,
                    "failed_correction": failed_correction,
                },
            )
            self.store.save_task(task)
            return

        # Determine the second arbiter's effective vote on the disputed span.
        # Priority: explicit corrected_annotation > explicit verdict on the
        # synthetic feedback > "silent" (treat as no opinion → HR).
        second_corrected = second_payload.get("corrected_annotation")
        if not isinstance(second_corrected, dict):
            second_corrected = None
        second_type: str | None = None
        affirmation_path: str = "none"
        if second_corrected is not None:
            second_type = self._extract_type_for_span(second_corrected, span)
            if second_type is not None:
                affirmation_path = "corrected_annotation"

        if second_type is None:
            # No explicit type from corrected_annotation. Look for an
            # explicit verdict on our synthetic feedback.
            for v in second_payload.get("verdicts") or []:
                if not isinstance(v, dict):
                    continue
                if v.get("feedback_id") != "prior_verifier_synth":
                    continue
                verdict = str(v.get("verdict") or "").lower()
                conf = _resolve_confidence_label(v.get("confidence"))
                if conf not in ("certain", "confident"):
                    continue
                if verdict == "annotator":
                    second_type = first_type
                    affirmation_path = "verdict_annotator"
                elif verdict == "qc":
                    second_type = prior_type
                    affirmation_path = "verdict_qc"
                # 'neither' without corrected_annotation: caller can't tell
                # what type → treat as silent.
                break

        if second_type is None:
            # Truly silent / uncertain. Route to HR rather than rubber-stamp.
            task.metadata["prior_verifier_action"] = "second_arbiter_silent"
            self._clear_divergence_flag(task)
            self.store.append_feedback(
                FeedbackRecord.new(
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    source_stage=FeedbackSource.HUMAN_REVIEW,
                    severity=FeedbackSeverity.ERROR,
                    category="prior_divergence",
                    message=(
                        f"Span {span!r}: arbiter selected {first_type!r} but project history "
                        f"dominant is {prior_type!r}. Second arbiter gave no confident verdict — choose the correct type."
                    ),
                    target={"span": span, "first_arbiter_type": first_type,
                            "prior_dominant_type": prior_type, "proposed_type": first_type},
                    suggested_action="human_select_type",
                    created_by="prior_verifier",
                )
            )
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason=(
                    f"Second arbiter returned no explicit verdict on {span!r}: "
                    f"corrected_annotation is null and no certain/confident "
                    f"verdict on the synthetic prior-disagreement feedback. "
                    f"Silence is not affirmation — needs human review (first "
                    f"arbiter said {first_type!r}, project prior dominant {prior_type!r})"
                ),
                stage="prior_verifier",
                attempt_id=attempt_id,
                metadata={
                    "prior_verifier_action": "second_arbiter_silent",
                    "first_arbiter_type": first_type,
                    "prior_dominant_type": prior_type,
                    "span": span,
                },
            )
            self.store.save_task(task)
            return

        if second_type == first_type:
            task.metadata["prior_verifier_action"] = "resolved_to_first"
            self._clear_divergence_flag(task)
            self._increment_entity_statistics_for_task(task, annotation_artifact, weight=1)
            self._transition(
                task,
                TaskStatus.ACCEPTED,
                reason=f"second arbiter explicitly affirms first ({affirmation_path}); override prior",
                stage="prior_verifier",
                attempt_id=attempt_id,
                metadata={
                    "prior_verifier_action": "resolved_to_first",
                    "affirmation_path": affirmation_path,
                },
            )
        elif second_type == prior_type:
            corrected_payload = self._load_annotation_payload(annotation_artifact)
            self._rewrite_span_type(corrected_payload, span, first_type, prior_type)
            new_artifact = self._write_corrected_annotation_artifact(
                task, corrected_payload, attempt_id=attempt_id,
            )
            task.metadata["prior_verifier_action"] = "resolved_to_prior"
            self._clear_divergence_flag(task)
            self._increment_entity_statistics_for_task(task, new_artifact, weight=1)
            self._transition(
                task,
                TaskStatus.ACCEPTED,
                reason=f"second arbiter agrees with prior ({affirmation_path}); flip first arbiter's call",
                stage="prior_verifier",
                attempt_id=attempt_id,
                metadata={
                    "prior_verifier_action": "resolved_to_prior",
                    "affirmation_path": affirmation_path,
                },
            )
        else:
            task.metadata["prior_verifier_action"] = "escalated_to_hr"
            self._clear_divergence_flag(task)
            # Generate a feedback record so the Manual Review panel has
            # structured content to display. Without this the operator sees
            # an empty review screen and has no way to know what to decide.
            self.store.append_feedback(
                FeedbackRecord.new(
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    source_stage=FeedbackSource.HUMAN_REVIEW,
                    severity=FeedbackSeverity.ERROR,
                    category="three_way_disagreement",
                    message=(
                        f"Span {span!r}: first arbiter selected {first_type!r}, "
                        f"second arbiter selected {second_type!r}, "
                        f"project history dominant is {prior_type!r}. "
                        f"All three signals disagree — choose the correct type."
                    ),
                    target={"span": span, "proposed_type": first_type,
                            "first_arbiter_type": first_type,
                            "second_arbiter_type": second_type,
                            "prior_dominant_type": prior_type},
                    suggested_action="human_select_type",
                    created_by="prior_verifier",
                )
            )
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason="Three-way disagreement: first arbiter, second arbiter (different LLM family), and project history each picked a different type for the same span; needs human review",
                stage="prior_verifier",
                attempt_id=attempt_id,
                metadata={
                    "first_arbiter_type": first_type,
                    "second_arbiter_type": second_type,
                    "prior_dominant_type": prior_type,
                    "span": span,
                },
            )
        self.store.save_task(task)

    @staticmethod
    def _extract_type_for_span(payload: Any, span: str) -> str | None:
        if not isinstance(payload, dict):
            return None
        for row in payload.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            entities = (row.get("output") or {}).get("entities")
            if not isinstance(entities, dict):
                continue
            for typ, items in entities.items():
                if isinstance(items, list) and span in items:
                    return typ
        return None

    @staticmethod
    def _rewrite_span_type(payload: Any, span: str, old_type: str, new_type: str) -> None:
        if not isinstance(payload, dict):
            return
        for row in payload.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            entities = (row.get("output") or {}).get("entities")
            if not isinstance(entities, dict):
                continue
            old_items = entities.get(old_type) or []
            if span in old_items:
                old_items.remove(span)
                if not old_items:
                    entities.pop(old_type, None)
                else:
                    entities[old_type] = old_items
                entities.setdefault(new_type, []).append(span)

    def _write_corrected_annotation_artifact(
        self,
        task: Task,
        payload: dict | None,
        *,
        attempt_id: str | None = None,
    ) -> ArtifactRef:
        """Persist a prior-verifier-corrected annotation as a new
        ``annotation_result`` artifact and return the ArtifactRef. Mirrors
        the on-disk shape used by ``_apply_arbiter_correction`` so the
        downstream ``_load_annotation_payload`` reader round-trips cleanly.
        """
        if attempt_id is None:
            attempt_id = self._next_attempt_id(task)
        rel = f"artifact_payloads/{task.task_id}/{attempt_id}_prior_verifier_fix.json"
        abs_path = self.store.root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        body = payload if isinstance(payload, dict) else {}
        abs_path.write_text(
            json.dumps(
                {
                    "text": json.dumps(body, ensure_ascii=False),
                    "source": "prior_verifier_fix",
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact = ArtifactRef.new(
            task_id=task.task_id,
            kind="annotation_result",
            path=rel,
            content_type="application/json",
            metadata={"source": "prior_verifier_fix", "attempt_id": attempt_id},
        )
        self.store.append_artifact(artifact)
        return artifact

    def _clear_divergence_flag(self, task: Task) -> None:
        task.metadata.pop("prior_verifier_first_arbiter_divergent", None)
        task.metadata.pop("prior_verifier_payload", None)

    def _resolve_uncertain_arbiter(self, task: Task) -> None:
        """Sync entry called by the scheduler when it sees a task with the
        ``arbiter_uncertain_needs_second`` flag set. Runs the second arbiter
        via arbiter_secondary and applies the resolution:

        - second arbiter resolves (unresolved == 0) → ACCEPTED or arbiter fix
        - second arbiter also uncertain (unresolved > 0) → HUMAN_REVIEW
        - second arbiter unavailable (exception) → HUMAN_REVIEW
        """
        asyncio.run(self._resolve_uncertain_arbiter_async(task))

    async def _resolve_uncertain_arbiter_async(self, task: Task) -> None:
        """Async implementation of the second-arbiter-for-uncertain path."""
        attempt_id = self._next_attempt_id(task)
        task.metadata.pop("arbiter_uncertain_needs_second", None)

        arb = await self._arbitrate_and_apply(
            task,
            attempt_id,
            stage="arbitration",
            require_rebuttal=False,
            target_name="arbiter_secondary",
        )

        # _arbitrate_and_apply catches _ArbiterClientUnavailable/_ArbiterCallFailed
        # internally and returns ran=False with exception_class set. Treat any
        # client/call failure as "second arbiter unavailable" → HR.
        if not arb.get("ran") and arb.get("exception_class"):
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason="Arbiter flagged its own answer as uncertain; second arbiter unavailable — needs human review",
                stage="arbitration",
                attempt_id=attempt_id,
            )
            self.store.save_task(task)
            return

        terminal = self._terminal_from_arbiter(task, attempt_id, "arbitration", arb)
        if terminal is not None:
            self.store.save_task(task)
            return

        if arb["unresolved"] > 0:
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason=(
                    "Both arbiters flagged their answers as uncertain "
                    "(tentative/unsure verdict); needs human review"
                ),
                stage="arbitration",
                attempt_id=attempt_id,
                metadata={
                    "auto_escalated": True,
                    "arbiter_ran": arb["ran"],
                    "arbiter_unresolved": arb["unresolved"],
                },
            )
        else:
            self._handle_arbiter_mechanical_fail(
                task, attempt_id, arb, stage="arbitration",
                hr_extra_metadata={},
            )
        self.store.save_task(task)

    async def _run_rearbitration(self, task: Task) -> None:
        """Worker entry for human-dragged REJECTED/HR → Arbitration cards.

        Task already has status=ARBITRATING (the manual-move API set it).
        We re-evaluate every QC/validation feedback (consensus-closed ones
        included) and let the arbiter decide. On no-fix outcome the task
        falls back to HUMAN_REVIEW.
        """
        attempt_id = self._next_attempt_id(task)
        # Wrap the arbiter call: any internal exception (LLM error, JSON
        # parse fail, SQLite IntegrityError on attempt insertion, etc.)
        # must be converted into a mechanical_fail outcome rather than
        # propagating up to the worker pool — the pool's catch-all
        # `except Exception: pass` would otherwise swallow the error,
        # release the lease without saving any metadata change, and let
        # the scheduler re-claim the task indefinitely (no audit event,
        # no retry counter advancing). Mechanical_fail goes through
        # _handle_arbiter_mechanical_fail which bumps the per-task retry
        # counter and forces HR at the 3-retry cap.
        try:
            arb = await self._arbitrate_and_apply(
                task,
                attempt_id,
                stage="arbitration",
                include_closed_feedbacks=True,
                require_rebuttal=False,
            )
        except Exception as exc:  # noqa: BLE001
            arb = {
                "ran": False,
                "closed": 0,
                "fixed": 0,
                "unresolved": 0,
                "mechanical_fail": 1,
                "corrected_annotation": None,
                "exception_class": type(exc).__name__,
                "exception_message": str(exc)[:200],
            }
        terminal = self._terminal_from_arbiter(task, attempt_id, "arbitration", arb)
        if terminal is None:
            # HR only on tentative/unsure arbiter verdicts. Mechanical
            # failures leave the task in ARBITRATING for re-pickup — no
            # point re-running the annotator since the annotation is fine,
            # we just need the arbiter to produce a coherent verdict.
            if arb["unresolved"] > 0:
                self._transition(
                    task,
                    TaskStatus.HUMAN_REVIEW,
                    reason="Arbiter flagged its own answer as uncertain (tentative/unsure verdict); needs human review",
                    stage="arbitration",
                    attempt_id=attempt_id,
                    metadata={
                        "rearbitrate": True,
                        "arbiter_ran": arb["ran"],
                        "arbiter_unresolved": arb["unresolved"],
                        "arbiter_closed": arb["closed"],
                        "arbiter_fixed": arb["fixed"],
                        "arbiter_mechanical_fail": arb["mechanical_fail"],
                    },
                )
            else:
                self._handle_arbiter_mechanical_fail(
                    task, attempt_id, arb, stage="arbitration",
                    hr_extra_metadata={"rearbitrate": True},
                )
        self.store.save_task(task)

    async def _arbitrate_and_apply(
        self,
        task: Task,
        attempt_id: str,
        stage: str,
        *,
        include_closed_feedbacks: bool = False,
        require_rebuttal: bool = True,
        target_name: str = "arbiter",
    ) -> dict[str, Any]:
        """Run the external arbiter as judge + fixer over open disputes.

        Returns:
            {
                "ran": bool,                 # arbiter was invoked
                "closed": int,               # annotator-wins verdicts (label certain/confident)
                "fixed": int,                # qc-wins verdicts where arbiter also provided a fix
                "unresolved": int,           # any verdict labeled tentative/unsure, or qc-wins without a fix
                "corrected_annotation": dict | None,  # full corrected annotation from arbiter, when provided
            }
        Callers decide the terminal transition based on these counts (with help
        from _terminal_from_arbiter, which applies the correction).

        ``require_rebuttal`` (default True): the auto pipeline gates the arbiter
        on the annotator having posted a discussion rebuttal — no rebuttal means
        the annotator gave up, no dispute to arbitrate. The human-dragged
        ``rearbitrate`` path overrides this to False: the human is explicitly
        asking the arbiter to look at the task again, even if the annotator
        never produced a coherent rebuttal. In that case the arbiter judges
        QC's complaint directly against the latest annotation artifact and may
        still produce a corrected annotation.
        """
        empty = {"ran": False, "closed": 0, "fixed": 0, "unresolved": 0, "mechanical_fail": 0, "corrected_annotation": None}
        discussions = self.store.list_feedback_discussions(task.task_id)
        replies_by_feedback = {
            d.feedback_id: d for d in discussions
            if d.role == "annotator"
        }
        if require_rebuttal and not replies_by_feedback:
            return empty
        consensus_ids = {d.feedback_id for d in discussions if d.consensus}
        open_feedbacks = [
            f for f in self.store.list_feedback(task.task_id)
            if (include_closed_feedbacks or f.feedback_id not in consensus_ids)
            and (not require_rebuttal or f.feedback_id in replies_by_feedback)
            and (f.source_stage is FeedbackSource.QC or f.source_stage is FeedbackSource.VALIDATION)
        ]
        # Send only the latest QC round's feedback. list_feedback returns records
        # ordered by seq (insertion order); feedbacks from the same QC run share an
        # attempt_id and are inserted together, so the last record's attempt_id is
        # the most recent round. Stale feedback from prior rounds references
        # annotation states that no longer exist, causing the arbiter to mark
        # verdicts tentative when it sees the mismatch with current_annotation.
        if open_feedbacks:
            latest_attempt_id = open_feedbacks[-1].attempt_id
            open_feedbacks = [f for f in open_feedbacks if f.attempt_id == latest_attempt_id]
        if not open_feedbacks:
            return empty
        # Promote the task into ARBITRATING — visible in the kanban while the
        # arbiter LLM is running. Idempotent: if a human (or a prior step) has
        # already moved the task into ARBITRATING, skip the transition.
        if task.status is not TaskStatus.ARBITRATING:
            self._transition(
                task,
                TaskStatus.ARBITRATING,
                reason="invoking arbiter to resolve QC / annotator disputes",
                stage="arbitration",
                attempt_id=attempt_id,
            )
        items: list[dict[str, Any]] = []
        for f in open_feedbacks:
            reply = replies_by_feedback.get(f.feedback_id)
            if reply is not None:
                annotator_view = {
                    "message": reply.message,
                    "confidence": reply.metadata.get("confidence"),
                    "disputed_points": reply.disputed_points,
                    "agreed_points": reply.agreed_points,
                }
            else:
                # Rearbitrate-without-rebuttal: annotator never posted an
                # explicit reply. Tell the arbiter to judge QC's complaint
                # against the current annotation directly.
                annotator_view = {
                    "message": "(no explicit rebuttal posted; refer to current_annotation for the annotator's position)",
                    "confidence": None,
                    "disputed_points": [],
                    "agreed_points": [],
                }
            items.append({
                "feedback_id": f.feedback_id,
                "category": f.category,
                "qc": {
                    "message": f.message,
                    "confidence": f.metadata.get("confidence"),
                    "target": f.target,
                },
                "annotator": annotator_view,
            })
        try:
            payload = await self._run_arbiter_llm(
                task=task,
                items=items,
                target_name=target_name,
            )
        except _ArbiterRateLimited as exc:
            empty["rate_limited"] = True
            empty["exception_class"] = type(exc).__name__
            empty["exception_message"] = str(exc)[:500]
            task.metadata["arbiter_last_exception_class"] = empty["exception_class"]
            task.metadata["arbiter_last_exception_message"] = empty["exception_message"]
            return empty
        except (_ArbiterClientUnavailable, _ArbiterCallFailed) as exc:
            # Preserve the actual failure cause so HR metadata + task.metadata
            # show whether it was client-unavailable, LLM exception, JSON
            # parse error, or missing-verdicts shape. Without this the outer
            # `_handle_arbiter_mechanical_fail` writes uninformative HR
            # entries (`arbiter_ran=false, arbiter_mechanical_fail=0`) and we
            # have to grep codex rollouts to guess the cause.
            empty["exception_class"] = type(exc).__name__
            empty["exception_message"] = str(exc)[:500]
            task.metadata["arbiter_last_exception_class"] = empty["exception_class"]
            task.metadata["arbiter_last_exception_message"] = empty["exception_message"]
            return empty
        verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
        if not isinstance(verdicts, list):
            return empty
        outcome = {
            "ran": True,
            "closed": 0,
            "fixed": 0,
            "unresolved": 0,
            "mechanical_fail": 0,
            "corrected_annotation": None,
            "verbatim_retry_exhausted": False,
            "failed_verbatim_correction": None,
            "failed_verbatim_target": None,
        }
        verbatim_exhausted = bool(payload.get("_verbatim_retry_exhausted"))
        corrected = payload.get("corrected_annotation") if isinstance(payload, dict) else None
        if isinstance(corrected, dict):
            if verbatim_exhausted:
                # Don't surface the bad correction as the applyable answer
                # (it'd fail downstream verbatim validation, silently
                # losing the arbiter's attempt). Preserve it separately so
                # the HR escalation can show the operator what the arbiter
                # tried, and let qc/neither verdicts route to mechanical_fail
                # so the per-task retry counter triggers HR after the cap.
                outcome["failed_verbatim_correction"] = corrected
                outcome["failed_verbatim_target"] = payload.get("_verbatim_failed_target")
                outcome["verbatim_retry_exhausted"] = True
            else:
                outcome["corrected_annotation"] = corrected
        arbiter_attempt_id = payload.get("_arbiter_attempt_id")
        arbiter_result_meta = payload.get("_arbiter_result_meta") or {}
        known_ids = {f.feedback_id for f in open_feedbacks}
        for verdict_entry in verdicts:
            if not isinstance(verdict_entry, dict):
                continue
            fid = verdict_entry.get("feedback_id")
            if not isinstance(fid, str) or fid not in known_ids:
                continue
            verdict = str(verdict_entry.get("verdict") or "").lower()
            conf_label = _resolve_confidence_label(verdict_entry.get("confidence"))
            reasoning = str(verdict_entry.get("reasoning") or "")
            provider = arbiter_result_meta.get("provider", "arbiter")
            model = arbiter_result_meta.get("model", "")
            base_metadata = {
                "attempt_id": arbiter_attempt_id,
                "resolution_source": "arbiter",
                "arbiter_confidence": conf_label,
                "arbiter_verdict": verdict,
                "arbiter_reasoning": reasoning,
            }
            if conf_label in (None, "tentative", "unsure"):
                outcome["unresolved"] += 1
                self.store.append_feedback_discussion(
                    FeedbackDiscussionEntry.new(
                        task_id=task.task_id,
                        feedback_id=fid,
                        role="qc",
                        stance="comment",
                        message=f"Arbiter ({provider}/{model}) uncertain: {reasoning}",
                        consensus=False,
                        created_by="arbiter",
                        metadata=base_metadata,
                    )
                )
                continue
            if verdict == "annotator":
                outcome["closed"] += 1
                self.store.append_feedback_discussion(
                    FeedbackDiscussionEntry.new(
                        task_id=task.task_id,
                        feedback_id=fid,
                        role="qc",
                        stance="agree",
                        message=f"Arbiter ({provider}/{model}) ruled in annotator's favor: {reasoning}",
                        consensus=True,
                        created_by="arbiter",
                        metadata=base_metadata,
                    )
                )
            elif verdict in {"qc", "neither"}:
                if outcome["corrected_annotation"] is not None:
                    outcome["fixed"] += 1
                    self.store.append_feedback_discussion(
                        FeedbackDiscussionEntry.new(
                            task_id=task.task_id,
                            feedback_id=fid,
                            role="qc",
                            stance="agree",
                            message=(
                                f"Arbiter ({provider}/{model}) ruled {verdict!r} "
                                f"and produced a fix: {reasoning}"
                            ),
                            consensus=True,
                            created_by="arbiter",
                            metadata=base_metadata,
                        )
                    )
                else:
                    outcome["mechanical_fail"] += 1
                    self.store.append_feedback_discussion(
                        FeedbackDiscussionEntry.new(
                            task_id=task.task_id,
                            feedback_id=fid,
                            role="qc",
                            stance="comment",
                            message=(
                                f"Arbiter ({provider}/{model}) ruled {verdict!r} but "
                                f"did not produce a fix: {reasoning}"
                            ),
                            consensus=False,
                            created_by="arbiter",
                            metadata=base_metadata,
                        )
                    )
            else:
                outcome["mechanical_fail"] += 1
        return outcome

    async def _run_arbiter_llm(
        self,
        *,
        task: Task,
        items: list[dict[str, Any]],
        target_name: str,
        attempt_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Shared arbiter LLM machinery used by both the first arbiter
        (`_arbitrate_and_apply`) and the second arbiter (prior-divergence
        resolver). Builds the same prompt shape, runs verbatim-retry loop,
        persists an arbiter_result artifact, and returns the parsed
        response with `_arbiter_attempt_id` / `_arbiter_result_meta` keys
        spliced in for the caller's discussion-posting bookkeeping.

        Raises `_ArbiterClientUnavailable` when client_factory can't build
        the target, and `_ArbiterCallFailed` for any other failure
        (network, unparseable JSON, missing verdicts). Callers convert
        these to outcome=empty / second_arbiter_unavailable as needed.
        """
        try:
            arbiter_client = self.client_factory(target_name)
        except Exception as exc:  # noqa: BLE001
            raise _ArbiterClientUnavailable(str(exc))
        latest_annotation_artifact = self._latest_annotation_artifact(task.task_id)
        current_annotation = self._slim_annotation_payload(latest_annotation_artifact)
        instructions = (
            "You are a senior arbiter AND fixer for an annotation pipeline. You receive the input task, "
            "the annotator's latest annotation, and a list of disputes between the automated QC "
            "reviewer and the annotator.\n\n"
            "Your response shape is ALWAYS:\n"
            "{\n"
            '  "verdicts": [{"feedback_id", "verdict", "confidence", "reasoning"}, ...],\n'
            '  "corrected_annotation": <full corrected annotation object> | null\n'
            "}\n"
            "`corrected_annotation` is a TOP-LEVEL key. "
            "OMIT IT ENTIRELY (do not output null) when ALL verdicts are 'annotator'. "
            "Include it as an object when ANY verdict is 'qc' or 'neither'.\n\n"
            "For EACH disputed feedback choose exactly one verdict:\n"
            "  - 'annotator': the annotator's current annotation IS correct on this item; QC is wrong.\n"
            "  - 'qc':        QC's complaint IS correct; the annotation has the defect QC describes — "
            "YOU MUST APPLY QC's REQUESTED FIX in corrected_annotation. (Add the missing entity, "
            "remove the wrong span, repopulate json_structures, whatever QC asked for.)\n"
            "  - 'neither':   both sides are wrong; YOU produce the right answer in corrected_annotation.\n"
            "Confidence: ONE of these strings (no numbers; the runtime won't accept them):\n"
            "  - \"certain\"   = evidence unambiguous; any reasonable reviewer would reach the same verdict.\n"
            "  - \"confident\" = strong case but a reasonable reviewer with different priors might rule differently.\n"
            "  - \"tentative\" = judgment call; you lean this way but admit another reading is defensible.\n"
            "  - \"unsure\"    = you don't really know; route to human.\n"
            "Pick the label that fits the evidence; don't default to \"certain\".\n\n"
            "OUTPUT SHAPE REQUIREMENTS:\n"
            "  - If ANY verdict is 'qc' or 'neither' (the annotation needs change), corrected_annotation "
            "MUST be a non-null object with the FULL corrected annotation. Describing the fix in "
            "reasoning while leaving corrected_annotation null wastes your verdicts.\n"
            "  - If ALL verdicts are 'annotator' (the annotation stands as-is), set corrected_annotation = null.\n"
            "There is no 'rejected' outcome.\n\n"
            "Shape of corrected_annotation when non-null:\n"
            "  {\"rows\": [{\"row_index\": int, \"row_id\": str, \"output\": "
            "{\"entities\": {\"<type>\": [\"span\", ...]}, "
            "\"json_structures\": {\"<type>\": [\"span\", ...]}}}, ...]}\n\n"
            "CRITICAL — entities / json_structures format is DICT-BY-TYPE (keys = type names, "
            "values = lists of verbatim strings). NEVER a list of objects.\n"
            "  CORRECT:  {\"entities\": {\"location\": [\"Paris\"], \"person\": [\"Alice\"]}}\n"
            "  WRONG:    {\"entities\": [{\"type\": \"location\", \"phrase\": \"Paris\"}, ...]}\n"
            "  WRONG:    {\"entities\": [{\"text\": \"Paris\", \"type\": \"location\"}, ...]}\n"
            "  WRONG:    {\"entities\": [{\"span\": \"Paris\", \"entity_type\": \"location\"}, ...]}\n"
            "  WRONG:    {\"entities\": [{\"value\": \"Paris\", \"type\": \"location\"}, ...]}\n"
            "If you output a list of objects instead of a dict, the pipeline rejects the entire "
            "corrected_annotation and the arbiter call is wasted.\n\n"
            "corrected_annotation MUST include EVERY source row, in order — not just the rows "
            "your verdicts touch. Rows you didn't change: copy the annotator's existing output "
            "(from current_annotation) verbatim. Rows you did change: apply your fix. The pipeline "
            "post-validates the entire payload (schema / verbatim / cross-type / trailing-punct / "
            "row-coverage); a partial corrected_annotation will fail the row-coverage check and "
            "burn the arbiter call.\n"
            "Entity / structure type names: use ONLY the values listed in "
            "`output_schema.entity_types[*].name` and `output_schema.json_structure_types[*].name`. "
            "Inventing types like 'attribute' or 'system' causes silent drops "
            "(runtime coerces invented keys out before validation, but you've "
            "wasted the verdict by emitting them).\n"
            "Each entity / phrase value is a bare VERBATIM string copied from the corresponding "
            "row's input.text (no character offsets, just the text itself). Pipeline validates: "
            "every span must appear in input.text via substring match.\n"
            "Preserve fields the annotator already had right WITHIN the rows you do change; "
            "only modify what your verdicts say needs changing.\n\n"
            "SELF-CHECK TOOL (MANDATORY when producing a corrected_annotation): when "
            "`check_annotation_draft` is in your tools list, you MUST validate your full "
            "corrected_annotation BEFORE submitting. The pipeline runs the same mechanical "
            "checks post-submit and rejects non-clean corrections, burning a full arbiter call.\n"
            "  Workflow:\n"
            "    1. Build the full corrected_annotation: copy current_annotation, apply your "
            "       fixes on the disputed rows.\n"
            "    2. Call `check_annotation_draft` with "
            "       `{task_id: <task_id>, payload: <your corrected_annotation>}`.\n"
            "    3. If ok=true → submit. If violations are non-empty, fix them in the draft "
            "       (use `lookup_row_text` for verbatim spans if needed) and re-call. Cap: 5 "
            "       iterations.\n"
            "Return raw JSON only, no markdown fences."
        )
        # All three agents (annotator, QC, arbiter) share the same cross-cutting
        # span rules — verbatim, no character substitution, no trailing punct,
        # no duplicates, one type per span. Single source of truth in code.
        instructions = instructions + "\n\n" + _SHARED_SPAN_RULES
        # NOTE: conventions_block is per-task and was previously appended to
        # `instructions` (= system prompt). That made the system prompt
        # task-specific and broke vLLM prefix-cache locality for the
        # arbiter_secondary (qwen) path. The block is now prepended to the
        # user prompt below; system stays bytestable.
        conventions_block = self._build_conventions_block(task)
        # Include the resolved output_schema so the arbiter doesn't invent
        # entity types, phrase types, or field shapes when constructing
        # corrected_annotation. Without this constraint, gpt-5.5 was emitting
        # entity names like "attribute" / "system" that the schema validator
        # rejected, causing the fix to silently fall back to HR.
        from annotation_pipeline_skill.core.schema_validation import (
            _schema_type_enums,
            resolve_output_schema,
        )
        from annotation_pipeline_skill.services.row_mask_service import apply_masks_to_task
        output_schema = resolve_output_schema(task, self.store)
        # Apply mask filter so the arbiter never sees (and never tries
        # to correct annotations for) rows the operator has masked.
        masked_task = apply_masks_to_task(self.store, task)

        # Arbiter input: full source rows + full current_annotation. The
        # ROW-FILTERING that earlier code did (SLIM-PROMPT CONTRACT) saved
        # ~5K on a ~30K prompt but caused 17% of all arbitration→HR
        # transitions in production: arbiter emitted a partial
        # corrected_annotation, the runtime merge-filled the unchanged
        # rows from the latest annotation, post-validation tripped on a
        # verbatim/cross-type violation in one of the merge-filled rows
        # that the arbiter never saw — silent mech_fail → cap=3 → HR.
        # Net cost of slimming was >$20/day in wasted LLM calls plus
        # ~200 tasks/day routed to manual review unnecessarily.
        #
        # The compact `slim_schema` and `slim_items` shapes below are
        # kept — those compress task-agnostic and duplicate content
        # respectively, with no analogous bug pattern.
        full_payload = masked_task.source_ref.get("payload", {}) or {}
        full_rows = full_payload.get("rows", []) if isinstance(full_payload, dict) else []
        arbiter_input = {
            **{k: v for k, v in full_payload.items() if k != "rows"},
            "rows": full_rows,
        }
        arbiter_current_annotation = (
            current_annotation if isinstance(current_annotation, dict) else {"rows": []}
        )
        # Compact schema: just the {name, desc} list per enum. The full
        # JSON Schema (with its $defs / oneOf / regex constraints) is
        # re-applied server-side at validation time; the arbiter only
        # needs the enum lists to avoid inventing types.
        slim_schema: dict[str, Any] = {
            "_note": "corrected_annotation entity keys must be drawn from entity_types[*].name; "
                     "json_structures keys from json_structure_types[*].name. Runtime "
                     "re-validates the full JSON Schema after submit.",
            "entity_types": [],
            "json_structure_types": [],
        }
        if isinstance(output_schema, dict):
            defs = output_schema.get("$defs") or output_schema.get("definitions") or {}
            for src_key, dst_key in (
                ("entityType", "entity_types"),
                ("jsonStructureType", "json_structure_types"),
            ):
                defn = defs.get(src_key) if isinstance(defs, dict) else None
                if isinstance(defn, dict):
                    one_of = defn.get("oneOf")
                    if isinstance(one_of, list):
                        slim_schema[dst_key] = [
                            {"name": x.get("const"), "desc": (x.get("description") or "")[:120]}
                            for x in one_of
                            if isinstance(x, dict) and isinstance(x.get("const"), str)
                        ]
                    elif isinstance(defn.get("enum"), list):
                        slim_schema[dst_key] = [{"name": v, "desc": ""} for v in defn["enum"]]
        # Disputed-items shaping: drop annotator's disputed_points /
        # agreed_points (duplicates qc.message + annotator.message). Also
        # cap qc.target.errors[].message — schema_invalid feedbacks
        # serialize the ENTIRE annotation payload into the error message
        # as a Python repr, which would re-bloat the prompt. Empirical:
        # one schema_invalid item used to take a 1-row prompt from
        # ~22KB back up to ~40KB before the cap.
        ERR_MSG_CAP = 400
        def _slim_qc(qc_dict: Any) -> Any:
            if not isinstance(qc_dict, dict):
                return qc_dict
            tgt = qc_dict.get("target")
            if not isinstance(tgt, dict):
                return qc_dict
            errs = tgt.get("errors")
            if not isinstance(errs, list) or not errs:
                return qc_dict
            new_errs: list[Any] = []
            for e in errs:
                if not isinstance(e, dict):
                    new_errs.append(e)
                    continue
                m = e.get("message", "")
                if isinstance(m, str) and len(m) > ERR_MSG_CAP:
                    new_errs.append({
                        **e,
                        "message": m[:ERR_MSG_CAP]
                        + "… [truncated — full payload elided to keep prompt slim]",
                    })
                else:
                    new_errs.append(e)
            return {**qc_dict, "target": {**tgt, "errors": new_errs}}
        slim_items = [{
            "feedback_id": it.get("feedback_id"),
            "category": it.get("category"),
            "qc": _slim_qc(it.get("qc")),
            "annotator_reply": {
                "message": (it.get("annotator") or {}).get("message", ""),
                "confidence": (it.get("annotator") or {}).get("confidence"),
            },
        } for it in items]

        # Key order matters for vLLM prefix-cache: stable fields head, volatile
        # tail. task_id + input + output_schema are stable across the arbiter's
        # verbatim-retry loop (max 3 attempts within a single arbiter call,
        # same prompt body); current_annotation + disputed_items change per
        # arbiter dispatch. Without this ordering sort_keys=True puts
        # current_annotation at the head and the retry-loop calls share no
        # prefix-cache prefix at all.
        prompt = json.dumps(
            {
                "task_id": task.task_id,
                "input": arbiter_input,
                "output_schema": slim_schema,
                "current_annotation": arbiter_current_annotation,
                "disputed_items": slim_items,
            },
            indent=2,
        )
        # Prepend per-task entity conventions to the user prompt (was
        # previously appended to system, which broke prefix-cache).
        if conventions_block:
            prompt = conventions_block + "\n\n" + prompt
        # Up to ``arbiter_verbatim_retries`` retry rounds if the arbiter's
        # corrected_annotation contains a non-verbatim span. After retries
        # exhausted, we PRESERVE the (still non-verbatim) correction and
        # flag it via `_verbatim_retry_exhausted` so callers can surface it
        # to HR. Previously the field was silently nulled out — that erased
        # the arbiter's intent (the operator couldn't see what it tried
        # to do) and let `_resolve_first_arbiter_divergence_async` fall
        # through to the verdict-only path, accepting based on the verdict
        # even though the underlying span is malformed.
        max_retries = getattr(self.config, "arbiter_verbatim_retries", 2)
        retry_note = ""
        result = None
        payload = None
        verdicts = None
        verbatim_exhausted_target: dict | None = None
        started_at = utc_now()
        for attempt_idx in range(max_retries + 1):
            attempt_instructions = instructions + retry_note
            try:
                # Request strict JSON when the client supports
                # response_format (OpenAI chat-completions API, used by
                # DeepSeek / OpenRouter / similar). Eliminates a whole
                # class of "arbiter returned prose preamble before JSON"
                # parse failures. Clients that don't honor it (codex CLI,
                # responses API .generate) just ignore the field.
                result = await arbiter_client.generate(LLMGenerateRequest(
                    instructions=attempt_instructions,
                    prompt=prompt,
                    continuity_handle=None,
                    response_format=self._build_response_format(
                        target_name, stage="arbiter", output_schema=output_schema
                    ),
                    task_id=task.task_id,
                ))
            except Exception as exc:  # noqa: BLE001
                # Tag with the underlying class — some clients raise empty-
                # message exceptions (asyncio.CancelledError / TimeoutError /
                # bare subprocess errors) where str(exc) is "" and we lose
                # the only clue about what went wrong.
                # ProviderCallError carries .diagnostics (returncode +
                # stderr + error_event with parsed api_error_status). Without
                # surfacing diagnostics the wrapped message is just "local
                # CLI provider failed" and the cause (auth/balance/wrong
                # model/etc.) is lost.
                diag = getattr(exc, "diagnostics", None)
                tail = ""
                if isinstance(diag, dict):
                    rc = diag.get("returncode")
                    err = (diag.get("stderr") or "")
                    if isinstance(err, str):
                        err = err.strip().replace("\n", " | ")[-300:]
                    err_ev = diag.get("error_event")
                    api_status = (
                        err_ev.get("api_error_status") if isinstance(err_ev, dict) else None
                    )
                    api_msg = (
                        err_ev.get("result_text") if isinstance(err_ev, dict) else None
                    )
                    tail = f" rc={rc} api_status={api_status} api_msg={str(api_msg)[:200]!r} stderr={err!r}"
                # Operator-actionable error → alert + try fallback target
                # for THIS arbiter call. If fallback also fails, fall
                # through and raise _ArbiterCallFailed as before.
                if _is_provider_permanent_error(exc):
                    self._emit_provider_alert(target_name, exc)
                    try:
                        fb_client = self.client_factory("fallback")
                        result = await fb_client.generate(LLMGenerateRequest(
                            instructions=attempt_instructions,
                            prompt=prompt,
                            continuity_handle=None,
                            response_format=self._build_response_format(
                                "fallback", stage="arbiter", output_schema=output_schema
                            ),
                            task_id=task.task_id,
                        ))
                        # fallback succeeded — continue to JSON parse step
                        # without raising. (Drop out of the except block.)
                    except Exception:  # noqa: BLE001
                        raise _ArbiterCallFailed(
                            f"llm_call/{type(exc).__name__}: {exc!s}{tail}"
                        ) from exc
                    else:
                        try:
                            close = getattr(fb_client, "aclose", None)
                            if close is not None:
                                await close()
                        except Exception:  # noqa: BLE001
                            pass
                        # `result` now holds fallback output — skip the raise.
                elif _is_provider_transient_error(exc):
                    raise _ArbiterRateLimited(
                        f"transient/{type(exc).__name__}: {exc!s}{tail}"
                    ) from exc
                else:
                    raise _ArbiterCallFailed(
                        f"llm_call/{type(exc).__name__}: {exc!s}{tail}"
                    ) from exc
            try:
                payload = _parse_llm_json(result.final_text)
            except (json.JSONDecodeError, ValueError) as exc:
                raise _ArbiterCallFailed(f"json_parse/{type(exc).__name__}: {exc!s}") from exc
            verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
            if not isinstance(verdicts, list):
                raise _ArbiterCallFailed("shape/missing_verdicts_list")
            corrected_check = payload.get("corrected_annotation") if isinstance(payload, dict) else None
            if isinstance(corrected_check, dict):
                # First try a safe whitespace/punctuation auto-alignment of
                # near-verbatim spans. Many "verbatim_exhausted" HR routes
                # were caused by trivial trailing-period / wrapping-quote
                # differences — fixing those locally avoids burning the
                # arbiter retry budget on a no-op re-emission.
                aligned_count = self._auto_align_corrected_annotation(task, corrected_check)
                if aligned_count:
                    payload["_verbatim_auto_aligned"] = (
                        payload.get("_verbatim_auto_aligned", 0) + aligned_count
                    )
                verbatim_failure = self._check_verbatim_spans(task, corrected_check)
                if verbatim_failure is not None:
                    if attempt_idx < max_retries:
                        target = verbatim_failure.get("target", {})
                        candidates = self._verbatim_candidate_spans(
                            task,
                            row_index=int(target.get("row_index") or 0),
                            failed_span=str(target.get("span") or ""),
                        )
                        candidates_block = ""
                        if candidates:
                            candidates_block = (
                                "\nCANDIDATE VERBATIM SPANS from this row's input.text "
                                "(you MUST use one of these exactly as-is, character-for-character — "
                                "do NOT paraphrase or choose your own wording):\n  - "
                                + "\n  - ".join(repr(c) for c in candidates)
                            )
                        retry_note = (
                            f"\n\nPREVIOUS ATTEMPT FAILED VERBATIM CHECK: "
                            f"span {target.get('span')!r} at {target.get('field')!r} "
                            f"is not a verbatim substring of the row's input.text. "
                            f"Re-emit corrected_annotation using only spans that appear "
                            f"VERBATIM (exact character match including punctuation, "
                            f"whitespace, traditional vs simplified Chinese, case) in "
                            f"input.text. Do not paraphrase, normalize, or invent spans."
                            + candidates_block
                        )
                        continue
                    # Retries exhausted. Before giving up, try stripping the
                    # non-verbatim spans — the arbiter's type correction for
                    # the target span is usually right even when it hallucinates
                    # an unrelated entity alongside it. Only mark as exhausted
                    # if violations remain after stripping.
                    from annotation_pipeline_skill.core.schema_validation import find_verbatim_violations
                    all_viols = find_verbatim_violations(task, corrected_check)
                    if all_viols:
                        _strip_non_verbatim_spans_in_place(corrected_check, all_viols)
                    if self._check_verbatim_spans(task, corrected_check) is not None:
                        verbatim_exhausted_target = verbatim_failure.get("target", {})
                else:
                    # Cross-type collision: same span tagged under two entity
                    # types. _apply_arbiter_correction blocks this silently;
                    # catch it here to give the arbiter immediate feedback.
                    from annotation_pipeline_skill.core.schema_validation import (
                        find_cross_type_collisions,
                        find_trailing_punctuation_spans,
                    )
                    cross_type = find_cross_type_collisions(corrected_check)
                    if cross_type and attempt_idx < max_retries:
                        collision_desc = "; ".join(
                            f"span {c.get('span')!r} appears under both "
                            + " and ".join(repr(t) for t in (c.get("types") or [])[:2])
                            for c in cross_type[:3]
                        )
                        retry_note = (
                            f"\n\nPREVIOUS ATTEMPT FAILED CROSS-TYPE CHECK: "
                            f"{collision_desc}. "
                            f"Each span must appear under exactly ONE entity type. "
                            f"Remove the duplicate and re-emit corrected_annotation."
                        )
                        continue
                    # Trailing-punctuation: span ends with a punctuation
                    # character that is not part of the token (e.g. "Apple.")
                    trailing = find_trailing_punctuation_spans(task, corrected_check)
                    if trailing and attempt_idx < max_retries:
                        tp_desc = "; ".join(
                            f"span {t.get('span')!r} at {t.get('field')!r}"
                            for t in trailing[:3]
                        )
                        retry_note = (
                            f"\n\nPREVIOUS ATTEMPT FAILED TRAILING-PUNCTUATION CHECK: "
                            f"{tp_desc}. "
                            f"Spans must not end with punctuation that is not part of "
                            f"the token itself. Strip the trailing punctuation and "
                            f"re-emit corrected_annotation."
                        )
                        continue
            else:
                needs_correction = any(
                    isinstance(v, dict)
                    and str(v.get("verdict") or "").lower() in {"qc", "neither"}
                    and _resolve_confidence_label(v.get("confidence")) in ("certain", "confident")
                    for v in verdicts
                )
                if needs_correction and attempt_idx < max_retries:
                    retry_note = (
                        "\n\nPREVIOUS ATTEMPT WAS MISSING corrected_annotation: you ruled "
                        "'qc' or 'neither' on at least one feedback (meaning the annotation "
                        "needs change) but set corrected_annotation to null. Re-emit your "
                        "full response with a non-null corrected_annotation: "
                        "{\"rows\": [...]} containing the FULL corrected annotation. "
                        "Your reasoning is wasted without it."
                    )
                    continue
            break
        finished_at = utc_now()
        arbiter_attempt_id = self._next_attempt_id(task)
        task.current_attempt += 1
        artifact_metadata = {"target": target_name}
        if attempt_metadata:
            artifact_metadata.update(attempt_metadata)
        arbiter_artifact = self._write_stage_artifact(
            task,
            result,
            kind="arbiter_result",
            attempt_id=arbiter_attempt_id,
            payload={"decision": payload, "items": items, "target": target_name},
            extra_metadata=artifact_metadata,
        )
        self._append_attempt(
            Attempt(
                attempt_id=arbiter_attempt_id,
                task_id=task.task_id,
                index=task.current_attempt,
                stage="arbitration",
                status=AttemptStatus.SUCCEEDED,
                started_at=started_at,
                finished_at=finished_at,
                provider_id=result.provider,
                model=result.model,
                effort=None,
                route_role=target_name,
                summary=result.final_text[:500],
                artifacts=[arbiter_artifact],
            ),
            arbiter_artifact,
        )
        # Splice bookkeeping for the caller; underscored to mark as
        # caller-only and not part of the LLM's response.
        payload["_arbiter_attempt_id"] = arbiter_attempt_id
        payload["_arbiter_result_meta"] = {
            "provider": result.provider,
            "model": result.model,
            "target": target_name,
            "artifact_path": arbiter_artifact.path,
        }
        if verbatim_exhausted_target is not None:
            # The arbiter tried but couldn't produce a verbatim correction
            # after `arbiter_verbatim_retries+1` attempts. The bad correction
            # is preserved in payload["corrected_annotation"] so callers
            # can surface it in HR metadata; this flag tells them NOT to
            # apply it as a real correction (it'd fail downstream verbatim
            # validation) and instead route the task to HR.
            payload["_verbatim_retry_exhausted"] = True
            payload["_verbatim_failed_target"] = verbatim_exhausted_target
        return payload

    def _record_confidence_sample(self, role: str, value: float) -> None:
        history = self._confidence_history.setdefault(role, [])
        history.append(value)
        if len(history) > self._confidence_window:
            del history[: len(history) - self._confidence_window]

    def _normalize_confidence(self, role: str, value: float) -> float:
        history = self._confidence_history.get(role, [])
        if len(history) < self._confidence_min_samples:
            return value
        lo, hi = min(history), max(history)
        if hi <= lo:
            return value
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    def _mark_early_hr(
        self,
        task: Task,
        feedback_id: str,
        reason: str,
        annotator_confidence: float,
        qc_confidence: float,
    ) -> None:
        task.metadata["needs_early_hr_low_confidence"] = True
        task.metadata.setdefault("early_hr_reason", reason)
        ids = list(task.metadata.get("low_confidence_feedback_ids", []))
        if feedback_id not in ids:
            ids.append(feedback_id)
        task.metadata["low_confidence_feedback_ids"] = ids
        confs = dict(task.metadata.get("early_hr_confidence", {}))
        confs[feedback_id] = {"annotator": annotator_confidence, "qc": qc_confidence}
        task.metadata["early_hr_confidence"] = confs

    def _resolved_qc_policy(self, task: Task) -> dict[str, Any]:
        return _resolve_qc_policy_from_task_or_config(task, self.config)

    def _qc_instructions(self, task: Task, *, guideline: str | None = None) -> str:
        return _build_qc_instructions(
            task,
            resolved_policy=self._resolved_qc_policy(task),
            guideline=guideline,
        )

    def _build_conventions_block(self, task: Task) -> str | None:
        return self._prompt_builder.build_conventions_block(task)

    def _annotation_prompt(self, task: Task, *, continuation_handle: str | None = None) -> str:
        return self._prompt_builder.build_annotation_prompt(task, continuation_handle=continuation_handle)

    def _delta_feedback_items(self, task: Task) -> list[dict]:
        return self._prompt_builder.delta_feedback_items(task)

    def _snapshot_sent_feedback(self, task: Task) -> None:
        self._prompt_builder.snapshot_sent_feedback(task)

    def _qc_prompt(self, task: Task, annotation_artifact: ArtifactRef) -> str:
        return self._prompt_builder.build_qc_prompt(task, annotation_artifact)

    def _slim_annotation_payload(self, artifact: ArtifactRef) -> Any:
        return self._prompt_builder.slim_annotation_payload(artifact)

    def _artifact_context(
        self, task_id: str, *, per_kind_limit: int = 1
    ) -> list[dict[str, Any]]:
        return self._prompt_builder._artifact_context(task_id, per_kind_limit=per_kind_limit)

    def _read_artifact_payload(self, artifact: ArtifactRef) -> Any:
        return self._prompt_builder._read_artifact_payload(artifact)

    def _transition(
        self,
        task: Task,
        next_status: TaskStatus,
        *,
        reason: str,
        stage: str,
        attempt_id: str,
        metadata: dict | None = None,
    ) -> None:
        event = transition_task(
            task,
            next_status,
            actor="subagent-runtime",
            reason=reason,
            stage=stage,
            attempt_id=attempt_id,
            metadata=metadata,
        )
        self.store.append_event(event)
        # Persist the new status immediately so the kanban (5s poll) can show
        # tasks transiting ANNOTATING → VALIDATING → QC, not just PENDING → ACCEPTED.
        self.store.save_task(task)

    def _write_stage_artifact(
        self,
        task: Task,
        result: LLMGenerateResult,
        *,
        kind: str,
        attempt_id: str,
        payload: dict[str, Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        relative_path = f"artifact_payloads/{task.task_id}/{attempt_id}_{kind}.json"
        path = self.store.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "task_id": task.task_id,
                    **payload,
                    "raw_response": result.raw_response,
                    "usage": result.usage,
                    "diagnostics": result.diagnostics,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata = {
            "runtime": result.runtime,
            "provider": result.provider,
            "model": result.model,
            "continuity_handle": result.continuity_handle,
            "diagnostics": result.diagnostics or {},
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return ArtifactRef.new(
            task_id=task.task_id,
            kind=kind,
            path=relative_path,
            content_type="application/json",
            metadata=metadata,
        )


_ANNOTATION_VALIDATOR_SUFFIX = (
    "SELF-CHECK TOOL (MANDATORY BEFORE SUBMIT): when "
    "`check_annotation_draft` is in your tools list, you MUST call it on "
    "your draft BEFORE you emit your final JSON answer. The pipeline runs the same mechanical checks "
    "after you submit and will REJECT non-clean drafts, costing a full re-annotation. Self-check "
    "lets you fix issues in this session.\n"
    "  Workflow:\n"
    "    1. Build your full draft `{rows: [...]}` payload (all source rows, not a subset).\n"
    "    2. Call `check_annotation_draft` with "
    "       `{task_id: <task_id from prompt input>, payload: <your draft>}`.\n"
    "    3. If `ok=true` and `violations={}`, emit your final JSON. Done.\n"
    "    4. If violations are non-empty, fix each one in your draft. For verbatim_violations: "
    "       the offending span is not a byte-for-byte substring of the row's input.text. Use "
    "       `lookup_row_text` with `{task_id, row_index}` to read the "
    "       original text and re-extract the correct verbatim substring (no character substitution, "
    "       no normalization). For row_coverage_missing: add the missing rows with at least empty "
    "       `{entities: {}, json_structures: {}}` outputs. For cross_type_collisions: pick ONE type "
    "       per span per row. For trailing_punctuation: drop the trailing punctuation from the span. "
    "       For schema_errors: align entity/structure type names with the schema enum.\n"
    "    5. Re-call check_annotation_draft. Loop until ok=true. Hard cap: 5 self-check iterations; "
    "       if still not clean, submit your best draft and the pipeline will route to human review.\n"
    "  Don't skip this even if you're confident — the dominant failure mode is verbatim violations "
    "  on spans you remembered slightly wrong (paraphrased / normalized / dropped article)."
)

_SHARED_SPAN_RULES = """\
CROSS-CUTTING SPAN RULES (every agent — annotator, QC, arbiter — enforces the same set):

1. VERBATIM: every entity span and json_structures phrase MUST appear byte-for-byte as a
   substring of input.text. The pipeline runs a deterministic substring check; any non-verbatim
   span is BLOCKING and bounces the task back.

2. NO CHARACTER SUBSTITUTION. Do not normalize:
   - Traditional ↔ Simplified Chinese (蕭 ≠ 萧, 盧 ≠ 卢) — copy whichever form is in input
   - LaTeX escapes (`7.68\\%`, `$\\alpha$`, `AdaB$^2$N`) — keep the escape, don't render
   - UTF-8 mojibake (`90Â°`) — keep as-is, do not "fix"
   - Defanged URLs/IPs (`hxxp://`, `[.]`) — keep defanged
   - Whitespace inside CJK character sequences (`5 7` ≠ `57`, `落 合` ≠ `落合`) — preserve

3. SPAN BOUNDARY: do NOT include sentence-ending punctuation (`.,;:!?。，；：！？`) at the
   END of an entity span, even when that character is verbatim in input.text. The entity is
   the name itself, not the sentence boundary. Emit `Mitul Mallik`, not `Mitul Mallik.`.

4. NO DUPLICATES: each (entity_type, span) pair appears at most once per row. The runtime
   dedupes at write time, but emit deduped to start with.

5. ONE TYPE PER SPAN PER ROW: an entity span may not be tagged under two entity types within
   the same row. Cross-type collisions are BLOCKING. json_structures fields may overlap (a
   phrase can be both a `goal` and a `constraint`).
"""


def _annotation_instructions(
    task: Task,
    *,
    guideline: str | None = None,
    conventions_block: str | None = None,  # DEPRECATED — kept for callers that still pass it; no-op
    output_schema: dict | None = None,
) -> str:
    """Build the system prompt for annotator subagents.

    Prefix-cache contract (see commit history around the vLLM 0% hit-rate
    fix): this string must be BYTE-STABLE across tasks of the same project
    so vLLM's prefix cache can hit on the system block. Anything per-task
    MUST be excluded:
      - conventions_block (per-task entity examples) is NOT appended here;
        the caller prepends it to the user message instead.
      - The "Modality: ... Requirements: ..." f-string at the end of base
        is constant for the v3 deployment (all tasks are text + entity
        extraction); if a future project varies these the caller should
        also move them out of system.

    output_schema, when supplied, is embedded as a stable JSON block.
    Schemas are per-project (not per-task), so embedding them here keeps
    system bytestable while making the schema part of the cacheable
    prefix instead of leaking into the variable user payload.

    The SELF-CHECK TOOL workflow text was historically appended to the
    user message (one source of per-call variability — its bytes shifted
    relative to the per-task content above it). Now baked into base.
    """
    base = (
        "You are an annotation subagent. Return raw JSON only, with no markdown fences or commentary. "
        "Follow the output_schema in this prompt: it is the JSON Schema your response must conform to, and its "
        "$defs section enumerates every allowed entity type (entityType enum) and json_structures "
        "phrase type (jsonStructureType enum or equivalent). "
        "Use ONLY those values — labels outside the schema's enums will be rejected by the validator. "
        "For text entity spans, copy exact contiguous text spans from task.source_ref.payload.text. "
        "MANDATORY ROW COVERAGE: the output rows array MUST contain an entry for EVERY row in "
        "task.source_ref.payload.rows, in the same order. If a row has no entities and no phrases, "
        "still include it with empty dicts: {\"output\": {\"entities\": {}, \"json_structures\": {}}}. "
        "Omitting any input row is a validation error that resets the task. "
        "For json_structures: on every row, scan the input text for instances of every phrase type the schema "
        "declares and populate json_structures with arrays of VERBATIM strings copied from the input — no "
        "character offsets, just the text itself. The pipeline rejects any span that isn't a substring of "
        "input.text, so do not paraphrase. Empty json_structures = {} is only acceptable when the input "
        "genuinely contains no instance of any declared type. "
        "\n\n"
        "HANDLING QC FEEDBACK: for each item in feedback_bundle, choose either to fix or to rebut:\n"
        "(a) if you accept the complaint — silently fix the annotation; no discussion_reply needed.\n"
        "(b) if you disagree — add a discussion_reply with a verbal confidence label.\n"
        "\n"
        "discussion_replies schema (each entry):\n"
        "  feedback_id: str (must match feedback_bundle.items)\n"
        "  confidence:  REQUIRED — one of these strings (no numbers; the runtime won't accept them):\n"
        "    - \"certain\"   = evidence unambiguous; you can quote the exact span/text proving QC is wrong; "
        "any reasonable reviewer would agree.\n"
        "    - \"confident\" = strong case but a reasonable reviewer with different priors might side with QC.\n"
        "    - \"tentative\" = judgment call; you lean against QC but admit the other reading is defensible.\n"
        "    - \"unsure\"    = you don't know — let the arbiter / human decide.\n"
        "    Don't anchor on \"certain\". Pick the label that actually fits the evidence strength.\n"
        "  message:     str, REQUIRED, your reasoning\n"
        "  disputed_points: list[str], optional\n"
        "  proposed_resolution: str, optional\n"
        "  stance:      str, optional — for human readability only. The label drives the decision.\n"
        "Omit discussion_replies on a first attempt with no prior feedback. Never set consensus yourself."
        "\n\n"
        "KNOWLEDGE BASE TOOL: when the `check_past_experience` tool appears in your "
        "tools list, the project has an accumulated annotation history you can consult before guessing. "
        "Call it for any candidate entity/phrase whose type is genuinely ambiguous in this row — typically: "
        "named-entity spans (proper nouns, products, organizations, technologies), tokens that could "
        "plausibly map to multiple types in the schema, or unfamiliar terms. The tool returns the current "
        "convention (status `active` / `disputed` / `none`), the distribution of past type proposals, up "
        "to 3 representative example sentences per type with `[task_id/row_id]` trace prefixes, and a "
        "wordfreq Zipf score. Prefer matching the project's established `active` convention. When the "
        "convention is `disputed`, use the per-type example sentences as analogies and pick whichever "
        "type's examples best match this row's surrounding context. When `meta.generic_word` is true with "
        "little evidence, the span is likely a function word and should usually be left untagged. Do NOT "
        "call the tool for tokens that are clearly not entities (function words, punctuation, common "
        "verbs) or for spans whose schema mapping is already obvious — over-calling wastes tokens and "
        "the tool has no useful information for spans you can already handle."
        "\n\n"
        + _ANNOTATION_VALIDATOR_SUFFIX
        + f"\n\nModality: {task.modality}. Requirements: {json.dumps(task.annotation_requirements, sort_keys=True)}."
    )
    parts = [base, _SHARED_SPAN_RULES]
    if output_schema is not None:
        # Schema bytes are stable per-project. Embedding into the system
        # prompt with sort_keys=True for deterministic byte layout.
        parts.append(
            "OUTPUT SCHEMA (your response MUST conform to this JSON Schema):\n"
            + json.dumps(output_schema, sort_keys=True)
        )
    if guideline:
        parts.append(guideline)
    # NOTE: conventions_block intentionally NOT appended here — caller
    # prepends it to the user message so the system stays bytestable.
    return "\n\n".join(parts)


def _qc_instructions(task: Task, *, guideline: str | None = None) -> str:
    """Legacy module-level helper retained for any external callers.

    The runtime now uses ``SubagentRuntime._qc_instructions``, which resolves
    the QC sampling policy from project config when the task has none. This
    fallback uses the default ``RuntimeConfig``.
    """
    return _build_qc_instructions(
        task,
        resolved_policy=_resolve_qc_policy_from_task_or_config(task, RuntimeConfig()),
        guideline=guideline,
    )


def _resolve_qc_policy_from_task_or_config(task: Task, config: RuntimeConfig) -> dict[str, Any]:
    """Build the QC sampling policy: legacy per-task override wins, else project default."""
    task_policy = task.metadata.get("qc_policy") if isinstance(task.metadata, dict) else None
    if isinstance(task_policy, dict) and task_policy:
        return task_policy
    return {
        "mode": config.qc_sample_mode,
        "sample_ratio": config.qc_sample_ratio,
        "sample_count": config.qc_sample_count,
    }


def _build_qc_instructions(
    task: Task,
    *,
    resolved_policy: dict[str, Any],
    guideline: str | None = None,
    conventions_block: str | None = None,  # DEPRECATED — kept for callers that still pass it; no-op
) -> str:
    base = (
        "You are a QC subagent. Inspect EVERY row of the task and the latest annotation artifact end-to-end. "
        "Return raw JSON with no markdown fences. Include a boolean field named passed. "
        "If passed is false, include message or failures. failures must be a list of objects with row_id or target, category, message, severity, and suggested_action. "
        "When feedback discussions or annotator rebuttals are present, include feedback_resolution as a list of row-level decisions with row_id, decision, and reason. "
        "Use the output_schema and annotation_guidance fields in this prompt as the quality policy when present. "
        "\n\n"
        "DETERMINISM: scan every row exactly once. Do not sample, do not pick random rows. "
        "If you fail this task, the NEXT QC pass on the same input MUST produce the same failure list — "
        "do not surface different missing types on different passes; that creates infinite retry loops. "
        "\n\n"
        "ROUND DISCIPLINE: The feedback_bundle in this prompt contains all prior feedback items for this task. "
        "If feedback_bundle.items is EMPTY, this is round 1 — scan every row exhaustively and report EVERY "
        "defect you detect. This is your only opportunity to flag issues anywhere in the annotation. "
        "If feedback_bundle.items is NON-EMPTY, this is a retry round. In retry rounds your failures list "
        "is STRICTLY RESTRICTED to: "
        "(1) rows or spans explicitly referenced by a prior feedback item where the annotator's fix is still "
        "incorrect or introduced a new defect; and "
        "(2) rows or spans that the annotator has visibly changed since the last round but that now contain "
        "a new defect not present before. "
        "Do NOT raise new failures on rows or spans that prior feedback did not cover — those regions were "
        "scanned in round 1 and implicitly approved. If you see issues there, accept them; the round-1 "
        "window has passed. Violating this rule causes the annotator to chase an ever-expanding target "
        "and creates infinite retry loops. "
        "\n\n"
        "json_structures recall: for each row, scan the input text for every phrase type declared in this "
        "prompt's output_schema (jsonStructureType enum or equivalent $defs). Each phrase is a verbatim string "
        "copied from input.text — no character offsets, and the pipeline rejects spans that aren't substrings "
        "of input.text. Use ONLY phrase types the schema declares; do not invent new ones. When a phrase type "
        "also appears as an entity type, treat the json_structures version as OPTIONAL — do NOT flag tasks for "
        "missing json_structures entries just because the same name appears in entities. "
        "\n\n"
        "SEVERITY: every entry in failures MUST include a severity field set to ONE of these "
        "strings exactly (the runtime rejects any other value):\n"
        "  - \"blocking\" = prevents downstream use; task cannot proceed without this fix.\n"
        "  - \"error\"    = clear defect that must be corrected (wrong type, verbatim mismatch, missing required entity).\n"
        "  - \"warning\"  = likely defect but debatable; annotator should review (e.g. borderline case, tentative type choice).\n"
        "  - \"info\"     = advisory note; does not require a fix but worth flagging.\n"
        "\n"
        "CONFIDENCE: every entry in failures MUST include a confidence field set to ONE of these "
        "strings (no numbers; the runtime won't accept them):\n"
        "  - \"certain\"   = you can quote the exact verbatim span the annotation got wrong; any reasonable "
        "reviewer would agree this is a defect.\n"
        "  - \"confident\" = strong defect but requires reading more than one sentence to confirm; reasonable "
        "reviewer with different priors might disagree.\n"
        "  - \"tentative\" = judgment call you'd defend but you admit a reasonable reviewer could disagree.\n"
        "  - \"unsure\"    = you're really not sure — at that point DO NOT FLAG. Just pass instead.\n"
        "Don't anchor on \"certain\". Pick the label that fits the evidence strength. If you only ever use "
        "\"certain\", you are miscalibrated.\n"
        "\n"
        "ANNOTATOR REBUTTALS: if feedback_bundle items carry annotator discussion_replies, each reply has a "
        "confidence label. Compare against your own label for that feedback:\n"
        "(1) annotator label is HIGHER than yours (e.g. annotator=\"certain\", you=\"tentative\") → the "
        "annotator is more sure; emit this feedback_id in consensus_acknowledgements (closes the dispute).\n"
        "(2) labels are equal → re-evaluate; if still defective keep the failure (same label); if you've "
        "changed your mind, ack it.\n"
        "(3) annotator label is LOWER than yours → keep the failure.\n"
        "\n\n"
        "KNOWLEDGE BASE TOOL: when `check_past_experience` is in your tools list, "
        "call it for any span where you're uncertain whether the annotator's type choice matches "
        "project convention — pass the span text as `entry`. The tool returns the established "
        "convention (if any) and example sentences from past tasks. Use the convention as the "
        "ground truth: if the annotator chose a type that contradicts an `active` convention with "
        "high evidence_count, flag it; if the convention is `disputed` or `none`, fall back to "
        "your own judgement and the per-type examples. Skip the tool for spans whose type is "
        "obvious from the schema and surrounding text — over-calling wastes tokens."
        "\n\n"
        f"qc_policy (informational): {json.dumps(resolved_policy, sort_keys=True)}. "
        f"Modality: {task.modality}. Requirements: {json.dumps(task.annotation_requirements, sort_keys=True)}."
    )
    parts = [base, _SHARED_SPAN_RULES]
    if guideline:
        parts.append(guideline)
    return "\n\n".join(parts)


def _task_payload(task: Task, store: "SqliteStore | None" = None) -> dict[str, Any]:
    """Build the prompt input for annotator / QC / arbiter subagents.

    When ``store`` is provided, masked rows are filtered out of
    ``source_ref.payload.rows`` before the prompt is built, so LLM
    subagents never see (and can never annotate) rows that have been
    masked at the operator's review step. Passing ``store=None`` keeps
    the legacy behaviour for any caller that still needs the unfiltered
    payload (rare — most call sites have a store handy).
    """
    sref = task.source_ref
    if store is not None:
        from annotation_pipeline_skill.services.row_mask_service import (
            apply_masks_to_task,
        )
        masked = apply_masks_to_task(store, task)
        sref = masked.source_ref
    return {
        "task_id": task.task_id,
        "source_ref": sref,
        "selected_annotator_id": task.selected_annotator_id,
        "metadata": task.metadata,
    }


def _parse_qc_decision(text: str) -> dict[str, Any]:
    try:
        payload = _parse_llm_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise QCParseError("QC response was not valid JSON.", raw_text=text) from exc
    if not isinstance(payload, dict):
        raise QCParseError("QC response JSON must be an object.", raw_text=text)
    if not isinstance(payload.get("passed"), bool):
        raise QCParseError("QC response JSON must include boolean passed.", raw_text=text)
    failures = payload.get("failures", [])
    if failures is not None and not isinstance(failures, list):
        raise QCParseError("QC response failures must be a list when present.", raw_text=text)
    feedback_resolution = payload.get("feedback_resolution", [])
    if feedback_resolution is not None and not isinstance(feedback_resolution, list):
        raise QCParseError("QC response feedback_resolution must be a list when present.", raw_text=text)
    if payload["passed"] is False and not str(payload.get("message") or payload.get("summary") or "").strip() and not failures:
        raise QCParseError("Rejected QC response must include message or failures.", raw_text=text)
    consensus_acks = payload.get("consensus_acknowledgements", [])
    if consensus_acks is not None and not isinstance(consensus_acks, list):
        consensus_acks = []
    return {
        "passed": bool(payload.get("passed", False)),
        "message": str(payload.get("message") or payload.get("summary") or ""),
        "category": str(payload.get("category") or "qc"),
        "severity": _severity_value(payload.get("severity")),
        "target": payload.get("target") if isinstance(payload.get("target"), dict) else {},
        "suggested_action": str(payload.get("suggested_action") or "annotator_rerun"),
        "failures": failures or [],
        "feedback_resolution": feedback_resolution or [],
        "consensus_acknowledgements": [str(x) for x in consensus_acks if isinstance(x, str)],
        "raw_response": payload,
    }


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks, returning stripped text or '' if nothing remains."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _parse_llm_json(text: str) -> Any:
    """Robust JSON parser for LLM-emitted text.

    Primary path: strip ``<think>...</think>`` blocks first (robust-json-parser
    scans for the largest/first JSON object in the full text and incorrectly
    picks up annotation JSON embedded inside thinking rather than the QC answer
    that follows ``</think>``), then call ``robust-json-parser`` which handles:
      - markdown code fences (```json ... ```)
      - prose preambles ("I'm rebuilding the annotations..." from codex CLI)
      - single quotes instead of doubles, trailing commas, inline comments
      - truncated / partial JSON (auto-closes braces)

    If stripping think blocks leaves nothing (model hit max_tokens mid-think and
    never produced an answer), fall back to the original full text so the library
    can attempt recovery via partial-JSON auto-close.

    Fallback path (when the library fails): scan for top-level ``{`` and
    use ``json.JSONDecoder.raw_decode`` to extract the first balanced JSON
    object, ignoring trailing prose. This covers the case where the LLM
    interleaves CoT prose and the JSON output — the library tripped on
    stray ``{`` chars inside the prose and returned "No JSON payload"
    even when valid JSON followed at the end.

    Raises ``ValueError`` (the base class of ``json.JSONDecodeError``) on
    unrecoverable input — call sites already catching ``json.JSONDecodeError``
    keep working because ``JSONDecodeError`` is a subclass.
    """
    stripped = _strip_think_blocks(text)
    parse_target = stripped if stripped else text
    try:
        return _robust_json_loads(parse_target)
    except (ValueError, TypeError) as primary_err:
        # Fallback: scan for every "{" position and try ``raw_decode``;
        # collect successfully-parsed candidates, then pick the one most
        # likely to be the actual payload. Picking the FIRST candidate
        # (the obvious approach) is wrong when the LLM interleaves prose
        # and JSON — ``raw_decode`` might land on an inner ``{...}`` (e.g.
        # one row's ``output`` field) before the outer ``{"rows": [...]}``.
        #
        # Selection rule:
        #   1. Prefer the candidate whose top-level dict contains key
        #      ``rows`` (annotation/correction shape) or ``verdicts``
        #      (arbiter shape) — these are the canonical envelopes.
        #   2. Otherwise pick the candidate with the largest text span
        #      (end - start), which is almost always the outer object.
        decoder = json.JSONDecoder()
        candidates: list[tuple[Any, int]] = []  # (parsed, span_len)
        for opener in ("{", "["):
            for i in range(len(parse_target)):
                if parse_target[i] != opener:
                    continue
                try:
                    obj, end = decoder.raw_decode(parse_target, i)
                except json.JSONDecodeError:
                    continue
                candidates.append((obj, end - i))
        # Only accept candidates that look like a real envelope. Falling
        # back to "largest random dict" would silently substitute one
        # row's `output` for the whole `{"rows": [...]}` envelope —
        # validator then complains about missing rows, masking the actual
        # truncated-JSON bug. Better to raise so the worker retries.
        ENVELOPE_KEYS = ("rows", "verdicts", "corrected_annotation")
        for obj, _span in sorted(candidates, key=lambda c: -c[1]):
            if isinstance(obj, dict) and any(k in obj for k in ENVELOPE_KEYS):
                return obj
        raise primary_err


def _auto_fill_missing_rows(task: Task, parsed: Any) -> int:
    """Insert empty `{entities: {}, json_structures: {}}` stubs for any
    source row the annotator dropped from its output. Returns the count
    of stubs added.

    Background: qwen3.6-35b-a3b on this annotation prompt produces 40%
    "missing rows" failures (8 of 32 attempts in a sampled 30 min
    window). The model knows what to do for the rows it emits — it's
    just sloppy about row coverage. Each dropped row triggers a full
    re-annotation (~220s) for one missing row, which is wasted work.

    Auto-fill is SAFE because:
      - The empty stub is structurally identical to what the prompt
        already documents ("MANDATORY ROW COVERAGE: ... If a row has
        no entities and no phrases, still include it with empty dicts").
      - QC still sees the row and can flag it for missing content if
        the row genuinely has annotatable spans the model overlooked.
      - We don't fabricate spans — only add scaffolding.
    """
    if not isinstance(parsed, dict):
        return 0
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        return 0
    try:
        src_rows = task.source_ref.get("payload", {}).get("rows", [])
    except (AttributeError, TypeError):
        return 0
    if not isinstance(src_rows, list) or not src_rows:
        return 0
    present_ids = {
        r.get("row_id") for r in rows
        if isinstance(r, dict) and isinstance(r.get("row_id"), str)
    }
    added = 0
    for src in src_rows:
        if not isinstance(src, dict):
            continue
        rid = src.get("row_id")
        if not isinstance(rid, str) or rid in present_ids:
            continue
        ridx = src.get("row_index")
        rows.append({
            "row_id": rid,
            "row_index": ridx,
            "output": {"entities": {}, "json_structures": {}},
        })
        present_ids.add(rid)
        added += 1
    if added:
        # Keep stable order by row_index so the artifact reads naturally.
        rows.sort(key=lambda r: r.get("row_index", 0) if isinstance(r, dict) and isinstance(r.get("row_index"), int) else 0)
    return added


def _serialize_llm_json(text: str, *, task: Task | None = None) -> str:
    """Parse LLM output, run safe auto-fix, and re-serialize as canonical
    JSON. Returns the original text if no JSON can be recovered (caller
    surfaces the error downstream).

    Auto-fix steps (boundary-only, NEVER touches letters):
      - Strip surrounding whitespace / sentence punctuation / quote chars
        from spans whose only defect is the boundary (``try_align_to_verbatim``).
      - For already-verbatim spans where the punct-trimmed form is ALSO
        verbatim, prefer the trimmed form — matches ``find_trailing_punctuation_spans``
        ("entity is the name, not the sentence boundary").
      - Auto-fill missing rows with empty `{entities:{}, json_structures:{}}`
        stubs (qwen 40% drop rate, see ``_auto_fill_missing_rows``).
      - Dedupe within-type duplicates.

    Cross-type collisions, hallucinated (genuinely-non-verbatim) spans,
    schema breakage, and empty annotations are NOT silently rewritten —
    they fail the verification gate and surface as feedback. Auto-fix only
    handles cases where the annotator's intent is clear but the boundary
    was off by trim-safe characters or rows were silently dropped.
    """
    try:
        parsed = _parse_llm_json(text)
    except (ValueError, TypeError):
        return text
    if task is not None:
        try:
            from annotation_pipeline_skill.core.schema_validation import (
                auto_fix_safe_spans_in_place,
            )
            auto_fix_safe_spans_in_place(task, parsed)
        except Exception:  # noqa: BLE001 — never block artifact write on auto-fix
            pass
        try:
            _auto_fill_missing_rows(task, parsed)
        except Exception:  # noqa: BLE001 — never block artifact write on auto-fix
            pass
    _dedupe_within_type_spans(parsed)
    try:
        return json.dumps(parsed, ensure_ascii=False)
    except (ValueError, TypeError):
        return text


def _coerce_to_enum_in_place(
    payload: Any, schema: Any
) -> tuple[dict[str, int], dict[str, int]]:
    """Remove entity / json_structure entries whose TYPE KEY is not in the
    resolved output_schema enum.

    Returns ``(dropped, rescued)`` where:
    - ``dropped``: types invalid in both fields, truly invented — spans lost.
      ``{"entities/<bad_type>": N, ...}``
    - ``rescued``: types valid in the OTHER field, misrouted — spans moved.
      ``{"entities/<bad_type>→json_structures": N, ...}``

    Arbiters occasionally put valid json_structures types (risk, goal, task…)
    under entities and vice versa. Rather than discarding those spans, we move
    them to the correct field so the correction is not degraded. Duplicates
    created by the move are cleaned up by _dedupe_within_type_spans downstream.

    Truly invented types (valid in neither field) are still dropped.

    Safe no-op when payload doesn't conform, or when the schema can't be
    resolved into enum sets (we leave the data alone and let the strict
    validator make the call instead of silently passing through).
    """
    from annotation_pipeline_skill.core.schema_validation import _schema_type_enums
    if not isinstance(payload, dict):
        return {}, {}
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}, {}
    entity_enum, phrase_enum = _schema_type_enums(schema)
    if not entity_enum and not phrase_enum:
        # Schema couldn't be resolved — don't coerce, let downstream
        # validation handle whatever is there.
        return {}, {}
    dropped: dict[str, int] = {}
    rescued: dict[str, int] = {}
    other_field: dict[str, tuple[str, set[str]]] = {
        "entities": ("json_structures", phrase_enum),
        "json_structures": ("entities", entity_enum),
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        for field_key, allowed in (
            ("entities", entity_enum),
            ("json_structures", phrase_enum),
        ):
            field = output.get(field_key)
            if not isinstance(field, dict) or not allowed:
                continue
            other_key, other_allowed = other_field[field_key]
            for bad_type in [t for t in field.keys() if t not in allowed]:
                spans = field.pop(bad_type)
                count = len(spans) if isinstance(spans, list) else 0
                if other_allowed and bad_type in other_allowed and isinstance(spans, list):
                    # Misrouted: move to the correct field.
                    target = output.setdefault(other_key, {})
                    if bad_type in target and isinstance(target[bad_type], list):
                        target[bad_type].extend(spans)
                    else:
                        target[bad_type] = spans
                    if count:
                        rkey = f"{field_key}/{bad_type}→{other_key}"
                        rescued[rkey] = rescued.get(rkey, 0) + count
                else:
                    if count:
                        dkey = f"{field_key}/{bad_type}"
                        dropped[dkey] = dropped.get(dkey, 0) + count
    return dropped, rescued


def _dedupe_within_type_spans(payload: Any) -> None:
    """In-place dedupe of duplicate string spans within the same
    entities/json_structures type. Preserves first occurrence order.
    Safe no-op for non-conforming payloads.
    """
    if not isinstance(payload, dict):
        return
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        for field_key in ("entities", "json_structures"):
            field = output.get(field_key)
            if not isinstance(field, dict):
                continue
            for typ, items in field.items():
                if not isinstance(items, list):
                    continue
                seen: set[str] = set()
                deduped: list[Any] = []
                for s in items:
                    if isinstance(s, str):
                        if s in seen:
                            continue
                        seen.add(s)
                    deduped.append(s)
                field[typ] = deduped


def _iter_verbatim_spans(output: dict) -> "list[tuple[str, str]]":
    """Yield (span_text, location) pairs from an annotation row's output for
    verbatim-against-input checking. location is a short label like
    'entities.number' or 'json_structures.constraint'.
    """
    spans: list[tuple[str, str]] = []
    entities = output.get("entities")
    if isinstance(entities, dict):
        for ent_type, items in entities.items():
            if not isinstance(items, list):
                continue
            for s in items:
                if isinstance(s, str):
                    spans.append((s, f"entities.{ent_type}"))
    js = output.get("json_structures")
    if isinstance(js, dict):
        for phrase_type, items in js.items():
            if not isinstance(items, list):
                continue
            for s in items:
                if isinstance(s, str):
                    spans.append((s, f"json_structures.{phrase_type}"))
                elif isinstance(s, dict) and isinstance(s.get("text"), str):
                    # Tolerate the legacy {text,start,end} shape too.
                    spans.append((s["text"], f"json_structures.{phrase_type}"))
    return spans


def _clamp_confidence(value: Any) -> float | None:
    """Coerce a model-provided confidence value to a clamped float in [0, 1].

    Accepts a verbal label (preferred) or a legacy numeric value. Labels map
    to bin midpoints so callers that still need a number get a comparable
    one. Returns None if the value can't be interpreted.
    """
    label = _resolve_confidence_label(value)
    if label is not None:
        return _LABEL_TO_NUMERIC[label]
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(0.0, min(1.0, f))


# Verbal confidence scale. Ordered high → low. Each label has an explicit
# semantic anchor written into the role prompts; the runtime treats them as
# categorical (no numeric comparison across roles). The numeric mapping is
# kept only for backward compat with historical samples and for legacy
# diagnostics — decisions should branch on the label.
CONFIDENCE_LABELS = ("certain", "confident", "tentative", "unsure")

_LABEL_TO_NUMERIC: dict[str, float] = {
    "certain": 0.97,
    "confident": 0.85,
    "tentative": 0.55,
    "unsure": 0.20,
}

# Coarse buckets to map legacy numeric values back into the label scale.
# Threshold is the inclusive lower bound.
_NUMERIC_TO_LABEL_BINS: list[tuple[float, str]] = [
    (0.85, "certain"),
    (0.65, "confident"),
    (0.40, "tentative"),
    (0.0, "unsure"),
]


def _resolve_confidence_label(value: Any) -> str | None:
    """Return one of CONFIDENCE_LABELS for any model-provided confidence value.

    Accepts the new verbal label or a legacy numeric value. Returns None
    if the value is missing or uninterpretable.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in CONFIDENCE_LABELS:
            return normalized
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    f = max(0.0, min(1.0, f))
    for threshold, label in _NUMERIC_TO_LABEL_BINS:
        if f >= threshold:
            return label
    return "unsure"


def _feedback_from_qc_decision(task: Task, attempt_id: str, decision: dict[str, Any]) -> list[FeedbackRecord]:
    failures = decision.get("failures") if isinstance(decision.get("failures"), list) else []
    valid_failures: list[dict[str, Any]] = [f for f in failures if isinstance(f, dict)]
    if not valid_failures:
        valid_failures = [{}]

    records: list[FeedbackRecord] = []
    for i, failure in enumerate(valid_failures):
        metadata: dict[str, Any] = {"qc_decision": decision}
        if i == 0:
            confidence_label = _resolve_confidence_label(failure.get("confidence"))
            if confidence_label is not None:
                metadata["confidence"] = confidence_label
        records.append(FeedbackRecord.new(
            task_id=task.task_id,
            attempt_id=attempt_id,
            source_stage=FeedbackSource.QC,
            severity=FeedbackSeverity(failure.get("severity") or decision.get("severity") or FeedbackSeverity.WARNING.value),
            category=str(failure.get("category") or decision.get("category") or "qc"),
            message=str(failure.get("message") or decision.get("message") or "QC rejected the annotation result."),
            target=failure.get("target") if isinstance(failure.get("target"), dict) else decision.get("target") if isinstance(decision.get("target"), dict) else {},
            suggested_action=str(failure.get("suggested_action") or decision.get("suggested_action") or "annotator_rerun"),
            created_by="qc",
            metadata=metadata,
        ))
    return records


def _severity_value(value: object) -> str:
    if isinstance(value, str):
        try:
            return FeedbackSeverity(value).value
        except ValueError:
            return FeedbackSeverity.WARNING.value
    return FeedbackSeverity.WARNING.value


def _strip_non_verbatim_spans_in_place(corrected: dict, violations: list[dict]) -> int:
    """Remove hallucinated (non-verbatim) spans from a corrected_annotation dict.

    Mutates ``corrected`` in-place. Returns the number of spans removed.
    Each violation is a dict with keys ``row_index``, ``field`` (e.g.
    ``"entities.organization"``), and ``span``.
    """
    removed = 0
    for v in violations:
        row_idx = v.get("row_index")
        field = v.get("field", "")   # e.g. "entities.organization"
        span = v.get("span")
        if span is None or not field:
            continue
        parts = field.split(".", 1)
        if len(parts) != 2:
            continue
        typ_dict_key, type_name = parts
        for row in corrected.get("rows", []):
            if not isinstance(row, dict) or row.get("row_index") != row_idx:
                continue
            output = row.get("output")
            if not isinstance(output, dict):
                continue
            items = output.get(typ_dict_key, {}).get(type_name)
            if isinstance(items, list) and span in items:
                items.remove(span)
                removed += 1
    return removed

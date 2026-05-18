from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
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


class QCParseError(ValueError):
    def __init__(self, message: str, *, raw_text: str):
        super().__init__(message)
        self.diagnostics = {"error_kind": "parse_error", "raw_text": raw_text}


def _is_rate_limited(exc: BaseException) -> bool:
    """Detect provider rate-limit / quota errors across SDKs and local-CLI clients.

    Covers openai.RateLimitError (status 429), generic APIStatusError with
    .status_code==429, and CLI-style errors that just carry a message — we
    inspect both the type name and the string representation.
    """
    name = type(exc).__name__
    if "RateLimit" in name:
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
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
    return any(s in text for s in ("not found", "unauthorized", "forbidden",
                                    " 400 ", " 401 ", " 403 ", " 404 ", " 422 "))


class SubagentRuntime:
    def __init__(
        self,
        store: SqliteStore,
        client_factory: Callable[[str], LLMClient],
        *,
        max_qc_rounds: int | None = None,
        config: RuntimeConfig | None = None,
    ):
        self.store = store
        self.client_factory = client_factory
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
        if not task.document_version_id:
            return None
        try:
            ver = self.store.load_document_version(task.document_version_id)
        except FileNotFoundError:
            return None
        return f"Annotation guideline ({ver.version}):\n{ver.content}"

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
            prelabeled = [
                artifact for artifact in self.store.list_artifacts(task.task_id)
                if artifact.kind == "annotation_result"
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
        annotation_result = await self._generate_async(
            stage_target,
            LLMGenerateRequest(
                instructions=_annotation_instructions(
                    task,
                    guideline=guideline,
                    conventions_block=conventions_block,
                ),
                prompt=self._annotation_prompt(task),
                continuity_handle=self._read_pinned_handle(task, "continuity_handle", stage_target),
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
        await self._run_validation_and_qc(
            task,
            annotation_artifact,
            annotation_attempt_id,
            annotation_result.final_text,
        )

    # Hard cap on consecutive arbiter mechanical retries. After this many
    # arbiter pickups produce no actionable verdict (codex error / no fix /
    # bad correction), give up and route to HR. Prevents a stuck task from
    # looping forever when the LLM consistently fails on it.
    ARBITER_MECHANICAL_RETRY_CAP = 3

    def _handle_arbiter_mechanical_fail(
        self,
        task: Task,
        attempt_id: str,
        arb: dict,
        stage: str,
        hr_extra_metadata: dict,
    ) -> None:
        """Bump the per-task mechanical-retry counter. If the cap is reached,
        transition to HR. Otherwise leave the task in ARBITRATING for re-pickup.
        """
        count = int(task.metadata.get("arbiter_mechanical_retries", 0)) + 1
        task.metadata["arbiter_mechanical_retries"] = count
        if count >= self.ARBITER_MECHANICAL_RETRY_CAP:
            metadata = {
                **hr_extra_metadata,
                "arbiter_mechanical_retries": count,
                "arbiter_ran": arb["ran"],
                "arbiter_unresolved": arb["unresolved"],
                "arbiter_mechanical_fail": arb["mechanical_fail"],
            }
            # Surface the arbiter's failed verbatim correction so HR can
            # see what it tried to fix and decide whether to apply it
            # manually (after fixing the verbatim issue) or send back to
            # annotation.
            verbatim_exhausted = bool(arb.get("verbatim_retry_exhausted"))
            if verbatim_exhausted:
                metadata["arbiter_verbatim_retry_exhausted"] = True
                metadata["arbiter_failed_correction"] = arb.get("failed_verbatim_correction")
                metadata["arbiter_failed_verbatim_target"] = arb.get("failed_verbatim_target")
            if verbatim_exhausted:
                target = arb.get("failed_verbatim_target") or {}
                reason_text = (
                    f"Arbiter ruled qc/neither but could not produce a verbatim-compliant "
                    f"correction after {count} pickup(s); routing to human review "
                    f"(failed span: {target.get('span')!r} at "
                    f"{target.get('field')!r} row {target.get('row_index')})"
                )
            else:
                reason_text = (
                    f"Arbiter retried {count} times but kept failing to return a usable answer "
                    f"(JSON parse / shape errors / LLM exception); routing to human review"
                )
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason=reason_text,
                stage=stage,
                attempt_id=attempt_id,
                metadata=metadata,
            )

    def _retry_round_count(self, task_id: str) -> int:
        """Count how many *open* retry rounds have happened for this task.

        A round is any QC/VALIDATION feedback that bounced the task back to
        PENDING. Feedbacks that have already been resolved by consensus
        (QC accepted an annotator rebuttal) are excluded — otherwise a
        single subjective complaint that both sides agreed to drop would
        still march the task toward HUMAN_REVIEW.
        """
        discussions = self.store.list_feedback_discussions(task_id)
        consensus_ids = {d.feedback_id for d in discussions if d.consensus}
        return sum(
            1 for f in self.store.list_feedback(task_id)
            if (f.source_stage is FeedbackSource.QC or f.source_stage is FeedbackSource.VALIDATION)
            and f.feedback_id not in consensus_ids
        )

    async def _run_validation_and_qc(
        self,
        task: Task,
        annotation_artifact: ArtifactRef,
        annotation_attempt_id: str,
        annotation_final_text: str,
    ) -> None:
        validation_failure = self._check_annotation_validation(task, annotation_final_text)
        if validation_failure is not None:
            self._record_validation_feedback(
                task,
                annotation_attempt_id,
                category=validation_failure["category"],
                message=validation_failure["message"],
                target=validation_failure.get("target", {}),
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
                    self._transition(
                        task,
                        TaskStatus.HUMAN_REVIEW,
                        reason="Arbiter flagged its own answer as uncertain (tentative/unsure verdict); needs human review",
                        stage="validation",
                        attempt_id=annotation_attempt_id,
                        metadata={
                            "auto_escalated": True,
                            "round_count": round_count,
                            "max_qc_rounds": self.max_qc_rounds,
                            "arbiter_ran": arb["ran"],
                            "arbiter_unresolved": arb["unresolved"],
                            "arbiter_mechanical_fail": arb["mechanical_fail"],
                        },
                    )
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
        # One feedback per attempt; aggregate count + first example in the message
        # so annotator sees the pattern without N feedbacks per task.
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
                target={"duplicates": dups[:5]},
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
        qc_result = await self._generate_async(
            "qc",
            LLMGenerateRequest(
                instructions=self._qc_instructions(task, guideline=guideline),
                prompt=self._qc_prompt(task, annotation_artifact),
                continuity_handle=self._read_pinned_handle(task, "qc_continuity_handle", "qc"),
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
                self.store.append_feedback(verifier_failure["feedback"])
                self._transition(
                    task,
                    TaskStatus.ARBITRATING,
                    reason="prior verifier flagged divergence at QC pass",
                    stage="prior_verifier",
                    attempt_id=qc_attempt_id,
                    metadata={
                        "qc_artifact_id": qc_artifact.artifact_id,
                        "prior_verifier_action": "qc_pass_divergent",
                        "verifier_payload": verifier_failure["payload"],
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
            feedback = _feedback_from_qc_decision(task, qc_attempt_id, qc_decision)
            self.store.append_feedback(feedback)
            qc_conf = _clamp_confidence(feedback.metadata.get("confidence"))
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
                    self._transition(
                        task,
                        TaskStatus.HUMAN_REVIEW,
                        reason="Arbiter flagged its own answer as uncertain (tentative/unsure verdict); needs human review",
                        stage="qc",
                        attempt_id=qc_attempt_id,
                        metadata={
                            "auto_escalated": True,
                            "round_count": round_count,
                            "max_qc_rounds": self.max_qc_rounds,
                            "feedback_id": feedback.feedback_id,
                            "qc_artifact_id": qc_artifact.artifact_id,
                            "arbiter_ran": arb["ran"],
                            "arbiter_unresolved": arb["unresolved"],
                            "arbiter_mechanical_fail": arb["mechanical_fail"],
                        },
                    )
                else:
                    self._handle_arbiter_mechanical_fail(
                        task, qc_attempt_id, arb, stage="qc",
                        hr_extra_metadata={
                            "round_count": round_count,
                            "max_qc_rounds": self.max_qc_rounds,
                            "feedback_id": feedback.feedback_id,
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
                    metadata={"feedback_id": feedback.feedback_id, "qc_artifact_id": qc_artifact.artifact_id},
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
            extract_entity_type_decisions,
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
            decisions = extract_entity_type_decisions(prelabel or {}, current)
            if not decisions:
                return
            svc = EntityConventionService(self.store)
            for span, entity_type in decisions:
                try:
                    svc.record_decision(
                        project_id=task.pipeline_id,
                        span=span,
                        entity_type=entity_type,
                        source="qc_consensus",
                        task_id=task.task_id,
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
    ) -> dict | None:
        """Return {feedback, payload} on the FIRST divergent (span, type), or
        None when every span is agree/cold_start.
        """
        from annotation_pipeline_skill.services.entity_statistics_service import (
            EntityStatisticsService,
            iter_span_decisions,
        )
        payload = self._load_annotation_payload(annotation_artifact)
        if payload is None:
            return None
        svc = EntityStatisticsService(self.store)
        for span, entity_type in iter_span_decisions(payload):
            result = svc.check(
                project_id=task.pipeline_id,
                span=span,
                proposed_type=entity_type,
            )
            if result.status != "divergent":
                continue
            attempts = self.store.list_attempts(task.task_id)
            attempt_id = attempts[-1].attempt_id if attempts else f"{task.task_id}-attempt-0"
            verifier_payload = {
                "span": result.span,
                "proposed_type": result.proposed_type,
                "dominant_type": result.dominant_type,
                "dominant_count": result.dominant_count,
                "total": result.total,
                "distribution": result.distribution,
            }
            return {
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
            }
        return None

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
        task.metadata["prior_verifier_payload"] = divergence["payload"]

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

    async def _generate_async(self, target: str, request: LLMGenerateRequest) -> LLMGenerateResult:
        try:
            return await self._call_client(target, request)
        except Exception as exc:  # noqa: BLE001 — try fallback on transient provider errors
            if target == "fallback" or not _is_provider_transient_error(exc):
                raise
            return await self._call_client("fallback", request)

    def _profile_name_for_target(self, target: str) -> str | None:
        """Return the LLM profile name the factory currently resolves for
        ``target``. Used to invalidate cross-provider continuity handles —
        a ``previous_response_id`` minted by codex (e.g.) is meaningless to
        a Qwen-backed gateway and causes the gateway to 404. Compares
        cheaply via a transient client construction; profile lookup itself
        does not open any network connections.
        """
        try:
            client = self.client_factory(target)
        except Exception:  # noqa: BLE001
            return None
        prof = getattr(client, "profile", None)
        return getattr(prof, "name", None)

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
        current = self._profile_name_for_target(target)
        if minted_by is None or current is None or minted_by == current:
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
            return await client.generate(request)
        finally:
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()

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
        schema = resolve_output_schema(task, self.store)
        if schema is None:
            return None
        try:
            payload = _parse_llm_json(final_text)
        except (json.JSONDecodeError, ValueError) as exc:
            return {
                "category": "schema_invalid",
                "message": f"Annotation result is not valid JSON: {exc}",
                "reason": "schema validation failed",
            }
        if isinstance(payload, dict):
            # Strip discussion_replies before schema validation: it's a
            # side-channel for QC dialogue, not part of the output schema.
            # May appear at top level or nested inside each row.
            payload.pop("discussion_replies", None)
            rows = payload.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        row.pop("discussion_replies", None)
        try:
            validate_payload_against_task_schema(task, payload, store=self.store)
        except SchemaValidationError as exc:
            return {
                "category": "schema_invalid",
                "message": f"Annotation result failed schema validation: {exc}",
                "reason": "schema validation failed",
                "target": {"errors": exc.errors},
            }
        # After the schema check, enforce verbatim — every annotated entity /
        # phrase string must exist in the corresponding input row's text.
        # Catches "annotator hallucinated a span" failures at validation time
        # instead of waiting for QC.
        verbatim_failure = self._check_verbatim_spans(task, payload)
        if verbatim_failure is not None:
            return verbatim_failure
        # Cross-type entity collision: same span tagged under two entity
        # types in the same row. Blocking — annotator must pick one.
        # json_structures collisions are NOT blocked (phrases can legitimately
        # play multiple roles).
        from annotation_pipeline_skill.core.schema_validation import (
            find_cross_type_collisions,
            find_trailing_punctuation_spans,
        )
        collisions = find_cross_type_collisions(payload)
        if collisions:
            first = collisions[0]
            return {
                "category": "cross_type_collision",
                "message": (
                    f"Row {first['row_index']} entity span {first['span']!r} is tagged as "
                    f"both {first['types'][0]!r} and {first['types'][1]!r}. Pick one type "
                    f"per span — the schema allows separate keys but a single occurrence "
                    f"should resolve to a single entity type."
                ),
                "reason": "cross-type entity collision",
                "target": {"row_index": first["row_index"], "span": first["span"], "types": first["types"]},
            }
        # Trailing-punctuation span boundary check: blocks "Mitul Mallik."
        # when "Mitul Mallik" is also in input. The entity is the name,
        # not the sentence boundary.
        trailing = find_trailing_punctuation_spans(task, payload)
        if trailing:
            first = trailing[0]
            return {
                "category": "trailing_punctuation_span",
                "message": (
                    f"Row {first['row_index']} {first['field']} span {first['span']!r} ends "
                    f"with sentence-ending punctuation that should not be part of the entity. "
                    f"Re-emit as {first['trimmed']!r} — the trimmed form is also verbatim in "
                    f"input.text and that's where the entity boundary belongs."
                ),
                "reason": "trailing-punctuation span boundary",
                "target": {"row_index": first["row_index"], "field": first["field"],
                           "span": first["span"], "trimmed": first["trimmed"]},
            }
        return None

    def _check_verbatim_spans(self, task: Task, payload: Any) -> dict | None:
        """Wrap the shared ``find_verbatim_violations`` helper in the
        validation-failure dict shape the pipeline uses (first mismatch only,
        so retry feedback stays focused on one issue at a time).
        """
        from annotation_pipeline_skill.core.schema_validation import find_verbatim_violations
        violations = find_verbatim_violations(task, payload)
        if not violations:
            return None
        first = violations[0]
        return {
            "category": "non_verbatim_span",
            "message": (
                f"Row {first['row_index']} {first['field']}: span {first['span']!r} "
                f"is not a verbatim substring of the input text."
            ),
            "reason": "verbatim check failed",
            "target": first,
        }

    def _auto_align_corrected_annotation(self, task: Task, corrected: dict) -> int:
        """Mutate ``corrected`` in place: replace each non-verbatim span that
        differs from a verbatim substring only by surrounding whitespace /
        punctuation / quotes with the aligned form. Returns the number of
        spans that were rewritten.

        Letter-level edits are NEVER performed — ``try_align_to_verbatim``
        only strips trim-safe characters from either end, so the entity's
        semantic content is preserved. Spans that can't be safely aligned
        are left untouched and will surface to the normal verbatim-failure
        path below.
        """
        from annotation_pipeline_skill.core.schema_validation import try_align_to_verbatim
        source_payload = task.source_ref.get("payload") if isinstance(task.source_ref, dict) else None
        if not isinstance(source_payload, dict):
            return 0
        source_rows = source_payload.get("rows")
        if not isinstance(source_rows, list):
            return 0
        input_by_index: dict[int, str] = {}
        for i, r in enumerate(source_rows):
            if not isinstance(r, dict):
                continue
            idx = r.get("row_index") if isinstance(r.get("row_index"), int) else i
            text = r.get("input")
            if isinstance(text, str):
                input_by_index[idx] = text
        rows_out = corrected.get("rows") if isinstance(corrected, dict) else None
        if not isinstance(rows_out, list):
            return 0
        rewrites = 0
        for r in rows_out:
            if not isinstance(r, dict):
                continue
            row_index = r.get("row_index") if isinstance(r.get("row_index"), int) else 0
            input_text = input_by_index.get(row_index)
            if not input_text:
                continue
            output = r.get("output")
            if not isinstance(output, dict):
                continue
            for typ_dict_key in ("entities", "json_structures"):
                type_dict = output.get(typ_dict_key)
                if not isinstance(type_dict, dict):
                    continue
                for type_name, items in type_dict.items():
                    if not isinstance(items, list):
                        continue
                    for i, span in enumerate(items):
                        if not isinstance(span, str) or not span:
                            continue
                        if span in input_text:
                            continue
                        aligned = try_align_to_verbatim(span, input_text)
                        if aligned is not None:
                            items[i] = aligned
                            rewrites += 1
        return rewrites

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
        source_payload = task.source_ref.get("payload") if isinstance(task.source_ref, dict) else None
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
            validate_payload_against_task_schema,
        )

        # Schema check the corrected annotation up front. If it fails we punt
        # back to HR rather than save a bad artifact.
        try:
            validate_payload_against_task_schema(task, corrected, store=self.store)
        except SchemaValidationError:
            return None
        # Verbatim check — arbiter sometimes paraphrases / normalizes spans
        # (e.g., traditional→simplified Chinese, dropped articles) that pass
        # schema but break the input.text substring guarantee. Without this
        # check, hallucinated/normalized spans landed in ACCEPTED tasks
        # (5% audit found ~11% violation rate). On failure, return None so
        # _terminal_from_arbiter falls through to HUMAN_REVIEW instead of
        # saving a bad corrected_annotation as the final artifact.
        verbatim_failure = self._check_verbatim_spans(task, corrected)
        if verbatim_failure is not None:
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
            task.metadata["prior_verifier_action"] = "second_arbiter_verbatim_failed"
            self._clear_divergence_flag(task)
            self._transition(
                task,
                TaskStatus.HUMAN_REVIEW,
                reason=(
                    f"Second arbiter tried to correct {span!r} but its "
                    f"corrected_annotation contains a non-verbatim span "
                    f"({failed_target.get('span')!r} at "
                    f"{failed_target.get('field')!r}) after retries exhausted. "
                    f"Bad correction preserved in event metadata; needs human "
                    f"review (first arbiter said {first_type!r}, project prior "
                    f"dominant {prior_type!r})"
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
                target_name="arbiter",
            )
        except (_ArbiterClientUnavailable, _ArbiterCallFailed):
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
            "`corrected_annotation` is a TOP-LEVEL REQUIRED key. Always present. Either an object "
            "matching current_annotation's shape, or literally null. Do not omit the field.\n\n"
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
            "Shape of corrected_annotation when non-null: a {\"rows\": [{\"row_index\": int, "
            "\"output\": {entities, classifications, relations, json_structures}}, ...]} object that "
            "preserves every row from current_annotation.\n"
            "MUST CONFORM TO output_schema (provided in the prompt). In particular: entity types are "
            "limited to the enum in $defs.entityType — do NOT invent new entity types like 'attribute' "
            "or 'system'. json_structures keys are limited to the enum in $defs.jsonStructureType. "
            "Each entity / phrase is a bare VERBATIM string copied from the corresponding row's "
            "input.text (no character offsets, just the text itself). Pipeline validates: every span "
            "must appear in input.text via substring match.\n"
            "Preserve fields the annotator already had right; only change what your verdicts say "
            "needs changing.\n\n"
            "Return raw JSON only, no markdown fences."
        )
        # All three agents (annotator, QC, arbiter) share the same cross-cutting
        # span rules — verbatim, no character substitution, no trailing punct,
        # no duplicates, one type per span. Single source of truth in code.
        instructions = instructions + "\n\n" + _SHARED_SPAN_RULES
        # Inject per-project entity conventions if any match the input text.
        conventions_block = self._build_conventions_block(task)
        if conventions_block:
            instructions = instructions + "\n\n" + conventions_block
        # Include the resolved output_schema so the arbiter doesn't invent
        # entity types, phrase types, or field shapes when constructing
        # corrected_annotation. Without this constraint, gpt-5.5 was emitting
        # entity names like "attribute" / "system" that the schema validator
        # rejected, causing the fix to silently fall back to HR.
        from annotation_pipeline_skill.core.schema_validation import resolve_output_schema
        output_schema = resolve_output_schema(task, self.store)
        prompt = json.dumps(
            {
                "task_id": task.task_id,
                "input": task.source_ref.get("payload", {}),
                "current_annotation": current_annotation,
                "output_schema": output_schema,
                "disputed_items": items,
            },
            indent=2,
            sort_keys=True,
        )
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
                result = await arbiter_client.generate(LLMGenerateRequest(
                    instructions=attempt_instructions,
                    prompt=prompt,
                    continuity_handle=None,
                ))
            except Exception as exc:  # noqa: BLE001
                raise _ArbiterCallFailed(str(exc))
            try:
                payload = _parse_llm_json(result.final_text)
            except (json.JSONDecodeError, ValueError) as exc:
                raise _ArbiterCallFailed(str(exc))
            verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
            if not isinstance(verdicts, list):
                raise _ArbiterCallFailed("arbiter response missing 'verdicts' list")
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
                                "\nCANDIDATE VERBATIM SPANS from this row's input.text that "
                                "share words with the failed span (copy one of these exactly, "
                                "or pick a different verbatim substring of input.text):\n  - "
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
                    # Retries exhausted. Keep the bad corrected_annotation
                    # so callers can preserve it in HR metadata; the flag
                    # tells them NOT to apply it as a real correction.
                    verbatim_exhausted_target = verbatim_failure.get("target", {})
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
            conventions_block=self._build_conventions_block(task),
        )

    def _build_conventions_block(self, task: Task) -> str | None:
        """Look up entity conventions established for this project that match
        any span in the task's input text. Returns a prompt block to inject
        into the annotator / QC / arbiter instructions, or None if no
        matches.
        """
        from annotation_pipeline_skill.services.entity_convention_service import (
            EntityConventionService,
        )
        try:
            payload = task.source_ref.get("payload") if isinstance(task.source_ref, dict) else None
            rows = payload.get("rows") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                return None
            # Concatenate all rows' input text for a single substring search.
            combined = "\n".join(
                r.get("input", "") for r in rows if isinstance(r, dict) and isinstance(r.get("input"), str)
            )
        except Exception:  # noqa: BLE001 — never let prompt build fail the task
            return None
        if not combined.strip():
            return None
        svc = EntityConventionService(self.store)
        try:
            matches = svc.find_matches_in_text(task.pipeline_id, combined)
        except Exception:  # noqa: BLE001
            return None
        if not matches:
            return None
        lines = []
        for m in matches[:50]:  # cap at 50 to bound prompt size
            note_suffix = f"  (notes: {m.notes})" if m.notes else ""
            if m.entity_type == "not_an_entity":
                lines.append(
                    f"  - {m.span_original!r} → DO NOT TAG (this span is "
                    f"intentionally NOT an entity in this project)" + note_suffix
                )
            else:
                lines.append(f"  - {m.span_original!r} → entities.{m.entity_type}" + note_suffix)
        return (
            "KNOWN ENTITY CONVENTIONS FOR THIS PROJECT (established by prior "
            "QC consensus, arbiter rulings, or human review — apply them so "
            "ambiguous spans get classified consistently across tasks):\n"
            + "\n".join(lines)
        )

    def _annotation_prompt(self, task: Task) -> str:
        return json.dumps(
            {
                "task": _task_payload(task),
                "feedback_bundle": build_feedback_bundle(self.store, task.task_id),
                "prior_artifacts": self._artifact_context(task.task_id),
                "output_schema": resolve_output_schema(task, self.store),
            },
            sort_keys=True,
        )

    def _qc_prompt(self, task: Task, annotation_artifact: ArtifactRef) -> str:
        return json.dumps(
            {
                "task": _task_payload(task),
                "annotation_artifact": {
                    **annotation_artifact.to_dict(),
                    "payload": self._slim_annotation_payload(annotation_artifact),
                },
                "feedback_bundle": build_feedback_bundle(self.store, task.task_id),
                "output_schema": resolve_output_schema(task, self.store),
            },
            sort_keys=True,
        )

    def _slim_annotation_payload(self, artifact: ArtifactRef) -> Any:
        """Return only the parsed annotation rows, dropping ``raw_response`` and
        other provider-side metadata that downstream consumers (QC, arbiter)
        don't read. The minimax HTTP response can be 20 KB on its own — 75% of
        the QC prompt — and contributes nothing to QC's actual job. The
        pre-parsed inner JSON also saves QC/arbiter from re-parsing the
        ``<think>``/markdown-fence wrapper."""
        raw = self._read_artifact_payload(artifact)
        if not isinstance(raw, dict):
            return raw
        text = raw.get("text")
        if not isinstance(text, str):
            # Fallback: keep the dict but drop the bulky raw_response.
            return {k: v for k, v in raw.items() if k != "raw_response"}
        try:
            return _parse_llm_json(text)
        except (json.JSONDecodeError, ValueError):
            return {"text": text}

    def _artifact_context(
        self, task_id: str, *, per_kind_limit: int = 1
    ) -> list[dict[str, Any]]:
        """Return artifacts grouped by kind, keeping only the most recent N per kind.

        Default ``per_kind_limit=1``: just the single latest artifact of each
        kind. The previous 3-per-kind cap was a conservative cushion from
        early development; in practice the annotator only needs the most
        recent annotation (to see its own last attempt) and the most recent
        qc_result (the latest reviewer output). Earlier attempts are stale —
        the active QC complaints come through feedback_bundle anyway, which
        names the specific row/target to fix. Cutting 3→1 saves another
        ~14 KB per prompt on a loop-heavy task.

        Prevents the annotator prompt from growing unbounded when a task loops
        through repeated annotation/QC retries (the 73-attempt case we hit in
        production blew past the LLM context window).

        Each artifact is slimmed down to the fields the annotator/QC/arbiter
        actually read. Things dropped:
          • wrapper.path, wrapper.content_type — filesystem detail
          • payload.raw_response — provider HTTP wrapper, single biggest bloat
          • payload.usage, payload.diagnostics, payload.task_id — runtime
            telemetry the LLM doesn't read (task_id is already on the wrapper)
          • payload.decision.raw_response (qc_result, arbiter_result) — the
            qc/arbiter parser stores its parsed JSON back under
            ``decision.raw_response`` in addition to lifting fields to the top
            level. Pure duplicate of decision.{failures,feedback_resolution,
            message,passed}; dropping saves ~900 chars per qc_result.

        Empirically the annotation prompt drops from 173 KB → ~50 KB after
        all of these trims on a task with the max 3+3+1 artifacts.
        """
        by_kind: dict[str, list[ArtifactRef]] = {}
        for artifact in self.store.list_artifacts(task_id):
            by_kind.setdefault(artifact.kind, []).append(artifact)
        selected: list[ArtifactRef] = []
        for arts in by_kind.values():
            # Artifacts are returned in insertion (seq) order — keep tail N
            selected.extend(arts[-per_kind_limit:])
        results: list[dict[str, Any]] = []
        for artifact in selected:
            payload = self._read_artifact_payload(artifact)
            if isinstance(payload, dict):
                payload = {
                    k: v for k, v in payload.items()
                    if k not in {"raw_response", "usage", "diagnostics", "task_id"}
                }
                decision = payload.get("decision")
                if isinstance(decision, dict) and "raw_response" in decision:
                    payload = {
                        **payload,
                        "decision": {k: v for k, v in decision.items() if k != "raw_response"},
                    }
            wrapper = {k: v for k, v in artifact.to_dict().items() if k not in {"path", "content_type"}}
            results.append({**wrapper, "payload": payload})
        return results

    def _read_artifact_payload(self, artifact: ArtifactRef) -> Any:
        path = self.store.root / artifact.path
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

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
    conventions_block: str | None = None,
) -> str:
    base = (
        "You are an annotation subagent. Return raw JSON only, with no markdown fences or commentary. "
        "Follow the output_schema and annotation_guidance fields in this prompt (output_schema is the JSON Schema your response must conform to). Honor allowed_entity_types and rules from annotation_guidance when present. "
        "For text entity spans, copy exact contiguous text spans from task.source_ref.payload.text. "
        "Do not add entity labels outside the configured allowed entity types. "
        "For json_structures: on every row, scan the input text for all 10 phrase types defined in annotation_guidance "
        "(status, risk, goal, strategy, constraint, decision, task, preference, reason, technology) and populate "
        "json_structures with arrays of VERBATIM strings copied from the input — no character offsets, just the text "
        "itself. The pipeline rejects any span that isn't a substring of input.text, so do not paraphrase. Building "
        "codes, requirements, must/shall statements are almost always constraints. Empty json_structures = {} is only "
        "acceptable when the input genuinely contains no instance of any type. "
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
        f"\n\nModality: {task.modality}. Requirements: {json.dumps(task.annotation_requirements, sort_keys=True)}."
    )
    parts = [base, _SHARED_SPAN_RULES]
    if conventions_block:
        parts.append(conventions_block)
    if guideline:
        parts.append(guideline)
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
    conventions_block: str | None = None,
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
        "json_structures recall: for each row, scan the input text for all 10 phrase types "
        "(status, risk, goal, strategy, constraint, decision, task, preference, reason, technology) defined "
        "in annotation_guidance. Each phrase is a verbatim string copied from input.text — no character offsets, "
        "and the pipeline rejects spans that aren't substrings of input.text. Building codes / must / shall / "
        "should statements are clear constraints. Note: json_structures.technology is OPTIONAL — do NOT flag "
        "tasks for missing technology phrases just because the same name appears in entities.technology. "
        "json_structures.technology is appropriate only when the technology is the structural subject of a "
        "phrase (decision about it, constraint on it, status update on it). "
        "\n\n"
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
        f"qc_policy (informational): {json.dumps(resolved_policy, sort_keys=True)}. "
        f"Modality: {task.modality}. Requirements: {json.dumps(task.annotation_requirements, sort_keys=True)}."
    )
    parts = [base, _SHARED_SPAN_RULES]
    if conventions_block:
        parts.append(conventions_block)
    if guideline:
        parts.append(guideline)
    return "\n\n".join(parts)


def _task_payload(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "source_ref": task.source_ref,
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


def _parse_llm_json(text: str) -> Any:
    """Robust JSON parser for LLM-emitted text.

    Backed by the ``robust-json-parser`` library, which handles:
      - <think>...</think> reasoning blocks (minimax / deepseek-reasoner / qwen-r1)
      - markdown code fences (```json ... ```)
      - prose preambles ("I'm rebuilding the annotations..." from codex CLI)
      - single quotes instead of doubles, trailing commas, inline comments
      - truncated / partial JSON (auto-closes braces)

    Raises ``ValueError`` (the base class of ``json.JSONDecodeError``) on
    unrecoverable input — call sites already catching ``json.JSONDecodeError``
    keep working because ``JSONDecodeError`` is a subclass.
    """
    return _robust_json_loads(text)


def _serialize_llm_json(text: str, *, task: Task | None = None) -> str:
    """Parse LLM output and re-serialize as canonical JSON. Returns the
    original text if no JSON can be recovered (caller surfaces the error
    downstream).

    Dedupes within-type spans (annotators / arbiter sometimes emit the same
    string twice under one entity type) but performs NO character-level
    normalization of spans. Per the downstream GLiNER pipeline, spans must
    be byte-for-byte substrings of input.text — any "cleanup" (CJK
    whitespace collapse, trailing punctuation strip, Traditional↔Simplified,
    LaTeX escape removal) breaks the verbatim guarantee training relies on.

    Cross-type collisions and non-verbatim spans are NOT silently rewritten;
    they fail the verification gate and surface as feedback.
    """
    # `task` is accepted for API compatibility with callers that already
    # pass it, but no normalization needs it today.
    del task
    try:
        parsed = _parse_llm_json(text)
    except (ValueError, TypeError):
        return text
    _dedupe_within_type_spans(parsed)
    try:
        return json.dumps(parsed, ensure_ascii=False)
    except (ValueError, TypeError):
        return text


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


def _feedback_from_qc_decision(task: Task, attempt_id: str, decision: dict[str, Any]) -> FeedbackRecord:
    failures = decision.get("failures") if isinstance(decision.get("failures"), list) else []
    first_failure = failures[0] if failures and isinstance(failures[0], dict) else {}
    confidence_label = _resolve_confidence_label(
        first_failure.get("confidence") if isinstance(first_failure, dict) else None
    )
    metadata: dict[str, Any] = {"qc_decision": decision}
    if confidence_label is not None:
        metadata["confidence"] = confidence_label
    return FeedbackRecord.new(
        task_id=task.task_id,
        attempt_id=attempt_id,
        source_stage=FeedbackSource.QC,
        severity=FeedbackSeverity(decision["severity"]),
        category=str(first_failure.get("category") or decision.get("category") or "qc"),
        message=str(first_failure.get("message") or decision.get("message") or "QC rejected the annotation result."),
        target=first_failure.get("target") if isinstance(first_failure.get("target"), dict) else decision.get("target") if isinstance(decision.get("target"), dict) else {},
        suggested_action=str(first_failure.get("suggested_action") or decision.get("suggested_action") or "annotator_rerun"),
        created_by="qc",
        metadata=metadata,
    )


def _severity_value(value: object) -> str:
    if isinstance(value, str):
        try:
            return FeedbackSeverity(value).value
        except ValueError:
            return FeedbackSeverity.WARNING.value
    return FeedbackSeverity.WARNING.value

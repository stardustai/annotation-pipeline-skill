from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from annotation_pipeline_skill.core.models import ArtifactRef, FeedbackRecord, Task
from annotation_pipeline_skill.core.schema_validation import (
    SchemaValidationError,
    find_cross_type_collisions,
    find_trailing_punctuation_spans,
    find_verbatim_violations,
    validate_payload_against_task_schema,
)
from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource, TaskStatus
from annotation_pipeline_skill.core.transitions import InvalidTransition, transition_task
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@dataclass(frozen=True)
class HumanReviewDecisionResult:
    task: Task
    decision: dict

    def to_dict(self) -> dict:
        return {
            "task": self.task.to_dict(),
            "decision": self.decision,
        }


@dataclass(frozen=True)
class HumanCorrectionResult:
    task: Task
    artifact: ArtifactRef
    answer: dict

    def to_dict(self) -> dict:
        return {
            "task": self.task.to_dict(),
            "artifact": self.artifact.to_dict(),
            "answer": self.answer,
        }


class HumanReviewService:
    def __init__(self, store: SqliteStore):
        self.store = store

    def decide(
        self,
        *,
        task_id: str,
        action: str,
        actor: str,
        feedback: str,
        correction_mode: str,
        picks: list[dict] | None = None,
    ) -> HumanReviewDecisionResult:
        task = self.store.load_task(task_id)
        if task.status is not TaskStatus.HUMAN_REVIEW:
            raise InvalidTransition(f"task {task_id} is not in human_review")

        next_status, reason = self._transition_for_action(action)
        # Accept-with-picks: operator clicked one or more entity-type picks
        # in Manual Review that must patch the current task's annotation.
        # Plain accept doesn't write a new artifact, so route through
        # submit_correction with the patched payload instead.
        if action == "accept" and picks:
            answer = self._latest_annotation_payload(task_id)
            if answer is not None:
                if isinstance(answer, dict):
                    answer.pop("discussion_replies", None)
                    rows = answer.get("rows")
                    if isinstance(rows, list):
                        for row in rows:
                            if isinstance(row, dict):
                                row.pop("discussion_replies", None)
                n_applied = _apply_operator_picks(answer, picks)
                if n_applied:
                    note = f"accept with {n_applied} operator pick(s) applied"
                    if feedback.strip():
                        note = f"{note} — {feedback.strip()}"
                    self.submit_correction(
                        task_id=task_id, answer=answer, actor=actor,
                        note=note, force=False,
                    )
                    return HumanReviewDecisionResult(
                        task=self.store.load_task(task_id),
                        decision={
                            "task_id": task_id, "action": action, "actor": actor,
                            "feedback": feedback, "correction_mode": correction_mode,
                            "picks_applied": n_applied,
                        },
                    )
        # Verbatim guard on the "accept underlying annotation as-is" path.
        # The task likely landed in HR because the arbiter's verbatim retries
        # exhausted on a hallucinated span — accepting blindly would commit
        # known-bad data. Operator must use submit_correction with verbatim
        # spans, or request_changes.
        if next_status is TaskStatus.ACCEPTED:
            latest_annotation = self._latest_annotation_payload(task_id)
            if latest_annotation is not None:
                # Same span checks the annotator / arbiter / submit_correction
                # paths run. Accepting "as-is" must NOT bypass them — the
                # underlying annotation is what would end up in the training
                # export, and the operator may not have noticed defects.
                violations = find_verbatim_violations(task, latest_annotation)
                if violations:
                    raise SchemaValidationError(
                        f"underlying annotation has {len(violations)} non-verbatim span(s); "
                        f"leave the task in Human Review until the annotator re-runs it (Request Changes), or move it to Rejected if it's unfixable",
                        [
                            {"kind": "non_verbatim_span",
                             "path": f"rows[{v['row_index']}].output.{v['field']}",
                             "message": f"span {v['span']!r} is not a verbatim substring of the row's input.text"}
                            for v in violations
                        ],
                    )
                collisions = find_cross_type_collisions(latest_annotation)
                if collisions:
                    raise SchemaValidationError(
                        f"underlying annotation has {len(collisions)} cross-type entity collision(s); "
                        f"leave the task in Human Review until the annotator re-runs it (Request Changes), or move it to Rejected if it's unfixable",
                        [
                            {"kind": "cross_type_collision",
                             "path": f"rows[{c['row_index']}].output.entities",
                             "message": f"span {c['span']!r} tagged as both {c['types'][0]!r} and {c['types'][1]!r}; pick one"}
                            for c in collisions
                        ],
                    )
                trailing = find_trailing_punctuation_spans(task, latest_annotation)
                if trailing:
                    raise SchemaValidationError(
                        f"underlying annotation has {len(trailing)} span(s) with trailing sentence punctuation; "
                        f"leave the task in Human Review until the annotator re-runs it (Request Changes), or move it to Rejected if it's unfixable",
                        [
                            {"kind": "trailing_punctuation_span",
                             "path": f"rows[{t['row_index']}].output.{t['field']}",
                             "message": f"span {t['span']!r} should be {t['trimmed']!r} — trim trailing punctuation"}
                            for t in trailing
                        ],
                    )
                # Prior verifier check — no force override here; operator must
                # use submit_correction(force=True) to override the prior.
                from annotation_pipeline_skill.services.entity_statistics_service import (
                    EntityStatisticsService,
                    iter_span_decisions,
                )
                _ess = EntityStatisticsService(self.store)
                _divergent = []
                for _span, _entity_type in iter_span_decisions(latest_annotation):
                    _r = _ess.check(
                        project_id=task.pipeline_id,
                        span=_span,
                        proposed_type=_entity_type,
                    )
                    if _r.status == "divergent":
                        _divergent.append(_r)
                if _divergent:
                    raise SchemaValidationError(
                        f"underlying annotation disagrees with project prior on "
                        f"{len(_divergent)} span(s); leave the task in Human Review until the annotator re-runs it (Request Changes), or move it to Rejected if it's unfixable",
                        [{
                            "kind": "prior_disagreement",
                            "path": f"output.entities[{_r.proposed_type}]",
                            "message": (
                                f"span {_r.span!r} proposed as {_r.proposed_type!r} but "
                                f"prior ({_r.dominant_count}/{_r.total}) → {_r.dominant_type!r}"
                            ),
                        } for _r in _divergent],
                    )
        decision = {
            "task_id": task_id,
            "action": action,
            "actor": actor,
            "feedback": feedback,
            "correction_mode": correction_mode,
        }
        event = transition_task(
            task,
            next_status,
            actor=actor,
            reason=reason,
            stage="human_review",
            metadata={
                "action": action,
                "correction_mode": correction_mode,
                "feedback": feedback,
            },
        )
        # request_changes means the operator wants the annotator to redo the
        # task — NOT to resume from the most recent annotation artifact.
        # Without this marker, the scheduler's _prepare_annotating_for_resume
        # would see the existing annotation_result and bounce the task into
        # QC ("resume at QC"), short-circuiting the re-annotation entirely.
        if action == "request_changes":
            task.metadata["hr_request_changes"] = True
            task.metadata.pop("runtime_next_stage", None)
        self.store.append_event(event)
        self.store.save_task(task)

        # Update stats with HR weight (5x) when accepting via decide.
        if next_status is TaskStatus.ACCEPTED:
            _ann = self._latest_annotation_payload(task_id)
            if _ann is not None:
                self._increment_stats_from_hr(task, _ann)

        # Persist the human reviewer's feedback as a first-class FeedbackRecord
        # so it appears in the Discussions tab alongside QC feedback.
        if feedback.strip() and action in {"request_changes", "reject"}:
            attempts = self.store.list_attempts(task_id)
            attempt_id = attempts[-1].attempt_id if attempts else f"{task_id}-attempt-0"
            severity = FeedbackSeverity.BLOCKING if action == "reject" else FeedbackSeverity.WARNING
            self.store.append_feedback(
                FeedbackRecord.new(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    source_stage=FeedbackSource.HUMAN_REVIEW,
                    severity=severity,
                    category="human_review_decision",
                    message=feedback,
                    target={},
                    suggested_action=action,
                    created_by=actor,
                    metadata={"correction_mode": correction_mode},
                )
            )
        return HumanReviewDecisionResult(task=task, decision=decision)

    def submit_correction(
        self,
        *,
        task_id: str,
        answer: dict,
        actor: str,
        note: str | None,
        force: bool = False,
        record_conventions: bool = True,
        stat_bumps: list[tuple[str, str | None, str | None]] | None = None,
    ) -> HumanCorrectionResult:
        task = self.store.load_task(task_id)
        if task.status is not TaskStatus.HUMAN_REVIEW:
            raise InvalidTransition(f"task {task_id} is not in human_review")

        # Schema-validate. Raises SchemaValidationError on failure (missing schema OR mismatch).
        validate_payload_against_task_schema(task, answer, store=self.store)
        # Verbatim check — operator-submitted corrections must use exact spans
        # from the input, same as annotator/arbiter outputs. Without this, an
        # operator could paste a normalized / paraphrased span and ACCEPT a
        # task with a non-verbatim span (the same defect we just fixed in
        # the arbiter path).
        violations = find_verbatim_violations(task, answer)
        if violations:
            raise SchemaValidationError(
                f"corrected answer has {len(violations)} non-verbatim span(s)",
                [
                    {"kind": "non_verbatim_span", "path": f"rows[{v['row_index']}].output.{v['field']}",
                     "message": f"span {v['span']!r} is not a verbatim substring of the row's input.text"}
                    for v in violations
                ],
            )
        # Cross-type collision — same span tagged as two entity types in one row.
        # Block, same as annotator/arbiter paths.
        collisions = find_cross_type_collisions(answer)
        if collisions:
            raise SchemaValidationError(
                f"corrected answer has {len(collisions)} cross-type entity collision(s)",
                [
                    {"kind": "cross_type_collision",
                     "path": f"rows[{c['row_index']}].output.entities",
                     "message": f"span {c['span']!r} tagged as both {c['types'][0]!r} and {c['types'][1]!r}; pick one"}
                    for c in collisions
                ],
            )
        # Trailing-punctuation span boundary — block "Mitul Mallik." when the
        # trimmed form is also verbatim in input.text.
        trailing = find_trailing_punctuation_spans(task, answer)
        if trailing:
            raise SchemaValidationError(
                f"corrected answer has {len(trailing)} span(s) with trailing sentence punctuation",
                [
                    {"kind": "trailing_punctuation_span",
                     "path": f"rows[{t['row_index']}].output.{t['field']}",
                     "message": f"span {t['span']!r} should be {t['trimmed']!r} — trim trailing punctuation"}
                    for t in trailing
                ],
            )

        # Prior verifier check — skipped on operator-force override.
        if not force:
            from annotation_pipeline_skill.services.entity_statistics_service import (
                EntityStatisticsService,
                iter_span_decisions,
            )
            svc = EntityStatisticsService(self.store)
            divergent = []
            for span, entity_type in iter_span_decisions(answer):
                r = svc.check(
                    project_id=task.pipeline_id,
                    span=span,
                    proposed_type=entity_type,
                )
                if r.status == "divergent":
                    divergent.append(r)
            if divergent:
                raise SchemaValidationError(
                    f"corrected answer disagrees with project prior on "
                    f"{len(divergent)} span(s); pass force=True to override",
                    [{
                        "kind": "prior_disagreement",
                        "path": f"output.entities[{r.proposed_type}]",
                        "message": (
                            f"span {r.span!r} proposed as {r.proposed_type!r} but "
                            f"prior ({r.dominant_count}/{r.total}) → {r.dominant_type!r}"
                        ),
                    } for r in divergent],
                )

        artifact = self._write_correction_artifact(task_id, answer, actor=actor, note=note)
        event = transition_task(
            task,
            TaskStatus.ACCEPTED,
            actor=actor,
            reason="human review submitted corrected answer",
            stage="human_review",
            metadata={
                "human_authored": True,
                "answer_artifact_id": artifact.artifact_id,
                "answer_artifact_path": artifact.path,
                "note": note,
            },
        )
        self.store.append_artifact(artifact)
        self.store.append_event(event)
        self.store.save_task(task)
        # Auto-record entity conventions for any entity-type changes the
        # operator made vs the latest annotation. Captured per-project so
        # future tasks in the same project benefit from the human's call.
        # Callers that want a one-off task fix without promoting it to a
        # project rule pass record_conventions=False.
        if record_conventions:
            self._record_conventions_from_correction(task, answer, actor)
        # Update stats with HR weight (5x).
        # - Default (stat_bumps=None): full-correction path. Bump EVERY
        #   (span, type) in the corrected answer — the operator authored
        #   the whole annotation, so every span carries their endorsement.
        # - Scoped (stat_bumps provided): targeted fix path. Only touch
        #   the explicit (span, old_type, new_type) tuples — used by
        #   apply_posterior_fix so a bulk-retroactive sweep doesn't
        #   blanket-bump unrelated spans across hundreds of tasks (which
        #   would inflate their HR-weighted counts and flip previously-
        #   settled spans back into "contested").
        if stat_bumps is None:
            self._increment_stats_from_hr(task, answer)
        else:
            self._apply_scoped_stat_bumps(task, stat_bumps)
        return HumanCorrectionResult(task=task, artifact=artifact, answer=answer)

    def _increment_stats_from_hr(self, task: Task, answer: dict) -> None:
        from annotation_pipeline_skill.services.entity_statistics_service import (
            HR_WEIGHT,
            EntityStatisticsService,
            iter_span_decisions,
        )
        svc = EntityStatisticsService(self.store)
        for span, entity_type in iter_span_decisions(answer):
            try:
                svc.increment(
                    project_id=task.pipeline_id, span=span,
                    entity_type=entity_type, weight=HR_WEIGHT,
                )
            except Exception:  # noqa: BLE001
                continue

    def _apply_scoped_stat_bumps(
        self,
        task: Task,
        bumps: list[tuple[str, str | None, str | None]],
    ) -> None:
        """Apply only the explicit (span, old_type, new_type) stat changes.

        For each entry:
          - Increment (span, new_type) by HR_WEIGHT when new_type is set
            and != "not_an_entity".
          - Decrement (span, old_type) by 1 (approximate: assumes the
            prior contribution from THIS task came from an annotator/QC
            vote of weight 1, which is the common case). Caps the
            resulting count at 0 to avoid negative aggregates.
        """
        from annotation_pipeline_skill.services.entity_statistics_service import (
            HR_WEIGHT,
            EntityStatisticsService,
        )
        svc = EntityStatisticsService(self.store)
        for span, old_type, new_type in bumps:
            if new_type and new_type != "not_an_entity":
                try:
                    svc.increment(
                        project_id=task.pipeline_id, span=span,
                        entity_type=new_type, weight=HR_WEIGHT,
                    )
                except Exception:  # noqa: BLE001
                    pass
            if old_type and old_type != "not_an_entity":
                # Approximate-decrement: subtract 1 (typical annotator/QC
                # vote weight) for the now-superseded type. We can't tell
                # the exact historical contribution from THIS task without
                # walking history, so this is an estimate that prevents
                # stale-type counts from drifting upward unboundedly.
                span_lower = span.strip().lower()
                if not span_lower:
                    continue
                try:
                    svc.store._conn.execute(
                        "UPDATE entity_statistics SET "
                        "count = max(0, count - 1), updated_at = ? "
                        "WHERE project_id=? AND span_lower=? AND entity_type=?",
                        (
                            datetime.now(timezone.utc).isoformat(),
                            task.pipeline_id, span_lower, old_type,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    pass

    def _record_conventions_from_correction(self, task: Task, answer: dict, actor: str) -> None:
        from annotation_pipeline_skill.services.entity_convention_service import (
            EntityConventionService,
            extract_entity_type_decisions,
        )
        prior = self._latest_annotation_payload(task.task_id)
        decisions = extract_entity_type_decisions(prior, answer)
        if not decisions:
            return
        svc = EntityConventionService(self.store)
        for span, entity_type in decisions:
            try:
                svc.record_decision(
                    project_id=task.pipeline_id,
                    span=span,
                    entity_type=entity_type,
                    source=f"hr_correction:{actor}",
                    task_id=task.task_id,
                )
            except (ValueError, TypeError):
                continue

    def _latest_annotation_payload(self, task_id: str) -> dict | None:
        """Load and parse the most recent annotation_result artifact's inner
        annotation JSON. Returns None when there's no annotation_result yet
        or when the inner text isn't parseable JSON.

        Strips ``<think>...</think>`` reasoning blocks and a single leading
        markdown fence — same wrapper handling the runtime uses.
        """
        import re
        artifacts = [a for a in self.store.list_artifacts(task_id) if a.kind == "annotation_result"]
        if not artifacts:
            return None
        path = self.store.root / artifacts[-1].path
        if not path.exists():
            return None
        outer = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(outer, dict):
            return None
        text = outer.get("text")
        if not isinstance(text, str):
            return None
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _write_correction_artifact(self, task_id: str, answer: dict, *, actor: str, note: str | None) -> ArtifactRef:
        relative_path = Path("artifact_payloads") / task_id / f"human_review_answer-{uuid4().hex}.json"
        absolute_path = self.store.root / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text(
            json.dumps({"answer": answer, "actor": actor, "note": note}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return ArtifactRef.new(
            task_id=task_id,
            kind="human_review_answer",
            path=relative_path.as_posix(),
            content_type="application/json",
            metadata={"actor": actor, "note": note},
        )

    def _transition_for_action(self, action: str) -> tuple[TaskStatus, str]:
        if action == "accept":
            return TaskStatus.ACCEPTED, "human review accepted task"
        if action == "reject":
            return TaskStatus.REJECTED, "human review rejected task"
        if action == "request_changes":
            return TaskStatus.ANNOTATING, "human review requested annotator changes"
        raise ValueError(f"unknown human review action: {action}")

    def apply_posterior_fix(
        self,
        *,
        task_id: str,
        span: str,
        current_type: str,
        new_type: str | None,
        actor: str,
        save_as_convention: bool = True,
    ) -> dict:
        """Operator-level in-place correction triggered from Posterior Audit.

        The task must currently be ACCEPTED. We:
          1. Move it to HUMAN_REVIEW with an audit reason
          2. Load the latest annotation, swap (span, current_type) for
             (span, new_type) in every row that contains it (or remove
             entirely if new_type is None — "not_an_entity" / delete)
          3. Call submit_correction(force=True) which validates and
             transitions the task back to ACCEPTED with the corrected
             artifact and HR_WEIGHT-tagged entity_statistics bump.
        """
        from annotation_pipeline_skill.core.transitions import transition_task
        task = self.store.load_task(task_id)
        if task.status is not TaskStatus.ACCEPTED:
            raise InvalidTransition(
                f"task {task_id} is not ACCEPTED (posterior fix only applies to ACCEPTED)"
            )

        annotation = self._latest_annotation_payload(task_id)
        if not isinstance(annotation, dict):
            raise ValueError("no parseable annotation_result on this task")

        new_answer = _swap_span_type_in_payload(
            annotation, span=span, current_type=current_type, new_type=new_type,
        )
        # Auto-clean pre-existing defects that submit_correction would
        # block on. The operator's posterior-fix targets one (span, type)
        # pair; pre-existing trailing-punctuation / cross-type-collision
        # issues in *other* spans of the same task aren't theirs to
        # adjudicate manually, but they will block submit_correction's
        # strict checks. We auto-fix them deterministically so the
        # operator's scoped edit doesn't get derailed by unrelated noise.
        new_answer = _autoclean_pre_existing_defects(task, new_answer)

        # Transition ACCEPTED → HUMAN_REVIEW so submit_correction's
        # required-status check passes. Reason surfaces in audit log.
        event = transition_task(
            task, TaskStatus.HUMAN_REVIEW,
            actor=actor,
            reason=(
                f"posterior audit fix: '{span}' / {current_type} → "
                f"{new_type or 'delete (not_an_entity)'}"
            ),
            stage="posterior_audit",
            metadata={
                "posterior_fix": True,
                "span": span,
                "from_type": current_type,
                "to_type": new_type or "not_an_entity",
            },
        )
        self.store.save_task(task)
        self.store.append_event(event)

        # Record the explicit convention BEFORE submit_correction —
        # but only when the operator opted in (save_as_convention=True).
        # When False, this is a one-off task fix: patch the annotation
        # without promoting the operator's call to a project-wide rule.
        #
        # submit_correction has its own convention-recording pass but it
        # only fires on diffs between prior and answer — when the operator
        # confirms the current type (new_type == current_type), the answer
        # doesn't change so no convention gets written and the deviation
        # re-flags forever. Recording here makes the operator's policy
        # decision durable regardless of swap delta.
        #
        # If the (span) already has a convention that was previously
        # marked DISPUTED (because an earlier source picked a different
        # type), the normal record_decision call only appends a proposal
        # — it doesn't reactivate. The operator's explicit posterior-audit
        # declaration should be the tiebreaker, so we force-clear any
        # existing dispute to the operator's pick.
        if save_as_convention:
            decided_type = new_type or "not_an_entity"
            try:
                from annotation_pipeline_skill.services.entity_convention_service import (
                    EntityConventionService,
                )
                conv_svc = EntityConventionService(self.store)
                conv = conv_svc.record_decision(
                    project_id=task.pipeline_id,
                    span=span,
                    entity_type=decided_type,
                    source="posterior_audit_operator",
                    task_id=task_id,
                )
                if conv.status == "disputed":
                    # Operator's explicit declaration overrides prior dispute.
                    conv_svc.clear_dispute(
                        convention_id=conv.convention_id,
                        resolved_type=decided_type,
                        actor=actor,
                        notes="resolved by posterior audit operator declaration",
                    )
            except Exception:  # noqa: BLE001 — never let convention recording fail the fix
                pass

        try:
            self.submit_correction(
                task_id=task_id,
                answer=new_answer,
                actor=actor,
                note=f"posterior audit fix: '{span}' / {current_type} → {new_type or 'delete'}",
                force=True,
                record_conventions=save_as_convention,
                # Scope stat changes to the one (span, type) pair this
                # fix actually touches — bumping the entire annotation
                # would inflate every other span's HR-weighted count
                # across hundreds of tasks during Apply-to-all and flip
                # previously-settled spans back into contested.
                stat_bumps=[(span, current_type, new_type)],
            )
        except Exception:
            # Submit failed (schema / verbatim / etc). Roll the task back
            # to ACCEPTED so it isn't stranded in HR. Re-raise so the
            # operator sees the original error.
            try:
                rolled = self.store.load_task(task_id)
                if rolled.status is TaskStatus.HUMAN_REVIEW:
                    rb = transition_task(
                        rolled, TaskStatus.ACCEPTED,
                        actor="posterior_audit_rollback",
                        reason="posterior fix submit failed; restoring ACCEPTED",
                        stage="posterior_audit",
                    )
                    self.store.save_task(rolled)
                    self.store.append_event(rb)
            except Exception:  # noqa: BLE001
                pass
            raise
        return {
            "task_id": task_id,
            "span": span,
            "from_type": current_type,
            "to_type": new_type or "not_an_entity",
        }


def _repair_non_verbatim_span(input_text: str, span: str) -> str | None:
    """Try to recover a verbatim substring of ``input_text`` that the model
    likely intended when it emitted ``span``. Returns the corrected substring
    on success, or ``None`` when no reasonable match exists (caller should
    then drop).

    Strategies, in order of preference:
      1. Case-insensitive match — keep input's original casing
      2. Strip leading/trailing whitespace
      3. Collapse runs of internal whitespace
      4. Longest common substring via difflib, accepted if it covers ≥70 %
         of the original span length (or 4 chars, whichever is larger)
    """
    import re
    import difflib

    if not span or not input_text:
        return None
    if span in input_text:
        return span

    lower_text = input_text.lower()
    lower_span = span.lower()
    idx = lower_text.find(lower_span)
    if idx >= 0:
        return input_text[idx : idx + len(span)]

    stripped = span.strip()
    if stripped and stripped != span:
        if stripped in input_text:
            return stripped
        idx = lower_text.find(stripped.lower())
        if idx >= 0:
            return input_text[idx : idx + len(stripped)]

    norm = re.sub(r"\s+", " ", span).strip()
    if norm and norm != span:
        if norm in input_text:
            return norm
        idx = lower_text.find(norm.lower())
        if idx >= 0:
            return input_text[idx : idx + len(norm)]

    m = difflib.SequenceMatcher(None, lower_text, lower_span).find_longest_match(
        0, len(lower_text), 0, len(lower_span)
    )
    if m.size >= max(4, int(len(span) * 0.7)):
        return input_text[m.a : m.a + m.size]
    return None


def _input_text_by_row(task) -> dict[int, str]:
    """Build a row_index → input.text map from the task's source payload.
    Mirrors the lookup ``find_verbatim_violations`` uses."""
    source_payload = task.source_ref.get("payload") if isinstance(task.source_ref, dict) else None
    if not isinstance(source_payload, dict):
        return {}
    rows = source_payload.get("rows")
    if not isinstance(rows, list):
        return {}
    out: dict[int, str] = {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        idx = r.get("row_index") if isinstance(r.get("row_index"), int) else i
        text = r.get("input")
        if isinstance(text, str):
            out[idx] = text
    return out


def _autoclean_pre_existing_defects(task, payload: dict) -> dict:
    """Mutate `payload` in place to clean defects that submit_correction
    rejects:
      - non-verbatim spans → first try to repair (case fix, whitespace
        normalization, fuzzy substring match); drop only if no repair works
      - trailing-sentence-punctuation spans → replace with trimmed form
      - cross-type collisions (same span tagged under multiple entity
        types in one row) → keep the first listed type, drop the others
    Returns the same `payload` reference for chaining.
    """
    # Non-verbatim spans: repair first, drop only as last resort.
    input_by_row = _input_text_by_row(task)
    for v in find_verbatim_violations(task, payload):
        input_text = input_by_row.get(v["row_index"], "")
        repaired = _repair_non_verbatim_span(input_text, v["span"])
        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            if row.get("row_index") != v["row_index"]:
                continue
            output = row.get("output")
            if not isinstance(output, dict):
                continue
            type_key, type_name = (
                v["field"].split(".", 1) if "." in v["field"] else (v["field"], "")
            )
            container = output.get(type_key)
            if not isinstance(container, dict):
                continue
            items = container.get(type_name)
            if not isinstance(items, list):
                continue
            if repaired is not None and repaired != v["span"]:
                # Substitute the repaired (verbatim) form in place.
                container[type_name] = [
                    (repaired if s == v["span"] else s) for s in items
                ]
            elif repaired is None:
                # No safe repair — drop the span entirely.
                container[type_name] = [s for s in items if s != v["span"]]
                if not container[type_name]:
                    container.pop(type_name, None)
    # Trailing punctuation: replace span with trimmed form in the same field.
    for f in find_trailing_punctuation_spans(task, payload):
        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            if row.get("row_index") != f["row_index"]:
                continue
            output = row.get("output")
            if not isinstance(output, dict):
                continue
            type_key, type_name = f["field"].split(".", 1) if "." in f["field"] else (f["field"], "")
            container = output.get(type_key)
            if not isinstance(container, dict):
                continue
            items = container.get(type_name)
            if not isinstance(items, list):
                continue
            container[type_name] = [
                (f["trimmed"] if s == f["span"] else s) for s in items
            ]
    # Cross-type collisions: deterministically keep the first type, drop
    # the span from the others. Operator can override later via Conventions.
    for c in find_cross_type_collisions(payload):
        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            if row.get("row_index") != c["row_index"]:
                continue
            entities = row.get("output", {}).get("entities")
            if not isinstance(entities, dict):
                continue
            keeper = c["types"][0]
            for typ in c["types"][1:]:
                if typ in entities and isinstance(entities[typ], list):
                    entities[typ] = [s for s in entities[typ] if s != c["span"]]
                    if not entities[typ]:
                        entities.pop(typ, None)
    return payload


def _swap_span_type_in_payload(
    payload: dict, *, span: str, current_type: str, new_type: str | None,
) -> dict:
    """Return a deep copy of `payload` where every row's entities[current_type]
    list has `span` removed; if `new_type` is provided AND it's not the
    sentinel "not_an_entity", `span` is appended to entities[new_type] in
    that row. JSON-structures and other fields are untouched.

    Strips non-schema-friendly top-level fields like ``discussion_replies``
    that the annotator/arbiter add for runtime communication — these are
    not part of the project's annotation output schema and would fail
    the submit_correction schema validation.
    """
    import copy
    out = copy.deepcopy(payload)
    if isinstance(out, dict):
        out.pop("discussion_replies", None)
    rows = out.get("rows") if isinstance(out, dict) else None
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        entities = output.get("entities")
        if not isinstance(entities, dict):
            continue
        cur_list = entities.get(current_type)
        if not isinstance(cur_list, list) or span not in cur_list:
            continue
        # Drop span from the old bucket.
        entities[current_type] = [s for s in cur_list if s != span]
        if not entities[current_type]:
            entities.pop(current_type, None)
        # Add to new bucket unless deleting.
        if new_type and new_type != "not_an_entity":
            target = entities.setdefault(new_type, [])
            if span not in target:
                target.append(span)
    return out


def _apply_operator_picks(payload: dict, picks: list[dict]) -> int:
    """Mutate `payload` in place applying operator picks from Manual Review.

    Each pick is a dict with keys:
      - ``span``: str — the span text to relabel
      - ``entity_type``: str | None — target type, or "not_an_entity" / None
        to delete the span from all buckets

    For each pick, walks every row's entities and removes ``span`` from any
    bucket whose key differs from the target type; if the target type is a
    real entity type, ensures the span is present in that bucket.

    Matching is case-sensitive on the span string. Returns the number of
    picks that actually mutated at least one row (useful for status notes).

    Picks with an empty span or that don't appear in any row are silently
    skipped — the operator may have clicked a span that the annotation
    doesn't actually contain (e.g. arbiter dropped it between sessions).
    """
    if not isinstance(payload, dict) or not picks:
        return 0
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return 0
    applied = 0
    for pick in picks:
        span = pick.get("span")
        if not isinstance(span, str) or not span:
            continue
        target = pick.get("entity_type")
        delete = target is None or target == "not_an_entity"
        mutated = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            output = row.get("output")
            if not isinstance(output, dict):
                continue
            entities = output.get("entities")
            if not isinstance(entities, dict):
                continue
            # Drop span from every bucket that isn't the target.
            for typ in list(entities.keys()):
                if not delete and typ == target:
                    continue
                bucket = entities.get(typ)
                if not isinstance(bucket, list):
                    continue
                if span in bucket:
                    entities[typ] = [s for s in bucket if s != span]
                    if not entities[typ]:
                        entities.pop(typ, None)
                    mutated = True
            # Ensure span is in the target bucket when not deleting.
            if not delete:
                tgt = entities.setdefault(target, [])
                if span not in tgt:
                    tgt.append(span)
                    mutated = True
        if mutated:
            applied += 1
    return applied

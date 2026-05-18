"""Integration tests for prior verifier wiring across the runtime."""
from __future__ import annotations

import asyncio
import json

import pytest

from annotation_pipeline_skill.core.models import ArtifactRef, Task
from annotation_pipeline_skill.core.states import FeedbackSource, TaskStatus
from annotation_pipeline_skill.llm.client import LLMGenerateResult
from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
from annotation_pipeline_skill.services.entity_statistics_service import (
    EntityStatisticsService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _seed_prior(store, *, project_id, span, type_to_count):
    svc = EntityStatisticsService(store)
    for typ, n in type_to_count.items():
        for _ in range(n):
            svc.increment(project_id=project_id, span=span, entity_type=typ)


def _make_task(task_id, *, input_text, project_id="p"):
    return Task.new(
        task_id=task_id,
        pipeline_id=project_id,
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": input_text,
                "rows": [{"row_index": 0, "input": input_text}],
                "annotation_guidance": {"output_schema": {"type": "object"}},
            },
        },
    )


class _RecorderClient:
    def __init__(self, qc_passed: bool, annotation: dict):
        self.qc_passed = qc_passed
        self.annotation = annotation

    async def generate(self, request):
        if "qc subagent" in request.instructions.lower():
            final = json.dumps({
                "passed": self.qc_passed,
                "message": "ok" if self.qc_passed else "issues",
                "failures": [] if self.qc_passed else [{"category": "x", "message": "bad", "confidence": "certain"}],
            })
        else:
            final = json.dumps(self.annotation)
        return LLMGenerateResult(
            final_text=final, raw_response={}, usage={}, diagnostics={},
            runtime="stub", provider="stub", model="stub", continuity_handle=None,
        )


def test_qc_pass_with_prior_agree_accepts_and_increments_stats(tmp_path):
    store = SqliteStore.open(tmp_path)
    project = "p"
    _seed_prior(store, project_id=project, span="Apple",
                type_to_count={"organization": 9, "project": 1})

    annotation = {
        "rows": [{
            "row_index": 0,
            "output": {"entities": {"organization": ["Apple"]}},
        }]
    }
    task = _make_task("t-agree", input_text="Apple is a company")
    task.status = TaskStatus.PENDING
    store.save_task(task)

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: _RecorderClient(qc_passed=True, annotation=annotation),
    )
    asyncio.run(runtime.run_task_async(store.load_task("t-agree")))

    after = store.load_task("t-agree")
    assert after.status is TaskStatus.ACCEPTED
    svc = EntityStatisticsService(store)
    # Original 9+1 from seed plus 1 increment from this acceptance.
    assert svc.distribution(project_id=project, span="Apple") == {
        "organization": 10, "project": 1,
    }


def test_qc_pass_with_prior_divergent_routes_to_arbitrating(tmp_path):
    store = SqliteStore.open(tmp_path)
    project = "p"
    _seed_prior(store, project_id=project, span="Apple",
                type_to_count={"organization": 10})

    annotation = {
        "rows": [{
            "row_index": 0,
            "output": {"entities": {"technology": ["Apple"]}},
        }]
    }
    task = _make_task("t-divergent", input_text="Apple is mentioned")
    task.status = TaskStatus.PENDING
    store.save_task(task)

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: _RecorderClient(qc_passed=True, annotation=annotation),
    )
    asyncio.run(runtime.run_task_async(store.load_task("t-divergent")))

    after = store.load_task("t-divergent")
    assert after.status is TaskStatus.ARBITRATING
    fbs = store.list_feedback("t-divergent")
    assert any(
        f.source_stage is FeedbackSource.VALIDATION and f.category == "prior_disagreement"
        for f in fbs
    )


def test_qc_pass_with_cold_start_accepts(tmp_path):
    store = SqliteStore.open(tmp_path)
    project = "p"
    # 5 samples — below MIN_PRIOR_SAMPLES (10) → cold_start
    _seed_prior(store, project_id=project, span="Apple",
                type_to_count={"organization": 5})

    annotation = {
        "rows": [{
            "row_index": 0,
            "output": {"entities": {"technology": ["Apple"]}},
        }]
    }
    task = _make_task("t-cold", input_text="Apple is referenced")
    task.status = TaskStatus.PENDING
    store.save_task(task)

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: _RecorderClient(qc_passed=True, annotation=annotation),
    )
    asyncio.run(runtime.run_task_async(store.load_task("t-cold")))

    after = store.load_task("t-cold")
    assert after.status is TaskStatus.ACCEPTED
    svc = EntityStatisticsService(store)
    assert svc.distribution(project_id=project, span="Apple") == {
        "organization": 5, "technology": 1,
    }


def test_arbiter_acceptance_increments_stats(tmp_path):
    """When arbiter rules annotator-wins on a task that was QC-rejected,
    the resulting ACCEPTED transition still increments stats so they
    reflect every accepted decision in the project."""
    store = SqliteStore.open(tmp_path)
    project = "p"
    # Seed a clear prior agreeing with the annotation under test.
    _seed_prior(store, project_id=project, span="Acme",
                type_to_count={"organization": 12})

    annotation = {
        "rows": [{
            "row_index": 0,
            "output": {"entities": {"organization": ["Acme"]}},
        }]
    }
    task = _make_task("t-arb", input_text="Acme is mentioned")
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)

    # Drop a final annotation artifact for the runtime to read.
    rel_path = "artifact_payloads/t-arb/final.json"
    abs_path = store.root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(
        json.dumps({"text": json.dumps(annotation)}), encoding="utf-8"
    )
    artifact = ArtifactRef.new(
        task_id="t-arb", kind="annotation_result", path=rel_path,
        content_type="application/json",
    )
    store.append_artifact(artifact)

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: _RecorderClient(qc_passed=True, annotation=annotation),
    )
    # Drive _terminal_from_arbiter through the closed-branch path.
    arb_outcome = {
        "ran": True, "closed": 1, "fixed": 0, "unresolved": 0,
        "mechanical_fail": 0, "corrected_annotation": None,
    }
    runtime._terminal_from_arbiter(
        store.load_task("t-arb"),
        attempt_id="t-arb-attempt-1", stage="arbitration", arb=arb_outcome,
    )

    svc = EntityStatisticsService(store)
    # 12 from seed + 1 from the arbiter-driven acceptance.
    assert svc.distribution(project_id=project, span="Acme") == {"organization": 13}


def test_arbiter_correction_records_divergent_payload(tmp_path):
    """When the arbiter writes a corrected_annotation whose final (span, type)
    still diverges from prior, the post-check marks the task metadata so
    the next task (invoke second arbiter) can pick it up."""
    store = SqliteStore.open(tmp_path)
    project = "p"
    _seed_prior(store, project_id=project, span="Apple",
                type_to_count={"organization": 12})

    task = _make_task("t-arb-fix", input_text="Apple is here")
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: _RecorderClient(qc_passed=True, annotation={}),
    )
    corrected = {
        "rows": [{
            "row_index": 0,
            "output": {"entities": {"technology": ["Apple"]}},  # diverges from prior
        }]
    }
    arb_outcome = {
        "ran": True, "closed": 0, "fixed": 1, "unresolved": 0,
        "mechanical_fail": 0, "corrected_annotation": corrected,
    }
    result = runtime._apply_arbiter_correction(
        store.load_task("t-arb-fix"),
        attempt_id="t-arb-fix-attempt-1",
        corrected=corrected,
        arb=arb_outcome,
    )
    # First arbiter post-check should mark the divergence in task metadata.
    after = store.load_task("t-arb-fix")
    assert after.metadata.get("prior_verifier_first_arbiter_divergent") is True
    assert "prior_verifier_payload" in after.metadata


def test_second_arbiter_invoked_when_first_diverges(tmp_path):
    """When the first arbiter's accepted annotation diverges from prior,
    the runtime invokes a SECOND arbiter via the arbiter_secondary target."""
    store = SqliteStore.open(tmp_path)
    project = "p"
    _seed_prior(store, project_id=project, span="Apple",
                type_to_count={"organization": 12})

    task = _make_task("t-second", input_text="Apple is here")
    task.status = TaskStatus.ARBITRATING
    task.metadata["prior_verifier_first_arbiter_divergent"] = True
    task.metadata["prior_verifier_payload"] = {
        "span": "Apple", "proposed_type": "technology",
        "dominant_type": "organization", "dominant_count": 12,
        "total": 12, "distribution": {"organization": 12},
    }
    store.save_task(task)

    # Drop an annotation_result so _latest_annotation_artifact returns something.
    rel = "artifact_payloads/t-second/final.json"
    abs_path = store.root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(
        json.dumps({"text": json.dumps({
            "rows": [{"row_index": 0, "output": {"entities": {"technology": ["Apple"]}}}]
        })}),
        encoding="utf-8",
    )
    store.append_artifact(ArtifactRef.new(
        task_id="t-second", kind="annotation_result", path=rel,
        content_type="application/json",
    ))

    invocations: list[str] = []

    class _MultiArbiterClient:
        def __init__(self, target):
            self.target = target
            invocations.append(target)
        async def generate(self, request):
            return LLMGenerateResult(
                final_text=json.dumps({
                    "verdicts": [],
                    "corrected_annotation": {
                        "rows": [{
                            "row_index": 0,
                            "output": {"entities": {"technology": ["Apple"]}},
                        }]
                    },
                }),
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider=self.target, model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda t: _MultiArbiterClient(t),
    )

    # Method-under-test: drives the second-arbiter invocation
    runtime._resolve_first_arbiter_divergence(store.load_task("t-second"))

    # Second arbiter must have been invoked via the arbiter_secondary target.
    assert "arbiter_secondary" in invocations


def _setup_post_first_arbiter(tmp_path, second_arbiter_type):
    """Fabricate a task post first-arbiter (divergent) with the second
    arbiter stubbed to return ``second_arbiter_type`` for "Apple"."""
    store = SqliteStore.open(tmp_path)
    project = "p"
    _seed_prior(store, project_id=project, span="Apple",
                type_to_count={"organization": 12})

    task = _make_task("t", input_text="Apple is referenced here")
    task.status = TaskStatus.ARBITRATING
    task.metadata["prior_verifier_first_arbiter_divergent"] = True
    task.metadata["prior_verifier_payload"] = {
        "span": "Apple", "proposed_type": "technology",
        "dominant_type": "organization", "dominant_count": 12,
        "total": 12, "distribution": {"organization": 12},
    }
    store.save_task(task)

    rel = "artifact_payloads/t/final.json"
    abs_path = store.root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(
        json.dumps({"text": json.dumps({
            "rows": [{
                "row_index": 0,
                "output": {"entities": {"technology": ["Apple"]}},
            }]
        })}),
        encoding="utf-8",
    )
    store.append_artifact(ArtifactRef.new(
        task_id="t", kind="annotation_result", path=rel,
        content_type="application/json",
    ))

    class _Client:
        async def generate(self, request):
            return LLMGenerateResult(
                final_text=json.dumps({
                    "verdicts": [],
                    "corrected_annotation": {
                        "rows": [{
                            "row_index": 0,
                            "output": {"entities": {second_arbiter_type: ["Apple"]}},
                        }]
                    },
                }),
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="arbiter_secondary", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _Client())
    return store, runtime


def test_second_arbiter_matches_first_accepts_with_first(tmp_path):
    """Second arbiter says technology (same as first). Two LLMs from
    different families agree -> ACCEPTED with technology, overriding prior."""
    store, runtime = _setup_post_first_arbiter(tmp_path, "technology")
    runtime._resolve_first_arbiter_divergence(store.load_task("t"))
    after = store.load_task("t")
    assert after.status is TaskStatus.ACCEPTED


def test_second_arbiter_matches_prior_flips_to_prior(tmp_path):
    """Second arbiter agrees with the prior (organization). First arbiter
    was the outlier -> ACCEPTED with organization."""
    store, runtime = _setup_post_first_arbiter(tmp_path, "organization")
    runtime._resolve_first_arbiter_divergence(store.load_task("t"))
    after = store.load_task("t")
    assert after.status is TaskStatus.ACCEPTED
    # The final annotation artifact should now have Apple = organization.
    arts = [a for a in store.list_artifacts("t") if a.kind == "annotation_result"]
    latest = arts[-1]
    outer = json.loads((store.root / latest.path).read_text())
    text = outer.get("text")
    inner = json.loads(text) if isinstance(text, str) else outer
    assert inner["rows"][0]["output"]["entities"] == {"organization": ["Apple"]}


def test_second_arbiter_third_option_routes_to_hr(tmp_path):
    """Second arbiter returns a third type (project) - three-way
    disagreement -> HUMAN_REVIEW."""
    store, runtime = _setup_post_first_arbiter(tmp_path, "project")
    runtime._resolve_first_arbiter_divergence(store.load_task("t"))
    after = store.load_task("t")
    assert after.status is TaskStatus.HUMAN_REVIEW


def test_scheduler_routes_divergent_task_to_resolver(tmp_path):
    """An ARBITRATING task with prior_verifier_first_arbiter_divergent=True
    should be picked up by the scheduler claim loop and resolved via
    _resolve_first_arbiter_divergence (not via the normal rearbitrate path)."""
    from annotation_pipeline_skill.runtime.local_scheduler import LocalRuntimeScheduler
    from annotation_pipeline_skill.core.runtime import RuntimeConfig

    store = SqliteStore.open(tmp_path)
    project = "p"
    _seed_prior(store, project_id=project, span="Apple",
                type_to_count={"organization": 12})

    task = _make_task("t-sched", input_text="Apple here")
    task.status = TaskStatus.ARBITRATING
    task.metadata["prior_verifier_first_arbiter_divergent"] = True
    task.metadata["prior_verifier_payload"] = {
        "span": "Apple", "proposed_type": "technology",
        "dominant_type": "organization", "dominant_count": 12,
        "total": 12, "distribution": {"organization": 12},
    }
    store.save_task(task)

    rel = "artifact_payloads/t-sched/final.json"
    abs_path = store.root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(
        json.dumps({"text": json.dumps({
            "rows": [{
                "row_index": 0,
                "output": {"entities": {"technology": ["Apple"]}},
            }]
        })}),
        encoding="utf-8",
    )
    store.append_artifact(ArtifactRef.new(
        task_id="t-sched", kind="annotation_result", path=rel,
        content_type="application/json",
    ))

    class _Stub:
        async def generate(self, request):
            return LLMGenerateResult(
                final_text=json.dumps({
                    "verdicts": [],
                    "corrected_annotation": {
                        "rows": [{"row_index": 0,
                                  "output": {"entities": {"technology": ["Apple"]}}}]
                    },
                }),
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="arbiter_secondary", model="stub", continuity_handle=None,
            )

    sched = LocalRuntimeScheduler(
        store=store, client_factory=lambda _t: _Stub(),
        config=RuntimeConfig(max_concurrent_tasks=1),
    )

    async def run_one():
        await sched.run_forever(stop_when_idle=True, max_tasks=1)

    asyncio.run(run_one())
    after = store.load_task("t-sched")
    assert after.status is TaskStatus.ACCEPTED


# -------------------------------------------------------------------------
# Regression tests for the "silent agreement" / "second arbiter rubber-stamp"
# bugs found in production (e.g. COVID-19 → technology accepted despite an
# 83/55 event prior, because the second arbiter returned corrected_annotation
# null and the code inferred "implicit agreement with first arbiter").
# -------------------------------------------------------------------------


def _setup_post_first_arbiter_with_response(tmp_path, response_payload, *, raise_exc=None):
    """Same skeleton as `_setup_post_first_arbiter` but lets the test
    supply the raw second-arbiter response payload (or raise an exception
    instead of generating). Used to drive the silent/uncertain/unavailable
    branches the old code path papered over."""
    store = SqliteStore.open(tmp_path)
    project = "p"
    _seed_prior(store, project_id=project, span="Apple",
                type_to_count={"organization": 12})

    task = _make_task("t", input_text="Apple is referenced here")
    task.status = TaskStatus.ARBITRATING
    task.metadata["prior_verifier_first_arbiter_divergent"] = True
    task.metadata["prior_verifier_payload"] = {
        "span": "Apple", "proposed_type": "technology",
        "dominant_type": "organization", "dominant_count": 12,
        "total": 12, "distribution": {"organization": 12},
    }
    store.save_task(task)

    rel = "artifact_payloads/t/final.json"
    abs_path = store.root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(
        json.dumps({"text": json.dumps({
            "rows": [{
                "row_index": 0,
                "output": {"entities": {"technology": ["Apple"]}},
            }]
        })}),
        encoding="utf-8",
    )
    store.append_artifact(ArtifactRef.new(
        task_id="t", kind="annotation_result", path=rel,
        content_type="application/json",
    ))

    class _Client:
        async def generate(self, request):
            if raise_exc is not None:
                raise raise_exc
            return LLMGenerateResult(
                final_text=json.dumps(response_payload),
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="arbiter_secondary", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _Client())
    return store, runtime


def test_second_arbiter_silent_null_routes_to_hr(tmp_path):
    """REGRESSION: second arbiter returns corrected_annotation=null with
    empty verdicts. Old code inferred 'implicit agreement with first' →
    override prior → ACCEPTED. Production bug: COVID-19 → technology
    accepted despite 83/55 event-dominant project prior.

    New behavior: silence is not affirmation → HUMAN_REVIEW."""
    store, runtime = _setup_post_first_arbiter_with_response(
        tmp_path,
        response_payload={"verdicts": [], "corrected_annotation": None},
    )
    runtime._resolve_first_arbiter_divergence(store.load_task("t"))
    after = store.load_task("t")
    assert after.status is TaskStatus.HUMAN_REVIEW
    assert after.metadata.get("prior_verifier_action") == "second_arbiter_silent"


def test_second_arbiter_tentative_verdict_routes_to_hr(tmp_path):
    """Verdict on the synthetic prior-disagreement feedback is 'annotator'
    but confidence is 'tentative' — explicit but low confidence. Old code
    treated null corrected_annotation as agreement regardless. New behavior:
    only certain/confident verdicts count → HUMAN_REVIEW."""
    store, runtime = _setup_post_first_arbiter_with_response(
        tmp_path,
        response_payload={
            "verdicts": [{
                "feedback_id": "prior_verifier_synth",
                "verdict": "annotator",
                "confidence": "tentative",
                "reasoning": "judgment call",
            }],
            "corrected_annotation": None,
        },
    )
    runtime._resolve_first_arbiter_divergence(store.load_task("t"))
    after = store.load_task("t")
    assert after.status is TaskStatus.HUMAN_REVIEW
    assert after.metadata.get("prior_verifier_action") == "second_arbiter_silent"


def test_second_arbiter_explicit_verdict_annotator_accepts_with_first(tmp_path):
    """Verdict 'annotator' with high confidence on the synthetic feedback
    is real affirmation even without a corrected_annotation — accept with
    first arbiter, override prior. Distinguishes 'I read the dispute and
    agree' from 'I returned null because I didn't notice'."""
    store, runtime = _setup_post_first_arbiter_with_response(
        tmp_path,
        response_payload={
            "verdicts": [{
                "feedback_id": "prior_verifier_synth",
                "verdict": "annotator",
                "confidence": "certain",
                "reasoning": "context clearly supports first arbiter's call",
            }],
            "corrected_annotation": None,
        },
    )
    runtime._resolve_first_arbiter_divergence(store.load_task("t"))
    after = store.load_task("t")
    assert after.status is TaskStatus.ACCEPTED
    assert after.metadata.get("prior_verifier_action") == "resolved_to_first"


def test_second_arbiter_explicit_verdict_qc_flips_to_prior(tmp_path):
    """Verdict 'qc' with high confidence: second arbiter sides with the
    prior. Should flip first arbiter's call to prior_type."""
    store, runtime = _setup_post_first_arbiter_with_response(
        tmp_path,
        response_payload={
            "verdicts": [{
                "feedback_id": "prior_verifier_synth",
                "verdict": "qc",
                "confidence": "certain",
                "reasoning": "prior dominance is decisive in this context",
            }],
            "corrected_annotation": None,
        },
    )
    runtime._resolve_first_arbiter_divergence(store.load_task("t"))
    after = store.load_task("t")
    assert after.status is TaskStatus.ACCEPTED
    assert after.metadata.get("prior_verifier_action") == "resolved_to_prior"
    # Annotation file should have Apple = organization (the prior).
    arts = [a for a in store.list_artifacts("t") if a.kind == "annotation_result"]
    outer = json.loads((store.root / arts[-1].path).read_text())
    text = outer.get("text")
    inner = json.loads(text) if isinstance(text, str) else outer
    assert inner["rows"][0]["output"]["entities"] == {"organization": ["Apple"]}


def test_second_arbiter_unavailable_routes_to_hr_not_stuck(tmp_path):
    """REGRESSION: when the second arbiter client raises (target missing,
    network error, etc.), old code set prior_verifier_action=
    'second_arbiter_unavailable' but never called _transition — leaving the
    task stuck in ARBITRATING. New behavior: route to HUMAN_REVIEW so the
    operator can adjudicate (and the task doesn't accumulate as a zombie)."""
    store, runtime = _setup_post_first_arbiter_with_response(
        tmp_path,
        response_payload=None,
        raise_exc=RuntimeError("provider 500"),
    )
    runtime._resolve_first_arbiter_divergence(store.load_task("t"))
    after = store.load_task("t")
    assert after.status is TaskStatus.HUMAN_REVIEW, (
        "task must not stay stuck in ARBITRATING when second arbiter fails"
    )
    assert after.metadata.get("prior_verifier_action") == "second_arbiter_unavailable"


def test_second_arbiter_writes_artifact_with_target_metadata(tmp_path):
    """Second arbiter response must be persisted as an artifact tagged with
    metadata.target='arbiter_secondary' so post-hoc audits can tell first
    and second arbiter responses apart. Old code didn't persist anything,
    making it impossible to reconstruct what the second arbiter actually
    said for tasks like COVID-19 → technology."""
    store, runtime = _setup_post_first_arbiter_with_response(
        tmp_path,
        response_payload={
            "verdicts": [{
                "feedback_id": "prior_verifier_synth",
                "verdict": "annotator",
                "confidence": "certain",
                "reasoning": "agree",
            }],
            "corrected_annotation": None,
        },
    )
    runtime._resolve_first_arbiter_divergence(store.load_task("t"))
    arbiter_artifacts = [a for a in store.list_artifacts("t") if a.kind == "arbiter_result"]
    secondaries = [
        a for a in arbiter_artifacts
        if (a.metadata or {}).get("target") == "arbiter_secondary"
    ]
    assert len(secondaries) == 1, (
        f"expected exactly one arbiter_secondary artifact, "
        f"got {[a.metadata for a in arbiter_artifacts]}"
    )
    art_meta = secondaries[0].metadata or {}
    assert art_meta.get("disputed_span") == "Apple"
    assert art_meta.get("first_arbiter_type") == "technology"
    assert art_meta.get("prior_dominant_type") == "organization"

import json
import pytest

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import FeedbackSource, TaskStatus
from annotation_pipeline_skill.llm.client import LLMGenerateResult
from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime, SubagentRuntimeResult
from annotation_pipeline_skill.runtime.local_scheduler import LocalRuntimeScheduler
from annotation_pipeline_skill.core.runtime import RuntimeConfig
from annotation_pipeline_skill.services.feedback_service import build_feedback_consensus_summary
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


class StubLLMClient:
    def __init__(self, final_text='{"labels":[{"text":"alpha","type":"ENTITY"}]}', provider="test_provider"):
        self.final_text = final_text
        self.provider = provider
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)

        return LLMGenerateResult(
            runtime="test_runtime",
            provider=self.provider,
            model="test-model",
            continuity_handle="thread-1",
            final_text=self.final_text,
            usage={"total_tokens": 10},
            raw_response={"id": "test"},
            diagnostics={"queue_wait_ms": 0},
        )


def test_subagent_runtime_runs_annotation_and_qc_before_accepting(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl", "payload": {"text": "alpha"}})
    task.status = TaskStatus.PENDING
    store.save_task(task)
    annotation_client = StubLLMClient(final_text='{"labels":[{"text":"alpha","type":"ENTITY"}]}', provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true, "summary": "acceptable"}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
    )

    result = runtime.run_once(stage_target="annotation")

    loaded = store.load_task("task-1")
    attempts = store.list_attempts("task-1")
    artifacts = store.list_artifacts("task-1")
    assert isinstance(result, SubagentRuntimeResult)
    assert result.started == 1
    assert result.accepted == 1
    assert loaded.status is TaskStatus.ACCEPTED
    assert [attempt.stage for attempt in attempts] == ["annotation", "qc"]
    assert [attempt.provider_id for attempt in attempts] == ["annotator", "qc"]
    assert [artifact.kind for artifact in artifacts] == ["annotation_result", "qc_result"]
    assert artifacts[0].metadata["continuity_handle"] == "thread-1"
    assert store.list_feedback("task-1") == []
    # Annotator prompt is schema-driven now; assert it references output_schema
    # rather than the legacy annotation_guidance field.
    assert "output_schema" in annotation_client.requests[0].instructions
    assert "raw JSON" in qc_client.requests[0].instructions
    assert '"annotation_result"' in qc_client.requests[0].prompt


def test_subagent_runtime_accepts_markdown_fenced_qc_json(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl", "payload": {"text": "alpha"}})
    task.status = TaskStatus.PENDING
    store.save_task(task)
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: StubLLMClient(final_text='```json\n{"passed": true, "summary": "acceptable"}\n```'),
    )

    result = runtime.run_once(stage_target="annotation")

    loaded = store.load_task("task-1")
    assert result.accepted == 1
    assert loaded.status is TaskStatus.ACCEPTED
    assert store.list_feedback("task-1") == []


def test_subagent_runtime_records_qc_feedback_and_returns_task_to_pending(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl", "payload": {"text": "alpha"}})
    task.status = TaskStatus.PENDING
    store.save_task(task)
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: StubLLMClient(
            final_text='{"passed": false, "message": "missing entity", "category": "quality", "severity": "warning", "suggested_action": "annotator_rerun", "target": {"field": "labels"}}',
            provider="qc",
        )
        if target == "qc"
        else StubLLMClient(final_text='{"labels":[]}', provider="annotator"),
    )

    result = runtime.run_once(stage_target="annotation")

    loaded = store.load_task("task-1")
    feedback = store.list_feedback("task-1")
    assert result.started == 1
    assert result.accepted == 0
    assert result.failed == 0
    assert loaded.status is TaskStatus.PENDING
    assert feedback[0].source_stage is FeedbackSource.QC
    assert feedback[0].message == "missing entity"
    assert feedback[0].suggested_action == "annotator_rerun"
    assert store.list_artifacts("task-1")[-1].kind == "qc_result"


def test_local_scheduler_records_qc_parse_error_without_annotator_feedback_and_retries_qc(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl", "payload": {"text": "alpha"}})
    task.status = TaskStatus.PENDING
    store.save_task(task)
    annotation_client = StubLLMClient(final_text='{"labels":[]}', provider="annotator")
    qc_clients = [
        StubLLMClient(final_text="not json", provider="qc"),
        StubLLMClient(final_text='{"passed": true, "summary": "acceptable"}', provider="qc"),
    ]

    def client_factory(target):
        return qc_clients.pop(0) if target == "qc" else annotation_client

    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=client_factory,
        config=RuntimeConfig(max_concurrent_tasks=1),
    )

    # First pass: QC parser fails so the qc attempt is marked failed and the
    # task bounces back to QC for another try. Second pass: second QC client
    # returns valid JSON, task accepts.
    scheduler.run_until_idle(stage_target="annotation")

    attempts = store.list_attempts("task-1")
    assert store.load_task("task-1").status is TaskStatus.ACCEPTED
    assert store.list_feedback("task-1") == []
    assert [attempt.stage for attempt in attempts] == ["annotation", "qc", "qc"]
    assert attempts[1].status.value == "failed"
    assert attempts[1].error["kind"] == "parse_error"
    assert len(annotation_client.requests) == 1


def test_subagent_runtime_qc_prompt_includes_task_qc_sampling_policy(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="task-1",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"rows": [{"text": "alpha"}, {"text": "beta"}]}},
        metadata={
            "qc_policy": {
                "mode": "sample_count",
                "row_count": 2,
                "sample_count": 1,
                "required_correct_rows": 1,
                "sample_scope": "per_task",
            }
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    annotation_client = StubLLMClient(final_text='{"labels":[]}', provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
    )

    runtime.run_once(stage_target="annotation")

    assert "qc_policy" in qc_client.requests[0].instructions
    assert "sample_count" in qc_client.requests[0].instructions
    assert '"sample_count": 1' in qc_client.requests[0].prompt


def test_subagent_runtime_rerun_prompt_includes_feedback_context(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl", "payload": {"text": "alpha"}})
    task.status = TaskStatus.PENDING
    store.save_task(task)
    first_annotation_client = StubLLMClient(final_text='{"labels":[]}', provider="annotator")
    second_annotation_client = StubLLMClient(
        final_text='{"labels":[{"text":"alpha","type":"ENTITY"}]}',
        provider="annotator",
    )
    annotation_clients = [first_annotation_client, second_annotation_client]
    qc_clients = [
        StubLLMClient(final_text='{"passed": false, "message": "missing entity"}', provider="qc"),
        StubLLMClient(final_text='{"passed": true, "summary": "fixed"}', provider="qc"),
    ]

    def client_factory(target):
        return qc_clients.pop(0) if target == "qc" else annotation_clients.pop(0)

    runtime = SubagentRuntime(store=store, client_factory=client_factory)

    first = runtime.run_once(stage_target="annotation")
    second = runtime.run_once(stage_target="annotation")

    loaded = store.load_task("task-1")
    attempts = store.list_attempts("task-1")
    artifacts = store.list_artifacts("task-1")
    assert first.accepted == 0
    assert second.accepted == 1
    assert loaded.status is TaskStatus.ACCEPTED
    assert loaded.current_attempt == 4
    assert [attempt.stage for attempt in attempts] == ["annotation", "qc", "annotation", "qc"]
    assert [artifact.kind for artifact in artifacts] == ["annotation_result", "qc_result", "annotation_result", "qc_result"]
    rerun_prompt = second_annotation_client.requests[0].prompt
    assert "missing entity" in rerun_prompt
    assert "feedback_bundle" in rerun_prompt
    # Continuation turn (StubLLMClient returned continuity_handle="thread-1")
    # intentionally omits prior_artifacts — the server-side KV cache from
    # turn 1 already holds the full task + prior context. Only the new
    # feedback delta is sent on the continuation. See
    # _annotation_prompt(continuation_handle=...) in subagent_cycle.
    assert "prior_artifacts" not in rerun_prompt
    discussions = store.list_feedback_discussions("task-1")
    assert len(discussions) == 1
    assert discussions[0].consensus is True
    assert discussions[0].stance == "resolved"
    assert discussions[0].metadata["resolution_source"] == "subsequent_qc_pass"
    assert build_feedback_consensus_summary(store, "task-1")["open_feedback"] == []
    # The persisted text is canonical-serialized JSON via _serialize_llm_json.
    # Compare semantically, not byte-for-byte (whitespace differs).
    assert json.loads(json.loads((tmp_path / artifacts[2].path).read_text(encoding="utf-8"))["text"]) == {"labels": [{"text": "alpha", "type": "ENTITY"}]}


def test_annotator_output_failing_schema_records_blocking_feedback_and_loops(tmp_path):
    """Annotator returns JSON that fails task.output_schema -> validation feedback + PENDING."""
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource, TaskStatus
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-1",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "Acme",
                "annotation_guidance": {
                    "output_schema": {
                        "type": "object",
                        "required": ["entities"],
                        "properties": {"entities": {"type": "array"}},
                    }
                },
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    class _StubClient:
        async def generate(self, request):
            return LLMGenerateResult(
                final_text='{"wrong_field": []}',
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient())
    runtime.run_task(store.load_task("t-1"))

    task_after = store.load_task("t-1")
    assert task_after.status is TaskStatus.PENDING

    feedbacks = store.list_feedback("t-1")
    schema_fb = [f for f in feedbacks if f.source_stage is FeedbackSource.VALIDATION and f.category == "schema_invalid"]
    assert schema_fb, f"expected schema_invalid feedback, got {[f.category for f in feedbacks]}"
    assert schema_fb[0].severity is FeedbackSeverity.BLOCKING


def test_annotator_output_invalid_json_records_validation_feedback(tmp_path):
    """Annotator returns non-JSON text -> schema_invalid feedback (parse error)."""
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import FeedbackSource, TaskStatus
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-3",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "Acme",
                "annotation_guidance": {
                    "output_schema": {"type": "object", "required": ["entities"]}
                },
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    class _StubClient:
        async def generate(self, request):
            return LLMGenerateResult(
                final_text="not json at all",
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient())
    runtime.run_task(store.load_task("t-3"))

    task_after = store.load_task("t-3")
    assert task_after.status is TaskStatus.PENDING
    feedbacks = store.list_feedback("t-3")
    assert any(f.source_stage is FeedbackSource.VALIDATION and f.category == "schema_invalid" for f in feedbacks)


def test_annotator_output_without_schema_is_passed_through(tmp_path):
    """Task with no output_schema does not trigger schema_invalid gate; reaches QC."""
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-noschema",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "Acme"}},  # no annotation_guidance
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    call = {"n": 0}
    class _StubClient:
        async def generate(self, request):
            call["n"] += 1
            if call["n"] == 1:
                return LLMGenerateResult(
                    final_text='{"entities": []}',
                    raw_response={}, usage={}, diagnostics={}, runtime="stub",
                    provider="stub", model="stub", continuity_handle=None,
                )
            return LLMGenerateResult(
                final_text='{"passed": true}',
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient())
    runtime.run_task(store.load_task("t-noschema"))

    task_after = store.load_task("t-noschema")
    assert task_after.status is TaskStatus.ACCEPTED


def test_qc_rejection_escalates_after_n_rounds_uncertain_sets_flag(tmp_path):
    """After max_qc_rounds, arbiter is invoked. When arbiter returns a
    tentative/unsure verdict (genuine uncertainty), the task stays in ARBITRATING
    with arbiter_uncertain_needs_second=True for the second arbiter to resolve.
    Mechanical arbiter failures route back to PENDING."""
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import FeedbackSource, TaskStatus
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-loop",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "Acme",
                "annotation_guidance": {
                    "output_schema": {"type": "object"}  # permissive: annotator always passes
                },
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    # Stub: annotations pass schema; QC rejects every round; arbiter (when
    # invoked at max_qc_rounds) returns a "tentative" verdict so HR triggers.
    def _build_arbiter_response():
        feedbacks = store.list_feedback("t-loop")
        if not feedbacks:
            return '{"verdicts": [], "corrected_annotation": null}'
        return json.dumps({
            "verdicts": [
                {"feedback_id": feedbacks[-1].feedback_id,
                 "verdict": "neither", "confidence": "tentative",
                 "reasoning": "judgment call"},
            ],
            "corrected_annotation": None,
        })

    class _StubClient:
        async def generate(self, request):
            instructions = request.instructions
            if "senior arbiter" in instructions.lower():
                final = _build_arbiter_response()
            elif "qc subagent" in instructions.lower():
                final = '{"passed": false, "message": "still bad", "failures": [{"category": "x", "message": "still bad", "confidence": "certain"}]}'
            else:
                final = '{"entities": []}'
            return LLMGenerateResult(
                final_text=final, raw_response={}, usage={}, diagnostics={},
                runtime="stub", provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient(), max_qc_rounds=3)

    for _ in range(3):
        runtime.run_once()

    task_after = store.load_task("t-loop")
    assert task_after.status is TaskStatus.ARBITRATING, (
        f"expected ARBITRATING+flag after uncertain arbiter; got {task_after.status}"
    )
    assert task_after.metadata.get("arbiter_uncertain_needs_second") is True

    qc_feedbacks = [f for f in store.list_feedback("t-loop") if f.source_stage is FeedbackSource.QC]
    assert len(qc_feedbacks) == 3


def test_qc_rejection_loops_normally_under_threshold(tmp_path):
    """1 or 2 QC rejections still go back to PENDING."""
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-loop2",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "x", "annotation_guidance": {"output_schema": {"type": "object"}}}},
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    class _StubClient:
        async def generate(self, request):
            instructions = request.instructions
            if "qc subagent" in instructions.lower():
                final = '{"passed": false, "message": "bad", "failures": [{"category": "x", "message": "bad"}]}'
            else:
                final = '{"entities": []}'
            return LLMGenerateResult(
                final_text=final, raw_response={}, usage={}, diagnostics={},
                runtime="stub", provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient(), max_qc_rounds=3)
    runtime.run_once()
    runtime.run_once()

    task_after = store.load_task("t-loop2")
    assert task_after.status is TaskStatus.PENDING


def test_subagent_runtime_defaults_max_qc_rounds_to_3(tmp_path):
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    store = SqliteStore.open(tmp_path)
    runtime = SubagentRuntime(store=store, client_factory=lambda _t: None)
    assert runtime.max_qc_rounds == 3


def test_validation_failures_count_toward_escalation_threshold(tmp_path):
    """Validation failures (schema invalid) count toward max_qc_rounds. After
    that threshold, arbiter runs; if it returns a tentative verdict, the task
    gets the second-arbiter flag (ARBITRATING) rather than going straight to HR."""
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-stuck",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "x",
                "annotation_guidance": {
                    "output_schema": {"type": "object", "required": ["entities"]}
                },
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    # Annotator always produces JSON that fails schema. Arbiter responds with
    # a tentative verdict so the post-threshold path lands in HR.
    def _build_arbiter_response():
        feedbacks = store.list_feedback("t-stuck")
        if not feedbacks:
            return '{"verdicts": [], "corrected_annotation": null}'
        return json.dumps({
            "verdicts": [
                {"feedback_id": feedbacks[-1].feedback_id,
                 "verdict": "neither", "confidence": "tentative",
                 "reasoning": "schema ambiguous"},
            ],
            "corrected_annotation": None,
        })

    class _StubClient:
        async def generate(self, request):
            if "senior arbiter" in request.instructions.lower():
                final = _build_arbiter_response()
            else:
                final = '{"wrong_field": []}'
            return LLMGenerateResult(
                final_text=final,
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient(), max_qc_rounds=3)
    # 3 rounds, each ends in validation failure -> 3rd triggers arbiter -> flag
    for _ in range(3):
        runtime.run_once()

    task_after = store.load_task("t-stuck")
    assert task_after.status is TaskStatus.ARBITRATING, (
        f"expected ARBITRATING+flag after uncertain arbiter; got {task_after.status}"
    )
    assert task_after.metadata.get("arbiter_uncertain_needs_second") is True


def test_mixed_qc_and_validation_failures_escalate_together(tmp_path):
    """QC + validation failures both count toward max_qc_rounds. After the
    threshold, arbiter is invoked; tentative verdicts set the second-arbiter flag."""
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-mixed",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "x",
                "annotation_guidance": {
                    "output_schema": {"type": "object", "required": ["entities"]}
                },
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    state = {"annotation_round": 0}

    def _build_arbiter_response():
        feedbacks = store.list_feedback("t-mixed")
        if not feedbacks:
            return '{"verdicts": [], "corrected_annotation": null}'
        return json.dumps({
            "verdicts": [
                {"feedback_id": feedbacks[-1].feedback_id,
                 "verdict": "neither", "confidence": "tentative",
                 "reasoning": "mixed signals"},
            ],
            "corrected_annotation": None,
        })

    class _StubClient:
        async def generate(self, request):
            instructions = request.instructions.lower()
            if "senior arbiter" in instructions:
                final = _build_arbiter_response()
            elif "qc subagent" in instructions:
                final = '{"passed": false, "message": "bad", "failures": [{"category": "x", "message": "bad", "confidence": "certain"}]}'
            else:
                state["annotation_round"] += 1
                if state["annotation_round"] == 1:
                    # First annotation: schema-valid, will reach QC which rejects.
                    final = '{"entities": []}'
                else:
                    # Subsequent annotations: schema-invalid (missing "entities").
                    final = '{"wrong_field": []}'
            return LLMGenerateResult(
                final_text=final, raw_response={}, usage={}, diagnostics={},
                runtime="stub", provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient(), max_qc_rounds=3)
    # Run enough cycles to exhaust the 3-round budget
    for _ in range(5):
        runtime.run_once()
        t = store.load_task("t-mixed")
        if t.status is TaskStatus.ARBITRATING and t.metadata.get("arbiter_uncertain_needs_second"):
            break

    task_after = store.load_task("t-mixed")
    assert task_after.status is TaskStatus.ARBITRATING, (
        f"expected ARBITRATING+flag after uncertain arbiter; got {task_after.status}"
    )
    assert task_after.metadata.get("arbiter_uncertain_needs_second") is True


def _seed_prelabeled_task(store, *, task_id, annotation_text, output_schema=None):
    """Create a PENDING task with a prelabeled annotation_result artifact already on disk."""
    from annotation_pipeline_skill.core.models import ArtifactRef, Attempt
    from annotation_pipeline_skill.core.states import AttemptStatus

    payload = {
        "text": "alpha",
    }
    if output_schema is not None:
        payload["annotation_guidance"] = {"output_schema": output_schema}
    task = Task.new(
        task_id=task_id,
        pipeline_id="pipe",
        source_ref={"kind": "jsonl_prelabeled", "payload": payload},
        metadata={"prelabeled": True},
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    artifact_path = f"artifact_payloads/{task_id}/prelabeled-annotation.json"
    full_path = store.root / artifact_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "text": annotation_text,
                "raw_response": {"source": "v2_prelabel"},
                "usage": {},
                "diagnostics": {"source": "prelabel"},
            },
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    artifact = ArtifactRef.new(
        task_id=task_id,
        kind="annotation_result",
        path=artifact_path,
        content_type="application/json",
        metadata={"runtime": "import", "provider": "prelabel"},
    )
    store.append_artifact(artifact)
    attempt = Attempt(
        attempt_id=f"attempt-prelabel-{task_id}",
        task_id=task_id,
        index=0,
        stage="annotation",
        status=AttemptStatus.SUCCEEDED,
        provider_id="prelabel",
        model="v2_baseline",
        summary="imported from v2 annotation",
    )
    store.append_attempt(attempt)
    return task


def test_prelabeled_task_skips_annotation_and_runs_qc(tmp_path):
    store = SqliteStore.open(tmp_path)
    _seed_prelabeled_task(
        store,
        task_id="pre-1",
        annotation_text='{"labels":[{"text":"alpha","type":"ENTITY"}]}',
    )

    call_log = []

    class TrackingClient:
        def __init__(self, target, final_text):
            self.target = target
            self.final_text = final_text

        async def generate(self, request):
            call_log.append(self.target)
            return LLMGenerateResult(
                runtime="test_runtime",
                provider=self.target,
                model="test-model",
                continuity_handle="handle",
                final_text=self.final_text,
                usage={},
                raw_response={},
                diagnostics={},
            )

    def client_factory(target):
        if target == "qc":
            return TrackingClient("qc", '{"passed": true, "summary": "acceptable"}')
        return TrackingClient("annotation", "SHOULD NOT BE CALLED")

    runtime = SubagentRuntime(store=store, client_factory=client_factory)
    runtime.run_once(stage_target="annotation")

    loaded = store.load_task("pre-1")
    assert loaded.status is TaskStatus.ACCEPTED
    # Annotation LLM never called
    assert "annotation" not in call_log
    assert call_log == ["qc"]
    # Stages recorded: existing prelabel attempt + new qc attempt only
    attempts = store.list_attempts("pre-1")
    stages = [a.stage for a in attempts]
    assert stages.count("annotation") == 1  # only the seeded prelabel attempt
    assert "qc" in stages
    artifacts = store.list_artifacts("pre-1")
    assert any(a.kind == "annotation_result" for a in artifacts)
    assert any(a.kind == "qc_result" for a in artifacts)


def test_prelabeled_task_falls_through_to_normal_annotation_after_qc_failure(tmp_path):
    store = SqliteStore.open(tmp_path)
    _seed_prelabeled_task(
        store,
        task_id="pre-2",
        annotation_text='{"labels":[]}',
    )

    annotation_calls = {"count": 0}

    def client_factory(target):
        if target == "qc":
            return StubLLMClient(
                final_text='{"passed": false, "message": "rejected", "category": "quality", "severity": "warning", "suggested_action": "annotator_rerun"}',
                provider="qc",
            )
        annotation_calls["count"] += 1
        return StubLLMClient(
            final_text='{"labels":[{"text":"alpha","type":"ENTITY"}]}',
            provider="annotator",
        )

    runtime = SubagentRuntime(store=store, client_factory=client_factory)

    # First run: prelabeled path, QC fails -> PENDING
    runtime.run_once(stage_target="annotation")
    loaded = store.load_task("pre-2")
    assert loaded.status is TaskStatus.PENDING
    assert loaded.current_attempt >= 1
    assert annotation_calls["count"] == 0  # prelabeled path skipped annotation LLM

    # Second run: now current_attempt > 0, prelabeled branch must NOT fire.
    # Annotation LLM must be invoked. Use a QC stub that rejects again to keep test simple.
    runtime.run_once(stage_target="annotation")
    assert annotation_calls["count"] == 1
    attempts = store.list_attempts("pre-2")
    annotation_stages = [a for a in attempts if a.stage == "annotation"]
    # Seeded prelabel + new annotation attempt
    assert len(annotation_stages) >= 2
    assert any(a.provider_id == "annotator" for a in annotation_stages)


def test_prelabeled_task_fails_schema_validation_on_existing_artifact(tmp_path):
    store = SqliteStore.open(tmp_path)
    _seed_prelabeled_task(
        store,
        task_id="pre-3",
        annotation_text='{"wrong_field": []}',
        output_schema={
            "type": "object",
            "required": ["labels"],
            "properties": {"labels": {"type": "array"}},
        },
    )

    qc_called = {"count": 0}

    class _StubClient:
        def __init__(self, target):
            self.target = target

        async def generate(self, request):
            if self.target == "qc":
                qc_called["count"] += 1
            return LLMGenerateResult(
                runtime="stub",
                provider=self.target,
                model="m",
                continuity_handle=None,
                final_text='{"passed": true}',
                usage={},
                raw_response={},
                diagnostics={},
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda t: _StubClient(t))
    runtime.run_once(stage_target="annotation")

    loaded = store.load_task("pre-3")
    assert loaded.status is TaskStatus.PENDING
    # Schema validation gate blocked QC entirely
    assert qc_called["count"] == 0
    feedbacks = store.list_feedback("pre-3")
    assert any(f.category == "schema_invalid" for f in feedbacks), f"got {[f.category for f in feedbacks]}"


def test_prior_artifacts_capped_per_kind(tmp_path):
    """Verify _artifact_context returns at most N artifacts per kind to bound prompt growth."""
    from annotation_pipeline_skill.core.models import ArtifactRef

    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-cap", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    store.save_task(task)

    # Write 5 annotation_result + 5 qc_result artifacts (10 total).
    for i in range(5):
        for kind in ("annotation_result", "qc_result"):
            rel_path = f"artifact_payloads/task-cap/attempt-{i}-{kind}.json"
            full_path = store.root / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(
                json.dumps({"kind": kind, "index": i}, sort_keys=True),
                encoding="utf-8",
            )
            store.append_artifact(
                ArtifactRef.new(
                    task_id="task-cap",
                    kind=kind,
                    path=rel_path,
                    content_type="application/json",
                    metadata={"index": i},
                )
            )

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: StubLLMClient(),
    )
    artifacts = runtime._artifact_context("task-cap", per_kind_limit=3)

    # Expect exactly 6: the last 3 of each kind.
    assert len(artifacts) == 6
    by_kind: dict[str, list] = {}
    for entry in artifacts:
        by_kind.setdefault(entry["kind"], []).append(entry)
    assert len(by_kind["annotation_result"]) == 3
    assert len(by_kind["qc_result"]) == 3
    # Verify the most-recent indexes are retained (2, 3, 4 — last 3).
    assert [a["payload"]["index"] for a in by_kind["annotation_result"]] == [2, 3, 4]
    assert [a["payload"]["index"] for a in by_kind["qc_result"]] == [2, 3, 4]


def test_annotation_prompt_includes_resolved_schema_from_project(tmp_path):
    """When a task lacks an inline schema, the annotator prompt must inline the project schema."""
    project_schema = {
        "type": "object",
        "required": ["labels"],
        "properties": {"labels": {"type": "array"}},
    }
    store = SqliteStore.open(tmp_path)
    (store.root / "output_schema.json").write_text(
        json.dumps(project_schema), encoding="utf-8"
    )
    # Task carries NO inline output_schema -- it should be picked up from the project file.
    task = Task.new(
        task_id="t-proj",
        pipeline_id="pipe",
        source_ref={
            "kind": "jsonl",
            "payload": {"text": "alpha", "annotation_guidance": {"rules_path": "annotation_rules.yaml"}},
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    annotation_client = StubLLMClient(final_text='{"labels": []}', provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
    )

    runtime.run_once(stage_target="annotation")

    assert annotation_client.requests, "annotator was not invoked"
    # output_schema lives in the SYSTEM prompt (instructions), not user JSON,
    # so the cacheable prefix stays bytestable across tasks. Verify the
    # serialized schema is embedded in instructions.
    annotator_instructions = annotation_client.requests[0].instructions or ""
    assert json.dumps(project_schema, sort_keys=True) in annotator_instructions
    # QC currently keeps schema in user JSON (codex_5.4_mini path doesn't
    # benefit from local prefix-cache); preserve that contract for now.
    qc_prompt_obj = json.loads(qc_client.requests[0].prompt)
    assert qc_prompt_obj.get("output_schema") is None or qc_prompt_obj["output_schema"] == project_schema


def test_annotation_prompt_uses_inline_schema_when_present(tmp_path):
    """Inline schema in source_ref must take precedence over the project file."""
    project_schema = {"type": "object", "required": ["from_project"]}
    inline_schema = {
        "type": "object",
        "required": ["labels"],
        "properties": {"labels": {"type": "array"}},
    }
    store = SqliteStore.open(tmp_path)
    (store.root / "output_schema.json").write_text(
        json.dumps(project_schema), encoding="utf-8"
    )
    task = Task.new(
        task_id="t-inline",
        pipeline_id="pipe",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "alpha",
                "annotation_guidance": {"output_schema": inline_schema},
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    annotation_client = StubLLMClient(final_text='{"labels": []}', provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
    )

    runtime.run_once(stage_target="annotation")

    # Schema-in-system contract — see preceding test's comment block.
    annotator_instructions = annotation_client.requests[0].instructions or ""
    assert json.dumps(inline_schema, sort_keys=True) in annotator_instructions


def test_next_attempt_id_uses_max_existing_index_not_current_attempt(tmp_path):
    """Regression for the import-overwrite scenario: handle_import_jsonl_prelabeled
    used to UPSERT a task with current_attempt=0 while leaving stale child rows
    in `attempts`. _next_attempt_id then formatted f"...-attempt-1", colliding
    with the leftover attempt-1 and tripping the UNIQUE constraint.

    Derive the next id from MAX(attempts.idx)+1 so we are robust to a
    desynchronized current_attempt counter.
    """
    from annotation_pipeline_skill.core.models import Attempt
    from annotation_pipeline_skill.core.states import AttemptStatus

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="task-resync",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "x"}},
    )
    # Simulate post-UPSERT state: parent says current_attempt=0, but
    # stale attempts at indices 1 and 2 remain in the children table.
    task.current_attempt = 0
    task.status = TaskStatus.PENDING
    store.save_task(task)
    store.append_attempt(
        Attempt(attempt_id="task-resync-attempt-1", task_id="task-resync",
                index=1, stage="annotation", status=AttemptStatus.SUCCEEDED)
    )
    store.append_attempt(
        Attempt(attempt_id="task-resync-attempt-2", task_id="task-resync",
                index=2, stage="annotation", status=AttemptStatus.SUCCEEDED)
    )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: StubLLMClient())

    next_id = runtime._next_attempt_id(task)
    assert next_id == "task-resync-attempt-3", (
        f"expected attempt-3 (MAX(idx)+1), got {next_id!r}; "
        "_next_attempt_id must not trust task.current_attempt after an import reset"
    )


def test_runtime_uses_project_qc_policy_when_task_lacks_one(tmp_path):
    """When a task carries no ``metadata.qc_policy``, the QC prompt must be built
    from the project-level RuntimeConfig.qc_sample_* fields."""
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="task-1",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"rows": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}},
        metadata={},  # NO qc_policy on the task itself
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    annotation_client = StubLLMClient(final_text='{"labels":[]}', provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
        config=RuntimeConfig(qc_sample_mode="sample_count", qc_sample_count=3, qc_sample_ratio=1.0),
    )

    runtime.run_once(stage_target="annotation")

    instructions = qc_client.requests[0].instructions
    assert "qc_policy" in instructions
    assert "sample_count" in instructions
    assert '"sample_count": 3' in instructions
    assert '"mode": "sample_count"' in instructions


def test_runtime_legacy_task_qc_policy_overrides_project_config(tmp_path):
    """A legacy task whose ``metadata.qc_policy`` is already populated must
    keep its per-task override (so the running 100-task import doesn't break)."""
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="legacy",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"rows": [{"text": "x"}]}},
        metadata={"qc_policy": {"mode": "sample_ratio", "sample_ratio": 0.25}},
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    annotation_client = StubLLMClient(final_text='{"labels":[]}', provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
        config=RuntimeConfig(qc_sample_mode="sample_count", qc_sample_count=9),
    )

    runtime.run_once(stage_target="annotation")

    instructions = qc_client.requests[0].instructions
    # Per-task legacy override wins over the project default.
    assert '"sample_ratio": 0.25' in instructions
    assert '"mode": "sample_ratio"' in instructions
    assert '"sample_count": 9' not in instructions


def test_annotator_discussion_replies_are_recorded_and_stripped_from_schema(tmp_path):
    """Annotator may push back on prior QC feedback via top-level discussion_replies.

    The replies must be persisted as FeedbackDiscussion rows (role='annotator')
    and must NOT cause schema validation to fail even when the output schema
    has additionalProperties: false at the top level.
    """
    from annotation_pipeline_skill.core.models import FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity

    store = SqliteStore.open(tmp_path)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["entities"],
        "properties": {"entities": {"type": "array"}},
    }
    task = Task.new(
        task_id="t-reply",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {"text": "Acme is a company", "annotation_guidance": {"output_schema": schema}},
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    prior = FeedbackRecord.new(
        task_id="t-reply",
        attempt_id="t-reply-attempt-prior",
        source_stage=FeedbackSource.QC,
        severity=FeedbackSeverity.WARNING,
        category="missing_entity",
        message="Missing entity 'Acme'.",
        target={},
        suggested_action="add_entity",
        created_by="qc-agent",
    )
    store.append_feedback(prior)

    annotation_payload = {
        "entities": [],
        "discussion_replies": [
            {
                "feedback_id": prior.feedback_id,
                "stance": "disagree",
                "message": "Acme is part of a different document; not in this row.",
                "disputed_points": ["entity scope"],
                "proposed_resolution": "Keep current annotation.",
            }
        ],
    }
    annotation_client = StubLLMClient(final_text=json.dumps(annotation_payload), provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true, "summary": "ok"}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
    )

    runtime.run_once(stage_target="annotation")

    discussions = store.list_feedback_discussions("t-reply")
    annotator_replies = [d for d in discussions if d.role == "annotator"]
    assert len(annotator_replies) == 1
    reply = annotator_replies[0]
    assert reply.feedback_id == prior.feedback_id
    assert reply.stance == "disagree"
    assert reply.disputed_points == ["entity scope"]
    assert reply.consensus is False
    assert reply.created_by == "annotator-agent"
    # Task reached ACCEPTED — schema validation did not trip on discussion_replies.
    assert store.load_task("t-reply").status is TaskStatus.ACCEPTED


def test_annotator_discussion_reply_for_unknown_feedback_id_is_ignored(tmp_path):
    """Bogus or stale feedback_ids in discussion_replies must not crash and must not be persisted."""
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-bogus",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "x"}},  # no schema → annotation passes through
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    annotation_payload = {
        "entities": [],
        "discussion_replies": [
            {"feedback_id": "feedback-does-not-exist", "stance": "agree", "message": "ack"},
            {"feedback_id": "another-ghost", "stance": "disagree"},  # missing message → skipped
        ],
    }
    annotation_client = StubLLMClient(final_text=json.dumps(annotation_payload), provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
    )

    runtime.run_once(stage_target="annotation")

    assert store.list_feedback_discussions("t-bogus") == []
    assert store.load_task("t-bogus").status is TaskStatus.ACCEPTED


def test_qc_unsure_label_closes_feedback_by_consensus(tmp_path):
    """When QC labeled its complaint 'unsure', the annotator's rebuttal closes
    it by consensus without burning more rounds — QC itself admitted noise."""
    from annotation_pipeline_skill.core.models import FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-qc-unsure",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "alpha"}},
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    fb = FeedbackRecord.new(
        task_id="t-qc-unsure",
        attempt_id="t-qc-unsure-attempt-prior",
        source_stage=FeedbackSource.QC,
        severity=FeedbackSeverity.WARNING,
        category="missing_phrase",
        message="Maybe missing something?",
        target={},
        suggested_action="annotator_rerun",
        created_by="qc-agent",
        metadata={"confidence": "unsure"},
    )
    store.append_feedback(fb)

    annotation_payload = {
        "entities": [],
        "discussion_replies": [
            {"feedback_id": fb.feedback_id, "confidence": "confident", "message": "Span isn't in the text."}
        ],
    }
    annotation_client = StubLLMClient(final_text=json.dumps(annotation_payload), provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
    )

    runtime.run_once(stage_target="annotation")

    discussions = store.list_feedback_discussions("t-qc-unsure")
    sources = [d.metadata.get("resolution_source") for d in discussions if d.role == "qc"]
    assert "qc_unsure" in sources, f"expected qc_unsure consensus, got {discussions}"


def test_arbiter_rules_in_annotator_favor_avoids_hr(tmp_path):
    """When the retry loop is about to escalate to HUMAN_REVIEW, the arbiter
    is consulted. If it rules in the annotator's favor with confidence >= 0.7,
    the disputed feedback is closed and the task is accepted without HR."""
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-arb",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "alpha"}},
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    def _build_arbiter_response():
        # P2: arbiter only sees the latest round's feedback; reference it dynamically.
        feedbacks = store.list_feedback("t-arb")
        if not feedbacks:
            return '{"verdicts": [], "corrected_annotation": null}'
        return json.dumps({
            "verdicts": [
                {"feedback_id": feedbacks[-1].feedback_id, "verdict": "annotator",
                 "confidence": 0.85, "reasoning": "stretched, not verbatim"}
            ]
        })

    class _StubClient:
        async def generate(self, request):
            if "senior arbiter" in request.instructions.lower():
                final = _build_arbiter_response()
            elif "quality" in request.instructions.lower():
                final = '{"passed": false, "failures": [{"category": "missing_phrase", "confidence": 0.92, "message": "still missing"}]}'
            else:
                final = '{"entities": []}'
            return LLMGenerateResult(
                final_text=final, raw_response={}, usage={}, diagnostics={},
                runtime="stub", provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient(), max_qc_rounds=3)
    for _ in range(3):
        runtime.run_once()

    discussions = store.list_feedback_discussions("t-arb")
    arbiter_closures = [d for d in discussions if d.metadata.get("resolution_source") == "arbiter"]
    assert len(arbiter_closures) == 1, f"expected 1 arbiter closure (latest round only); got {len(arbiter_closures)}"
    assert store.load_task("t-arb").status is not TaskStatus.HUMAN_REVIEW


def test_arbiter_low_confidence_sets_second_arbiter_flag(tmp_path):
    """When the arbiter's verdict is uncertain (tentative/unsure), the task must
    stay in ARBITRATING with arbiter_uncertain_needs_second=True rather than
    going straight to HUMAN_REVIEW. The second-arbiter resolver handles HR."""
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-arb-low",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "alpha"}},
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    def _build_arbiter_response():
        feedbacks = store.list_feedback("t-arb-low")
        if not feedbacks:
            return '{"verdicts": [], "corrected_annotation": null}'
        # 0.4 confidence → "tentative" label → unresolved
        return json.dumps({
            "verdicts": [
                {"feedback_id": feedbacks[-1].feedback_id, "verdict": "annotator",
                 "confidence": 0.4, "reasoning": "unclear"}
            ]
        })

    class _StubClient:
        async def generate(self, request):
            if "senior arbiter" in request.instructions.lower():
                final = _build_arbiter_response()
            elif "quality" in request.instructions.lower():
                final = '{"passed": false, "failures": [{"category": "missing_phrase", "confidence": 0.92, "message": "still missing"}]}'
            else:
                final = '{"entities": []}'
            return LLMGenerateResult(
                final_text=final, raw_response={}, usage={}, diagnostics={},
                runtime="stub", provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient(), max_qc_rounds=3)
    for _ in range(3):
        runtime.run_once()

    after = store.load_task("t-arb-low")
    assert after.status is TaskStatus.ARBITRATING, (
        f"expected ARBITRATING (waiting for second arbiter), got {after.status}"
    )
    assert after.metadata.get("arbiter_uncertain_needs_second") is True, (
        "arbiter_uncertain_needs_second flag must be set when first arbiter is uncertain"
    )
    discussions = store.list_feedback_discussions("t-arb-low")
    arbiter_entries = [d for d in discussions if d.metadata.get("resolution_source") == "arbiter"]
    assert arbiter_entries, "arbiter must leave an audit trail even when uncertain"
    assert not any(d.consensus for d in arbiter_entries), "uncertain arbiter must not write consensus"


def test_arbiter_qc_wins_with_fix_accepts_task(tmp_path):
    """When arbiter rules 'qc' (or 'neither') with confidence >= 0.7 AND provides a
    corrected_annotation, the runtime writes the correction as the final
    annotation and ACCEPTS the task (no Rejected outcome)."""
    from annotation_pipeline_skill.core.models import FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity

    store = SqliteStore.open(tmp_path)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["entities"],
        "properties": {"entities": {"type": "array"}},
    }
    task = Task.new(
        task_id="t-arb-fix",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {"text": "alpha 42", "annotation_guidance": {"output_schema": schema}},
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    for index in range(3):
        fb = FeedbackRecord.new(
            task_id="t-arb-fix",
            attempt_id=f"t-arb-fix-attempt-{index}",
            source_stage=FeedbackSource.QC,
            severity=FeedbackSeverity.WARNING,
            category="missing_entity",
            message=f"Real defect #{index}",
            target={},
            suggested_action="annotator_rerun",
            created_by="qc-agent",
            metadata={"confidence": 0.9},
        )
        store.append_feedback(fb)
    feedback_ids = [f.feedback_id for f in store.list_feedback("t-arb-fix")]
    arbiter_response = json.dumps({
        "verdicts": [
            {"feedback_id": fid, "verdict": "qc", "confidence": 0.92, "reasoning": "defect is real"}
            for fid in feedback_ids
        ],
        "corrected_annotation": {"entities": [{"text": "42", "type": "number"}]},
    })
    annotation_payload = {
        "entities": [],
        "discussion_replies": [
            {"feedback_id": fid, "confidence": 0.6, "message": "I disagree."}
            for fid in feedback_ids
        ],
    }
    qc_response = '{"passed": false, "failures": [{"category": "missing_entity", "confidence": 0.95, "message": "still missing"}]}'
    annotation_client = StubLLMClient(final_text=json.dumps(annotation_payload), provider="annotator")
    qc_client = StubLLMClient(final_text=qc_response, provider="qc")
    arbiter_client = StubLLMClient(final_text=arbiter_response, provider="arbiter")

    def factory(target: str):
        if target == "arbiter":
            return arbiter_client
        if target == "qc":
            return qc_client
        return annotation_client

    runtime = SubagentRuntime(store=store, client_factory=factory, max_qc_rounds=3)
    runtime.run_once(stage_target="annotation")

    loaded = store.load_task("t-arb-fix")
    assert loaded.status is TaskStatus.ACCEPTED
    artifacts = [a for a in store.list_artifacts("t-arb-fix") if a.kind == "annotation_result"]
    arbiter_correction = [a for a in artifacts if a.metadata.get("source") == "arbiter_correction"]
    assert arbiter_correction, "expected an arbiter_correction artifact to be saved"


def test_confidence_normalization_uses_per_role_history(tmp_path):
    """When each role has a stable scale (QC outputs 0.85-0.99, annotator
    outputs 0.7-0.95), normalization re-maps both to [0,1] so the comparison
    isn't dominated by one role's habitual range."""
    store = SqliteStore.open(tmp_path)
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: StubLLMClient(),
    )

    # Below the min-samples threshold: raw value is returned unchanged.
    assert runtime._normalize_confidence("qc", 0.9) == 0.9

    for v in [0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.99, 0.86, 0.93, 0.96]:
        runtime._record_confidence_sample("qc", v)
    for v in [0.70, 0.75, 0.80, 0.85, 0.90, 0.78, 0.82, 0.88, 0.74, 0.92]:
        runtime._record_confidence_sample("annotator", v)

    # 0.99 is the top of QC's range → 1.0 normalized; 0.85 is the bottom → 0.0.
    assert runtime._normalize_confidence("qc", 0.99) == 1.0
    assert runtime._normalize_confidence("qc", 0.85) == 0.0
    # 0.92 is the top of annotator's range → 1.0 normalized.
    assert runtime._normalize_confidence("annotator", 0.92) == 1.0
    assert runtime._normalize_confidence("annotator", 0.70) == 0.0
    # An out-of-window value is clamped.
    assert runtime._normalize_confidence("annotator", 1.0) == 1.0
    assert runtime._normalize_confidence("annotator", 0.0) == 0.0


def test_verbatim_check_strips_hallucinated_span_at_write_time(tmp_path):
    """Non-verbatim spans are now stripped in _serialize_llm_json at artifact
    write time rather than surfaced as feedback. This prevents runaway
    annotations from generating thousands of feedback records that blow the
    context window on every subsequent round.

    Verifies:
    - The hallucinated span ('GPT-J') is absent from the written artifact.
    - No non_verbatim_span feedback record is generated (span was already gone
      by the time validation ran).
    - The verbatim span ('deployed v1 today') survives in the artifact.
    """
    from annotation_pipeline_skill.core.models import Task as _Task

    store = SqliteStore.open(tmp_path)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["rows"],
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["row_index", "output"],
                    "properties": {
                        "row_index": {"type": "integer"},
                        "output": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "entities": {
                                    "type": "object",
                                    "additionalProperties": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "json_structures": {
                                    "type": "object",
                                    "additionalProperties": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    task = _Task.new(
        task_id="t-verbatim",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "annotation_guidance": {"output_schema": schema},
                "rows": [{"row_index": 0, "input": "Acme deployed v1 today."}],
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    # 'GPT-J' is not in the input — it should be stripped at write time.
    # 'deployed v1 today' IS in the input — it should survive.
    annotation_payload = {
        "rows": [
            {
                "row_index": 0,
                "output": {
                    "entities": {"technology": ["GPT-J"]},
                    "json_structures": {"decision": ["deployed v1 today"]},
                },
            }
        ]
    }
    annotation_client = StubLLMClient(final_text=json.dumps(annotation_payload), provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
        max_qc_rounds=3,
    )

    runtime.run_once(stage_target="annotation")

    # The hallucinated span is stripped silently — no non_verbatim_span feedback.
    feedbacks = store.list_feedback("t-verbatim")
    categories = {f.category for f in feedbacks}
    assert "non_verbatim_span" not in categories, (
        f"non_verbatim_span feedback should not be generated after write-time strip; got {categories}"
    )

    # The written artifact must NOT contain 'GPT-J'.
    artifacts = store.list_artifacts("t-verbatim")
    ann_artifact = next((a for a in artifacts if a.kind == "annotation_result"), None)
    assert ann_artifact is not None, "annotation_result artifact must exist"
    artifact_text = (store.root / ann_artifact.path).read_text(encoding="utf-8")
    artifact_payload = json.loads(artifact_text)
    written_text = artifact_payload.get("text", artifact_text)
    assert "GPT-J" not in written_text, (
        "hallucinated span 'GPT-J' must be stripped from the artifact text"
    )
    # The verbatim span must still be present.
    assert "deployed v1 today" in written_text, (
        "verbatim span 'deployed v1 today' must survive in the artifact"
    )


def test_rearbitration_invokes_arbiter_without_annotator_rebuttal(tmp_path):
    """Human-dragged HR → Arbitration tasks usually have NO annotator rebuttal
    (they escalated precisely because the annotator gave up). The rearbitrate
    path must override the rebuttal gate and call the arbiter anyway — the
    arbiter judges QC's complaint directly against the latest annotation.

    Regression for: round-4 bulk rearbitrate ran on 32 HR tasks and produced
    zero arbiter calls because `_arbitrate_and_apply` short-circuited on
    `not replies_by_feedback`.
    """
    from annotation_pipeline_skill.core.models import ArtifactRef, FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-rearb",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "alpha 42"}},
    )
    task.status = TaskStatus.ARBITRATING  # human dragged the card
    store.save_task(task)
    # Seed a prior annotation artifact (the annotator's last word).
    artifact_path = "artifact_payloads/t-rearb/annotation.json"
    (store.root / artifact_path).parent.mkdir(parents=True, exist_ok=True)
    (store.root / artifact_path).write_text(
        json.dumps({"text": json.dumps({"entities": ["alpha"]})}),
        encoding="utf-8",
    )
    store.append_artifact(ArtifactRef.new(
        task_id="t-rearb",
        kind="annotation_result",
        path=artifact_path,
        content_type="application/json",
        metadata={"provider": "annotator"},
    ))
    # Single QC complaint, no annotator discussion reply at all.
    store.append_feedback(
        FeedbackRecord.new(
            task_id="t-rearb",
            attempt_id="t-rearb-attempt-1",
            source_stage=FeedbackSource.QC,
            severity=FeedbackSeverity.WARNING,
            category="missing_entity",
            message="Missing entity 'alpha'",
            target={},
            suggested_action="annotator_rerun",
            created_by="qc-agent",
            metadata={"confidence": 0.9},
        )
    )
    arbiter_response = json.dumps({
        "verdicts": [
            {"feedback_id": store.list_feedback("t-rearb")[0].feedback_id,
             "verdict": "annotator", "confidence": 0.9,
             "reasoning": "annotation actually contained alpha"}
        ],
        "corrected_annotation": None,
    })
    arbiter_client = StubLLMClient(final_text=arbiter_response, provider="arbiter")

    def factory(target: str):
        if target == "arbiter":
            return arbiter_client
        return StubLLMClient(provider=target)

    runtime = SubagentRuntime(store=store, client_factory=factory, max_qc_rounds=3)
    import asyncio
    asyncio.run(runtime.run_task_async(task, stage_target="annotation"))

    # The arbiter was actually invoked — exactly one prompt landed on the stub.
    assert len(arbiter_client.requests) == 1, (
        f"arbiter should have been invoked despite no annotator rebuttal, "
        f"got {len(arbiter_client.requests)} requests"
    )
    # Annotator-wins conf 0.9 → ACCEPTED.
    loaded = store.load_task("t-rearb")
    assert loaded.status is TaskStatus.ACCEPTED


def test_arbiter_retries_on_non_verbatim_correction_and_accepts_if_second_attempt_clean(tmp_path):
    """If the arbiter's first corrected_annotation has a non-verbatim span,
    the runtime retries the arbiter (up to arbiter_verbatim_retries) with a
    note telling it which span failed. If a later attempt produces a clean
    correction, the task ACCEPTS with the cleaned version."""
    from annotation_pipeline_skill.core.models import ArtifactRef, FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity

    store = SqliteStore.open(tmp_path)
    # Batched schema with rows wrapper — matches what the pipeline produces.
    row_schema = {
        "type": "object", "additionalProperties": False, "required": ["row_index", "output"],
        "properties": {
            "row_index": {"type": "integer"},
            "output": {"type": "object",
                       "properties": {"entities": {"type": "object"}}},
        },
    }
    schema = {
        "type": "object", "additionalProperties": False, "required": ["rows"],
        "properties": {"rows": {"type": "array", "items": row_schema}},
    }
    task = Task.new(
        task_id="t-arb-retry",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "text": "alpha is mentioned",
            "rows": [{"row_index": 0, "input": "alpha is mentioned"}],
            "annotation_guidance": {"output_schema": schema},
        }},
    )
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)
    artifact_path = "artifact_payloads/t-arb-retry/annotation.json"
    (store.root / artifact_path).parent.mkdir(parents=True, exist_ok=True)
    (store.root / artifact_path).write_text(
        json.dumps({"text": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {}}}]})}),
        encoding="utf-8",
    )
    store.append_artifact(ArtifactRef.new(
        task_id="t-arb-retry", kind="annotation_result", path=artifact_path,
        content_type="application/json",
    ))
    fb = FeedbackRecord.new(
        task_id="t-arb-retry", attempt_id="t-arb-retry-attempt-1",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_entity", message="missing alpha",
        target={}, suggested_action="annotator_rerun",
        created_by="qc-agent", metadata={"confidence": 0.9},
    )
    store.append_feedback(fb)
    fid = store.list_feedback("t-arb-retry")[0].feedback_id

    # First arbiter call returns a NON-verbatim span ("beta"); the second
    # call returns a clean span ("alpha"). Cycle through responses.
    bad_response = json.dumps({
        "verdicts": [{"feedback_id": fid, "verdict": "qc", "confidence": 0.9, "reasoning": "fix it"}],
        "corrected_annotation": {"rows": [{"row_index": 0, "output": {"entities": {"name": ["beta"]}}}]},
    })
    good_response = json.dumps({
        "verdicts": [{"feedback_id": fid, "verdict": "qc", "confidence": 0.9, "reasoning": "fix it (retry)"}],
        "corrected_annotation": {"rows": [{"row_index": 0, "output": {"entities": {"name": ["alpha"]}}}]},
    })
    responses = [bad_response, good_response]

    class CyclingArbiterClient:
        def __init__(self):
            self.requests = []
        async def generate(self, request):
            self.requests.append(request)
            text = responses[min(len(self.requests) - 1, len(responses) - 1)]
            return LLMGenerateResult(
                runtime="test", provider="arbiter", model="test-model",
                continuity_handle=None, final_text=text,
                usage={"total_tokens": 1}, raw_response={"id": "test"}, diagnostics={},
            )

    arbiter_client = CyclingArbiterClient()
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda t: arbiter_client if t == "arbiter" else StubLLMClient(provider=t),
        max_qc_rounds=3,
    )
    import asyncio
    asyncio.run(runtime.run_task_async(task, stage_target="annotation"))

    # Second arbiter call (with retry-note in instructions) produced a clean
    # correction. Task ACCEPTED.
    loaded = store.load_task("t-arb-retry")
    assert loaded.status is TaskStatus.ACCEPTED, (
        f"expected ACCEPTED after retry succeeded, got {loaded.status}"
    )
    assert len(arbiter_client.requests) == 2, (
        f"expected exactly 2 arbiter calls (initial + 1 retry); got {len(arbiter_client.requests)}"
    )
    # Retry call must include the failure-feedback note in its instructions.
    assert "PREVIOUS ATTEMPT FAILED VERBATIM CHECK" in arbiter_client.requests[1].instructions
    assert "'beta'" in arbiter_client.requests[1].instructions


def test_arbiter_retries_when_verdict_qc_but_corrected_annotation_missing(tmp_path):
    """Arbiter sometimes writes 'qc wins, here's the fix' in reasoning but
    leaves corrected_annotation = null. Without retry, every such verdict
    became unresolved → HR. The runtime now retries with a note pointing
    out the missing field. If the retry produces a usable correction, the
    task ACCEPTs."""
    from annotation_pipeline_skill.core.models import ArtifactRef, FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity

    store = SqliteStore.open(tmp_path)
    row_schema = {
        "type": "object", "additionalProperties": False, "required": ["row_index", "output"],
        "properties": {"row_index": {"type": "integer"},
                       "output": {"type": "object", "properties": {"entities": {"type": "object"}}}},
    }
    schema = {
        "type": "object", "additionalProperties": False, "required": ["rows"],
        "properties": {"rows": {"type": "array", "items": row_schema}},
    }
    task = Task.new(
        task_id="t-arb-null-fix",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "text": "alpha is mentioned",
            "rows": [{"row_index": 0, "input": "alpha is mentioned"}],
            "annotation_guidance": {"output_schema": schema},
        }},
    )
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)
    artifact_path = "artifact_payloads/t-arb-null-fix/annotation.json"
    (store.root / artifact_path).parent.mkdir(parents=True, exist_ok=True)
    (store.root / artifact_path).write_text(
        json.dumps({"text": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {}}}]})}),
        encoding="utf-8",
    )
    store.append_artifact(ArtifactRef.new(
        task_id="t-arb-null-fix", kind="annotation_result", path=artifact_path,
        content_type="application/json",
    ))
    store.append_feedback(FeedbackRecord.new(
        task_id="t-arb-null-fix", attempt_id="t-arb-null-fix-attempt-1",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_entity", message="missing alpha",
        target={}, suggested_action="annotator_rerun",
        created_by="qc-agent", metadata={"confidence": 0.9},
    ))
    fid = store.list_feedback("t-arb-null-fix")[0].feedback_id

    # First arbiter call: high-conf qc verdict but corrected_annotation = null
    bad = json.dumps({
        "verdicts": [{"feedback_id": fid, "verdict": "qc", "confidence": 0.9,
                      "reasoning": "the corrected annotation uses the verbatim sentence"}],
        "corrected_annotation": None,  # forgot to actually emit
    })
    # Second arbiter call: properly emits corrected_annotation
    good = json.dumps({
        "verdicts": [{"feedback_id": fid, "verdict": "qc", "confidence": 0.9, "reasoning": "fix included now"}],
        "corrected_annotation": {"rows": [{"row_index": 0, "output": {"entities": {"name": ["alpha"]}}}]},
    })
    responses = [bad, good]

    class CyclingArbiter:
        def __init__(self): self.requests = []
        async def generate(self, request):
            self.requests.append(request)
            return LLMGenerateResult(
                runtime="test", provider="arbiter", model="test-model",
                continuity_handle=None, final_text=responses[min(len(self.requests)-1, len(responses)-1)],
                usage={"total_tokens": 1}, raw_response={"id": "test"}, diagnostics={},
            )

    arb = CyclingArbiter()
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda t: arb if t == "arbiter" else StubLLMClient(provider=t),
        max_qc_rounds=3,
    )
    import asyncio
    asyncio.run(runtime.run_task_async(task, stage_target="annotation"))

    loaded = store.load_task("t-arb-null-fix")
    assert loaded.status is TaskStatus.ACCEPTED, (
        f"expected ACCEPTED after arbiter delivered the missing fix on retry, got {loaded.status}"
    )
    assert len(arb.requests) == 2, (
        f"expected exactly 2 arbiter calls (initial + 1 retry); got {len(arb.requests)}"
    )
    assert "PREVIOUS ATTEMPT WAS MISSING corrected_annotation" in arb.requests[1].instructions


def test_apply_arbiter_correction_rejects_non_verbatim_spans(tmp_path):
    """The arbiter can write a corrected_annotation whose entity / phrase
    strings aren't substrings of input.text (e.g., paraphrased or normalized
    Chinese characters). Schema-valid corrections must STILL pass the
    verbatim check; otherwise the corrected_annotation is discarded and the
    task routes to PENDING for a mechanical retry (was HUMAN_REVIEW; the
    new policy reserves HR strictly for arbiter tentative/unsure verdicts —
    verbatim failures are mechanical, not genuine uncertainty).

    Regression for: 5% audit on a 1882-task run found ~11% verbatim
    violations in accepted tasks where the arbiter wrote a corrected
    annotation containing hallucinated/normalized spans.
    """
    from annotation_pipeline_skill.core.models import ArtifactRef, FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity

    store = SqliteStore.open(tmp_path)
    schema = {
        "type": "object", "additionalProperties": False, "required": ["entities"],
        "properties": {"entities": {"type": "object"}},
    }
    # Input text contains "alpha" verbatim; the arbiter will (wrongly) emit
    # "beta" which is not in the input — verbatim check must catch it.
    task = Task.new(
        task_id="t-arb-verbatim",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "text": "alpha is mentioned here",
            "rows": [{"row_index": 0, "input": "alpha is mentioned here"}],
            "annotation_guidance": {"output_schema": schema},
        }},
    )
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)
    # Seed an annotation artifact so _latest_annotation_artifact has something.
    artifact_path = "artifact_payloads/t-arb-verbatim/annotation.json"
    (store.root / artifact_path).parent.mkdir(parents=True, exist_ok=True)
    (store.root / artifact_path).write_text(
        json.dumps({"text": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {}}}]})}),
        encoding="utf-8",
    )
    store.append_artifact(ArtifactRef.new(
        task_id="t-arb-verbatim", kind="annotation_result", path=artifact_path,
        content_type="application/json",
    ))
    store.append_feedback(FeedbackRecord.new(
        task_id="t-arb-verbatim", attempt_id="t-arb-verbatim-attempt-1",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_entity", message="missing beta",
        target={}, suggested_action="annotator_rerun",
        created_by="qc-agent", metadata={"confidence": 0.9},
    ))
    # Arbiter rules qc-wins with conf 0.9 and emits "beta" as the entity —
    # but "beta" is not in the input text.
    arbiter_response = json.dumps({
        "verdicts": [
            {"feedback_id": store.list_feedback("t-arb-verbatim")[0].feedback_id,
             "verdict": "qc", "confidence": 0.9, "reasoning": "missing"},
        ],
        "corrected_annotation": {"rows": [
            {"row_index": 0, "output": {"entities": {"name": ["beta"]}}},
        ]},
    })
    arbiter_client = StubLLMClient(final_text=arbiter_response, provider="arbiter")

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda t: arbiter_client if t == "arbiter" else StubLLMClient(provider=t),
        max_qc_rounds=3,
    )
    import asyncio
    asyncio.run(runtime.run_task_async(task, stage_target="annotation"))

    # Non-verbatim corrected_annotation must NOT be accepted. Under the new
    # HR-only-for-unresolved policy, mechanical failures leave the task in
    # ARBITRATING for re-pickup (let the arbiter try again on the same
    # annotation), not in PENDING.
    loaded = store.load_task("t-arb-verbatim")
    assert loaded.status is TaskStatus.ARBITRATING, (
        f"expected ARBITRATING (re-pickup), got {loaded.status}; the verbatim "
        f"guard in _apply_arbiter_correction should reject the 'beta' correction "
        f"because 'beta' is not in the input text 'alpha is mentioned here'"
    )


def test_permanent_classifier_catches_cli_auth_failure_in_stderr():
    """ProviderCallError surfaces CLI exit failures with diagnostics
    (stderr, returncode). The bug we fixed: OAuth-broken claude exits with
    'unauthorized' / '401' in stderr but the classifier only looked at
    exception name + status_code, classifying it as transient and looping
    forever instead of escalating to HR."""
    from annotation_pipeline_skill.llm.local_cli import ProviderCallError
    from annotation_pipeline_skill.runtime.subagent_cycle import (
        _is_provider_permanent_error,
    )

    auth_err = ProviderCallError(
        "local CLI provider failed",
        {"stderr": "API Error: 401 Unauthorized — invalid api key",
         "returncode": 1},
    )
    assert _is_provider_permanent_error(auth_err) is True

    fwd_err = ProviderCallError(
        "local CLI provider failed",
        {"stderr": "Authentication failed", "returncode": 1},
    )
    assert _is_provider_permanent_error(fwd_err) is True

    # 5xx in stderr should still be transient (not permanent).
    transient = ProviderCallError(
        "local CLI provider failed",
        {"stderr": "API Error: 503 Service Unavailable", "returncode": 1},
    )
    assert _is_provider_permanent_error(transient) is False

    # Generic error with no recognizable auth signal stays transient.
    unknown = ProviderCallError(
        "local CLI provider failed",
        {"stderr": "connection reset by peer", "returncode": 1},
    )
    assert _is_provider_permanent_error(unknown) is False


def test_generate_async_reraises_original_429_when_fallback_not_configured(tmp_path):
    """_generate_async must re-raise the original 429 when the 'fallback' target
    is not in llm_profiles.yaml (ProfileValidationError from client_factory).

    Regression: previously the ProfileValidationError replaced the 429, causing
    the scheduler to classify the error as unknown-transient and loop forever
    instead of surfacing the real cause.
    """
    import asyncio
    from annotation_pipeline_skill.llm.local_cli import ProviderCallError
    from annotation_pipeline_skill.llm.profiles import ProfileValidationError
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime

    rate_limit_exc = ProviderCallError(
        "local CLI provider failed",
        {"error_event": {"api_error_status": 429, "result_text": "usage limit exceeded"}},
    )

    def client_factory(target: str):
        if target == "annotation":
            raise rate_limit_exc
        raise ProfileValidationError(f"LLM target is not configured: {target}")

    store_path = tmp_path
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    runtime = SubagentRuntime(store=SqliteStore.open(store_path), client_factory=client_factory)

    from annotation_pipeline_skill.llm.client import LLMGenerateRequest
    req = LLMGenerateRequest(instructions="test", prompt="ping")

    with pytest.raises(ProviderCallError) as exc_info:
        asyncio.run(runtime._generate_async("annotation", req))

    assert exc_info.value is rate_limit_exc


# ---------------------------------------------------------------------------
# Task 1 (P3): arbiter slim prompt must include ALL source rows
# ---------------------------------------------------------------------------

def test_arbiter_slim_prompt_includes_all_input_rows_regardless_of_target(tmp_path):
    """Rows whose row_index is NOT referenced in any qc.target must still
    appear in the arbiter prompt's input.rows so the arbiter has full context.
    Previously, a feedback item with no target.row_index caused its row to be
    silently omitted, making the arbiter mark verdicts tentative."""
    import asyncio
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import (
        Task, FeedbackRecord, ArtifactRef, FeedbackDiscussionEntry,
    )
    from annotation_pipeline_skill.core.states import (
        FeedbackSeverity, FeedbackSource, TaskStatus,
    )
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    source_payload = {
        "rows": [
            {"row_id": "r0", "row_index": 0, "input": {"text": "row zero text"}},
            {"row_id": "r1", "row_index": 1, "input": {"text": "row one text"}},
            {"row_id": "r2", "row_index": 2, "input": {"text": "row two text"}},
        ]
    }
    task = Task.new(
        task_id="t-slim",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": source_payload},
    )
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)

    ann_payload = {"rows": [
        {"row_id": "r0", "row_index": 0, "output": {"entities": {}}},
        {"row_id": "r1", "row_index": 1, "output": {"entities": {}}},
        {"row_id": "r2", "row_index": 2, "output": {"entities": {}}},
    ]}
    rel = "artifact_payloads/t-slim/ann.json"
    (store.root / "artifact_payloads" / "t-slim").mkdir(parents=True, exist_ok=True)
    (store.root / rel).write_text(json.dumps({"text": json.dumps(ann_payload)}), encoding="utf-8")
    store.append_artifact(ArtifactRef.new(
        task_id="t-slim", kind="annotation_result", path=rel, content_type="application/json",
    ))

    # Feedback with NO target.row_index → this row was previously omitted.
    fb_no_row = FeedbackRecord.new(
        task_id="t-slim", attempt_id="t-slim-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="schema_invalid", message="annotation is empty",
        target={}, suggested_action="annotator_rerun", created_by="qc",
    )
    # Feedback WITH target.row_index=1.
    fb_with_row = FeedbackRecord.new(
        task_id="t-slim", attempt_id="t-slim-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="span not labelled",
        target={"row_index": 1}, suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb_no_row)
    store.append_feedback(fb_with_row)

    for fid in [fb_no_row.feedback_id, fb_with_row.feedback_id]:
        store.append_feedback_discussion(FeedbackDiscussionEntry.new(
            task_id="t-slim", feedback_id=fid, role="annotator", stance="disagree",
            message="I disagree.", consensus=False, created_by="annotator",
        ))

    captured_prompts: list[str] = []

    class _CapturingClient:
        async def generate(self, request):
            captured_prompts.append(request.prompt)
            return LLMGenerateResult(
                final_text=json.dumps({
                    "verdicts": [
                        {"feedback_id": fb_no_row.feedback_id, "verdict": "annotator",
                         "confidence": "confident", "reasoning": "ok"},
                        {"feedback_id": fb_with_row.feedback_id, "verdict": "annotator",
                         "confidence": "confident", "reasoning": "ok"},
                    ]
                }),
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="arbiter", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _CapturingClient())
    asyncio.run(runtime._arbitrate_and_apply(task, "t-slim-attempt-0", stage="qc"))

    assert captured_prompts, "arbiter must have been called"
    prompt_data = json.loads(captured_prompts[0])
    row_indices_in_prompt = {r["row_index"] for r in prompt_data["input"]["rows"]}
    assert row_indices_in_prompt == {0, 1, 2}, (
        f"all 3 rows must appear in input.rows; got {row_indices_in_prompt}"
    )


# ---------------------------------------------------------------------------
# Task 2 (P2): arbiter must only see the latest QC round's feedback
# ---------------------------------------------------------------------------

def test_arbiter_receives_only_latest_qc_round_feedback(tmp_path):
    """Stale feedback from an earlier QC round must NOT be sent to the arbiter.
    Only the most recent round's feedback (by attempt_id) should appear in the
    disputed_items the arbiter sees."""
    import asyncio
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import (
        Task, FeedbackRecord, ArtifactRef, FeedbackDiscussionEntry,
    )
    from annotation_pipeline_skill.core.states import (
        FeedbackSeverity, FeedbackSource, TaskStatus,
    )
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    source_payload = {"rows": [{"row_id": "r0", "row_index": 0, "input": {"text": "hello"}}]}
    task = Task.new(
        task_id="t-stale", pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": source_payload},
    )
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)

    rel = "artifact_payloads/t-stale/ann.json"
    (store.root / "artifact_payloads" / "t-stale").mkdir(parents=True, exist_ok=True)
    ann = {"rows": [{"row_id": "r0", "row_index": 0, "output": {"entities": {}}}]}
    (store.root / rel).write_text(json.dumps({"text": json.dumps(ann)}), encoding="utf-8")
    store.append_artifact(ArtifactRef.new(
        task_id="t-stale", kind="annotation_result", path=rel, content_type="application/json",
    ))

    # Round 1: stale feedback (old attempt_id).
    fb_stale = FeedbackRecord.new(
        task_id="t-stale", attempt_id="t-stale-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="schema_invalid", message="annotation was empty (stale)",
        target={}, suggested_action="annotator_rerun", created_by="qc",
    )
    # Round 2: current feedback (newer attempt_id).
    fb_current = FeedbackRecord.new(
        task_id="t-stale", attempt_id="t-stale-attempt-1",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="Acme should be labelled org",
        target={"row_index": 0}, suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb_stale)
    store.append_feedback(fb_current)

    for fid in [fb_stale.feedback_id, fb_current.feedback_id]:
        store.append_feedback_discussion(FeedbackDiscussionEntry.new(
            task_id="t-stale", feedback_id=fid, role="annotator", stance="disagree",
            message="addressed", consensus=False, created_by="annotator",
        ))

    seen_feedback_ids: list[list[str]] = []

    class _CapturingClient:
        async def generate(self, request):
            data = json.loads(request.prompt)
            seen_feedback_ids.append([it["feedback_id"] for it in data["disputed_items"]])
            return LLMGenerateResult(
                final_text=json.dumps({
                    "verdicts": [{"feedback_id": fb_current.feedback_id,
                                  "verdict": "annotator", "confidence": "confident",
                                  "reasoning": "correct"}]
                }),
                raw_response={}, usage={}, diagnostics={}, runtime="stub",
                provider="arbiter", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _CapturingClient())
    asyncio.run(runtime._arbitrate_and_apply(task, "t-stale-attempt-1", stage="qc"))

    assert seen_feedback_ids, "arbiter must have been called"
    ids_sent = seen_feedback_ids[0]
    assert fb_stale.feedback_id not in ids_sent, (
        "stale round-1 feedback must NOT be sent to arbiter"
    )
    assert fb_current.feedback_id in ids_sent, (
        "current round-2 feedback must be sent to arbiter"
    )


# ---------------------------------------------------------------------------
# Task 4: second arbiter for uncertain — flag setting and resolver methods
# ---------------------------------------------------------------------------

def test_arbiter_uncertain_sets_flag_instead_of_immediate_hr(tmp_path):
    """When the first arbiter is uncertain (tentative/unsure verdict), the task
    must stay in ARBITRATING with arbiter_uncertain_needs_second=True, NOT go
    straight to HUMAN_REVIEW. The second-arbiter resolver handles the HR decision."""
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-unc",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "alpha beta"}},
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)

    def _build_arbiter_response():
        feedbacks = store.list_feedback("t-unc")
        if not feedbacks:
            return '{"verdicts": [], "corrected_annotation": null}'
        return json.dumps({
            "verdicts": [
                {"feedback_id": feedbacks[-1].feedback_id,
                 "verdict": "annotator", "confidence": "tentative",
                 "reasoning": "not sure"}
            ],
            "corrected_annotation": None,
        })

    class _StubClient:
        async def generate(self, request):
            if "senior arbiter" in request.instructions.lower():
                final = _build_arbiter_response()
            elif "quality" in request.instructions.lower():
                final = '{"passed": false, "failures": [{"category": "missing_phrase", "confidence": 0.92, "message": "still missing"}]}'
            else:
                final = '{"entities": []}'
            return LLMGenerateResult(
                final_text=final, raw_response={}, usage={}, diagnostics={},
                runtime="stub", provider="stub", model="stub", continuity_handle=None,
            )

    runtime = SubagentRuntime(store=store, client_factory=lambda _t: _StubClient(), max_qc_rounds=3)
    for _ in range(3):
        runtime.run_once()

    after = store.load_task("t-unc")
    assert after.status is TaskStatus.ARBITRATING, (
        f"expected ARBITRATING (waiting for second arbiter), got {after.status}"
    )
    assert after.metadata.get("arbiter_uncertain_needs_second") is True, (
        "arbiter_uncertain_needs_second flag must be set"
    )


def _make_uncertain_task(store, task_id: str):
    """Helper: task in ARBITRATING with arbiter_uncertain_needs_second flag and
    an annotation_result artifact."""
    from annotation_pipeline_skill.core.models import ArtifactRef, Task
    from annotation_pipeline_skill.core.states import TaskStatus

    task = Task.new(
        task_id=task_id, pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_id": "r0", "row_index": 0, "input": {"text": "hello"}}]
        }},
    )
    task.status = TaskStatus.ARBITRATING
    task.metadata["arbiter_uncertain_needs_second"] = True
    store.save_task(task)

    rel = f"artifact_payloads/{task_id}/ann.json"
    (store.root / "artifact_payloads" / task_id).mkdir(parents=True, exist_ok=True)
    ann = {"rows": [{"row_id": "r0", "row_index": 0, "output": {"entities": {}}}]}
    (store.root / rel).write_text(json.dumps({"text": json.dumps(ann)}), encoding="utf-8")
    store.append_artifact(ArtifactRef.new(
        task_id=task_id, kind="annotation_result", path=rel, content_type="application/json",
    ))
    return store.load_task(task_id)


def test_resolve_uncertain_arbiter_second_confident_accepts(tmp_path):
    """When the second arbiter responds with confident verdicts (annotator wins),
    the task is ACCEPTED and the flag is cleared."""
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import FeedbackRecord, FeedbackDiscussionEntry
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource, TaskStatus
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    fb = FeedbackRecord.new(
        task_id="t-unc2", attempt_id="t-unc2-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="missing span",
        target={}, suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb)
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id="t-unc2", feedback_id=fb.feedback_id,
        role="annotator", stance="disagree", message="ok", consensus=False, created_by="annotator",
    ))
    task = _make_uncertain_task(store, "t-unc2")

    second_resp = json.dumps({
        "verdicts": [{"feedback_id": fb.feedback_id, "verdict": "annotator",
                      "confidence": "confident", "reasoning": "clear"}]
    })

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: StubLLMClient(final_text=second_resp, provider="arbiter_secondary"),
    )
    runtime._resolve_uncertain_arbiter(task)

    after = store.load_task("t-unc2")
    assert after.status is TaskStatus.ACCEPTED, f"expected ACCEPTED, got {after.status}"
    assert not after.metadata.get("arbiter_uncertain_needs_second"), "flag must be cleared"


def test_resolve_uncertain_arbiter_second_also_uncertain_goes_to_hr(tmp_path):
    """When the second arbiter is ALSO uncertain (tentative/unsure), the task
    escalates to HUMAN_REVIEW with a clear reason."""
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import FeedbackRecord, FeedbackDiscussionEntry
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource, TaskStatus
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.llm.client import LLMGenerateResult

    store = SqliteStore.open(tmp_path)
    fb = FeedbackRecord.new(
        task_id="t-unc3", attempt_id="t-unc3-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="missing span",
        target={}, suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb)
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id="t-unc3", feedback_id=fb.feedback_id,
        role="annotator", stance="disagree", message="ok", consensus=False, created_by="annotator",
    ))
    task = _make_uncertain_task(store, "t-unc3")

    second_resp = json.dumps({
        "verdicts": [{"feedback_id": fb.feedback_id, "verdict": "annotator",
                      "confidence": "tentative", "reasoning": "still unsure"}]
    })

    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda _t: StubLLMClient(final_text=second_resp, provider="arbiter_secondary"),
    )
    runtime._resolve_uncertain_arbiter(task)

    after = store.load_task("t-unc3")
    assert after.status is TaskStatus.HUMAN_REVIEW, f"expected HUMAN_REVIEW, got {after.status}"
    assert not after.metadata.get("arbiter_uncertain_needs_second"), "flag must be cleared"
    events = store.list_events("t-unc3")
    hr_event = next((e for e in events if e.next_status.value == "human_review"), None)
    assert hr_event is not None
    assert "both arbiters" in hr_event.reason.lower() or "second arbiter" in hr_event.reason.lower()


def test_resolve_uncertain_arbiter_unavailable_goes_to_hr(tmp_path):
    """When the second arbiter client raises (unavailable), the task goes to
    HUMAN_REVIEW rather than staying stuck in ARBITRATING."""
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import FeedbackRecord, FeedbackDiscussionEntry
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource, TaskStatus
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime

    store = SqliteStore.open(tmp_path)
    fb = FeedbackRecord.new(
        task_id="t-unc4", attempt_id="t-unc4-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="missing",
        target={}, suggested_action="annotator_rerun", created_by="qc",
    )
    store.append_feedback(fb)
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id="t-unc4", feedback_id=fb.feedback_id,
        role="annotator", stance="disagree", message="ok", consensus=False, created_by="annotator",
    ))
    task = _make_uncertain_task(store, "t-unc4")

    def bad_factory(target):
        raise RuntimeError("provider not configured")

    runtime = SubagentRuntime(store=store, client_factory=bad_factory)
    runtime._resolve_uncertain_arbiter(task)

    after = store.load_task("t-unc4")
    assert after.status is TaskStatus.HUMAN_REVIEW, f"expected HUMAN_REVIEW, got {after.status}"
    assert not after.metadata.get("arbiter_uncertain_needs_second")


def test_arbiter_transient_500_does_not_escalate_to_human_review(tmp_path):
    """Arbiter 5xx transient errors must NOT count toward ARBITER_MECHANICAL_RETRY_CAP.
    A task that keeps seeing 500s should stay in ARBITRATING indefinitely, not go to HR.
    Each failure must stamp next_retry_at with exponential backoff (30s × n, cap 300s)."""
    from annotation_pipeline_skill.llm.local_cli import ProviderCallError
    from annotation_pipeline_skill.core.models import ArtifactRef, Task, FeedbackRecord, FeedbackDiscussionEntry
    from annotation_pipeline_skill.core.states import FeedbackSource, FeedbackSeverity, TaskStatus
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    class Transient500Client:
        async def generate(self, request):
            raise ProviderCallError(
                "local CLI provider failed",
                {
                    "runtime": "anthropic_sdk",
                    "error_event": {"api_error_status": 500, "result_text": "Internal Server Error"},
                },
            )

    store = SqliteStore.open(tmp_path)

    # Create task in ARBITRATING state with an annotation artifact so the arbiter
    # can build its prompt (mirrors _make_uncertain_task pattern).
    task = Task.new(
        task_id="t-500", pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {
            "rows": [{"row_id": "r0", "row_index": 0, "input": "alpha"}],
        }},
    )
    task.status = TaskStatus.ARBITRATING
    store.save_task(task)

    ann_dir = store.root / "artifact_payloads" / "t-500"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_payload = {"rows": [{"row_id": "r0", "row_index": 0, "output": {"entities": {}}}]}
    ann_path = ann_dir / "ann.json"
    ann_path.write_text(json.dumps({"text": json.dumps(ann_payload)}), encoding="utf-8")
    store.append_artifact(ArtifactRef.new(
        task_id="t-500", kind="annotation_result",
        path=f"artifact_payloads/t-500/ann.json", content_type="application/json",
    ))

    fb = FeedbackRecord.new(
        task_id="t-500", attempt_id="t-500-attempt-0",
        source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
        category="missing_span", message="missing",
        target={}, suggested_action="arbiter", created_by="qc",
    )
    store.append_feedback(fb)
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id="t-500", feedback_id=fb.feedback_id,
        role="annotator", stance="disagree", message="nope", consensus=False, created_by="annotator",
    ))

    runtime = SubagentRuntime(store=store, client_factory=lambda target: Transient500Client())

    # run_task drives the arbiter directly (run_once only picks up PENDING)
    cap = SubagentRuntime.ARBITER_MECHANICAL_RETRY_CAP
    n_runs = cap + 2
    for _ in range(n_runs):
        task = store.load_task("t-500")
        runtime.run_task(task, stage_target="arbiter")

    after = store.load_task("t-500")
    assert after.status is TaskStatus.ARBITRATING, (
        f"task should stay ARBITRATING on transient 500s, got {after.status}"
    )
    # Backoff must be set (30s × bail#, cap 300s)
    assert after.next_retry_at is not None, "next_retry_at should be set after transient bail"
    bail_n = after.metadata.get("arbiter_transient_bail_count", 0)
    assert bail_n == n_runs, f"expected {n_runs} transient bails, got {bail_n}"


# ---------------------------------------------------------------------------
# Fix 2: _count_annotation_spans helper
# ---------------------------------------------------------------------------

def test_count_annotation_spans_empty():
    from annotation_pipeline_skill.runtime.subagent_cycle import _count_annotation_spans
    assert _count_annotation_spans({}) == 0
    assert _count_annotation_spans({"rows": []}) == 0


def test_count_annotation_spans_mixed():
    from annotation_pipeline_skill.runtime.subagent_cycle import _count_annotation_spans
    payload = {
        "rows": [
            {
                "row_index": 0,
                "output": {
                    "entities": {"org": ["Acme", "Corp"], "tech": ["Python"]},
                    "json_structures": {"decision": ["go live"]},
                },
            },
            {
                "row_index": 1,
                "output": {
                    "entities": {"org": []},
                    "json_structures": {},
                },
            },
        ]
    }
    # 2 org + 1 tech + 1 decision + 0 (second row) = 4
    assert _count_annotation_spans(payload) == 4


# ---------------------------------------------------------------------------
# Fix 2: high-hallucination reset (clears feedback + resets to PENDING)
# ---------------------------------------------------------------------------

def _make_schema_task(store, task_id: str, input_text: str) -> Task:
    """Helper: create and persist a PENDING task with a minimal NER schema."""
    from annotation_pipeline_skill.core.models import Task as _Task
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_index": {"type": "integer"},
                        "output": {
                            "type": "object",
                            "properties": {
                                "entities": {
                                    "type": "object",
                                    "additionalProperties": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "json_structures": {
                                    "type": "object",
                                    "additionalProperties": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                    "required": ["row_index", "output"],
                },
            },
        },
        "required": ["rows"],
    }
    task = _Task.new(
        task_id=task_id,
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "annotation_guidance": {"output_schema": schema},
                "rows": [{"row_index": 0, "input": input_text}],
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    return task


def _make_hallucinated_annotation(count: int, *, verbatim_span: str | None = None) -> str:
    """Build an annotation with `count` hallucinated spans + 1 optional verbatim span."""
    entities = {f"fake_type_{i}": [f"hallucinated_span_{i}"] for i in range(count)}
    js: dict = {}
    if verbatim_span:
        js["decision"] = [verbatim_span]
    return json.dumps({
        "rows": [{"row_index": 0, "output": {"entities": entities, "json_structures": js}}]
    })


def test_high_hallucination_resets_to_pending(tmp_path):
    """When stripped ratio >= threshold AND count >= threshold, task resets to PENDING.

    The reset wipes feedback for that attempt so the next annotation attempt
    starts with a clean context window.
    """
    store = SqliteStore.open(tmp_path)
    input_text = "Short input."
    _make_schema_task(store, "t-reset", input_text)

    # 60 hallucinated spans + 1 verbatim: ratio = 60/61 > 0.7, count = 60 >= 50
    annotation_text = _make_hallucinated_annotation(60, verbatim_span="Short")
    annotation_client = StubLLMClient(final_text=annotation_text, provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
        max_qc_rounds=3,
    )

    runtime.run_once(stage_target="annotation")

    task = store.load_task("t-reset")
    assert task.status is TaskStatus.PENDING, (
        f"task should be reset to PENDING after high-hallucination, got {task.status}"
    )
    assert task.metadata.get("high_hallucination_reset_count") == 1

    # Feedback for this attempt must have been wiped.
    feedbacks = store.list_feedback("t-reset")
    assert len(feedbacks) == 0, (
        f"feedback should be cleared after hallucination reset; got {len(feedbacks)} records"
    )


def test_high_hallucination_below_threshold_does_not_reset(tmp_path):
    """When count < threshold, no reset — normal flow proceeds."""
    store = SqliteStore.open(tmp_path)
    input_text = "Acme deployed v1 today."
    _make_schema_task(store, "t-no-reset", input_text)

    # Only 3 hallucinated spans — below HALLUCINATION_COUNT_RESET_THRESHOLD=50
    annotation_text = _make_hallucinated_annotation(3, verbatim_span="deployed v1 today")
    annotation_client = StubLLMClient(final_text=annotation_text, provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
        max_qc_rounds=3,
    )

    runtime.run_once(stage_target="annotation")

    task = store.load_task("t-no-reset")
    # Task should NOT be PENDING — it should have progressed (ANNOTATING or further)
    assert task.status is not TaskStatus.PENDING, (
        f"task with only 3 hallucinated spans should NOT be reset, got {task.status}"
    )
    assert "high_hallucination_reset_count" not in task.metadata


def test_high_hallucination_reset_cap_escalates_to_hr(tmp_path):
    """After HALLUCINATION_RESET_CAP resets, escalate to HUMAN_REVIEW."""
    store = SqliteStore.open(tmp_path)
    input_text = "Short input."
    task = _make_schema_task(store, "t-cap", input_text)

    # Pre-seed the reset counter at cap - 1 (2 previous resets already)
    task.metadata["high_hallucination_reset_count"] = SubagentRuntime.HALLUCINATION_RESET_CAP - 1
    store.save_task(task)

    annotation_text = _make_hallucinated_annotation(60, verbatim_span="Short")
    annotation_client = StubLLMClient(final_text=annotation_text, provider="annotator")
    qc_client = StubLLMClient(final_text='{"passed": true}', provider="qc")
    runtime = SubagentRuntime(
        store=store,
        client_factory=lambda target: qc_client if target == "qc" else annotation_client,
        max_qc_rounds=3,
    )

    runtime.run_once(stage_target="annotation")

    task = store.load_task("t-cap")
    assert task.status is TaskStatus.HUMAN_REVIEW, (
        f"task should escalate to HUMAN_REVIEW after {SubagentRuntime.HALLUCINATION_RESET_CAP} resets, "
        f"got {task.status}"
    )
    assert task.metadata["high_hallucination_reset_count"] == SubagentRuntime.HALLUCINATION_RESET_CAP


# ---------------------------------------------------------------------------
# Fix 4: latest_attempt_only in build_feedback_bundle
# ---------------------------------------------------------------------------

def test_latest_attempt_only_filters_to_last_attempt():
    """latest_attempt_only=True keeps only records whose attempt_id matches
    the most recently created record."""
    from annotation_pipeline_skill.core.models import FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource
    from annotation_pipeline_skill.services.feedback_service import build_feedback_bundle
    import datetime

    store = SqliteStore.open("/tmp")  # not used for I/O in this test
    # Patch list_feedback to return two attempts worth of records
    attempt_1_records = [
        FeedbackRecord.new(
            task_id="t-1",
            attempt_id="t-1-attempt-1",
            source_stage=FeedbackSource.VALIDATION,
            severity=FeedbackSeverity.ERROR,
            category="non_verbatim_span",
            message="old attempt feedback",
            target={},
            suggested_action=None,
            created_by="test",
        )
        for _ in range(5)
    ]
    attempt_2_records = [
        FeedbackRecord.new(
            task_id="t-1",
            attempt_id="t-1-attempt-2",
            source_stage=FeedbackSource.QC,
            severity=FeedbackSeverity.WARNING,
            category="schema_invalid",
            message="new attempt feedback",
            target={},
            suggested_action=None,
            created_by="test",
        )
        for _ in range(2)
    ]
    # Simulate created_at ordering: attempt_1 is older
    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    for i, r in enumerate(attempt_1_records):
        r.created_at = base + datetime.timedelta(seconds=i)
    for i, r in enumerate(attempt_2_records):
        r.created_at = base + datetime.timedelta(seconds=100 + i)

    all_records = attempt_1_records + attempt_2_records

    class _MockStore:
        def list_feedback(self, task_id):
            return all_records
        def list_feedback_discussions(self, task_id):
            return []

    bundle = build_feedback_bundle(_MockStore(), "t-1", latest_attempt_only=True)
    items = bundle["items"]
    assert len(items) == 2, f"expected 2 items (attempt-2 only), got {len(items)}"
    assert all(item["attempt_id"] == "t-1-attempt-2" for item in items)


def test_latest_attempt_only_false_returns_all():
    """latest_attempt_only=False (default) returns all attempts."""
    from annotation_pipeline_skill.core.models import FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource
    from annotation_pipeline_skill.services.feedback_service import build_feedback_bundle
    import datetime

    records = []
    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    for i in range(3):
        r = FeedbackRecord.new(
            task_id="t-2",
            attempt_id=f"t-2-attempt-{i}",
            source_stage=FeedbackSource.QC,
            severity=FeedbackSeverity.WARNING,
            category="schema_invalid",
            message=f"feedback {i}",
            target={},
            suggested_action=None,
            created_by="test",
        )
        r.created_at = base + datetime.timedelta(seconds=i)
        records.append(r)

    class _MockStore:
        def list_feedback(self, task_id):
            return records
        def list_feedback_discussions(self, task_id):
            return []

    bundle = build_feedback_bundle(_MockStore(), "t-2", latest_attempt_only=False)
    assert len(bundle["items"]) == 3


def test_latest_attempt_only_handles_attempt_10_correctly():
    """attempt_id at attempt-10 must not be confused by lexicographic ordering.

    With lexicographic sort, 'attempt-9' > 'attempt-10'. Using records[-1] after
    sorting by created_at is immune to this.
    """
    from annotation_pipeline_skill.core.models import FeedbackRecord
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource
    from annotation_pipeline_skill.services.feedback_service import build_feedback_bundle
    import datetime

    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    records = []
    for i in range(1, 11):  # attempt-1 through attempt-10
        r = FeedbackRecord.new(
            task_id="t-10",
            attempt_id=f"t-10-attempt-{i}",
            source_stage=FeedbackSource.QC,
            severity=FeedbackSeverity.WARNING,
            category="schema_invalid",
            message=f"feedback attempt-{i}",
            target={},
            suggested_action=None,
            created_by="test",
        )
        r.created_at = base + datetime.timedelta(seconds=i)
        records.append(r)

    class _MockStore:
        def list_feedback(self, task_id):
            return records
        def list_feedback_discussions(self, task_id):
            return []

    bundle = build_feedback_bundle(_MockStore(), "t-10", latest_attempt_only=True)
    items = bundle["items"]
    assert len(items) == 1
    # Must be attempt-10 (most recent by created_at), NOT attempt-9
    assert items[0]["attempt_id"] == "t-10-attempt-10"


# ---------------------------------------------------------------------------
# Fix 2 + Fix 4 integration: SqliteStore.clear_feedback_for_attempt
# ---------------------------------------------------------------------------

def test_clear_feedback_for_attempt(tmp_path):
    """clear_feedback_for_attempt deletes only records for the given attempt,
    including their discussion entries, and returns the deleted count."""
    from annotation_pipeline_skill.core.models import FeedbackRecord, FeedbackDiscussionEntry
    from annotation_pipeline_skill.core.states import FeedbackSeverity, FeedbackSource

    store = SqliteStore.open(tmp_path)

    task_id = "t-clear"
    attempt_a = "t-clear-attempt-1"
    attempt_b = "t-clear-attempt-2"

    # Append 3 records for attempt A, 2 for attempt B
    for i in range(3):
        store.append_feedback(FeedbackRecord.new(
            task_id=task_id, attempt_id=attempt_a,
            source_stage=FeedbackSource.VALIDATION, severity=FeedbackSeverity.ERROR,
            category="non_verbatim_span", message=f"a-{i}", target={},
            suggested_action="fix", created_by="test",
        ))
    for i in range(2):
        store.append_feedback(FeedbackRecord.new(
            task_id=task_id, attempt_id=attempt_b,
            source_stage=FeedbackSource.QC, severity=FeedbackSeverity.WARNING,
            category="schema_invalid", message=f"b-{i}", target={},
            suggested_action="fix", created_by="test",
        ))

    # Add a discussion entry for one of the attempt-A records
    feedback_a_ids = [r.feedback_id for r in store.list_feedback(task_id) if r.attempt_id == attempt_a]
    store.append_feedback_discussion(FeedbackDiscussionEntry.new(
        task_id=task_id, feedback_id=feedback_a_ids[0],
        role="annotator", stance="disagree",
        message="I disagree", agreed_points=[], disputed_points=[],
        proposed_resolution=None, consensus=False, created_by="annotator",
    ))

    # Clear attempt A
    deleted = store.clear_feedback_for_attempt(task_id, attempt_a)
    assert deleted == 3, f"expected 3 deleted, got {deleted}"

    # Only attempt B remains
    remaining = store.list_feedback(task_id)
    assert len(remaining) == 2
    assert all(r.attempt_id == attempt_b for r in remaining)

    # Discussion for attempt A's record must also be gone
    discussions = store.list_feedback_discussions(task_id)
    assert len(discussions) == 0, f"discussions for deleted records should be gone, got {len(discussions)}"


def test_qc_instructions_scan_exhaustively_when_feedback_is_validation_only():
    """Regression for task-000238: when all prior feedback has source_stage=='validation'
    (none from 'qc'), the QC instructions must tell the model to scan ALL rows
    exhaustively, not restrict to rows referenced by prior feedback."""
    from annotation_pipeline_skill.runtime.subagent_cycle import _build_qc_instructions
    from annotation_pipeline_skill.core.models import Task

    task = Task.new(
        task_id="t-rd-1",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"text": "hello"}},
        modality="text",
        annotation_requirements={"annotation_types": ["entity_span"]},
    )
    instructions = _build_qc_instructions(
        task,
        resolved_policy={"mode": "all", "sample_ratio": 1.0},
    )
    # Must instruct QC to treat validation-only bundles as round 1
    assert 'source_stage=="validation"' in instructions, (
        "QC instructions must distinguish validation-only feedback from QC feedback "
        "so prelabeled tasks get a real full-scan QC pass"
    )
    # Must also state the condition under which retry mode applies
    assert 'source_stage=="qc"' in instructions, (
        "QC instructions must explicitly state that retry mode requires at least one "
        "source_stage==\"qc\" item in the feedback bundle"
    )
    # ROUND-1 EXCEPTION must appear before "retry round" in the instructions
    assert instructions.index("ROUND-1 EXCEPTION") < instructions.index("retry round"), (
        "ROUND-1 EXCEPTION clause must precede the retry-mode clause"
    )
    # The exception clause must instruct exhaustive scanning
    assert "scan every row exhaustively" in instructions


def test_qc_instructions_enter_retry_mode_when_qc_feedback_exists():
    """When at least one feedback item has source_stage=='qc', QC must enter
    retry mode (STRICTLY RESTRICTED). The ROUND-1 EXCEPTION must not apply."""
    from annotation_pipeline_skill.runtime.subagent_cycle import _build_qc_instructions
    from annotation_pipeline_skill.core.models import Task

    task = Task.new(
        task_id="t-rd-2",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"text": "hello"}},
        modality="text",
        annotation_requirements={"annotation_types": ["entity_span"]},
    )
    instructions = _build_qc_instructions(
        task,
        resolved_policy={"mode": "all", "sample_ratio": 1.0},
    )
    # When QC feedback exists, the retry-mode restriction must be present
    assert "STRICTLY RESTRICTED" in instructions, (
        "Retry-mode STRICTLY RESTRICTED clause must be present in QC instructions"
    )
    # The triggering condition must reference source_stage=="qc"
    assert 'source_stage=="qc"' in instructions
    # human_review is also a real production source_stage; retry must trigger on it too
    assert 'source_stage=="human_review"' in instructions, (
        "Retry mode must also trigger on human_review feedback — "
        "a future refactor dropping the human_review branch would otherwise pass silently"
    )


def test_clear_feedback_for_attempt_nonexistent_returns_zero(tmp_path):
    """Clearing a non-existent attempt returns 0 (not an error)."""
    store = SqliteStore.open(tmp_path)
    result = store.clear_feedback_for_attempt("t-none", "t-none-attempt-99")
    assert result == 0


def test_annotation_instructions_include_baseline_preservation_rule():
    """Regression for task-000238: annotator must copy unchanged rows from
    prior_artifacts when feedback_bundle only references specific rows.
    Without this, Qwen drops rows it wasn't asked about."""
    from annotation_pipeline_skill.runtime.subagent_cycle import _annotation_instructions
    from annotation_pipeline_skill.core.models import Task

    task = Task.new(
        task_id="t-bp-1",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"text": "hello"}},
        modality="text",
        annotation_requirements={"annotation_types": ["entity_span"]},
    )
    instructions = _annotation_instructions(task)
    assert "BASELINE PRESERVATION" in instructions, (
        "Annotation instructions must include a BASELINE PRESERVATION rule so "
        "the model knows to copy unchanged rows from prior_artifacts"
    )
    assert "prior_artifacts" in instructions, (
        "BASELINE PRESERVATION rule must explicitly reference prior_artifacts "
        "so the model knows where to look for the baseline"
    )

    # Structural ordering: BASELINE PRESERVATION must sit between
    # MANDATORY ROW COVERAGE and HANDLING QC FEEDBACK
    bp_pos = instructions.index("BASELINE PRESERVATION")
    qc_pos = instructions.index("HANDLING QC FEEDBACK")
    cov_pos = instructions.index("genuinely contains no instance")
    assert cov_pos < bp_pos < qc_pos, (
        "BASELINE PRESERVATION must appear between MANDATORY ROW COVERAGE and "
        "HANDLING QC FEEDBACK paragraphs"
    )

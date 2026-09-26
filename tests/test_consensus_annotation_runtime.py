from annotation_pipeline_skill.core.runtime import AnnotationConfig
from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def test_runtime_defaults_to_single_annotation(tmp_path):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    rt = SubagentRuntime(store, client_factory=lambda t: None)
    assert rt.annotation_config.replicas == 1


def test_runtime_accepts_annotation_config(tmp_path):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2})
    rt = SubagentRuntime(store, client_factory=lambda t: None, annotation_config=cfg)
    assert rt.annotation_config.replicas == 2


import asyncio, json
from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus


class _StubClient:
    """Returns a canned annotation JSON for whichever target built it."""
    def __init__(self, payload_text): self._t = payload_text
    async def generate(self, request):
        from annotation_pipeline_skill.llm.client import LLMGenerateResult
        return LLMGenerateResult(runtime="stub", provider="stub", model="stub",
                                 continuity_handle=None, final_text=self._t,
                                 raw_response={}, usage={}, diagnostics={})


class _BadJsonStubClient:
    """Returns non-JSON text — simulates an annotator that fails to produce
    valid output so the consensus gather sees a partial failure."""
    async def generate(self, request):
        from annotation_pipeline_skill.llm.client import LLMGenerateResult
        return LLMGenerateResult(runtime="stub", provider="stub", model="stub",
                                 continuity_handle=None, final_text="not json",
                                 raw_response={}, usage={}, diagnostics={})


def _ann(person_rows):
    return json.dumps({"rows": [{"row_index": i, "output": {"entities": {"person": p}}}
                                for i, p in enumerate(person_rows)]})


def test_consensus_two_drafts_produces_final_artifact(tmp_path):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t1", pipeline_id="p",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    canned = {
        "a": _ann([["Alice", "Bob"]]),
        "b": _ann([["Alice"]]),
        "arbiter": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"person": ["Alice", "Bob"]}}}]}),
    }
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2,
                                      "arbiter_target": "arbiter"})
    rt = SubagentRuntime(store, client_factory=lambda target: _StubClient(canned[target]),
                         annotation_config=cfg)
    task = store.load_task("t1")
    asyncio.run(rt._produce_consensus_annotation(task))
    from annotation_pipeline_skill.services.entity_statistics_service import _load_latest_annotation
    final = _load_latest_annotation(store, "t1")
    persons = final["rows"][0]["output"]["entities"]["person"]
    assert set(persons) == {"Alice", "Bob"}
    assert final["rows"][0].get("row_id")  # schema requires row_id on every row
    # An annotation-stage attempt must be recorded for the dashboard timeline.
    attempts = store.list_attempts("t1")
    ann_attempts = [a for a in attempts if a.stage == "annotation"]
    assert ann_attempts, "consensus annotation should record an annotation-stage attempt"
    assert ann_attempts[-1].provider_id == "consensus"


def test_run_task_consensus_reaches_qc(tmp_path, monkeypatch):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t2", pipeline_id="p",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    canned = {"a": _ann([["Alice", "Bob"]]), "b": _ann([["Alice"]]),
              "arbiter": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"person": ["Alice", "Bob"]}}}]})}
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2, "arbiter_target": "arbiter", "accept_directly": False})  # opt into QC (multi-annotation skips it by default)
    rt = SubagentRuntime(store, client_factory=lambda target: _StubClient(canned[target]), annotation_config=cfg)

    called = {}
    async def fake_qc(task, artifact, attempt_id, text):
        called["ok"] = True
    monkeypatch.setattr(rt, "_run_validation_and_qc", fake_qc)

    asyncio.run(rt._run_task(store.load_task("t2"), "annotation"))
    assert called.get("ok") is True


def test_consensus_tolerates_one_failed_annotator(tmp_path):
    """One annotator returns invalid JSON; with keep_threshold=1 the one good
    draft's spans survive and the task still reaches a terminal state."""
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t_partial", pipeline_id="p",
                 source_ref={"kind": "jsonl",
                             "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)

    # "a" returns valid JSON with verbatim-safe spans; "b" returns garbage.
    good = _ann([["Alice", "Bob"]])

    def factory(target):
        return _StubClient(good) if target == "a" else _BadJsonStubClient()

    cfg = AnnotationConfig.from_dict({
        "replicas": 2, "targets": ["a", "b"], "keep_threshold": 1,
        "on_disagree": "drop", "accept_directly": True,
    })
    rt = SubagentRuntime(store, client_factory=factory, annotation_config=cfg)

    asyncio.run(rt._run_task(store.load_task("t_partial"), "annotation"))

    task = store.load_task("t_partial")
    assert task.status is TaskStatus.ACCEPTED
    from annotation_pipeline_skill.services.entity_statistics_service import _load_latest_annotation
    persons = _load_latest_annotation(store, "t_partial")["rows"][0]["output"]["entities"]["person"]
    assert set(persons) == {"Alice", "Bob"}


def test_consensus_aborts_when_too_few_valid_drafts(tmp_path):
    """When valid drafts fall below keep_threshold, a RuntimeError is raised so
    the scheduler's bail/escalation handles it (no silent accept)."""
    import pytest
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t_toofew", pipeline_id="p",
                 source_ref={"kind": "jsonl",
                             "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)

    def factory(target):
        return _StubClient(_ann([["Alice"]])) if target == "a" else _BadJsonStubClient()

    cfg = AnnotationConfig.from_dict({
        "replicas": 2, "targets": ["a", "b"], "keep_threshold": 2,
    })
    rt = SubagentRuntime(store, client_factory=factory, annotation_config=cfg)
    with pytest.raises(RuntimeError, match="keep_threshold"):
        asyncio.run(rt._produce_consensus_annotation(store.load_task("t_toofew")))


def test_accept_directly_routes_invalid_annotation_to_human_review(tmp_path):
    """accept_directly must still run deterministic validation. A cross-type
    collision in the arbiter's output must land the task in HUMAN_REVIEW, not
    ACCEPTED."""
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t_collide", pipeline_id="p",
                 source_ref={"kind": "jsonl",
                             "payload": {"rows": [{"row_index": 0, "input": "Apple shipped it"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    # Drafts disagree (forces the arbiter), and the arbiter tags the SAME span
    # ("Apple") as two entity types in one row -> cross-type collision.
    bad_arbiter = json.dumps({"rows": [{"row_index": 0, "output": {"entities": {
        "organization": ["Apple"], "technology": ["Apple"]}}}]})
    canned = {
        "a": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"organization": ["Apple"]}}}]}),
        "b": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"technology": ["Apple"]}}}]}),
        "arbiter": bad_arbiter,
    }
    cfg = AnnotationConfig.from_dict({
        "replicas": 2, "targets": ["a", "b"], "keep_threshold": 2,
        "arbiter_target": "arbiter", "on_disagree": "arbiter", "accept_directly": True,
    })
    rt = SubagentRuntime(store, client_factory=lambda target: _StubClient(canned[target]),
                         annotation_config=cfg)

    asyncio.run(rt._run_task(store.load_task("t_collide"), "annotation"))

    task = store.load_task("t_collide")
    assert task.status is TaskStatus.HUMAN_REVIEW
    assert task.status is not TaskStatus.ACCEPTED


def test_scheduler_threads_annotation_config(tmp_path):
    from annotation_pipeline_skill.runtime.local_scheduler import LocalRuntimeScheduler
    from annotation_pipeline_skill.core.runtime import RuntimeConfig
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2})
    sched = LocalRuntimeScheduler(store=store, client_factory=lambda t: None,
                                  config=RuntimeConfig(), annotation_config=cfg)
    assert sched.annotation_config.replicas == 2


def test_accept_directly_skips_qc_and_accepts(tmp_path, monkeypatch):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t3", pipeline_id="p",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    canned = {"a": _ann([["Alice", "Bob"]]), "b": _ann([["Alice"]]),
              "arbiter": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"person": ["Alice", "Bob"]}}}]})}
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2,
                                      "arbiter_target": "arbiter", "accept_directly": True})
    rt = SubagentRuntime(store, client_factory=lambda target: _StubClient(canned[target]), annotation_config=cfg)

    qc_called = {"n": 0}
    async def fake_qc(*a, **k): qc_called["n"] += 1
    monkeypatch.setattr(rt, "_run_validation_and_qc", fake_qc)

    asyncio.run(rt._run_task(store.load_task("t3"), "annotation"))
    assert qc_called["n"] == 0
    assert store.load_task("t3").status is TaskStatus.ACCEPTED
    from annotation_pipeline_skill.services.entity_statistics_service import _load_latest_annotation
    assert set(_load_latest_annotation(store, "t3")["rows"][0]["output"]["entities"]["person"]) == {"Alice", "Bob"}


def _v6_like_store(tmp_path, task_id):
    """A project whose output schema mirrors zh_chunks_v6: row_index + row_id +
    output required on every row, and maxItems = rows in the batch."""
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    (store.root / "output_schema.json").write_text(json.dumps({
        "type": "object", "required": ["rows"], "additionalProperties": False,
        "properties": {"rows": {"type": "array", "minItems": 1, "maxItems": 2, "items": {
            "type": "object", "required": ["row_index", "row_id", "output"],
            "properties": {"row_index": {"type": "integer"}, "row_id": {"type": "string"},
                           "output": {"type": "object", "required": ["entities", "json_structures"]}},
        }}},
    }))
    t = Task.new(task_id=task_id, pipeline_id="p", source_ref={"kind": "jsonl", "payload": {"rows": [
        {"row_index": 40, "row_id": "zh6-0022-1", "input": "Friday 和 ChatGPT"},
        {"row_index": 41, "row_id": "zh6-0023-0", "input": "阿里云 折扣"},
    ]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    return store


def test_consensus_result_rows_are_the_source_rows(tmp_path):
    """Regression for zh_chunks_v6 (-000005/-000011/-000018): MiniMax emitted its
    first row without row_index/row_id, consensus defaulted it to a phantom row
    0, and the arbiter — shown only row_index — wrote row_id = str(row_index).
    The final artifact must be exactly the source rows with their source ids,
    and pass deterministic validation (was schema_invalid / missing_rows)."""
    store = _v6_like_store(tmp_path, "t_rows")
    qwen = json.dumps({"rows": [
        {"row_index": 40, "row_id": "zh6-0022-1", "output": {"entities": {"product": ["Friday", "ChatGPT"]}, "json_structures": {}}},
        {"row_index": 41, "row_id": "zh6-0023-0", "output": {"entities": {"organization": ["阿里云"]}, "json_structures": {}}},
    ]})
    minimax = json.dumps({"rows": [
        {"output": {"entities": {"product": ["Friday"]}, "json_structures": {}}},
        {"row_index": 41, "row_id": "zh6-0023-0", "output": {"entities": {"organization": ["阿里云"]}, "json_structures": {}}},
    ]})
    arbiter = json.dumps({"rows": [
        {"row_index": 0, "row_id": "0", "output": {"entities": {}, "json_structures": {}}},
        {"row_index": 40, "row_id": "40", "output": {"entities": {"product": ["Friday", "ChatGPT"]}, "json_structures": {}}},
        {"row_index": 41, "row_id": "41", "output": {"entities": {"organization": ["阿里云"]}, "json_structures": {}}},
    ]})
    canned = {"qwen": qwen, "minimax": minimax, "arbiter": arbiter}
    cfg = AnnotationConfig.from_dict({
        "replicas": 2, "targets": ["qwen", "minimax"], "keep_threshold": 2,
        "arbiter_target": "arbiter", "on_disagree": "arbiter", "accept_directly": True,
    })
    rt = SubagentRuntime(store, client_factory=lambda target: _StubClient(canned[target]),
                         annotation_config=cfg)

    asyncio.run(rt._run_task(store.load_task("t_rows"), "annotation"))

    assert store.load_task("t_rows").status is TaskStatus.ACCEPTED
    from annotation_pipeline_skill.services.entity_statistics_service import _load_latest_annotation
    final = _load_latest_annotation(store, "t_rows")
    assert [(r["row_index"], r["row_id"]) for r in final["rows"]] == [
        (40, "zh6-0022-1"), (41, "zh6-0023-0")]
    assert final["rows"][0]["output"]["entities"]["product"] == ["ChatGPT", "Friday"]


def test_arbiter_prompt_carries_source_row_ids(tmp_path):
    store = _v6_like_store(tmp_path, "t_prompt")
    seen = {}

    class _Recording(_StubClient):
        async def generate(self, request):
            seen["prompt"] = request.prompt
            return await super().generate(request)

    drafts = {
        "a": json.dumps({"rows": [{"row_index": 40, "row_id": "zh6-0022-1", "output": {"entities": {"product": ["Friday"]}, "json_structures": {}}}]}),
        "b": json.dumps({"rows": [{"row_index": 40, "row_id": "zh6-0022-1", "output": {"entities": {}, "json_structures": {}}}]}),
    }
    arbiter = json.dumps({"rows": [{"row_index": 40, "row_id": "zh6-0022-1", "output": {"entities": {}, "json_structures": {}}}]})
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2,
                                      "arbiter_target": "arbiter", "on_disagree": "arbiter"})
    rt = SubagentRuntime(store, client_factory=lambda t: _Recording(arbiter) if t == "arbiter" else _StubClient(drafts[t]),
                         annotation_config=cfg)
    asyncio.run(rt._produce_consensus_annotation(store.load_task("t_prompt")))
    assert '"row_id": "zh6-0022-1"' in seen["prompt"]
    assert '"row_id": "zh6-0023-0"' in seen["prompt"]


def test_annotation_instructions_empty_row_example_carries_row_keys():
    """The empty-row example used to be {"output": {...}} with no row keys; MiniMax
    copied it verbatim for the first row (2 of 8 fresh v6 tasks)."""
    from annotation_pipeline_skill.runtime.subagent_cycle import _annotation_instructions
    t = Task.new(task_id="t", pipeline_id="p", source_ref={"kind": "jsonl", "payload": {"rows": []}})
    instr = _annotation_instructions(t)
    assert '{"output": {"entities": {}, "json_structures": {}}}' not in instr
    assert '"row_index": <that row\'s row_index>, "row_id": "<that row\'s row_id>"' in instr


def _substage(store, task_id):
    return store.load_task(task_id).metadata.get("runtime_substage")


def test_substage_marks_annotating_then_arbitrating_and_is_cleared_when_done(tmp_path, monkeypatch):
    """The dashboard places an in-flight multi-annotation task by runtime_substage: the annotators
    run in parallel ("annotating"), the arbiter after them ("arbitrating"); it is cleared afterwards."""
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t_sub", pipeline_id="p",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    seen: dict[str, list] = {"a": [], "b": [], "arbiter": []}
    canned = {"a": _ann([["Alice", "Bob"]]), "b": _ann([["Alice"]]),
              "arbiter": json.dumps({"rows": [{"row_index": 0, "output": {"entities": {"person": ["Alice", "Bob"]}}}]})}

    class _Observing(_StubClient):
        def __init__(self, target):
            super().__init__(canned[target])
            self._target = target

        async def generate(self, request):
            seen[self._target].append(_substage(store, "t_sub"))
            return await super().generate(request)

    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2,
                                      "arbiter_target": "arbiter", "accept_directly": True})
    rt = SubagentRuntime(store, client_factory=lambda target: _Observing(target), annotation_config=cfg)
    asyncio.run(rt._run_task(store.load_task("t_sub"), "annotation"))

    assert seen["a"] == ["annotating"] and seen["b"] == ["annotating"]
    assert seen["arbiter"] == ["arbitrating"]
    assert _substage(store, "t_sub") is None
    assert store.load_task("t_sub").status is TaskStatus.ACCEPTED


def test_substage_is_cleared_when_the_consensus_cycle_raises(tmp_path):
    """A failed cycle must not leave the task stamped "annotating"/"arbitrating" on the board."""
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id="t_fail", pipeline_id="p",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2,
                                      "accept_directly": True})
    rt = SubagentRuntime(store, client_factory=lambda target: _BadJsonStubClient(), annotation_config=cfg)
    import pytest
    with pytest.raises(RuntimeError, match="keep_threshold"):
        asyncio.run(rt._run_task(store.load_task("t_fail"), "annotation"))
    assert _substage(store, "t_fail") is None


class _TruncatingThenCompleteClient:
    """Answers with a cut-off response (the SDK client's diagnostics["truncated"]) `truncated` times, then a
    complete one — or never completes when `truncated` is None."""
    def __init__(self, complete_text, truncated):
        self._complete, self._truncated, self.calls = complete_text, truncated, 0

    async def generate(self, request):
        from annotation_pipeline_skill.llm.client import LLMGenerateResult
        self.calls += 1
        cut = self._truncated is None or self.calls <= self._truncated
        # a loop runs to the limit: the text is a fragment that the lenient parser would happily auto-close
        text = '{"rows": [{"row_index": 0, "row_id": "r0", "output": {"entities": {"person": ["Alice", "Alice", "Alice"' if cut else self._complete
        return LLMGenerateResult(runtime="stub", provider="stub", model="stub", continuity_handle=None,
                                 final_text=text, raw_response={}, usage={},
                                 diagnostics={"truncated": True, "stop_reason": "max_tokens"} if cut else {})


def _two_annotator_task(tmp_path, task_id):
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    t = Task.new(task_id=task_id, pipeline_id="p",
                 source_ref={"kind": "jsonl", "payload": {"rows": [{"row_index": 0, "row_id": "r0", "input": "Alice and Bob"}]}})
    t.status = TaskStatus.PENDING
    store.save_task(t)
    return store


def test_truncated_response_is_generated_again_and_only_the_complete_one_is_used(tmp_path):
    store = _two_annotator_task(tmp_path, "t_trunc_retry")
    good = _ann([["Alice", "Bob"]])
    flaky = _TruncatingThenCompleteClient(good, truncated=2)
    steady = _StubClient(good)
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2})
    rt = SubagentRuntime(store, client_factory=lambda target: flaky if target == "a" else steady, annotation_config=cfg)
    asyncio.run(rt._produce_consensus_annotation(store.load_task("t_trunc_retry")))
    assert flaky.calls == 3  # two cut-off responses were thrown away
    from annotation_pipeline_skill.services.entity_statistics_service import _load_latest_annotation
    assert set(_load_latest_annotation(store, "t_trunc_retry")["rows"][0]["output"]["entities"]["person"]) == {"Alice", "Bob"}


def test_response_that_always_truncates_fails_the_draft_instead_of_being_auto_closed(tmp_path):
    import pytest
    store = _two_annotator_task(tmp_path, "t_trunc_fail")
    looping = _TruncatingThenCompleteClient(None, truncated=None)
    steady = _StubClient(_ann([["Alice", "Bob"]]))
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2})
    rt = SubagentRuntime(store, client_factory=lambda target: looping if target == "a" else steady, annotation_config=cfg)
    # the truncated draft is dropped (not repaired into a one-row draft), so only one valid draft remains
    with pytest.raises(RuntimeError, match="keep_threshold"):
        asyncio.run(rt._produce_consensus_annotation(store.load_task("t_trunc_fail")))
    assert looping.calls == SubagentRuntime.TRUNCATED_RESPONSE_ATTEMPTS

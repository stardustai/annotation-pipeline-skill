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
    cfg = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 2, "arbiter_target": "arbiter"})
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

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

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

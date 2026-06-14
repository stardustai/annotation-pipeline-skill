import pytest
from annotation_pipeline_skill.core.runtime import AnnotationConfig


def test_defaults_to_single_annotation():
    c = AnnotationConfig.from_dict({})
    assert c.replicas == 1
    assert c.targets == ["annotation"]
    assert c.keep_threshold == 1
    assert c.on_disagree == "arbiter"
    assert c.arbiter_target == "arbiter"


def test_dual_explicit_targets():
    c = AnnotationConfig.from_dict({
        "replicas": 2,
        "targets": ["annotator_a", "annotator_b"],
        "keep_threshold": 2,
        "arbiter_target": "arbiter",
    })
    assert c.replicas == 2
    assert c.targets == ["annotator_a", "annotator_b"]
    assert c.keep_threshold == 2


def test_single_target_broadcast_to_n_replicas():
    c = AnnotationConfig.from_dict({"replicas": 3, "targets": ["annotation"]})
    assert c.targets == ["annotation", "annotation", "annotation"]
    assert c.replicas == 3


def test_keep_threshold_defaults_to_replicas():
    c = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"]})
    assert c.keep_threshold == 2


def test_validate_rejects_bad_threshold():
    c = AnnotationConfig.from_dict({"replicas": 2, "targets": ["a", "b"], "keep_threshold": 3})
    with pytest.raises(ValueError, match="keep_threshold"):
        c.validate()


def test_validate_rejects_target_count_mismatch():
    c = AnnotationConfig.from_dict({"replicas": 3, "targets": ["a", "b"]})
    with pytest.raises(ValueError, match="targets"):
        c.validate()


def test_validate_accepts_single():
    AnnotationConfig.from_dict({}).validate()  # no raise

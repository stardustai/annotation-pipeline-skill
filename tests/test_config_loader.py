import pytest

from annotation_pipeline_skill.config.loader import ConfigValidationError, load_project_config


def write_config(root):
    config_root = root / ".annotation-pipeline"
    config_root.mkdir()
    (config_root / "llm_profiles.yaml").write_text(
        """
profiles:
  local_codex:
    runtime: codex_cli
    model: gpt-5.4-mini
    base_url: https://api.example.com/codex
    api_key_env: CODEX_API_KEY
targets:
  annotation: local_codex
""",
        encoding="utf-8",
    )
    (config_root / "workflow.yaml").write_text(
        """
stages:
  annotation:
    target: annotation
human_review:
  required: false
""",
        encoding="utf-8",
    )
    (config_root / "annotators.yaml").write_text(
        """
annotators:
  text_annotator:
    display_name: Text Annotator
    modalities: [text]
    annotation_types: [entity_span]
    input_artifact_kinds: [raw_slice]
    output_artifact_kinds: [annotation_result]
    provider_target: annotation
    enabled: true
""",
        encoding="utf-8",
    )
    (config_root / "external_tasks.yaml").write_text(
        """
external_tasks:
  default:
    enabled: false
""",
        encoding="utf-8",
    )


def test_load_project_config_reads_yaml_files(tmp_path):
    write_config(tmp_path)

    config = load_project_config(tmp_path)

    assert config.workflow["stages"]["annotation"]["target"] == "annotation"
    assert config.annotators["text_annotator"].modalities == ["text"]
    assert config.annotators["text_annotator"].provider_target == "annotation"


def test_load_project_config_rejects_missing_provider_target(tmp_path):
    write_config(tmp_path)
    annotators_file = tmp_path / ".annotation-pipeline" / "annotators.yaml"
    annotators_file.write_text(
        """
annotators:
  text_annotator:
    display_name: Text Annotator
    modalities: [text]
    annotation_types: [entity_span]
    input_artifact_kinds: [raw_slice]
    output_artifact_kinds: [annotation_result]
    provider_target: missing_target
    enabled: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="missing_target"):
        load_project_config(tmp_path)


def test_load_runtime_config_picks_up_max_qc_rounds(tmp_path):
    from annotation_pipeline_skill.config.loader import load_runtime_config
    root = tmp_path / "proj"
    cfg_dir = root / ".annotation-pipeline"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "workflow.yaml").write_text(
        "runtime:\n  max_qc_rounds: 7\n",
        encoding="utf-8",
    )
    runtime_cfg = load_runtime_config(root)
    assert runtime_cfg.max_qc_rounds == 7


def test_load_runtime_config_picks_up_project_qc_sampling_fields(tmp_path):
    """Project-level QC sampling knobs land on RuntimeConfig.qc_sample_*."""
    from annotation_pipeline_skill.config.loader import load_runtime_config
    root = tmp_path / "proj"
    cfg_dir = root / ".annotation-pipeline"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "workflow.yaml").write_text(
        "runtime:\n"
        "  qc_sample_mode: sample_count\n"
        "  qc_sample_ratio: 0.4\n"
        "  qc_sample_count: 5\n",
        encoding="utf-8",
    )
    runtime_cfg = load_runtime_config(root)
    assert runtime_cfg.qc_sample_mode == "sample_count"
    assert runtime_cfg.qc_sample_ratio == 0.4
    assert runtime_cfg.qc_sample_count == 5


def test_runtime_config_defaults_when_yaml_omits_qc_fields(tmp_path):
    """A workflow.yaml without any qc_sample_* keys yields the documented defaults."""
    from annotation_pipeline_skill.config.loader import load_runtime_config
    root = tmp_path / "proj"
    cfg_dir = root / ".annotation-pipeline"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "workflow.yaml").write_text("runtime:\n  max_concurrent_tasks: 2\n", encoding="utf-8")
    runtime_cfg = load_runtime_config(root)
    assert runtime_cfg.qc_sample_mode == "sample_ratio"
    assert runtime_cfg.qc_sample_ratio == 1.0
    assert runtime_cfg.qc_sample_count is None

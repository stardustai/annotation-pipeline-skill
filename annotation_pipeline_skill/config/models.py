from dataclasses import dataclass, field

from annotation_pipeline_skill.core.runtime import AnnotationConfig, RuntimeConfig


@dataclass(frozen=True)
class AnnotatorConfig:
    annotator_id: str
    display_name: str
    modalities: list[str]
    annotation_types: list[str]
    input_artifact_kinds: list[str]
    output_artifact_kinds: list[str]
    provider_target: str | None = None
    external_tool_id: str | None = None
    preview_renderer_id: str | None = None
    human_review_policy_id: str | None = None
    fallback_annotator_id: str | None = None
    enabled: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectConfig:
    annotators: dict[str, AnnotatorConfig]
    external_tasks: dict
    callbacks: dict
    workflow: dict
    runtime: RuntimeConfig
    annotation: AnnotationConfig = field(default_factory=lambda: AnnotationConfig())

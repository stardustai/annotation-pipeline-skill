from __future__ import annotations

from pathlib import Path

import yaml

from annotation_pipeline_skill.config.models import (
    AnnotatorConfig,
    ProjectConfig,
)
from annotation_pipeline_skill.core.runtime import AnnotationConfig, RuntimeConfig
from annotation_pipeline_skill.llm.profiles import (
    ProfileValidationError,
    load_llm_registry,
    resolve_llm_profiles_path,
)


class ConfigValidationError(ValueError):
    pass


def load_project_config(
    project_root: Path | str,
    *,
    workspace_root: Path | str | None = None,
) -> ProjectConfig:
    config_root = Path(project_root) / ".annotation-pipeline"
    annotators_data = read_yaml(config_root / "annotators.yaml")
    external_data = read_yaml(config_root / "external_tasks.yaml")
    callbacks_data = read_yaml(config_root / "callbacks.yaml")
    workflow_data = read_yaml(config_root / "workflow.yaml")

    config = build_project_config_from_data(
        annotators_data=annotators_data,
        external_data=external_data,
        callbacks_data=callbacks_data,
        workflow_data=workflow_data,
    )
    # Default workspace_root to the parent of project_root so the workspace-global
    # llm_profiles.yaml is discovered without callers needing to pass it explicitly.
    if workspace_root is not None:
        resolved_workspace = Path(workspace_root)
    else:
        resolved_workspace = Path(project_root).parent
    validate_project_config(config, config_root, workspace_root=resolved_workspace)
    return config


def build_project_config_from_data(
    *,
    annotators_data: dict,
    external_data: dict,
    callbacks_data: dict,
    workflow_data: dict,
) -> ProjectConfig:
    return ProjectConfig(
        annotators=_load_annotators(annotators_data.get("annotators", {})),
        external_tasks=external_data.get("external_tasks", {}),
        callbacks=callbacks_data.get("callbacks", {}),
        workflow=workflow_data,
        runtime=RuntimeConfig.from_dict(workflow_data.get("runtime") or {}),
        annotation=AnnotationConfig.from_dict(
            (workflow_data.get("stages") or {}).get("annotation") or {}
        ),
    )


def load_runtime_config(project_root: Path | str) -> RuntimeConfig:
    config_root = Path(project_root) / ".annotation-pipeline"
    workflow_data = read_yaml(config_root / "workflow.yaml")
    return RuntimeConfig.from_dict(workflow_data.get("runtime") or {})


def validate_project_config(
    config: ProjectConfig,
    config_root: Path,
    llm_registry=None,
    *,
    workspace_root: Path | None = None,
) -> None:
    if llm_registry is None:
        profiles_path = resolve_llm_profiles_path(
            workspace_root=workspace_root,
            project_config_root=config_root,
        )
        if profiles_path is None:
            raise ConfigValidationError(
                f"no llm_profiles.yaml found under workspace_root={workspace_root} "
                f"or project_config_root={config_root}"
            )
        try:
            llm_registry = load_llm_registry(profiles_path)
        except (OSError, ProfileValidationError) as exc:
            raise ConfigValidationError(str(exc)) from exc
    for annotator_id, annotator in config.annotators.items():
        if annotator.provider_target and annotator.provider_target not in llm_registry.targets:
            raise ConfigValidationError(
                f"annotator {annotator_id} references missing provider target {annotator.provider_target}"
            )


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_annotators(data: dict) -> dict[str, AnnotatorConfig]:
    return {
        annotator_id: AnnotatorConfig(
            annotator_id=annotator_id,
            display_name=values.get("display_name", annotator_id),
            modalities=list(values.get("modalities", [])),
            annotation_types=list(values.get("annotation_types", [])),
            input_artifact_kinds=list(values.get("input_artifact_kinds", [])),
            output_artifact_kinds=list(values.get("output_artifact_kinds", [])),
            provider_target=values.get("provider_target"),
            external_tool_id=values.get("external_tool_id"),
            preview_renderer_id=values.get("preview_renderer_id"),
            human_review_policy_id=values.get("human_review_policy_id"),
            fallback_annotator_id=values.get("fallback_annotator_id"),
            enabled=bool(values.get("enabled", True)),
            metadata=dict(values.get("metadata", {})),
        )
        for annotator_id, values in data.items()
    }

import json

from annotation_pipeline_skill.interfaces.cli import main


def test_provider_doctor_validates_llm_profiles(tmp_path):
    project = tmp_path / "proj"
    main(["init", "--project-root", str(project)])
    # llm_profiles.yaml is workspace-global; overwrite the seeded default with a
    # custom valid registry to exercise validation against the resolver.
    profiles = tmp_path / "llm_profiles.yaml"
    profiles.write_text(
        """
profiles:
  local_codex:
    runtime: codex_cli
    model: gpt-5.4-mini
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
targets:
  annotation: local_codex
""",
        encoding="utf-8",
    )

    assert main(["provider", "doctor", "--project-root", str(project)]) == 0


def test_provider_doctor_rejects_invalid_llm_profiles(tmp_path):
    project = tmp_path / "proj"
    main(["init", "--project-root", str(project)])
    # Workspace-global file is the source of truth; clobber it with an invalid
    # registry to make sure `doctor` reports failure.
    profiles = tmp_path / "llm_profiles.yaml"
    profiles.write_text(
        """
profiles:
  broken:
    provider: openai_responses
    model: gpt-5.4-mini
targets:
  annotation: broken
""",
        encoding="utf-8",
    )

    assert main(["provider", "doctor", "--project-root", str(project)]) == 1


def test_provider_targets_exposes_ui_relevant_profile_fields(tmp_path, capsys):
    project = tmp_path / "proj"
    main(["init", "--project-root", str(project)])

    assert main(["provider", "targets", "--project-root", str(project)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["annotation"] == {
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-6",
        "profile": "local_claude",
        "runtime": "anthropic_sdk",
    }
    assert payload["qc"] == {
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-6",
        "profile": "local_claude",
        "runtime": "anthropic_sdk",
    }

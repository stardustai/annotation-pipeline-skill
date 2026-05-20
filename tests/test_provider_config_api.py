import json

from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.interfaces.cli import main
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def test_provider_config_api_returns_profiles_targets_and_local_diagnostics(tmp_path, monkeypatch):
    main(["init", "--project-root", str(tmp_path)])
    profiles = tmp_path / ".annotation-pipeline" / "llm_profiles.yaml"
    profiles.write_text(
        """
profiles:
  local_python:
    runtime: codex_cli
    model: test-model
    base_url: https://api.example.com/codex
    api_key_env: CODEX_API_KEY
  missing_api:
    runtime: claude_cli
    model: deepseek-chat
    api_key_env: DEEPSEEK_API_KEY
    base_url: https://api.deepseek.com
targets:
  annotation: local_python
  qc: missing_api
limits:
  local_cli_global_concurrency: 2
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    # No workspace-global file → falls back to project-local llm_profiles.yaml.
    api = DashboardApi(
        SqliteStore.open(tmp_path / ".annotation-pipeline"),
        workspace_root=tmp_path / "workspace",
    )

    status, _headers, body = api.handle_get("/api/providers")

    payload = json.loads(body.decode("utf-8"))
    assert status == 200
    assert payload["config_valid"] is True
    assert payload["targets"] == {"annotation": "local_python", "qc": "missing_api"}
    assert payload["limits"] == {"local_cli_global_concurrency": 2}
    assert payload["profiles"][0]["name"] == "local_python"
    assert payload["profiles"][1]["runtime"] == "claude_cli"
    assert payload["diagnostics"]["local_python"]["status"] in ("ok", "error")
    assert payload["diagnostics"]["missing_api"]["status"] == "error"
    assert payload["diagnostics"]["missing_api"]["checks"][2]["id"] == "api_key_env_present"
    assert payload["diagnostics"]["missing_api"]["checks"][2]["status"] == "error"


def test_provider_config_api_saves_structured_provider_configuration(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    api = DashboardApi(
        SqliteStore.open(tmp_path / ".annotation-pipeline"),
        workspace_root=workspace_root,
    )

    status, _headers, body = api.handle_put(
        "/api/providers",
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "local_codex",
                        "runtime": "codex_cli",
                        "model": "gpt-5.4-mini",
                        "base_url": "https://api.example.com/codex",
                        "api_key_env": "CODEX_API_KEY",
                        "reasoning_effort": "none",
                        "timeout_seconds": 900,
                    },
                    {
                        "name": "deepseek_default",
                        "runtime": "claude_cli",
                        "model": "deepseek-chat",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "base_url": "https://api.deepseek.com",
                        "timeout_seconds": 300,
                    },
                ],
                "targets": {
                    "annotation": "local_codex",
                    "qc": "deepseek_default",
                    "coordinator": "local_codex",
                },
                "limits": {"local_cli_global_concurrency": 3},
            }
        ).encode("utf-8"),
    )

    payload = json.loads(body.decode("utf-8"))
    # Save always writes to workspace-global, never to project-local.
    saved = (workspace_root / "llm_profiles.yaml").read_text(encoding="utf-8")
    assert status == 200
    assert payload["targets"]["qc"] == "deepseek_default"
    assert "runtime: codex_cli" in saved
    assert "local_cli_global_concurrency: 3" in saved


def test_provider_config_api_reads_workspace_global_when_present(tmp_path, monkeypatch):
    main(["init", "--project-root", str(tmp_path)])
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    # Workspace-global takes precedence over the project-local file written by init.
    (workspace_root / "llm_profiles.yaml").write_text(
        """
profiles:
  global_python:
    runtime: codex_cli
    model: workspace-model
    base_url: https://api.example.com/codex
    api_key_env: CODEX_API_KEY
targets:
  annotation: global_python
""",
        encoding="utf-8",
    )
    api = DashboardApi(
        SqliteStore.open(tmp_path / ".annotation-pipeline"),
        workspace_root=workspace_root,
    )

    status, _headers, body = api.handle_get("/api/providers")

    payload = json.loads(body.decode("utf-8"))
    assert status == 200
    assert payload["profiles"][0]["name"] == "global_python"
    assert payload["profiles"][0]["model"] == "workspace-model"


def test_provider_config_api_save_creates_workspace_file_when_absent(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    api = DashboardApi(
        SqliteStore.open(tmp_path / ".annotation-pipeline"),
        workspace_root=workspace_root,
    )
    assert not (workspace_root / "llm_profiles.yaml").exists()

    status, _headers, _body = api.handle_put(
        "/api/providers",
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "local_codex",
                        "runtime": "codex_cli",
                        "model": "gpt-5.4-mini",
                        "base_url": "https://api.example.com/codex",
                        "api_key_env": "CODEX_API_KEY",
                    }
                ],
                "targets": {"annotation": "local_codex"},
                "limits": {"local_cli_global_concurrency": None},
            }
        ).encode("utf-8"),
    )

    assert status == 200
    assert (workspace_root / "llm_profiles.yaml").exists()


def test_provider_config_api_persists_inline_api_key_to_yaml(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    api = DashboardApi(
        SqliteStore.open(tmp_path / ".annotation-pipeline"),
        workspace_root=workspace_root,
    )

    status, _headers, _body = api.handle_put(
        "/api/providers",
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "deepseek_inline",
                        "runtime": "claude_cli",
                        "model": "deepseek-chat",
                        "api_key": "sk-secret-abc123",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "base_url": "https://api.deepseek.com",
                    }
                ],
                "targets": {"annotation": "deepseek_inline"},
                "limits": {"local_cli_global_concurrency": None},
            }
        ).encode("utf-8"),
    )
    assert status == 200

    saved = (workspace_root / "llm_profiles.yaml").read_text(encoding="utf-8")
    assert "api_key: sk-secret-abc123" in saved


def test_provider_config_api_get_masks_api_key_as_set_flag(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "llm_profiles.yaml").write_text(
        """
profiles:
  deepseek_inline:
    runtime: claude_cli
    model: deepseek-chat
    api_key: sk-secret-abc123
    api_key_env: DEEPSEEK_API_KEY
    base_url: https://api.deepseek.com
targets:
  annotation: deepseek_inline
""",
        encoding="utf-8",
    )
    api = DashboardApi(
        SqliteStore.open(tmp_path / ".annotation-pipeline"),
        workspace_root=workspace_root,
    )

    status, _headers, body = api.handle_get("/api/providers")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    profile = payload["profiles"][0]
    assert profile["api_key_set"] is True
    # Raw key MUST NOT appear in any GET response.
    assert "api_key" not in profile
    assert "sk-secret-abc123" not in body.decode("utf-8")


def test_provider_config_api_save_preserves_existing_api_key_when_blank(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "llm_profiles.yaml").write_text(
        """
profiles:
  deepseek_inline:
    runtime: claude_cli
    model: deepseek-chat
    api_key: sk-existing-xyz
    api_key_env: DEEPSEEK_API_KEY
    base_url: https://api.deepseek.com
targets:
  annotation: deepseek_inline
""",
        encoding="utf-8",
    )
    api = DashboardApi(
        SqliteStore.open(tmp_path / ".annotation-pipeline"),
        workspace_root=workspace_root,
    )

    # PUT without api_key (UI default when user leaves field blank) and with
    # an empty-string api_key both preserve the stored secret.
    for api_key_value in (None, ""):
        profile_payload = {
            "name": "deepseek_inline",
            "runtime": "claude_cli",
            "model": "deepseek-chat",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com",
        }
        if api_key_value is not None:
            profile_payload["api_key"] = api_key_value
        status, _headers, _body = api.handle_put(
            "/api/providers",
            json.dumps(
                {
                    "profiles": [profile_payload],
                    "targets": {"annotation": "deepseek_inline"},
                    "limits": {"local_cli_global_concurrency": None},
                }
            ).encode("utf-8"),
        )
        assert status == 200, _body
        saved = (workspace_root / "llm_profiles.yaml").read_text(encoding="utf-8")
        assert "api_key: sk-existing-xyz" in saved


def test_provider_config_api_save_overwrites_api_key_when_provided(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "llm_profiles.yaml").write_text(
        """
profiles:
  deepseek_inline:
    runtime: claude_cli
    model: deepseek-chat
    api_key: sk-old-key
    api_key_env: DEEPSEEK_API_KEY
    base_url: https://api.deepseek.com
targets:
  annotation: deepseek_inline
""",
        encoding="utf-8",
    )
    api = DashboardApi(
        SqliteStore.open(tmp_path / ".annotation-pipeline"),
        workspace_root=workspace_root,
    )

    status, _headers, _body = api.handle_put(
        "/api/providers",
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "deepseek_inline",
                        "runtime": "claude_cli",
                        "model": "deepseek-chat",
                        "api_key": "sk-new-key",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "base_url": "https://api.deepseek.com",
                    }
                ],
                "targets": {"annotation": "deepseek_inline"},
                "limits": {"local_cli_global_concurrency": None},
            }
        ).encode("utf-8"),
    )
    assert status == 200

    saved = (workspace_root / "llm_profiles.yaml").read_text(encoding="utf-8")
    assert "api_key: sk-new-key" in saved
    assert "sk-old-key" not in saved

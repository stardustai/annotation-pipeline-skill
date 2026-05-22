import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import annotation_pipeline_skill.config.loader as config_loader
import annotation_pipeline_skill.interfaces.cli as cli
from annotation_pipeline_skill.interfaces.cli import main
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@contextmanager
def external_pull_server(response_payload):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.dumps(response_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/pull"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_cli_init_creates_project_layout(tmp_path):
    project = tmp_path / "proj"
    exit_code = main(["init", "--project-root", str(project)])

    assert exit_code == 0
    config_root = project / ".annotation-pipeline"
    assert not (config_root / "providers.yaml").exists()
    assert not (config_root / "stage_routes.yaml").exists()
    assert (config_root / "workflow.yaml").exists()
    assert (config_root / "annotators.yaml").exists()
    assert (config_root / "tasks").is_dir()
    assert (config_root / "exports").is_dir()
    assert (config_root / "coordination").is_dir()
    # llm_profiles.yaml is workspace-global, NOT per-project.
    assert not (config_root / "llm_profiles.yaml").exists()
    assert (tmp_path / "llm_profiles.yaml").exists()


def test_cli_init_does_not_overwrite_existing_workspace_llm_profiles(tmp_path):
    workspace_profiles = tmp_path / "llm_profiles.yaml"
    workspace_profiles.write_text("existing: content\n", encoding="utf-8")

    main(["init", "--project-root", str(tmp_path / "proj")])

    assert workspace_profiles.read_text(encoding="utf-8") == "existing: content\n"


def test_cli_init_seeds_workspace_llm_profiles_when_absent(tmp_path):
    project = tmp_path / "proj"
    main(["init", "--project-root", str(project)])

    content = (tmp_path / "llm_profiles.yaml").read_text(encoding="utf-8")
    assert "profiles:" in content
    assert "targets:" in content
    assert "local_claude" in content


def test_cli_init_accepts_explicit_workspace_flag(tmp_path):
    ws = tmp_path / "shared"
    main(
        [
            "init",
            "--project-root",
            str(tmp_path / "proj-a"),
            "--workspace",
            str(ws),
        ]
    )
    assert (ws / "llm_profiles.yaml").exists()
    # Per-project dir must NOT also contain it.
    assert not (tmp_path / "proj-a" / ".annotation-pipeline" / "llm_profiles.yaml").exists()


def test_cli_init_writes_runtime_config(tmp_path):
    main(["init", "--project-root", str(tmp_path)])

    workflow = (tmp_path / ".annotation-pipeline" / "workflow.yaml").read_text(encoding="utf-8")

    assert "runtime:" in workflow
    assert "max_concurrent_tasks: 8" in workflow


def test_cli_doctor_succeeds_after_init(tmp_path):
    main(["init", "--project-root", str(tmp_path)])

    exit_code = main(["doctor", "--project-root", str(tmp_path)])

    assert exit_code == 0


def test_cli_runtime_status_returns_snapshot_after_init(tmp_path, capsys):
    main(["init", "--project-root", str(tmp_path)])

    exit_code = main(["runtime", "status", "--project-root", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "runtime_status" in payload
    assert payload["capacity"]["max_concurrent_tasks"] == 8


def test_cli_runtime_status_does_not_load_llm_registry(tmp_path, capsys, monkeypatch):
    main(["init", "--project-root", str(tmp_path)])

    def fail_load_llm_registry(path):
        raise AssertionError("runtime status should not load llm registry")

    monkeypatch.setattr(config_loader, "load_llm_registry", fail_load_llm_registry)

    exit_code = main(["runtime", "status", "--project-root", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["capacity"]["max_concurrent_tasks"] == 8


def test_cli_runtime_context_reuses_loaded_llm_registry(tmp_path, monkeypatch):
    main(["init", "--project-root", str(tmp_path)])
    calls = []
    real_load_llm_registry = cli.load_llm_registry

    def counted_load_llm_registry(path):
        calls.append(path)
        return real_load_llm_registry(path)

    monkeypatch.setattr(cli, "load_llm_registry", counted_load_llm_registry)

    context = cli._runtime_context(tmp_path)
    cli._build_runtime_scheduler(context)

    assert len(calls) == 1


def test_cli_create_tasks_from_jsonl(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"text": "alpha"}),
                json.dumps({"text": "beta", "modality": "text", "annotation_types": ["entity_span"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "create-tasks",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "demo",
        ]
    )

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    tasks = store.list_tasks()
    assert exit_code == 0
    assert [task.task_id for task in tasks] == ["demo-000001", "demo-000002"]
    assert tasks[1].annotation_requirements == {"annotation_types": ["entity_span"]}


def test_cli_create_batched_jsonl_tasks(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"text": f"row {index}", "source_dataset": "demo_source"})
            for index in range(1, 6)
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "create-tasks",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v2",
            "--task-prefix",
            "memory-ner-v2",
            "--batch-size",
            "2",
            "--annotation-type",
            "entity_span",
            "--annotation-type",
            "structured_json",
        ]
    )

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    tasks = store.list_tasks()
    assert exit_code == 0
    assert [task.task_id for task in tasks] == [
        "memory-ner-v2-000001",
        "memory-ner-v2-000002",
        "memory-ner-v2-000003",
    ]
    assert [task.source_ref["row_count"] for task in tasks] == [2, 2, 1]
    assert tasks[0].source_ref["line_start"] == 1
    assert tasks[0].source_ref["line_end"] == 2
    assert len(tasks[0].source_ref["payload"]["rows"]) == 2
    assert tasks[0].annotation_requirements == {"annotation_types": ["entity_span", "structured_json"]}
    # qc_policy moved to project-level RuntimeConfig (workflow.yaml). No per-task injection.
    assert "qc_policy" not in tasks[0].metadata
    assert tasks[0].metadata["sources"] == ["demo_source"]


def test_cli_create_batched_jsonl_tasks_with_qc_sample_count(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(json.dumps({"text": f"row {index}"}) for index in range(1, 6)) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "create-tasks",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "sample-count",
            "--batch-size",
            "5",
            "--qc-sample-count",
            "2",
        ]
    )

    task = SqliteStore.open(tmp_path / ".annotation-pipeline").load_task("sample-count-000001")
    assert exit_code == 0
    # qc_policy moved to project-level RuntimeConfig. --qc-sample-count is accepted
    # by the CLI for backward compat but no longer populates per-task metadata.
    assert "qc_policy" not in task.metadata


def test_cli_create_batched_jsonl_tasks_with_qc_sample_ratio(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(json.dumps({"text": f"row {index}"}) for index in range(1, 6)) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "create-tasks",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "sample-ratio",
            "--batch-size",
            "5",
            "--qc-sample-ratio",
            "0.4",
        ]
    )

    task = SqliteStore.open(tmp_path / ".annotation-pipeline").load_task("sample-ratio-000001")
    assert exit_code == 0
    # qc_policy moved to project-level RuntimeConfig.
    assert "qc_policy" not in task.metadata


def test_cli_create_tasks_rejects_conflicting_qc_sample_options(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"text": "alpha"}) + "\n", encoding="utf-8")

    try:
        main(
            [
                "create-tasks",
                "--project-root",
                str(tmp_path),
                "--source",
                str(source),
                "--pipeline-id",
                "bad",
                "--qc-sample-count",
                "1",
                "--qc-sample-ratio",
                "0.5",
            ]
        )
    except ValueError as exc:
        assert str(exc) == "--qc-sample-count and --qc-sample-ratio are mutually exclusive"
    else:
        raise AssertionError("expected conflicting QC sample options to fail")


def test_cli_create_batched_jsonl_tasks_does_not_cross_group_boundaries(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    source = tmp_path / "input.jsonl"
    rows = [
        {"text": "a1", "source_dataset": "a"},
        {"text": "a2", "source_dataset": "a"},
        {"text": "a3", "source_dataset": "a"},
        {"text": "b1", "source_dataset": "b"},
        {"text": "b2", "source_dataset": "b"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "create-tasks",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v2",
            "--batch-size",
            "2",
            "--group-by",
            "source_dataset",
        ]
    )

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    tasks = store.list_tasks()
    assert exit_code == 0
    assert [task.source_ref["row_count"] for task in tasks] == [2, 1, 2]
    assert [task.metadata["sources"] for task in tasks] == [["a"], ["a"], ["b"]]


def test_cli_import_annotation_manager_v2_queues_imported_annotations_for_qc(tmp_path, capsys):
    from annotation_pipeline_skill.core.states import TaskStatus

    main(["init", "--project-root", str(tmp_path)])
    source_root = tmp_path / "manager-v2" / "tasks"
    source_root.mkdir(parents=True)
    output_file = source_root / "legacy_task_001.annotated.jsonl"
    output_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "input": "Repo: nodejs/node\nReviewed-By: Ada Lovelace",
                        "output": {
                            "entities": {"organization": ["nodejs"], "person": ["Ada Lovelace"]},
                            "classifications": [],
                            "json_structures": [],
                            "relations": [],
                        },
                        "source_dataset": "github",
                        "source_path": "github.jsonl",
                    }
                ),
                json.dumps({"input": "missing output"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_file = source_root / "legacy_task_001.task.json"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "legacy_task_001",
                "status": "merged",
                "output_file": str(output_file),
            }
        ),
        encoding="utf-8",
    )
    missing_output_task = source_root / "legacy_task_002.task.json"
    missing_output_task.write_text(
        json.dumps({"task_id": "legacy_task_002", "status": "merged", "output_file": ""}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "import",
            "annotation-manager-v2",
            "--project-root",
            str(tmp_path),
            "--source-task-root",
            str(source_root),
            "--pipeline-id",
            "memory-ner-v2",
            "--task-prefix",
            "memory-ner-v2-review",
            "--qc-sample-count",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    task = store.load_task("memory-ner-v2-review-000001")
    artifacts = store.list_artifacts(task.task_id)
    attempts = store.list_attempts(task.task_id)
    artifact_payload = json.loads((store.root / artifacts[0].path).read_text(encoding="utf-8"))
    events = store.list_events(task.task_id)
    assert exit_code == 0
    assert payload == {"imported": 1, "pipeline_id": "memory-ner-v2", "skipped": 1}
    assert task.status is TaskStatus.QC
    assert task.current_attempt == 1
    assert task.metadata["runtime_next_stage"] == "qc"
    assert task.metadata["source_task_id"] == "legacy_task_001"
    # qc_policy moved to project-level RuntimeConfig; v2 import no longer injects it.
    assert "qc_policy" not in task.metadata
    assert task.source_ref["kind"] == "annotation_manager_v2"
    assert task.source_ref["payload"]["rows"][0]["text"].startswith("Repo: nodejs/node")
    assert [event.next_status.value for event in events] == ["pending", "annotating", "qc"]
    assert attempts[0].provider_id == "annotation_manager_v2"
    assert artifacts[0].kind == "annotation_result"
    assert artifact_payload["imported_annotation"]["rows"][0]["output"]["entities"]["person"] == ["Ada Lovelace"]
    assert json.loads(artifact_payload["text"])["rows"][0]["output"]["entities"]["organization"] == ["nodejs"]


def test_cli_export_training_data_writes_manifest(tmp_path, capsys):
    from annotation_pipeline_skill.core.models import ArtifactRef
    from annotation_pipeline_skill.core.states import TaskStatus

    main(["init", "--project-root", str(tmp_path)])
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"text": "alpha"}) + "\n", encoding="utf-8")
    main(
        [
            "create-tasks",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "pipe",
        ]
    )
    task = store.load_task("pipe-000001")
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    payload_path = store.root / "artifact_payloads/pipe-000001/pipe-000001-attempt-1_annotation_result.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(json.dumps({"text": '{"labels":[]}'}), encoding="utf-8")
    store.append_artifact(
        ArtifactRef.new(
            task_id="pipe-000001",
            kind="annotation_result",
            path="artifact_payloads/pipe-000001/pipe-000001-attempt-1_annotation_result.json",
            content_type="application/json",
        )
    )

    exit_code = main(
        [
            "export",
            "training-data",
            "--project-root",
            str(tmp_path),
            "--project-id",
            "pipe",
            "--export-id",
            "export-1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["export_id"] == "export-1"
    assert payload["task_ids_included"] == ["pipe-000001"]
    assert (tmp_path / ".annotation-pipeline" / "exports" / "export-1" / "training_data.jsonl").exists()


def test_cli_report_readiness_returns_project_action(tmp_path, capsys):
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus

    main(["init", "--project-root", str(tmp_path)])
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)

    exit_code = main(["report", "readiness", "--project-root", str(tmp_path), "--project-id", "pipe"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["project_id"] == "pipe"
    assert payload["accepted_count"] == 1
    assert payload["recommended_next_action"] == "fix_export_blockers"


def test_cli_outbox_status_reports_counts(tmp_path, capsys):
    from annotation_pipeline_skill.core.models import OutboxRecord
    from annotation_pipeline_skill.core.states import OutboxKind

    main(["init", "--project-root", str(tmp_path)])
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    store.save_outbox(OutboxRecord.new(task_id="task-1", kind=OutboxKind.SUBMIT, payload={}))

    exit_code = main(["outbox", "status", "--project-root", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["counts"] == {"dead_letter": 0, "pending": 1, "sent": 0}
    assert payload["records"][0]["kind"] == "submit"


def test_cli_human_review_request_changes_returns_task_to_annotating(tmp_path, capsys):
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus

    main(["init", "--project-root", str(tmp_path)])
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    exit_code = main(
        [
            "human-review",
            "decide",
            "--project-root",
            str(tmp_path),
            "--task-id",
            "task-1",
            "--action",
            "request_changes",
            "--actor",
            "algorithm-engineer",
            "--feedback",
            "Run the batch boundary update.",
            "--correction-mode",
            "batch_code_update",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["task"]["status"] == "annotating"
    assert payload["decision"]["correction_mode"] == "batch_code_update"
    assert store.load_task("task-1").status is TaskStatus.ANNOTATING


def test_cli_external_pull_uses_configured_http_source(tmp_path, capsys):
    main(["init", "--project-root", str(tmp_path)])
    config_path = tmp_path / ".annotation-pipeline" / "external_tasks.yaml"
    with external_pull_server({"tasks": [{"external_task_id": "ext-1", "payload": {"text": "alpha"}}]}) as pull_url:
        config_path.write_text(
            "\n".join(
                [
                    "external_tasks:",
                    "  default:",
                    "    enabled: true",
                    "    system_id: vendor",
                    f"    pull_url: {pull_url}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        exit_code = main(
            [
                "external",
                "pull",
                "--project-root",
                str(tmp_path),
                "--project-id",
                "pipe",
                "--source-id",
                "default",
                "--limit",
                "1",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    assert exit_code == 0
    assert payload["created"] == 1
    assert store.list_tasks()[0].pipeline_id == "pipe"


def test_cli_db_init_creates_db(tmp_path, monkeypatch):
    from annotation_pipeline_skill.interfaces.cli import main
    monkeypatch.chdir(tmp_path)

    rc = main(["db", "init", "--root", str(tmp_path / "ws")])

    assert rc == 0
    assert (tmp_path / "ws" / "db.sqlite").exists()


def test_cli_db_backup_creates_snapshot(tmp_path):
    from annotation_pipeline_skill.interfaces.cli import main
    rc = main(["db", "init", "--root", str(tmp_path / "ws")])
    assert rc == 0

    rc = main(["db", "backup", "--root", str(tmp_path / "ws")])
    assert rc == 0
    snaps = list((tmp_path / "ws" / "backups").glob("sqlite-*.sqlite"))
    assert len(snaps) == 1


def test_cli_db_dump_json_round_trips(tmp_path):
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.interfaces.cli import main
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore

    rc = main(["db", "init", "--root", str(tmp_path / "ws")])
    assert rc == 0
    store = SqliteStore.open(tmp_path / "ws")
    store.save_task(Task.new(task_id="t-1", pipeline_id="p", source_ref={}))
    store.close()

    rc = main(["db", "dump-json",
               "--root", str(tmp_path / "ws"),
               "--out", str(tmp_path / "out")])
    assert rc == 0
    assert (tmp_path / "out" / "tasks" / "t-1.json").exists()


def test_cli_db_status_prints_counts(tmp_path, capsys):
    from annotation_pipeline_skill.interfaces.cli import main
    rc = main(["db", "init", "--root", str(tmp_path / "ws")])
    assert rc == 0

    rc = main(["db", "status", "--root", str(tmp_path / "ws")])
    assert rc == 0
    captured = capsys.readouterr()
    assert "tasks: 0" in captured.out


def test_cli_human_review_correct_accepts_answer_file(tmp_path):
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus

    root = tmp_path / "ws"
    rc = main(["db", "init", "--root", str(root)])
    assert rc == 0
    store = SqliteStore.open(root)
    task = Task.new(
        task_id="t-cli-hr",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "x",
                "annotation_guidance": {
                    "output_schema": {
                        "type": "object",
                        "required": ["entities"],
                        "properties": {"entities": {"type": "array"}},
                    }
                },
            },
        },
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)
    store.close()

    answer_path = tmp_path / "answer.json"
    answer_path.write_text(json.dumps({"entities": []}), encoding="utf-8")

    rc = main([
        "human-review", "correct",
        "--root", str(root),
        "--task", "t-cli-hr",
        "--answer-file", str(answer_path),
        "--actor", "reviewer-1",
    ])
    assert rc == 0
    store = SqliteStore.open(root)
    assert store.load_task("t-cli-hr").status is TaskStatus.ACCEPTED


def test_cli_human_review_correct_returns_nonzero_on_schema_fail(tmp_path):
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus

    root = tmp_path / "ws"
    main(["db", "init", "--root", str(root)])
    store = SqliteStore.open(root)
    task = Task.new(
        task_id="t-cli-bad",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {"annotation_guidance": {"output_schema": {"type": "object", "required": ["entities"]}}},
        },
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)
    store.close()

    answer_path = tmp_path / "bad.json"
    answer_path.write_text(json.dumps({"wrong": []}), encoding="utf-8")

    rc = main([
        "human-review", "correct",
        "--root", str(root),
        "--task", "t-cli-bad",
        "--answer-file", str(answer_path),
        "--actor", "r",
    ])
    assert rc != 0


def test_default_llm_profiles_template_covers_memory_ner_models():
    """The init template should include a claude_cli profile."""
    import yaml
    from annotation_pipeline_skill.interfaces.cli import CONFIG_FILES

    template = yaml.safe_load(CONFIG_FILES["llm_profiles.yaml"])
    profile_models = {p["model"] for p in template["profiles"].values()}
    profile_runtimes = {p.get("runtime") for p in template["profiles"].values()}
    # flat schema: claude_cli runtime
    assert "claude_cli" in profile_runtimes
    # claude-sonnet-4-6 is the default model
    assert "claude-sonnet-4-6" in profile_models


def test_default_llm_profiles_glm_coding_models_use_coding_endpoint_with_fallback_key():
    """glm-4.5-air, glm-4.6, glm-5.1 must use the coding endpoint + GLM_CODING_API_KEY fallback chain."""
    import yaml
    from annotation_pipeline_skill.interfaces.cli import CONFIG_FILES

    template = yaml.safe_load(CONFIG_FILES["llm_profiles.yaml"])
    coding_models = {"glm-4.5-air", "glm-4.6", "glm-5.1"}
    for profile_name, profile in template["profiles"].items():
        if profile.get("model") in coding_models:
            assert profile["base_url"] == "https://open.bigmodel.cn/api/coding/paas/v4", (
                f"{profile_name} should use coding endpoint, got {profile.get('base_url')}"
            )
            key_env = profile["api_key_env"]
            # Must be a list and include GLM_CODING_API_KEY first, BIGMODEL_MCP_API_KEY second
            assert isinstance(key_env, list), f"{profile_name} api_key_env should be a list"
            assert key_env[0] == "GLM_CODING_API_KEY", f"{profile_name} primary env must be GLM_CODING_API_KEY"
            assert "BIGMODEL_MCP_API_KEY" in key_env, f"{profile_name} should include BIGMODEL_MCP_API_KEY fallback"


def test_default_llm_profiles_glm_non_coding_models_use_public_endpoint():
    """If a glm-4-flash profile exists in the template it must use the public endpoint.
    The simplified default template no longer includes GLM profiles, so this test
    passes vacuously; individual project llm_profiles.yaml files with GLM profiles
    are validated by test_llm_profiles.py."""
    import yaml
    from annotation_pipeline_skill.interfaces.cli import CONFIG_FILES

    template = yaml.safe_load(CONFIG_FILES["llm_profiles.yaml"])
    for profile_name, profile in template["profiles"].items():
        if profile.get("model") == "glm-4-flash":
            assert profile["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
            return
    # No glm-4-flash in the simplified template — that's fine.


def test_default_llm_profiles_template_is_valid_yaml_and_registry():
    """Template must load cleanly through load_llm_registry."""
    from annotation_pipeline_skill.interfaces.cli import CONFIG_FILES
    from annotation_pipeline_skill.llm.profiles import load_llm_registry

    # Round-trip through file because load_llm_registry expects a path
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(CONFIG_FILES["llm_profiles.yaml"])
        path = Path(f.name)
    try:
        registry = load_llm_registry(path)
        # All declared targets must resolve to existing profiles.
        for target in registry.targets:
            registry.resolve(target)
    finally:
        path.unlink(missing_ok=True)


def _write_prelabeled_fixture(tmp_path, row_count):
    source = tmp_path / "v3_tasks.jsonl"
    lines = []
    for i in range(row_count):
        lines.append(
            json.dumps(
                {
                    "task_id": f"row-{i:03d}",
                    "source_id": f"src-{i}",
                    "input": f"text body {i}",
                    "output": {"labels": [{"text": f"e{i}", "type": "ENTITY"}]},
                }
            )
        )
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source


def _write_minimal_schema_file(tmp_path):
    """Write a small JSON Schema with $defs/output that contains a $ref to a spanList."""
    schema = {
        "$defs": {
            "spanList": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "type"],
                    "properties": {
                        "text": {"type": "string"},
                        "type": {"type": "string"},
                    },
                },
            },
            "output": {
                "type": "object",
                "required": ["labels"],
                "properties": {"labels": {"$ref": "#/$defs/spanList"}},
            },
        }
    }
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(schema), encoding="utf-8")
    return schema_file


def test_cli_import_jsonl_prelabeled_creates_tasks_with_prelabel_metadata(tmp_path, capsys):
    from annotation_pipeline_skill.core.states import TaskStatus

    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=15)
    schema_file = _write_minimal_schema_file(tmp_path)

    exit_code = main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3_initial_deployment",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
            "--output-schema-pointer",
            "$defs/output",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {"imported": 2, "pipeline_id": "v3_initial_deployment", "skipped": 0}

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    task0 = store.load_task("v3_initial_deployment-000000")
    task1 = store.load_task("v3_initial_deployment-000001")
    assert task0.status is TaskStatus.PENDING
    assert task1.status is TaskStatus.PENDING
    assert task0.current_attempt == 0
    assert task0.metadata["prelabeled"] is True
    assert task0.metadata["batch_size"] == 10
    assert task1.metadata["batch_size"] == 5
    assert len(task0.metadata["row_ids"]) == 10
    assert len(task1.metadata["row_ids"]) == 5

    rows0 = task0.source_ref["payload"]["rows"]
    assert len(rows0) == 10
    assert rows0[0] == {
        "row_index": 0,
        "row_id": "row-000",
        "source_id": "src-0",
        "input": "text body 0",
    }
    guidance = task0.source_ref["payload"]["annotation_guidance"]
    # Per-task source_ref no longer carries the schema -- it lives at the project level.
    assert "output_schema" not in guidance
    assert guidance == {"rules_path": "annotation_rules.yaml"}
    project_schema_path = tmp_path / ".annotation-pipeline" / "output_schema.json"
    assert project_schema_path.exists()
    batched = json.loads(project_schema_path.read_text(encoding="utf-8"))
    # Project-level schema accepts partial-final batches (minItems=1, maxItems=batch_size).
    assert batched["properties"]["rows"]["minItems"] == 1
    assert batched["properties"]["rows"]["maxItems"] == 10
    # $defs hoisted to root of batched schema so $refs in per-row output resolve.
    assert "$defs" in batched
    assert "spanList" in batched["$defs"]
    item_output = batched["properties"]["rows"]["items"]["properties"]["output"]
    assert "$defs" not in item_output  # do not duplicate at the nested level

    # End-to-end: validating the actual annotation artifact against the batched schema must pass.
    from jsonschema import Draft202012Validator
    artifact = store.list_artifacts(task0.task_id)[0]
    artifact_payload = json.loads((store.root / artifact.path).read_text(encoding="utf-8"))
    inner = json.loads(artifact_payload["text"])
    errors = list(Draft202012Validator(batched).iter_errors(inner))
    assert errors == [], f"expected no validation errors, got {[e.message for e in errors]}"


def test_cli_import_jsonl_prelabeled_writes_annotation_artifact(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=3)
    schema_file = _write_minimal_schema_file(tmp_path)

    main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    artifacts = store.list_artifacts("v3-000000")
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.kind == "annotation_result"
    artifact_payload = json.loads((store.root / artifact.path).read_text(encoding="utf-8"))
    assert artifact_payload["raw_response"] == {"source": "v2_prelabel"}
    inner = json.loads(artifact_payload["text"])
    assert len(inner["rows"]) == 3
    assert inner["rows"][0]["row_index"] == 0
    assert inner["rows"][0]["row_id"] == "row-000"
    assert inner["rows"][0]["output"] == {"labels": [{"text": "e0", "type": "ENTITY"}]}


def test_cli_import_writes_project_output_schema_file(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=3)
    schema_file = _write_minimal_schema_file(tmp_path)

    rc = main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )
    assert rc == 0
    project_schema_path = tmp_path / ".annotation-pipeline" / "output_schema.json"
    assert project_schema_path.exists()
    schema = json.loads(project_schema_path.read_text(encoding="utf-8"))
    assert schema["type"] == "object"
    assert schema["properties"]["rows"]["maxItems"] == 10
    assert schema["properties"]["rows"]["minItems"] == 1
    assert "$defs" in schema


def test_cli_import_omits_per_task_inline_schema(tmp_path):
    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=3)
    schema_file = _write_minimal_schema_file(tmp_path)

    main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    task = store.load_task("v3-000000")
    guidance = task.source_ref["payload"]["annotation_guidance"]
    assert "output_schema" not in guidance
    # rules_path is still present for the annotator instructions.
    assert guidance.get("rules_path") == "annotation_rules.yaml"


def test_cli_import_normalizes_empty_json_structures_array(tmp_path):
    """v2 prelabeled rows often had ``output.json_structures: []``; v3 requires an object."""
    source = tmp_path / "prelabel.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "row-000",
                "input": "hello",
                "output": {"labels": [], "json_structures": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    schema_file = _write_minimal_schema_file(tmp_path)
    main(["init", "--project-root", str(tmp_path)])
    rc = main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )
    assert rc == 0
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    artifact = store.list_artifacts("v3-000000")[0]
    payload = json.loads((store.root / artifact.path).read_text(encoding="utf-8"))
    inner = json.loads(payload["text"])
    assert inner["rows"][0]["output"]["json_structures"] == {}


def test_cli_import_normalizes_non_empty_json_structures_array(tmp_path, capsys):
    """Non-empty legacy list is dropped (with warning) since v2->v3 types do not auto-translate."""
    source = tmp_path / "prelabel.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "row-000",
                "input": "hello",
                "output": {
                    "labels": [],
                    "json_structures": [{"phrase": "x", "type": "LEGACY_TYPE"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    schema_file = _write_minimal_schema_file(tmp_path)
    main(["init", "--project-root", str(tmp_path)])
    rc = main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "warning" in captured.lower() and "json_structures" in captured
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    artifact = store.list_artifacts("v3-000000")[0]
    payload = json.loads((store.root / artifact.path).read_text(encoding="utf-8"))
    inner = json.loads(payload["text"])
    assert inner["rows"][0]["output"]["json_structures"] == {}


def test_cli_import_preserves_dict_json_structures(tmp_path):
    """v3-shaped dict json_structures must pass through unchanged."""
    js_dict = {"PHRASE_TYPE_A": [{"phrase": "alpha"}]}
    source = tmp_path / "prelabel.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "row-000",
                "input": "hello",
                "output": {"labels": [], "json_structures": js_dict},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    schema_file = _write_minimal_schema_file(tmp_path)
    main(["init", "--project-root", str(tmp_path)])
    main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )
    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    artifact = store.list_artifacts("v3-000000")[0]
    payload = json.loads((store.root / artifact.path).read_text(encoding="utf-8"))
    inner = json.loads(payload["text"])
    assert inner["rows"][0]["output"]["json_structures"] == js_dict


def test_cli_import_jsonl_prelabeled_appends_attempt_record(tmp_path):
    from annotation_pipeline_skill.core.states import AttemptStatus

    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=12)
    schema_file = _write_minimal_schema_file(tmp_path)

    main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    for task_id in ("v3-000000", "v3-000001"):
        attempts = store.list_attempts(task_id)
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt.status is AttemptStatus.SUCCEEDED
        assert attempt.provider_id == "prelabel"
        assert attempt.stage == "annotation"
        assert attempt.summary == "imported from v2 annotation"
        # attempt_id must be scoped by task_id so subsequent pipelines / re-runs
        # don't collide on the globally-unique attempts.attempt_id primary key.
        assert attempt.attempt_id == f"{task_id}-attempt-0-prelabel"


def test_cli_import_jsonl_prelabeled_two_pipelines_no_attempt_id_collision(tmp_path):
    """Importing two pipelines with overlapping batch indices must succeed.

    Regression test: previously attempt_id was `attempt-prelabel-{batch_idx}`
    which collided whenever batch_idx repeated across imports.
    """
    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=10)
    schema_file = _write_minimal_schema_file(tmp_path)

    for pipeline in ("pipeline_a", "pipeline_b"):
        rc = main(
            [
                "import",
                "jsonl-prelabeled",
                "--project-root",
                str(tmp_path),
                "--source",
                str(source),
                "--pipeline-id",
                pipeline,
                "--batch-size",
                "10",
                "--output-schema-file",
                str(schema_file),
            ]
        )
        assert rc == 0

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    # Both pipelines created their batch 0 task successfully — no collision.
    assert store.load_task("pipeline_a-000000").pipeline_id == "pipeline_a"
    assert store.load_task("pipeline_b-000000").pipeline_id == "pipeline_b"
    a_attempts = store.list_attempts("pipeline_a-000000")
    b_attempts = store.list_attempts("pipeline_b-000000")
    assert len(a_attempts) == 1 and len(b_attempts) == 1
    assert a_attempts[0].attempt_id != b_attempts[0].attempt_id


def test_cli_import_refuses_on_collision_without_force_rewrite(tmp_path, capsys):
    """Re-importing the same pipeline_id without --force-rewrite must refuse
    cleanly (exit 1, task_id_collision JSON) rather than silently UPSERT-ing
    the task and corrupting its children records."""
    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=10)
    schema_file = _write_minimal_schema_file(tmp_path)

    first_rc = main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "dup_pipeline",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )
    capsys.readouterr()  # drain first import's output
    assert first_rc == 0

    second_rc = main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "dup_pipeline",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )
    assert second_rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "task_id_collision"
    assert payload["pipeline_id"] == "dup_pipeline"
    assert "dup_pipeline-000000" in payload["collisions"]
    assert "--force-rewrite" in payload["hint"]


def test_cli_import_jsonl_prelabeled_start_batch_offset_appends_without_collision(tmp_path, capsys):
    """With --start-batch-offset N, the import skips N batches of input rows
    and numbers task_ids starting at N — letting a follow-up import append
    without colliding with previously-imported tasks.

    Concretely: first import gets task_id ...-000000 from rows 0-9. Second
    import with --start-batch-offset 1 skips rows 0-9 and writes task_id
    ...-000001 from rows 10-19. No --force-rewrite needed."""
    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=20)
    schema_file = _write_minimal_schema_file(tmp_path)

    # First import: rows 0-9 → task -000000 only (limit=10).
    rc = main(
        [
            "import", "jsonl-prelabeled",
            "--project-root", str(tmp_path),
            "--source", str(source),
            "--pipeline-id", "offset_pipeline",
            "--batch-size", "10",
            "--limit", "10",
            "--output-schema-file", str(schema_file),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload == {"imported": 1, "pipeline_id": "offset_pipeline", "skipped": 0}

    # Second import: skip first batch (rows 0-9), then take the next batch.
    rc = main(
        [
            "import", "jsonl-prelabeled",
            "--project-root", str(tmp_path),
            "--source", str(source),
            "--pipeline-id", "offset_pipeline",
            "--batch-size", "10",
            "--start-batch-offset", "1",
            "--output-schema-file", str(schema_file),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload == {"imported": 1, "pipeline_id": "offset_pipeline", "skipped": 0}

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    # Both tasks exist with non-colliding task_ids derived from the absolute index.
    t0 = store.load_task("offset_pipeline-000000")
    t1 = store.load_task("offset_pipeline-000001")
    assert t0.metadata["row_ids"][0] == "row-000"  # first batch starts at row 0
    assert t1.metadata["row_ids"][0] == "row-010"  # second batch starts at row 10 (offset*size)


def test_cli_import_force_rewrite_cascade_deletes_then_imports(tmp_path, capsys):
    """With --force-rewrite, colliding tasks (plus children + on-disk payload
    files) must be cascade-deleted before the import re-creates them."""
    from annotation_pipeline_skill.core.models import (
        ArtifactRef,
        Attempt,
        AuditEvent,
        FeedbackRecord,
    )
    from annotation_pipeline_skill.core.states import (
        AttemptStatus,
        FeedbackSeverity,
        FeedbackSource,
        TaskStatus,
    )

    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=10)
    schema_file = _write_minimal_schema_file(tmp_path)

    # First import populates a single task with batch_idx=0.
    rc = main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "fr_pipeline",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )
    capsys.readouterr()
    assert rc == 0

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    task_id = "fr_pipeline-000000"

    # Simulate runtime activity on the task: a second attempt + feedback +
    # extra audit event + extra artifact. Without cascade-delete these would
    # survive a re-import and collide with the new run.
    store.append_event(
        AuditEvent.new(task_id, TaskStatus.ANNOTATING, TaskStatus.PENDING,
                       actor="qc", reason="rejected", stage="qc")
    )
    store.append_attempt(
        Attempt(attempt_id=f"{task_id}-attempt-1", task_id=task_id, index=1,
                stage="annotation", status=AttemptStatus.SUCCEEDED)
    )
    store.append_feedback(
        FeedbackRecord.new(
            task_id=task_id, attempt_id=f"{task_id}-attempt-1",
            source_stage=FeedbackSource.QC, severity=FeedbackSeverity.ERROR,
            category="cat", message="bad", target={"x": "y"},
            suggested_action="rerun", created_by="qc",
        )
    )
    store.append_artifact(
        ArtifactRef.new(
            task_id=task_id, kind="qc_report",
            path=f"artifact_payloads/{task_id}/qc_report.json",
            content_type="application/json",
        )
    )
    # Place a stray file inside the on-disk payload dir.
    stray = store.root / "artifact_payloads" / task_id / "stray.json"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text("{}", encoding="utf-8")

    # Sanity: pre-rewrite state has multiple attempts/events/feedback/artifacts.
    assert len(store.list_attempts(task_id)) >= 2
    assert len(store.list_feedback(task_id)) >= 1
    assert stray.exists()

    # Re-import with --force-rewrite: cascade-delete then re-create.
    rc = main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "fr_pipeline",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
            "--force-rewrite",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload == {"imported": 1, "pipeline_id": "fr_pipeline", "skipped": 0}

    # Re-open the store so we observe the post-rewrite state.
    store2 = SqliteStore.open(tmp_path / ".annotation-pipeline")
    fresh = store2.load_task(task_id)
    assert fresh.status is TaskStatus.PENDING
    assert fresh.current_attempt == 0

    # Old children are gone; only the fresh prelabel attempt + artifact remain.
    attempts = store2.list_attempts(task_id)
    assert len(attempts) == 1
    assert attempts[0].attempt_id == f"{task_id}-attempt-0-prelabel"
    assert store2.list_feedback(task_id) == []
    # Audit events: only the freshly-emitted PENDING event from import.
    assert len(store2.list_events(task_id)) == 1
    # Stray file from the prior run is gone; only the fresh prelabeled
    # annotation payload is present.
    assert not stray.exists()
    artifacts = store2.list_artifacts(task_id)
    assert len(artifacts) == 1
    assert artifacts[0].kind == "annotation_result"


def test_cli_pipeline_delete_preview_then_force(tmp_path, capsys):
    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=3)
    schema_file = _write_minimal_schema_file(tmp_path)

    rc = main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3",
            "--batch-size",
            "1",
            "--output-schema-file",
            str(schema_file),
        ]
    )
    assert rc == 0
    capsys.readouterr()  # drain import output

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    pre_tasks = [t for t in store.list_tasks() if t.pipeline_id == "v3"]
    assert len(pre_tasks) == 3

    # Preview (no --force): tasks still present.
    rc = main(
        [
            "pipeline",
            "delete",
            "--project-root",
            str(tmp_path),
            "--pipeline-id",
            "v3",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "would_delete" in payload
    assert payload["would_delete"]["tasks"] == 3
    store2 = SqliteStore.open(tmp_path / ".annotation-pipeline")
    assert len([t for t in store2.list_tasks() if t.pipeline_id == "v3"]) == 3

    # With --force: tasks gone.
    rc = main(
        [
            "pipeline",
            "delete",
            "--project-root",
            str(tmp_path),
            "--pipeline-id",
            "v3",
            "--force",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["deleted"]["tasks"] == 3

    store3 = SqliteStore.open(tmp_path / ".annotation-pipeline")
    assert [t for t in store3.list_tasks() if t.pipeline_id == "v3"] == []


def test_cli_pipeline_delete_nonexistent_returns_1(tmp_path, capsys):
    main(["init", "--project-root", str(tmp_path)])

    rc = main(
        [
            "pipeline",
            "delete",
            "--project-root",
            str(tmp_path),
            "--pipeline-id",
            "does-not-exist",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    payload = json.loads(out)
    assert payload == {"error": "pipeline_not_found", "pipeline_id": "does-not-exist"}


def test_inspect_prints_task_summary(tmp_path):
    """inspect prints status, attempts, and feedback for a task."""
    from annotation_pipeline_skill.interfaces.cli import main
    from annotation_pipeline_skill.store.sqlite_store import SqliteStore
    from annotation_pipeline_skill.core.models import Task
    from annotation_pipeline_skill.core.states import TaskStatus
    from annotation_pipeline_skill.core.transitions import transition_task

    config_root = tmp_path / ".annotation-pipeline"
    config_root.mkdir()
    store = SqliteStore.open(config_root)
    task = Task.new(
        task_id="test-000001",
        pipeline_id="test-pipe",
        source_ref={"kind": "jsonl", "payload": {"rows": []}},
        modality="text",
        annotation_requirements={"annotation_types": ["extraction"]},
        metadata={},
    )
    event = transition_task(task, TaskStatus.PENDING, actor="test", reason="init", stage="prepare")
    store.save_task(task)
    store.append_event(event)
    store.close()

    result = main(["inspect", "test-000001", "--project-root", str(tmp_path)])
    assert result == 0


def test_cli_import_does_not_inject_qc_policy_into_task_metadata(tmp_path):
    """After the QC-config lift, ``apl import jsonl-prelabeled`` must NOT write
    per-task ``metadata.qc_policy`` — the policy now lives at project level in
    workflow.yaml > runtime.qc_*. (Legacy tasks may still carry it; new ones
    must not.)"""
    main(["init", "--project-root", str(tmp_path)])
    source = _write_prelabeled_fixture(tmp_path, row_count=4)
    schema_file = _write_minimal_schema_file(tmp_path)

    main(
        [
            "import",
            "jsonl-prelabeled",
            "--project-root",
            str(tmp_path),
            "--source",
            str(source),
            "--pipeline-id",
            "v3_no_qc",
            "--batch-size",
            "10",
            "--output-schema-file",
            str(schema_file),
        ]
    )

    store = SqliteStore.open(tmp_path / ".annotation-pipeline")
    task = store.load_task("v3_no_qc-000000")
    assert "qc_policy" not in task.metadata, (
        f"task.metadata still carries qc_policy={task.metadata.get('qc_policy')!r}; "
        "the QC policy must come from project workflow.yaml, not per-task injection"
    )

import json

from annotation_pipeline_skill.core.models import ArtifactRef, Attempt, FeedbackDiscussionEntry, FeedbackRecord, Task
from annotation_pipeline_skill.core.states import AttemptStatus, FeedbackSeverity, FeedbackSource, TaskStatus
from annotation_pipeline_skill.core.transitions import transition_task
from annotation_pipeline_skill.interfaces.api import DashboardApi
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def test_dashboard_api_returns_kanban_snapshot_json(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    store.save_task(task)
    api = DashboardApi(store)

    status, headers, body = api.handle_get("/api/kanban")

    assert status == 200
    assert headers["content-type"] == "application/json"
    payload = json.loads(body.decode("utf-8"))
    assert payload["columns"][0]["id"] == "pending"
    assert payload["columns"][0]["cards"][0]["task_id"] == "task-1"


def test_dashboard_api_returns_projects_and_filters_kanban_by_project(tmp_path):
    store = SqliteStore.open(tmp_path)
    alpha = Task.new(task_id="alpha-1", pipeline_id="project-alpha", source_ref={"kind": "jsonl"})
    beta = Task.new(task_id="beta-1", pipeline_id="project-beta", source_ref={"kind": "jsonl"})
    alpha.status = TaskStatus.PENDING
    beta.status = TaskStatus.PENDING
    store.save_task(alpha)
    store.save_task(beta)
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/projects")
    projects_payload = json.loads(body.decode("utf-8"))
    assert status == 200
    assert projects_payload["projects"] == [
        {"project_id": "project-alpha", "status_counts": {"pending": 1}, "task_count": 1},
        {"project_id": "project-beta", "status_counts": {"pending": 1}, "task_count": 1},
    ]

    status, _headers, body = api.handle_get("/api/kanban?project=project-alpha")
    kanban_payload = json.loads(body.decode("utf-8"))
    visible_task_ids = [
        card["task_id"]
        for column in kanban_payload["columns"]
        for card in column["cards"]
    ]
    assert status == 200
    assert kanban_payload["project_id"] == "project-alpha"
    assert visible_task_ids == ["alpha-1"]


def test_dashboard_api_returns_operator_stage_kanban_view(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.ANNOTATING
    store.save_task(task)
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/kanban?stage_view=operator")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert payload["stage_view"] == "operator"
    assert payload["columns"][1]["id"] == "annotation"
    assert payload["columns"][1]["cards"][0]["operator_stage"] == "annotation"


def test_dashboard_api_returns_stores_list_with_task_count(tmp_path):
    project_a = tmp_path / "project-a" / ".annotation-pipeline"
    project_b = tmp_path / "project-b" / ".annotation-pipeline"
    store_a = SqliteStore.open(project_a)
    store_b = SqliteStore.open(project_b)
    # Two tasks across two pipelines in project-a; one task in project-b.
    t1 = Task.new(task_id="a-1", pipeline_id="pipe-1", source_ref={"kind": "jsonl"})
    t2 = Task.new(task_id="a-2", pipeline_id="pipe-2", source_ref={"kind": "jsonl"})
    t3 = Task.new(task_id="b-1", pipeline_id="pipe-x", source_ref={"kind": "jsonl"})
    store_a.save_task(t1)
    store_a.save_task(t2)
    store_b.save_task(t3)
    api = DashboardApi(
        store_a,
        stores={"project-a": store_a, "project-b": store_b},
        default_store_key="project-a",
    )

    status, _headers, body = api.handle_get("/api/stores")

    assert status == 200
    payload = json.loads(body.decode("utf-8"))
    by_key = {s["key"]: s for s in payload["stores"]}
    assert by_key["project-a"]["task_count"] == 2
    assert by_key["project-a"]["pipeline_count"] == 2
    assert by_key["project-b"]["task_count"] == 1
    assert by_key["project-b"]["pipeline_count"] == 1


def test_dashboard_api_returns_404_for_unknown_route(tmp_path):
    api = DashboardApi(SqliteStore.open(tmp_path))

    status, headers, body = api.handle_get("/api/missing")

    assert status == 404
    assert headers["content-type"] == "application/json"
    assert json.loads(body.decode("utf-8")) == {"error": "not_found"}


def test_dashboard_api_returns_runtime_snapshot(tmp_path):
    store = SqliteStore.open(tmp_path)
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/runtime")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert "runtime_status" in payload
    assert "queue_counts" in payload


def test_dashboard_api_returns_runtime_monitor_report(tmp_path):
    store = SqliteStore.open(tmp_path)
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/runtime/monitor")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert payload["ok"] is False
    assert payload["failures"] == ["runtime_unhealthy"]
    assert "heartbeat_missing" in payload["details"]["runtime_unhealthy"]["errors"]
    assert "runtime_snapshot_missing" in payload["details"]["runtime_unhealthy"]["errors"]


def test_dashboard_api_runs_one_runtime_cycle_with_injected_runner(tmp_path):
    store = SqliteStore.open(tmp_path)
    called = {"count": 0}

    def run_once():
        called["count"] += 1
        from annotation_pipeline_skill.core.runtime import RuntimeConfig
        from annotation_pipeline_skill.runtime.snapshot import build_runtime_snapshot

        snapshot = build_runtime_snapshot(store, RuntimeConfig())
        store.save_runtime_snapshot(snapshot)
        return snapshot

    api = DashboardApi(store, runtime_once=run_once)

    status, _headers, body = api.handle_post("/api/runtime/run-once", b"{}")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert called["count"] == 1
    assert payload["ok"] is True
    assert "runtime_status" in payload["snapshot"]


def test_dashboard_api_returns_409_when_runtime_runner_is_unavailable(tmp_path):
    api = DashboardApi(SqliteStore.open(tmp_path))

    status, _headers, body = api.handle_post("/api/runtime/run-once", b"{}")

    assert status == 409
    assert json.loads(body.decode("utf-8")) == {"error": "runtime_runner_unavailable"}


def test_dashboard_api_returns_readiness_report_for_project(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.ACCEPTED
    store.save_task(task)
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/readiness?project=pipe")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert payload["project_id"] == "pipe"
    assert payload["accepted_count"] == 1
    assert payload["recommended_next_action"] == "fix_export_blockers"


def test_dashboard_api_returns_outbox_summary(tmp_path):
    from annotation_pipeline_skill.core.models import OutboxRecord
    from annotation_pipeline_skill.core.states import OutboxKind

    store = SqliteStore.open(tmp_path)
    store.save_outbox(OutboxRecord.new(task_id="task-1", kind=OutboxKind.SUBMIT, payload={"result": {}}))
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/outbox")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert payload["counts"] == {"dead_letter": 0, "pending": 1, "sent": 0}
    assert payload["records"][0]["task_id"] == "task-1"


def test_dashboard_api_filters_outbox_summary_by_project(tmp_path):
    from annotation_pipeline_skill.core.models import OutboxRecord
    from annotation_pipeline_skill.core.states import OutboxKind

    store = SqliteStore.open(tmp_path)
    alpha = Task.new(task_id="alpha-1", pipeline_id="project-alpha", source_ref={"kind": "jsonl"})
    beta = Task.new(task_id="beta-1", pipeline_id="project-beta", source_ref={"kind": "jsonl"})
    store.save_task(alpha)
    store.save_task(beta)
    store.save_outbox(OutboxRecord.new(task_id="alpha-1", kind=OutboxKind.SUBMIT, payload={}))
    store.save_outbox(OutboxRecord.new(task_id="beta-1", kind=OutboxKind.SUBMIT, payload={}))
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/outbox?project=project-beta")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert payload["counts"] == {"dead_letter": 0, "pending": 1, "sent": 0}
    assert [record["task_id"] for record in payload["records"]] == ["beta-1"]


def test_dashboard_api_returns_task_detail_with_source_attempts_artifacts_events_and_feedback(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="task-1",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"text": "Alice joined OpenAI."}},
    )
    task.status = TaskStatus.PENDING
    event = transition_task(task, TaskStatus.ANNOTATING, actor="test", reason="started", stage="annotation")
    artifact = ArtifactRef.new(
        task_id="task-1",
        kind="annotation_result",
        path="artifact_payloads/task-1/annotation_result.json",
        content_type="application/json",
        metadata={"provider": "local_codex"},
    )
    payload_path = tmp_path / artifact.path
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text('{"text":"{\\"entities\\":[{\\"text\\":\\"Alice\\"}]}"}\n', encoding="utf-8")
    attempt = Attempt(
        attempt_id="attempt-1",
        task_id="task-1",
        index=1,
        stage="annotation",
        status=AttemptStatus.SUCCEEDED,
        provider_id="local_codex",
        model="gpt-5.4-mini",
        artifacts=[artifact],
    )
    feedback = FeedbackRecord.new(
        task_id="task-1",
        attempt_id="attempt-1",
        source_stage=FeedbackSource.QC,
        severity=FeedbackSeverity.WARNING,
        category="boundary",
        message="Check entity span boundary.",
        target={"entity": "Alice"},
        suggested_action="manual_edit",
        created_by="qc",
    )
    store.save_task(task)
    store.append_event(event)
    store.append_artifact(artifact)
    store.append_attempt(attempt)
    store.append_feedback(feedback)
    store.append_feedback_discussion(
        FeedbackDiscussionEntry.new(
            task_id="task-1",
            feedback_id=feedback.feedback_id,
            role="annotator",
            stance="partial_agree",
            message="I agree with the boundary issue only.",
            created_by="annotator",
        )
    )
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/tasks/task-1")

    payload = json.loads(body.decode("utf-8"))
    assert status == 200
    assert payload["task"]["source_ref"]["payload"]["text"] == "Alice joined OpenAI."
    assert payload["attempts"][0]["provider_id"] == "local_codex"
    assert payload["artifacts"][0]["payload"]["text"] == '{"entities":[{"text":"Alice"}]}'
    assert payload["events"][0]["next_status"] == "annotating"
    assert payload["feedback"][0]["message"] == "Check entity span boundary."
    assert payload["feedback_discussions"][0]["stance"] == "partial_agree"
    assert payload["feedback_consensus"]["can_accept_by_consensus"] is False


def test_dashboard_api_posts_feedback_discussion_and_accepts_by_consensus(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.QC
    feedback = FeedbackRecord.new(
        task_id="task-1",
        attempt_id="attempt-1",
        source_stage=FeedbackSource.QC,
        severity=FeedbackSeverity.WARNING,
        category="span",
        message="Span boundary issue.",
        target={"entity": "OpenAI"},
        suggested_action="manual_annotation",
        created_by="qc",
    )
    store.save_task(task)
    store.append_feedback(feedback)
    api = DashboardApi(store)

    status, _headers, body = api.handle_post(
        "/api/tasks/task-1/feedback-discussions",
        json.dumps(
            {
                "feedback_id": feedback.feedback_id,
                "role": "qc",
                "stance": "agree",
                "message": "Annotator and QC agree on the final span.",
                "agreed_points": ["span corrected"],
                "consensus": True,
                "created_by": "qc",
            }
        ).encode("utf-8"),
    )

    payload = json.loads(body.decode("utf-8"))
    assert status == 200
    assert payload["feedback_consensus"]["can_accept_by_consensus"] is True
    assert store.load_task("task-1").status is TaskStatus.ACCEPTED
    assert store.list_events("task-1")[-1].reason == "feedback consensus accepted by annotator and qc"


def test_dashboard_api_posts_human_review_decision(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)
    api = DashboardApi(store)

    status, _headers, body = api.handle_post(
        "/api/tasks/task-1/human-review",
        json.dumps(
            {
                "action": "request_changes",
                "actor": "algorithm-engineer",
                "feedback": "Use batch code to apply the new rule.",
                "correction_mode": "batch_code_update",
            }
        ).encode("utf-8"),
    )

    payload = json.loads(body.decode("utf-8"))
    assert status == 200
    assert payload["task"]["status"] == "annotating"
    assert payload["decision"]["action"] == "request_changes"
    assert store.list_events("task-1")[-1].reason == "human review requested annotator changes"


def test_dashboard_api_updates_task_qc_policy_and_appends_event(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="task-1",
        pipeline_id="pipe",
        source_ref={"kind": "jsonl", "payload": {"rows": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}},
        metadata={
            "row_count": 3,
            "qc_policy": {
                "mode": "all_rows",
                "required_correct_rows": 3,
                "feedback_loop": "annotator_may_accept_or_dispute_qc_items",
            },
        },
    )
    task.status = TaskStatus.PENDING
    store.save_task(task)
    api = DashboardApi(store)

    status, _headers, body = api.handle_put(
        "/api/tasks/task-1/qc-policy",
        json.dumps({"mode": "sample_count", "sample_count": 2, "actor": "algorithm-engineer"}).encode("utf-8"),
    )

    payload = json.loads(body.decode("utf-8"))
    assert status == 200
    assert payload["task"]["metadata"]["qc_policy"]["mode"] == "sample_count"
    assert payload["task"]["metadata"]["qc_policy"]["sample_count"] == 2
    assert store.load_task("task-1").metadata["qc_policy"]["required_correct_rows"] == 2
    assert store.list_events("task-1")[-1].reason == "qc policy updated"
    assert store.list_events("task-1")[-1].metadata["previous_qc_policy"]["mode"] == "all_rows"


def test_dashboard_api_rejects_invalid_task_qc_policy(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    store.save_task(task)
    api = DashboardApi(store)

    status, _headers, body = api.handle_put(
        "/api/tasks/task-1/qc-policy",
        json.dumps({"mode": "sample_ratio", "sample_ratio": 1.5}).encode("utf-8"),
    )

    payload = json.loads(body.decode("utf-8"))
    assert status == 400
    assert payload["error"] == "invalid_qc_policy"
    assert store.list_events("task-1") == []


def test_dashboard_api_returns_config_files_and_can_update_allowed_yaml(tmp_path):
    store = SqliteStore.open(tmp_path)
    (tmp_path / "callbacks.yaml").write_text("callbacks: []\n", encoding="utf-8")
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/config")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    ids = {item["id"] for item in payload["files"]}
    assert ids == {"annotators.yaml", "workflow.yaml", "external_tasks.yaml", "callbacks.yaml"}
    # annotation rules are versioned DB documents, not an editable config file
    assert "annotation_rules.yaml" not in ids
    callbacks = next(item for item in payload["files"] if item["id"] == "callbacks.yaml")
    assert callbacks["exists"] is True

    status, _headers, body = api.handle_put(
        "/api/config/callbacks.yaml",
        b"callbacks:\n  - id: updated\n",
    )

    assert status == 200
    assert json.loads(body.decode("utf-8"))["ok"] is True
    assert "updated" in (tmp_path / "callbacks.yaml").read_text(encoding="utf-8")


def test_dashboard_api_rejects_invalid_config_name(tmp_path):
    api = DashboardApi(SqliteStore.open(tmp_path))

    status, _headers, body = api.handle_put("/api/config/../bad.yaml", b"ok: true\n")

    assert status == 404
    assert json.loads(body.decode("utf-8")) == {"error": "config_not_found"}


def test_dashboard_api_returns_event_log_across_tasks(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    event = transition_task(task, TaskStatus.ANNOTATING, actor="test", reason="started", stage="annotation")
    store.save_task(task)
    store.append_event(event)
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/events")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert payload["events"][0]["task_id"] == "task-1"
    assert payload["events"][0]["next_status"] == "annotating"


def test_dashboard_api_filters_event_log_by_project(tmp_path):
    store = SqliteStore.open(tmp_path)
    alpha = Task.new(task_id="alpha-1", pipeline_id="project-alpha", source_ref={"kind": "jsonl"})
    beta = Task.new(task_id="beta-1", pipeline_id="project-beta", source_ref={"kind": "jsonl"})
    alpha.status = TaskStatus.PENDING
    beta.status = TaskStatus.PENDING
    alpha_event = transition_task(alpha, TaskStatus.ANNOTATING, actor="test", reason="started", stage="annotation")
    beta_event = transition_task(beta, TaskStatus.ANNOTATING, actor="test", reason="started", stage="annotation")
    store.save_task(alpha)
    store.save_task(beta)
    store.append_event(alpha_event)
    store.append_event(beta_event)
    api = DashboardApi(store)

    status, _headers, body = api.handle_get("/api/events?project=project-beta")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert [event["task_id"] for event in payload["events"]] == ["beta-1"]


def test_post_human_review_correction_accepts_valid_answer(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-api",
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

    api = DashboardApi(store)
    body = json.dumps({"actor": "r", "answer": {"entities": []}, "note": "ok"}).encode("utf-8")
    status, _headers, response = api.handle_post("/api/tasks/t-api/human_review_correction", body)
    assert status == 200, response
    payload = json.loads(response)
    assert payload["task"]["status"] == "accepted"


def test_post_human_review_correction_rejects_invalid_answer_400(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-api-bad",
        pipeline_id="p",
        source_ref={
            "kind": "jsonl",
            "payload": {
                "text": "x",
                "annotation_guidance": {
                    "output_schema": {"type": "object", "required": ["entities"]}
                },
            },
        },
    )
    task.status = TaskStatus.HUMAN_REVIEW
    store.save_task(task)

    api = DashboardApi(store)
    body = json.dumps({"actor": "r", "answer": {"wrong": []}, "note": None}).encode("utf-8")
    status, _headers, response = api.handle_post("/api/tasks/t-api-bad/human_review_correction", body)
    assert status == 400
    payload = json.loads(response)
    assert payload["error"] == "schema_validation_failed"
    assert isinstance(payload["details"], list) and payload["details"]


def test_post_human_review_correction_rejects_invalid_state_409(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-api-state",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"annotation_guidance": {"output_schema": {"type": "object"}}}},
    )
    task.status = TaskStatus.PENDING  # not HUMAN_REVIEW
    store.save_task(task)

    api = DashboardApi(store)
    body = json.dumps({"actor": "r", "answer": {}, "note": None}).encode("utf-8")
    status, _headers, response = api.handle_post("/api/tasks/t-api-state/human_review_correction", body)
    assert status == 409
    payload = json.loads(response)
    assert payload["error"] == "invalid_transition"


def test_dashboard_api_manual_move_rejected_to_arbitration(tmp_path):
    """REJECTED → ARBITRATING via the manual-move endpoint is in the whitelist
    and flips the task into the Arbitration queue for the worker to pick up."""
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="t-rej", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.REJECTED
    store.save_task(task)
    api = DashboardApi(store)

    body = json.dumps({"target_status": "arbitrating", "reason": "rearbitrate the rejection"}).encode("utf-8")
    status, _headers, response = api.handle_post("/api/tasks/t-rej/move", body)

    assert status == 200
    assert store.load_task("t-rej").status is TaskStatus.ARBITRATING


def test_dashboard_api_manual_move_blocks_disallowed_transition(tmp_path):
    """PENDING → ANNOTATING is not in the manual-move whitelist (runtime owns
    the in-flight pipeline). The endpoint rejects it with 400."""
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="t-pend", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    store.save_task(task)
    api = DashboardApi(store)

    body = json.dumps({"target_status": "annotating", "reason": "force"}).encode("utf-8")
    status, _headers, response = api.handle_post("/api/tasks/t-pend/move", body)

    assert status == 400
    payload = json.loads(response)
    assert payload["error"] == "manual_move_not_allowed"


def test_dashboard_api_manual_move_blocks_in_flight_task(tmp_path):
    """A task currently held by an active runtime lease cannot be manually
    moved (would race with the worker)."""
    from datetime import datetime, timedelta, timezone
    from annotation_pipeline_skill.core.runtime import RuntimeLease

    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="t-busy", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.REJECTED
    store.save_task(task)
    now = datetime.now(timezone.utc)
    store.save_runtime_lease(
        RuntimeLease(
            lease_id="lease-1",
            task_id="t-busy",
            stage="qc",
            acquired_at=now,
            heartbeat_at=now,
            expires_at=now + timedelta(seconds=600),
            owner="worker-x",
        )
    )
    api = DashboardApi(store)

    body = json.dumps({"target_status": "arbitrating", "reason": "rearbitrate"}).encode("utf-8")
    status, _headers, response = api.handle_post("/api/tasks/t-busy/move", body)

    assert status == 409
    payload = json.loads(response)
    assert payload["error"] == "task_in_flight"

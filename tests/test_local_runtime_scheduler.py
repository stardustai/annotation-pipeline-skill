from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.runtime import RuntimeConfig
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.llm.client import LLMGenerateResult
from annotation_pipeline_skill.runtime.local_scheduler import LocalRuntimeScheduler
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


class StubLLMClient:
    def __init__(self, final_text: str):
        self.final_text = final_text

    async def generate(self, request):
        return LLMGenerateResult(
            runtime="test_runtime",
            provider="test_provider",
            model="test-model",
            continuity_handle=None,
            final_text=self.final_text,
            usage={"total_tokens": 10},
            raw_response={"id": "test"},
            diagnostics={},
        )


def passing_client_factory(target):
    if target == "qc":
        return StubLLMClient('{"passed": true, "summary": "acceptable"}')
    return StubLLMClient('{"labels":[]}')


class FailingLLMClient:
    async def generate(self, request):
        raise RuntimeError("provider unavailable")


class DiagnosticProviderError(RuntimeError):
    def __init__(self):
        super().__init__("local CLI provider failed")
        self.diagnostics = {"stderr": "resume thread not found", "returncode": 1}


class FailingDiagnosticLLMClient:
    async def generate(self, request):
        raise DiagnosticProviderError()


def test_local_runtime_scheduler_drains_pending_within_capacity(tmp_path):
    """run_until_idle keeps recruiting PENDING tasks until the queue empties,
    bounded only by max_concurrent_tasks worker coroutines."""
    store = SqliteStore.open(tmp_path)
    for index in range(1, 4):
        task = Task.new(task_id=f"task-{index}", pipeline_id="pipe", source_ref={"kind": "jsonl"})
        task.status = TaskStatus.PENDING
        store.save_task(task)
    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=passing_client_factory,
        config=RuntimeConfig(max_concurrent_tasks=4),
    )

    snapshot = scheduler.run_until_idle(stage_target="annotation")

    assert snapshot.queue_counts.accepted == 3
    assert snapshot.queue_counts.pending == 0


def test_local_runtime_scheduler_cleans_active_run_after_success(tmp_path):
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    store.save_task(task)
    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=passing_client_factory,
        config=RuntimeConfig(max_concurrent_tasks=1),
    )

    scheduler.run_until_idle(stage_target="annotation")

    assert store.list_active_runs() == []
    assert store.load_task("task-1").status is TaskStatus.ACCEPTED


def test_local_runtime_scheduler_cleans_records_after_failure(tmp_path):
    """A worker that crashes during the pipeline still releases its lease /
    active_run, so the worker pool stays healthy and the failed task remains
    on the queue for the next attempt."""
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    store.save_task(task)
    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=lambda target: FailingLLMClient(),
        config=RuntimeConfig(max_concurrent_tasks=1),
    )

    snapshot = scheduler.run_until_idle(stage_target="annotation", max_tasks=1)

    assert store.list_active_runs() == []
    assert store.list_runtime_leases() == []
    assert snapshot is not None
    assert store.load_runtime_snapshot() == snapshot


def test_scheduler_clears_stale_active_runs_on_construction(tmp_path):
    from datetime import datetime, timedelta, timezone
    from annotation_pipeline_skill.core.runtime import ActiveRun, RuntimeLease

    store = SqliteStore.open(tmp_path)
    fixed_now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    stale_after = 600  # seconds (default)
    # Stale heartbeat: now - (stale_after + 60s).
    stale_heartbeat = fixed_now - timedelta(seconds=stale_after + 60)
    store.save_active_run(
        ActiveRun(
            run_id="run-stale",
            task_id="ghost-task",
            stage="annotation",
            attempt_id="attempt-stale",
            provider_target="annotation",
            started_at=stale_heartbeat,
            heartbeat_at=stale_heartbeat,
        )
    )
    store.save_runtime_lease(
        RuntimeLease(
            lease_id="lease-stale",
            task_id="ghost-task",
            stage="annotation",
            acquired_at=stale_heartbeat,
            heartbeat_at=stale_heartbeat,
            expires_at=stale_heartbeat + timedelta(seconds=stale_after),
            owner="dead-scheduler",
        )
    )

    LocalRuntimeScheduler(
        store=store,
        client_factory=passing_client_factory,
        config=RuntimeConfig(stale_after_seconds=stale_after),
        now_fn=lambda: fixed_now,
    )

    assert store.list_active_runs() == []
    assert store.list_runtime_leases() == []


def test_workers_run_in_parallel(tmp_path):
    """Worker pool runs tasks concurrently. Each LLM call sleeps 0.5s — serial
    execution of 4 tasks would be ~4 * 1.0s = 4s; the pool should finish in
    ~1s (one annotation + one QC round-trip overlapping across workers)."""
    import asyncio as _asyncio
    import time

    sleep_seconds = 0.5

    class SlowClient:
        def __init__(self, final_text: str):
            self.final_text = final_text

        async def generate(self, request):
            await _asyncio.sleep(sleep_seconds)
            return LLMGenerateResult(
                runtime="test_runtime",
                provider="test_provider",
                model="test-model",
                continuity_handle=None,
                final_text=self.final_text,
                usage={"total_tokens": 1},
                raw_response={"id": "test"},
                diagnostics={},
            )

    def slow_factory(target):
        if target == "qc":
            return SlowClient('{"passed": true, "summary": "ok"}')
        return SlowClient('{"labels":[]}')

    store = SqliteStore.open(tmp_path)
    for index in range(1, 5):
        task = Task.new(
            task_id=f"task-{index}",
            pipeline_id="pipe",
            source_ref={"kind": "jsonl"},
        )
        task.status = TaskStatus.PENDING
        store.save_task(task)

    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=slow_factory,
        config=RuntimeConfig(max_concurrent_tasks=8),
    )

    t0 = time.monotonic()
    snapshot = scheduler.run_until_idle(stage_target="annotation")
    wall_seconds = time.monotonic() - t0

    assert snapshot.queue_counts.accepted == 4
    # Serial would be 4 * (0.5 + 0.5) = 4.0s; parallel ~1.0s. Allow generous slack.
    assert wall_seconds < 2.0, f"expected parallel speedup, wall={wall_seconds:.2f}s"


def test_scheduler_does_not_clear_fresh_active_runs_on_construction(tmp_path):
    from datetime import datetime, timedelta, timezone
    from annotation_pipeline_skill.core.runtime import ActiveRun, RuntimeLease

    store = SqliteStore.open(tmp_path)
    fixed_now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    stale_after = 600
    fresh_heartbeat = fixed_now - timedelta(seconds=30)  # well within threshold
    store.save_active_run(
        ActiveRun(
            run_id="run-fresh",
            task_id="live-task",
            stage="annotation",
            attempt_id="attempt-fresh",
            provider_target="annotation",
            started_at=fresh_heartbeat,
            heartbeat_at=fresh_heartbeat,
        )
    )
    store.save_runtime_lease(
        RuntimeLease(
            lease_id="lease-fresh",
            task_id="live-task",
            stage="annotation",
            acquired_at=fresh_heartbeat,
            heartbeat_at=fresh_heartbeat,
            expires_at=fresh_heartbeat + timedelta(seconds=stale_after),
            owner="live-scheduler",
        )
    )

    LocalRuntimeScheduler(
        store=store,
        client_factory=passing_client_factory,
        config=RuntimeConfig(stale_after_seconds=stale_after),
        now_fn=lambda: fixed_now,
    )

    assert len(store.list_active_runs()) == 1
    assert len(store.list_runtime_leases()) == 1


def test_workers_drain_many_tasks_with_small_pool(tmp_path):
    """A pool of just 2 workers still drains 10 PENDING tasks — each worker
    claims the next task as soon as it's free. There's no batch boundary."""
    store = SqliteStore.open(tmp_path)
    for index in range(1, 11):
        task = Task.new(task_id=f"task-{index}", pipeline_id="pipe", source_ref={"kind": "jsonl"})
        task.status = TaskStatus.PENDING
        store.save_task(task)
    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=passing_client_factory,
        config=RuntimeConfig(max_concurrent_tasks=2),
    )

    snapshot = scheduler.run_until_idle(stage_target="annotation")

    assert snapshot.queue_counts.accepted == 10
    assert snapshot.queue_counts.pending == 0


def test_scheduler_preserves_in_flight_tasks_on_init(tmp_path):
    """Scheduler init does NOT touch in-flight task status. ANNOTATING / QC /
    ARBITRATING are all preserved so the smart-resume claim path (and, for
    ARBITRATING, the rearbitration runner) can pick up where the previous
    session left off. Auto-routing ARBITRATING zombies to HR on restart was
    incorrect under the current arbiter rules — ARBITRATING is now a
    legitimate mechanical-retry state, and the per-task
    arbiter_mechanical_retries counter caps the loop without needing
    init-time intervention."""
    from datetime import datetime, timezone
    from annotation_pipeline_skill.core.models import Task as _Task
    from annotation_pipeline_skill.core.states import TaskStatus as _TS

    store = SqliteStore.open(tmp_path)
    now = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    annot = _Task.new(task_id="zombie-annot", pipeline_id="p", source_ref={"kind": "jsonl"})
    annot.status = _TS.ANNOTATING
    arb = _Task.new(task_id="zombie-arb", pipeline_id="p", source_ref={"kind": "jsonl"})
    arb.status = _TS.ARBITRATING
    store.save_task(annot)
    store.save_task(arb)

    LocalRuntimeScheduler(
        store=store,
        client_factory=passing_client_factory,
        config=RuntimeConfig(stale_after_seconds=600),
        now_fn=lambda: now,
    )

    # Both stay in their in-flight status; the claim loop will pick them up.
    assert store.load_task("zombie-annot").status is _TS.ANNOTATING
    assert store.load_task("zombie-arb").status is _TS.ARBITRATING


def test_try_claim_resumes_annotating_to_qc_when_annotation_artifact_exists(tmp_path):
    """An ANNOTATING task with an annotation_result artifact but no qc_result
    after it should be resumed at the QC stage on next claim: status → QC,
    metadata.runtime_next_stage = "qc"."""
    from annotation_pipeline_skill.core.models import ArtifactRef

    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="resume-qc", pipeline_id="p", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.ANNOTATING
    store.save_task(task)
    # Seed an annotation_result artifact — would normally exist from a
    # half-finished pipeline cycle before a restart.
    artifact_path = "artifact_payloads/resume-qc/annotation.json"
    (store.root / artifact_path).parent.mkdir(parents=True, exist_ok=True)
    (store.root / artifact_path).write_text('{"text": "{}"}', encoding="utf-8")
    store.append_artifact(ArtifactRef.new(
        task_id="resume-qc", kind="annotation_result", path=artifact_path,
        content_type="application/json",
    ))

    scheduler = LocalRuntimeScheduler(
        store=store, client_factory=passing_client_factory,
        config=RuntimeConfig(max_concurrent_tasks=1),
    )
    claim = scheduler._try_claim_task("annotation")
    assert claim is not None
    claimed_task, _, _ = claim
    assert claimed_task.status is TaskStatus.QC
    assert claimed_task.metadata.get("runtime_next_stage") == "qc"


def test_try_claim_resets_annotating_to_pending_when_no_annotation_artifact(tmp_path):
    """An ANNOTATING task with NO annotation_result yet must restart from
    annotation — _try_claim_task transitions it to PENDING so a worker picks
    it up via the normal entry path."""
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="resume-pending", pipeline_id="p", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.ANNOTATING
    store.save_task(task)

    scheduler = LocalRuntimeScheduler(
        store=store, client_factory=passing_client_factory,
        config=RuntimeConfig(max_concurrent_tasks=1),
    )
    claim = scheduler._try_claim_task("annotation")
    assert claim is not None
    claimed_task, _, _ = claim
    assert claimed_task.status is TaskStatus.PENDING


def test_delayed_sweep_resets_truly_orphaned_in_flight_tasks(tmp_path):
    """_delayed_sweep_unclaimed_orphans is the safety net: any ANNOTATING /
    QC task with no lease and no active_run gets reset to PENDING.
    """
    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="sweep-me", pipeline_id="p", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.ANNOTATING
    store.save_task(task)

    scheduler = LocalRuntimeScheduler(
        store=store, client_factory=passing_client_factory,
        config=RuntimeConfig(max_concurrent_tasks=1),
    )
    # Don't go through _try_claim_task — exercise the sweep directly.
    scheduler._delayed_sweep_unclaimed_orphans()

    assert store.load_task("sweep-me").status is TaskStatus.PENDING


def test_worker_task_timeout_releases_lease_on_hung_llm_call(tmp_path):
    """If an LLM call hangs forever, the worker's asyncio.wait_for kicks in,
    cancels the task, and the finally clause releases the lease/active_run.
    The task stays claimable for the next worker run."""
    import asyncio

    class HangingLLMClient:
        async def generate(self, request):
            # Simulate an HTTP/CLI call that never returns.
            await asyncio.sleep(60)
            return None  # never reached

    store = SqliteStore.open(tmp_path)
    task = Task.new(task_id="hang-task", pipeline_id="p", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    store.save_task(task)

    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=lambda target: HangingLLMClient(),
        config=RuntimeConfig(
            max_concurrent_tasks=1,
            worker_task_timeout_seconds=1,  # 1s — wait_for fires fast
        ),
    )

    # max_tasks=1 stops the pool after one completion (timeout counts as one)
    scheduler.run_until_idle(stage_target="annotation", max_tasks=1)

    # Lease/active_run released even though the LLM never returned
    assert store.list_runtime_leases() == []
    assert store.list_active_runs() == []


def test_total_bail_cap_circuit_breaker_escalates_to_human_review(tmp_path):
    """Bug 2b regression: when the permanent-error classifier doesn't catch
    a failure mode (corrupted auth surfacing as a generic RuntimeError,
    subprocess hangs that keep timing out, etc.), tasks would loop forever
    in PENDING with growing backoff and never escalate. The TOTAL_BAIL_CAP
    circuit breaker fires on any kind of bail."""
    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    # Seed the bail counter just below the cap so a single failing run trips
    # it — keeps the test fast. The first claim cycle will see PENDING with
    # bail_count=24, run the failing LLM, hit the worker_bail path with
    # bails=25, which equals TOTAL_BAIL_CAP and escalates.
    task.metadata["worker_bail_count"] = LocalRuntimeScheduler.TOTAL_BAIL_CAP - 1
    store = SqliteStore.open(tmp_path)
    store.save_task(task)

    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=lambda target: FailingLLMClient(),
        config=RuntimeConfig(max_concurrent_tasks=1),
    )
    scheduler.run_until_idle(stage_target="annotation", max_tasks=1)

    loaded = store.load_task("task-1")
    assert loaded.status is TaskStatus.HUMAN_REVIEW, (
        f"expected HUMAN_REVIEW after {LocalRuntimeScheduler.TOTAL_BAIL_CAP} "
        f"bails, got {loaded.status}"
    )
    assert loaded.metadata.get("worker_bail_count") >= LocalRuntimeScheduler.TOTAL_BAIL_CAP


def test_stop_mid_task_exits_fast_and_does_not_bump_bail_counter(tmp_path):
    """Bug 1 regression: with the old wait_for-only worker, SIGTERM during a
    long LLM call meant the worker kept waiting until worker_task_timeout
    fired (default 900s). The race-against-stop refactor makes shutdown
    propagate within the cancellation latency of one async hop.

    Also asserts the stop-set exit path does NOT count as a bail — graceful
    restart should not bump every in-flight task's worker_bail_count, which
    would otherwise trip TOTAL_BAIL_CAP after a few clean restarts.
    """
    import asyncio
    import time

    task = Task.new(task_id="task-1", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    initial_bail = 3
    task.metadata["worker_bail_count"] = initial_bail
    store = SqliteStore.open(tmp_path)
    store.save_task(task)

    # Client that sleeps "forever" so the worker is definitely mid-call when
    # stop fires. 30s ceiling is well above the test's tolerance window but
    # safe in case the cancellation path is broken — we don't want a hung
    # test indefinitely.
    class HangingClient:
        async def generate(self, request):
            await asyncio.sleep(30)
            raise AssertionError("HangingClient.generate should have been cancelled")

    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=lambda target: HangingClient(),
        config=RuntimeConfig(
            max_concurrent_tasks=1,
            worker_task_timeout_seconds=60,  # well above the test's stop fire
        ),
    )

    async def drive() -> float:
        stop = asyncio.Event()
        # Give the worker time to claim the task and start the LLM call before
        # we signal stop. 0.3s is plenty for the claim + dispatch overhead.
        async def trigger_stop_after_delay():
            await asyncio.sleep(0.3)
            stop.set()
        trigger = asyncio.create_task(trigger_stop_after_delay())
        t0 = time.monotonic()
        await scheduler.run_forever(stage_target="annotation", stop_event=stop)
        elapsed = time.monotonic() - t0
        await trigger
        return elapsed

    elapsed = asyncio.run(drive())

    # Cancellation latency budget — must exit way faster than the 60s
    # worker_task_timeout, proving the race-against-stop is doing its job.
    assert elapsed < 5.0, (
        f"runtime took {elapsed:.1f}s to honor stop; before the race-against-"
        f"stop fix this was 900s (worker_task_timeout). Should be sub-second."
    )

    loaded = store.load_task("task-1")
    # Stop-mid-task: bail counter MUST NOT increment. The task is interrupted,
    # not failed. Graceful restart that bumped every task's bail count would
    # trip TOTAL_BAIL_CAP after a handful of clean restarts.
    assert loaded.metadata.get("worker_bail_count") == initial_bail, (
        f"stop-mid-task wrongly bumped bail counter from {initial_bail} to "
        f"{loaded.metadata.get('worker_bail_count')}"
    )
    # Lease/active_run released cleanly so next runtime can re-claim.
    assert store.list_runtime_leases() == []
    assert store.list_active_runs() == []

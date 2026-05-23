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


def test_singleton_owner_is_acquired_and_released_around_run_forever(tmp_path):
    """A continuous run_forever invocation writes scheduler_owner.json on
    entry and removes it on clean shutdown. Operators (and the next start
    attempt) can read this file to identify the live owner."""
    import asyncio

    store = SqliteStore.open(tmp_path)
    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=passing_client_factory,
        config=RuntimeConfig(max_concurrent_tasks=1),
    )
    owner_path = scheduler._singleton_owner_path

    async def drive():
        stop = asyncio.Event()

        async def stopper():
            # Let the scheduler's setup write the owner file before we stop.
            await asyncio.sleep(0.05)
            assert owner_path.exists(), "owner sentinel not written on startup"
            stop.set()

        await asyncio.gather(
            scheduler.run_forever(stage_target="annotation", stop_event=stop),
            stopper(),
        )

    asyncio.run(drive())

    # Clean shutdown deletes the owner file so the next start isn't blocked.
    assert not owner_path.exists(), "owner sentinel not removed on shutdown"


def test_run_forever_rejects_second_instance_when_first_is_alive(tmp_path):
    """When a fresh owner sentinel already exists, a second run_forever
    raises SchedulerAlreadyRunningError before workers are spawned. The
    error carries the owner record so the CLI can print PID/host."""
    import asyncio
    import json
    import os

    from annotation_pipeline_skill.runtime.local_scheduler import (
        SchedulerAlreadyRunningError,
    )

    store = SqliteStore.open(tmp_path)
    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=passing_client_factory,
        config=RuntimeConfig(max_concurrent_tasks=1, snapshot_interval_seconds=30),
    )
    # Plant a fresh owner sentinel as if another scheduler is alive.
    owner_path = scheduler._singleton_owner_path
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    owner_path.write_text(
        json.dumps({
            "pid": os.getpid() + 1,  # pretend another process owns it
            "hostname": "test-host",
            "started_at": now_iso,
            "last_heartbeat_at": now_iso,
        }),
        encoding="utf-8",
    )

    try:
        asyncio.run(scheduler.run_forever(stage_target="annotation", stop_when_idle=True))
    except SchedulerAlreadyRunningError as exc:
        assert exc.owner.get("hostname") == "test-host"
        # Did not commit suicide on the planted sentinel.
        assert owner_path.exists()
        return
    # stop_when_idle bypasses the guard, so this call should NOT raise.
    # Run again in continuous mode to confirm the guard fires.
    async def drive_continuous():
        with __import__("pytest").raises(SchedulerAlreadyRunningError) as caught:
            await scheduler.run_forever(stage_target="annotation")
        return caught.value
    err = asyncio.run(drive_continuous())
    assert err.owner.get("hostname") == "test-host"
    assert owner_path.exists()


def test_stale_owner_sentinel_is_overwritten(tmp_path):
    """An owner sentinel older than the staleness threshold (max(2*interval,
    120s)) is presumed to belong to a dead scheduler — the new scheduler
    overwrites it and starts."""
    import asyncio
    import json
    import os
    from datetime import datetime, timedelta, timezone

    store = SqliteStore.open(tmp_path)
    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=passing_client_factory,
        config=RuntimeConfig(max_concurrent_tasks=1, snapshot_interval_seconds=30),
    )
    owner_path = scheduler._singleton_owner_path
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    stale_iso = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat()
    owner_path.write_text(
        json.dumps({
            "pid": os.getpid() + 1,
            "hostname": "ghost-host",
            "started_at": stale_iso,
            "last_heartbeat_at": stale_iso,
        }),
        encoding="utf-8",
    )

    async def drive():
        stop = asyncio.Event()

        async def stopper():
            await asyncio.sleep(0.05)
            # By now this scheduler should have claimed the sentinel.
            payload = json.loads(owner_path.read_text(encoding="utf-8"))
            assert payload["pid"] == os.getpid()
            assert payload["hostname"] != "ghost-host"
            stop.set()

        await asyncio.gather(
            scheduler.run_forever(stage_target="annotation", stop_event=stop),
            stopper(),
        )

    asyncio.run(drive())


def test_transient_rate_limit_bails_never_escalate_to_human_review(tmp_path):
    """Rate-limit (429) errors are transient — they must NOT trigger HUMAN_REVIEW
    even after more bails than TOTAL_BAIL_CAP.  Tasks should stay in PENDING
    with exponential backoff so the scheduler can retry once the provider
    rate-limit window resets.

    Regression: previously `total_cap_hit = bails >= TOTAL_BAIL_CAP` applied
    regardless of error type, routing tasks to HR after 10 consecutive 429s.
    """
    from annotation_pipeline_skill.llm.local_cli import LocalCLIExecutionError

    class RateLimitClient:
        """Raises a 429 wrapped exactly as AnthropicSDKClient does."""
        async def generate(self, request):
            raise LocalCLIExecutionError(
                "local CLI provider failed",
                {
                    "runtime": "anthropic_sdk",
                    "error_event": {
                        "api_error_status": 429,
                        "result_text": "rate limit exceeded",
                    },
                },
            )

    task = Task.new(task_id="task-rl", pipeline_id="pipe", source_ref={"kind": "jsonl"})
    task.status = TaskStatus.PENDING
    # Seed well above TOTAL_BAIL_CAP; the next bail should still not hit HR.
    task.metadata["worker_bail_count"] = LocalRuntimeScheduler.TOTAL_BAIL_CAP + 5
    store = SqliteStore.open(tmp_path)
    store.save_task(task)

    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=lambda target: RateLimitClient(),
        config=RuntimeConfig(max_concurrent_tasks=1),
    )
    scheduler.run_until_idle(stage_target="annotation", max_tasks=1)

    loaded = store.load_task("task-rl")
    assert loaded.status is TaskStatus.PENDING, (
        f"rate-limit bails must not escalate to HR; got {loaded.status}"
    )
    assert loaded.next_retry_at is not None, "task should have a backoff next_retry_at"


# ---------------------------------------------------------------------------
# Task 5: scheduler must route arbiter_uncertain_needs_second to second arbiter
# ---------------------------------------------------------------------------

def test_scheduler_routes_uncertain_flag_to_second_arbiter(tmp_path):
    """An ARBITRATING task with arbiter_uncertain_needs_second=True must be
    routed to _resolve_uncertain_arbiter, not the normal run_task_async path."""
    import asyncio
    from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime

    store = SqliteStore.open(tmp_path)
    task = Task.new(
        task_id="t-sched-unc",
        pipeline_id="p",
        source_ref={"kind": "jsonl", "payload": {"text": "hello"}},
    )
    task.status = TaskStatus.ARBITRATING
    task.metadata["arbiter_uncertain_needs_second"] = True
    store.save_task(task)

    resolver_called: list[str] = []

    class _PatchedRuntime(SubagentRuntime):
        async def _resolve_uncertain_arbiter_async(self, t):
            resolver_called.append(t.task_id)
            t.metadata.pop("arbiter_uncertain_needs_second", None)
            t.status = TaskStatus.ACCEPTED
            self.store.save_task(t)

        async def _resolve_first_arbiter_divergence_async(self, t):
            raise AssertionError("wrong resolver called")

    runtime = _PatchedRuntime(store=store, client_factory=lambda _t: None)
    scheduler = LocalRuntimeScheduler(
        store=store,
        client_factory=lambda _t: None,
        config=RuntimeConfig(max_concurrent_tasks=1),
        runtime=runtime,
    )
    scheduler.run_until_idle(max_tasks=1)

    assert resolver_called == ["t-sched-unc"], (
        f"_resolve_uncertain_arbiter must be called for the flagged task; got {resolver_called}"
    )


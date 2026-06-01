from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.runtime import ActiveRun, RuntimeConfig, RuntimeLease, RuntimeSnapshot
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.llm.client import LLMClient
from annotation_pipeline_skill.runtime.snapshot import build_runtime_snapshot
from annotation_pipeline_skill.runtime.subagent_cycle import SubagentRuntime
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


class SchedulerAlreadyRunningError(RuntimeError):
    """Raised when a continuous scheduler tries to start while another live
    scheduler owns this project's store. Carries the owner record so the
    caller can surface PID/hostname/age in an operator-facing message."""

    def __init__(self, message: str, *, owner: dict | None = None) -> None:
        super().__init__(message)
        self.owner = owner or {}


class LocalRuntimeScheduler:
    """Worker-pool runtime.

    ``max_concurrent_tasks`` worker coroutines run in parallel. Each worker
    claims one PENDING (or QC-resume) task at a time, runs the full
    annotation→validation→QC pipeline through ``SubagentRuntime``, releases
    its lease, then immediately claims the next task. A separate observer
    coroutine snapshots the runtime state every
    ``snapshot_interval_seconds``. There are no cycles, no batches, and no
    drain barriers — a slow task only ties up one worker slot.
    """

    # After this many CONSECUTIVE permanent (4xx) bails — e.g. 400
    # ContextWindowExceeded that retrying against the same model cannot fix —
    # escalate to HUMAN_REVIEW. Transient (5xx / rate-limit / timeout) bails
    # never escalate; they keep backing off indefinitely because the upstream
    # is expected to self-heal. A single transient bail resets the permanent
    # counter so an intermittent 4xx after gateway downtime doesn't escalate.
    PERMANENT_BAIL_CAP: int = 5
    # Circuit breaker on TOTAL bails (transient + permanent + unclassified).
    # Catches failure modes the permanent classifier doesn't recognize — e.g.
    # corrupted local auth, subprocess hangs that timeout-bail forever, exotic
    # provider error shapes. After this many consecutive bails on the same
    # task, give up and escalate to HUMAN_REVIEW. Reset on any successful
    # transition out of ANNOTATING.
    #
    # Lowered 25 → 10 (2026-05-22): in production the high cap caused
    # hundreds of "hopeless" tasks (broken UTF-8, malformed source rows,
    # repeated truncated JSON from MiniMax) to occupy worker slots for
    # hours retrying without ever progressing. By the time bail count
    # reaches ~5-6, the task has tried every fallback profile in the
    # rotation; 10 gives enough headroom for transient outages while
    # cutting effective worker-time-wasted-per-hopeless-task by 60%.
    TOTAL_BAIL_CAP: int = 10

    # Provider health probe: every PROBE_INTERVAL the observer sends a minimal
    # ping ("ok") through each common target. If a target returns a permanent
    # error (4xx), write a provider_health alert to alerts.jsonl. Catches
    # things like the 2026-05-21 DeepSeek 402 "Insufficient Balance" outage
    # an hour BEFORE the queue would have surfaced it through HR.
    PROBE_INTERVAL_SECONDS: int = 300
    PROBE_TARGETS: tuple[str, ...] = (
        "annotation", "qc", "arbiter", "arbiter_secondary", "fallback",
    )

    # Cap on simultaneously-running ARBITRATING tasks as a fraction of
    # max_concurrent_tasks. Today's incident: ~100 tasks queued in
    # ARBITRATING after a bulk rewind, each arbiter call ran 200-800s on
    # deepseek_flash, all 16 worker slots saturated on arbiter calls,
    # annotator + qc completely starved. Capping arbiter slots to half
    # the pool guarantees forward progress on fresh annotation work.
    # 0.0 disables the cap entirely (any worker can pick up any role).
    ARBITER_SLOT_FRACTION: float = 0.5

    # Hot-reload: how often the dedicated config-watcher task re-reads the
    # LLM profiles yaml to pick up changes to `max_concurrent_tasks`
    # without restarting the scheduler. yaml stat() is cheap (single
    # syscall, no I/O); we only fully re-parse when mtime actually
    # changed, so polling fast is safe. Runs as its own asyncio.Task so
    # the cadence is independent of the much slower observer
    # (snapshot_interval_seconds, default 30s).
    CONFIG_RELOAD_INTERVAL_SECONDS: float = 2.0

    def __init__(
        self,
        store: SqliteStore,
        client_factory: Callable[[str], LLMClient],
        config: RuntimeConfig,
        *,
        now_fn: Callable[[], datetime] | None = None,
        profiles_yaml_path: Path | None = None,
        registry: Any | None = None,
        client_builder: Callable[[Any], LLMClient] | None = None,
        runtime: Any | None = None,
    ):
        self.store = store
        # Two binding modes for client_factory:
        #   - legacy: caller supplies `client_factory`; scheduler uses it
        #     verbatim. Targets are baked in at startup, no hot-reload.
        #   - hot-reload: caller supplies `registry` (the resolved LLMRegistry)
        #     + `client_builder` (a function that turns one profile into a
        #     client, default = LocalCLIClient). Scheduler builds its own
        #     factory that calls `self._registry.resolve(target)` every
        #     invocation, so a yaml-change → registry-swap takes effect on
        #     the next worker request without restarting.
        self._registry = registry
        self._client_builder = client_builder
        if registry is not None and client_builder is not None:
            self.client_factory: Callable[[str], LLMClient] = self._resolve_client_from_registry
        else:
            self.client_factory = client_factory
        self.config = config
        self._runtime_override = runtime
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        # Source-of-truth for hot-reloadable `max_concurrent_tasks`. Workers
        # check this on every claim loop; observer rewrites it when yaml
        # changes. Initial value comes from the frozen config; if the yaml
        # path is provided, the observer can grow/shrink it at runtime.
        self._runtime_max_workers: int = config.max_concurrent_tasks
        self._profiles_yaml_path = profiles_yaml_path
        self._profiles_yaml_mtime: float = 0.0
        # Per-target cooldown for the provider health probe — once the
        # observer ticks at PROBE_INTERVAL_SECONDS it walks PROBE_TARGETS
        # in order; this tracks last successful probe timestamps so a
        # single observer tick doesn't re-fire all targets in a tight
        # loop after restart.
        self._last_provider_probe: float = 0.0
        # Pre-flight cleanup at init: clear stale leases/active_runs from
        # a previous (dead) scheduler. In-flight tasks (ANNOTATING / QC /
        # ARBITRATING) are NOT touched here — _try_claim_task picks them up
        # via smart-resume and either resumes from the right stage or
        # transitions back to PENDING. ARBITRATING in particular is a
        # legitimate mechanical-retry state under the current arbiter
        # rules (subagent_cycle._handle_arbiter_mechanical_fail) and the
        # per-task arbiter_mechanical_retries counter in task metadata
        # caps the retries; auto-routing to HR on restart would discard
        # that budget and contradict the "HR = arbiter uncertain only"
        # invariant.
        self._clear_stale_records()

    def _resolve_client_from_registry(self, target: str) -> "LLMClient":
        """Used as ``self.client_factory`` when the scheduler was built in
        registry-binding mode (``registry`` + ``client_builder`` supplied
        at init). Resolves the target against the CURRENT registry — which
        may have been swapped by the yaml hot-reload — so target re-bindings
        in projects/llm_profiles.yaml take effect without a restart.
        """
        if self._registry is None or self._client_builder is None:
            raise RuntimeError(
                "scheduler not in registry-binding mode; cannot resolve target"
            )
        return self._client_builder(self._registry.resolve(target))

    def _structured_output_targets(self) -> frozenset[str]:
        """Return the set of target names whose profile has structured_output=True."""
        if self._registry is None:
            return frozenset()
        return frozenset(
            target
            for target, profile_name in self._registry.targets.items()
            if self._registry.profiles.get(profile_name) and
               self._registry.profiles[profile_name].structured_output
        )

    def _clear_stale_records(self) -> None:
        """Drop leases / active_runs whose heartbeat is older than the stale window.

        Called at construction so a freshly-restarted scheduler doesn't count
        leftover rows from a previously-killed instance toward in-flight
        capacity. Fresh rows from a still-live scheduler are left alone.
        """
        threshold = self._now_fn() - timedelta(seconds=self.config.stale_after_seconds)
        cleared_leases = 0
        cleared_runs = 0
        for lease in self.store.list_runtime_leases():
            if lease.heartbeat_at < threshold:
                self.store.delete_runtime_lease(lease.lease_id)
                cleared_leases += 1
        for run in self.store.list_active_runs():
            if run.heartbeat_at < threshold:
                self.store.delete_active_run(run.run_id)
                cleared_runs += 1
        if cleared_leases or cleared_runs:
            import sys
            print(
                f"[scheduler] cleared {cleared_leases} stale leases, "
                f"{cleared_runs} stale active_runs",
                file=sys.stderr,
            )

    async def _probe_providers(self) -> None:
        """Ping each configured target with a 1-token request. If the
        provider returns ``is_error=true`` with a 4xx api_error_status,
        append a ``provider_health`` row to ``<store_root>/alerts.jsonl``.
        Runs at ``PROBE_INTERVAL_SECONDS`` cadence; idempotent if called
        more often (returns early).

        Catches the case the 2026-05-21 outage made painful: DeepSeek
        returned 402 silently on every arbiter call, no operator alert,
        ~200 tasks misrouted to HR before anyone noticed. With this
        probe an alert lands ~5 min after the wallet empties, regardless
        of whether any task happens to invoke that target.
        """
        import time
        import json as _json
        now = time.time()
        if now - self._last_provider_probe < self.PROBE_INTERVAL_SECONDS:
            return
        self._last_provider_probe = now

        from annotation_pipeline_skill.llm.client import LLMGenerateRequest

        # We import here, not at module-top, to dodge a circular import
        # (subagent_cycle imports local_scheduler indirectly).
        ping_request = LLMGenerateRequest(
            instructions="Respond with exactly the JSON object {\"ok\":true} and nothing else.",
            prompt="ping",
            continuity_handle=None,
            response_format={"type": "json_object"},
        )
        for target in self.PROBE_TARGETS:
            try:
                client = self.client_factory(target)
            except Exception:  # noqa: BLE001 — target may not be configured
                continue
            # codex_cli authenticates via OAuth auth.json, not an API key,
            # and spawns a full CLI process — the probe's ephemeral isolated
            # home lacks project-trust config, causing false-positive failures.
            # Balance / auth errors don't apply to OAuth sessions, so skip.
            if getattr(getattr(client, "profile", None), "runtime", None) == "codex_cli":
                continue
            try:
                result = await client.generate(ping_request)
            except Exception as exc:  # noqa: BLE001
                diag = getattr(exc, "diagnostics", None) or {}
                err_ev = diag.get("error_event") if isinstance(diag, dict) else None
                status = err_ev.get("api_error_status") if isinstance(err_ev, dict) else None
                msg = (err_ev.get("result_text") if isinstance(err_ev, dict) else None) or str(exc)[:200]
                self._write_health_alert(target, status, msg, exc_class=type(exc).__name__)
                continue
            finally:
                close = getattr(client, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:  # noqa: BLE001
                        pass
            # Successful subprocess BUT result may carry an error event
            # (claude CLI exits rc=0 ONLY on success; rc=1 raises above,
            # so if we got here result is clean. Still defensively check.)
            diag = getattr(result, "diagnostics", None) or {}
            err_ev = diag.get("error_event") if isinstance(diag, dict) else None
            if isinstance(err_ev, dict):
                status = err_ev.get("api_error_status")
                msg = err_ev.get("result_text") or "(no message)"
                self._write_health_alert(target, status, msg, exc_class=None)

    def _write_health_alert(
        self,
        target: str,
        api_error_status: Any,
        message: str,
        *,
        exc_class: str | None,
    ) -> None:
        """Best-effort write to <store_root>/alerts.jsonl. Also prints to
        stderr so an operator tailing the runtime log sees the banner.
        Not deduped — each probe failure is one line; downstream tooling
        can dedup on (target, api_error_status) if it cares.
        """
        import sys
        from annotation_pipeline_skill.runtime.alerts import append_alert
        banner = (
            f"\n🚨 PROVIDER HEALTH  target={target}  status={api_error_status}  "
            f"class={exc_class or 'is_error'}\n   {str(message)[:300]}\n"
            f"   (probe fired by scheduler; operator action required)\n"
        )
        try:
            print(banner, file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001
            pass
        append_alert(self.store.root, {
            "ts": self._now_fn().isoformat(),
            "kind": "provider_health",
            "target": target,
            "api_error_status": api_error_status,
            "exception_class": exc_class,
            "message": str(message)[:500],
        })
    def _delayed_sweep_unclaimed_orphans(self) -> None:
        """Catch ANNOTATING / QC tasks that no worker claimed during the
        settle window. Called periodically by the observer coroutine.

        A task is an "unclaimed orphan" if:
          - status is ANNOTATING or QC
          - has NO runtime_lease pointing at it
          - has NO active_run pointing at it

        Such a task slipped past ``_try_claim_task`` (e.g., because its
        artifacts were in a weird state, or it lost a race) — reset to
        PENDING so the natural pipeline retries it from the top.
        """
        from annotation_pipeline_skill.core.transitions import InvalidTransition, transition_task

        leased = {l.task_id for l in self.store.list_runtime_leases()}
        active = {r.task_id for r in self.store.list_active_runs()}
        recovered = 0
        for task in self.store.list_tasks_by_status({TaskStatus.ANNOTATING, TaskStatus.QC}):
            if task.task_id in leased or task.task_id in active:
                continue
            # QC tasks with runtime_next_stage=qc are waiting for QC-only re-claim,
            # not orphaned annotation tasks — leave them for the scheduler's normal
            # pickup loop instead of resetting to PENDING.
            if task.status is TaskStatus.QC and task.metadata.get("runtime_next_stage") == "qc":
                continue
            try:
                event = transition_task(
                    task, TaskStatus.PENDING,
                    actor="scheduler",
                    reason=f"delayed sweep: still unclaimed in {task.status.value} after settle window; resetting to pending",
                    stage="recovery",
                    metadata={"recovery": "delayed_sweep", "previous_status": task.status.value},
                )
            except InvalidTransition:
                continue
            self.store.save_task(task)
            self.store.append_event(event)
            recovered += 1
        if recovered:
            import sys
            print(f"[scheduler] delayed-sweep reset {recovered} unclaimed orphans → pending", file=sys.stderr)

    def _reap_stale_leases(self) -> None:
        """Reclaim leases whose expires_at has passed. Called periodically by
        the observer coroutine.

        When a worker hangs (subprocess deadlocked, asyncio.wait_for failed
        to propagate cancellation), its lease stays held forever and the
        task is invisible to other workers' claim cycles. The reaper
        deletes the stale lease + active_run and resets ANNOTATING tasks
        back to PENDING so another worker can pick them up.

        Idempotent with the worker's own finally-block cleanup: if the
        hung worker eventually returns and tries to delete the lease/run,
        the delete is a no-op (already gone).
        """
        from annotation_pipeline_skill.core.transitions import InvalidTransition, transition_task

        now = datetime.now(timezone.utc)
        reaped = 0
        reset = 0
        stale_lease_task_ids: set[str] = set()
        for lease in self.store.list_runtime_leases():
            if lease.expires_at and lease.expires_at < now:
                self.store.delete_runtime_lease(lease.lease_id)
                stale_lease_task_ids.add(lease.task_id)
                reaped += 1
        for run in self.store.list_active_runs():
            if run.task_id in stale_lease_task_ids:
                self.store.delete_active_run(run.run_id)
        for task_id in stale_lease_task_ids:
            try:
                task = self.store.load_task(task_id)
            except (FileNotFoundError, KeyError):
                continue
            if task.status is not TaskStatus.ANNOTATING:
                # QC / ARBITRATING tasks are picked up by smart resume on the
                # next claim cycle without needing a status reset.
                continue
            try:
                event = transition_task(
                    task, TaskStatus.PENDING,
                    actor="scheduler",
                    reason="stale-lease reaper: lease expired with task in annotating; resetting to pending",
                    stage="recovery",
                    metadata={"recovery": "stale_lease_reap", "previous_status": "annotating"},
                )
            except InvalidTransition:
                continue
            self.store.save_task(task)
            self.store.append_event(event)
            reset += 1
        if reaped:
            import sys
            print(
                f"[scheduler] stale-lease reaper: dropped {reaped} expired lease(s), "
                f"reset {reset} ANNOTATING task(s) to pending",
                file=sys.stderr,
            )

    async def run_forever(
        self,
        *,
        stage_target: str = "annotation",
        stop_event: asyncio.Event | None = None,
        max_tasks: int | None = None,
        stop_when_idle: bool = False,
    ) -> int:
        """Spin up the worker pool and run until ``stop_event`` is set.

        - ``max_tasks``: optional ceiling — stop after that many task
          completions (useful for sized smoke runs).
        - ``stop_when_idle``: stop once no PENDING tasks remain and no worker
          is busy (used by tests and one-shot CLI helpers).

        Returns the number of tasks processed.
        """
        # Singleton guard: only one continuous scheduler per project store.
        # Skip for on-demand modes (stop_when_idle=True is used by the
        # dashboard "run once" button and by tests, which legitimately
        # co-exist with a long-running scheduler — the SQLite lease layer
        # handles their race).
        owns_singleton = not stop_when_idle
        if owns_singleton:
            self._acquire_singleton()

        stop = stop_event or asyncio.Event()
        runtime = self._runtime_override or SubagentRuntime(
            store=self.store,
            client_factory=self.client_factory,
            max_qc_rounds=self.config.max_qc_rounds,
            config=self.config,
            structured_output_targets=self._structured_output_targets(),
        )

        completed = 0
        busy_workers = 0

        async def worker(worker_idx: int) -> None:
            nonlocal completed, busy_workers
            while not stop.is_set():
                # Hot-reload self-cap: if the desired worker pool shrank
                # (observer reloaded yaml, max_concurrent_tasks went down),
                # the surplus workers idle here instead of claiming. Cheap
                # 5s sleep — no DB hit. They wake up immediately if the
                # pool grows back.
                if worker_idx >= self._runtime_max_workers:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass
                    continue
                claim = self._try_claim_task(stage_target)
                if claim is None:
                    if stop_when_idle and busy_workers == 0:
                        stop.set()
                        return
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass
                    continue
                task, lease, run = claim
                busy_workers += 1
                # Defaults read by the bail-counter logic in `finally`.
                # Overwritten by the except branches when a worker exception
                # actually fires.
                last_exception_was_permanent = False
                last_exception_was_transient = False
                last_exception_summary = ""
                # Set when stop fires mid-task. Tells `finally` to skip the
                # bail-counter logic — shutdown isn't a failure, and counting
                # it would bump every in-flight task's bail count on every
                # graceful restart, potentially tripping TOTAL_BAIL_CAP for
                # tasks that were progressing fine.
                stop_signaled_mid_task = False
                try:
                    # Hard upper bound on a single task's run. If an LLM call
                    # (codex subprocess, HTTP stream) hangs past this, we cancel
                    # so the finally clause releases the lease/active_run and
                    # the task gets recycled instead of zombifying the worker.
                    if (
                        task.status is TaskStatus.ARBITRATING
                        and task.metadata.get("prior_verifier_first_arbiter_divergent")
                    ):
                        # Divergent-flag path: first arbiter accepted an annotation
                        # that diverges from project prior; second arbiter adjudicates.
                        work_coro = runtime._resolve_first_arbiter_divergence_async(task)
                    elif (
                        task.status is TaskStatus.ARBITRATING
                        and task.metadata.get("arbiter_uncertain_needs_second")
                    ):
                        # Uncertain-flag path: first arbiter was tentative/unsure;
                        # second arbiter gets a fresh attempt before escalating to HR.
                        work_coro = runtime._resolve_uncertain_arbiter_async(task)
                    else:
                        work_coro = runtime.run_task_async(task, stage_target=stage_target)
                    # Race the task against the stop event so SIGTERM is
                    # honored immediately, not after `worker_task_timeout`.
                    # Plain `wait_for(coro, timeout=...)` ignores stop entirely:
                    # 24 workers all mid-call meant shutdown sat for ~15min
                    # before timeout reaped them, and SIGTERM looked dead.
                    work_task = asyncio.create_task(work_coro)
                    stop_wait_task = asyncio.create_task(stop.wait())
                    try:
                        done, _pending = await asyncio.wait(
                            {work_task, stop_wait_task},
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=self.config.worker_task_timeout_seconds,
                        )
                        if not done:
                            # Hard timeout — same semantics as old wait_for.
                            work_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await work_task
                            raise asyncio.TimeoutError
                        if stop_wait_task in done and work_task not in done:
                            # Shutdown signaled mid-task. Cancel work and drain
                            # — _generate_codex catches CancelledError and
                            # SIGKILLs the subprocess, SDK runtimes close
                            # their httpx connections on cancel, so the
                            # provider call dies promptly.
                            stop_signaled_mid_task = True
                            work_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await work_task
                        else:
                            # Work completed (success or exception). Re-raise
                            # any exception so the except handlers below fire
                            # with the same semantics as the old wait_for path.
                            await work_task
                    finally:
                        if not stop_wait_task.done():
                            stop_wait_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await stop_wait_task
                except asyncio.TimeoutError:
                    import sys
                    print(
                        f"[scheduler] worker_task_timeout: task={task.task_id} "
                        f"after {self.config.worker_task_timeout_seconds}s; "
                        f"releasing lease and recycling",
                        file=sys.stderr,
                    )
                    last_exception_was_permanent = False
                except Exception as worker_exc:  # noqa: BLE001
                    # SubagentRuntime captures errors on the attempt record; the
                    # worker only needs to release records and keep going.
                    # We do classify the exception though — permanent errors
                    # (404 wrong endpoint, 401 bad key, 400 schema) skip the
                    # 5-bail retry dance and go straight to HR.
                    from annotation_pipeline_skill.runtime.subagent_cycle import (
                        _is_provider_permanent_error,
                        _is_provider_transient_error,
                    )
                    last_exception_was_permanent = _is_provider_permanent_error(worker_exc)
                    last_exception_was_transient = _is_provider_transient_error(worker_exc)
                    # Mirror of the arbiter wrap site: ProviderCallError
                    # only stringifies to "local CLI provider failed"; the
                    # actual cause (auth, model name, OOM, API 5xx) is in
                    # .diagnostics (returncode + last 4KB of stderr). Without
                    # this the `last_provider_error` task metadata gives
                    # operators no clue what to fix.
                    diag = getattr(worker_exc, "diagnostics", None)
                    tail = ""
                    if isinstance(diag, dict):
                        rc = diag.get("returncode")
                        err = (diag.get("stderr") or "")
                        if isinstance(err, str):
                            err = err.strip().replace("\n", " | ")[-300:]
                        tail = f" rc={rc} stderr={err!r}"
                    last_exception_summary = (
                        f"{type(worker_exc).__name__}: {str(worker_exc)[:200]}{tail}"
                    )
                finally:
                    self.store.delete_active_run(run.run_id)
                    self.store.delete_runtime_lease(lease.lease_id)
                    # Shutdown-mid-task: release records and exit, but DO NOT
                    # touch the bail counter. The task is interrupted, not
                    # failed; the stale-lease reaper will pick it back up on
                    # the next runtime start. Skipping the whole bail block
                    # (rather than `continue`-ing) keeps control flow out of
                    # the finally clause — that's a SyntaxWarning hazard in
                    # newer Pythons and a hard error in some versions.
                    do_bail_logic = not stop_signaled_mid_task
                    # If run_task_async bailed before reaching a terminal
                    # transition (LLM error, timeout, parse failure), the
                    # task is left in whatever in-flight state it last hit
                    # (typically ANNOTATING from the early annotator
                    # transition). Without resetting, the next claim cycle
                    # sees ANNOTATING-without-lease and triggers
                    # _prepare_annotating_for_resume → PENDING → re-claim →
                    # PENDING→ANNOTATING → LLM fails again, infinite loop
                    # at ~700 spurious audit events/min. Reset here closes
                    # the loop: next claim sees a clean PENDING task.
                    try:
                        latest = self.store.load_task(task.task_id) if do_bail_logic else None
                        # Only reset ANNOTATING. QC with runtime_next_stage=qc
                        # is a legitimate "wait for QC re-claim" exit state
                        # used by the QC parse-error retry path; leaving it
                        # alone lets the next worker run QC-only as designed.
                        if latest is not None and latest.status is TaskStatus.ANNOTATING:
                            from annotation_pipeline_skill.core.transitions import (
                                InvalidTransition,
                                transition_task,
                            )
                            # Per-task worker-bail counter + exponential
                            # backoff via next_retry_at. Transient (5xx /
                            # rate-limit / timeout) failures stay in PENDING
                            # with growing backoff — the upstream is
                            # expected to self-heal. Permanent (4xx) failures
                            # are capped: after PERMANENT_BAIL_CAP they
                            # escalate to HR with the provider error message
                            # so an operator can decide (fix config / swap
                            # model / give up on the task). Without the cap,
                            # ContextWindowExceeded and similar task-level
                            # permanent errors loop forever at 10-min
                            # intervals.
                            from datetime import timedelta
                            bails = int(latest.metadata.get("worker_bail_count", 0)) + 1
                            latest.metadata["worker_bail_count"] = bails
                            if last_exception_summary:
                                latest.metadata["last_provider_error"] = last_exception_summary

                            # Escalate to HR when permanent-error budget is
                            # exhausted. Transient errors never escalate —
                            # they just keep backing off.
                            permanent_bails = int(latest.metadata.get(
                                "worker_permanent_bail_count", 0
                            ))
                            if last_exception_was_permanent:
                                permanent_bails += 1
                                latest.metadata["worker_permanent_bail_count"] = permanent_bails
                            else:
                                # Reset permanent counter on any transient bail —
                                # an intermittent 4xx after gateway downtime
                                # shouldn't escalate.
                                latest.metadata["worker_permanent_bail_count"] = 0
                                permanent_bails = 0

                            # Two escalation triggers, either hits → HR:
                            #   1. permanent error cap (5 consecutive 4xx-like)
                            #   2. total bail cap — circuit breaker for failure
                            #      modes the permanent classifier doesn't catch
                            #      (corrupted auth, subprocess hangs, exotic
                            #      provider error shapes). Without (2),
                            #      misclassified-transient bugs loop forever.
                            # Transient errors (429 / 5xx) are explicitly
                            # excluded from the total cap: they self-heal
                            # and should back off indefinitely rather than
                            # escalate to HR.
                            permanent_cap_hit = (
                                last_exception_was_permanent
                                and permanent_bails >= self.PERMANENT_BAIL_CAP
                            )
                            total_cap_hit = (
                                not last_exception_was_transient
                                and bails >= self.TOTAL_BAIL_CAP
                            )
                            if permanent_cap_hit or total_cap_hit:
                                latest.next_retry_at = None
                                escalation_kind = (
                                    "permanent_bail_cap"
                                    if permanent_cap_hit
                                    else "total_bail_cap"
                                )
                                if permanent_cap_hit:
                                    reason = (
                                        f"worker bailed with permanent provider error "
                                        f"{permanent_bails} consecutive times "
                                        f"(cap={self.PERMANENT_BAIL_CAP}); routing to "
                                        f"human review "
                                        f"(last: {(last_exception_summary or '')[:200]})"
                                    )
                                else:
                                    reason = (
                                        f"worker bailed {bails} consecutive times "
                                        f"(total cap={self.TOTAL_BAIL_CAP}); routing "
                                        f"to human review — likely systemic issue "
                                        f"the permanent classifier didn't catch "
                                        f"(last: {(last_exception_summary or '')[:200]})"
                                    )
                                try:
                                    event = transition_task(
                                        latest, TaskStatus.HUMAN_REVIEW,
                                        actor="scheduler",
                                        reason=reason,
                                        stage="recovery",
                                        metadata={"recovery": escalation_kind,
                                                  "previous_status": "annotating",
                                                  "worker_bail_count": bails,
                                                  "worker_permanent_bail_count": permanent_bails,
                                                  "permanent_error": last_exception_was_permanent},
                                    )
                                    self.store.save_task(latest)
                                    self.store.append_event(event)
                                except InvalidTransition:
                                    pass
                            else:
                                base = 60 if last_exception_was_permanent else 30
                                backoff_seconds = min(base * bails, 600)
                                next_retry_at = self._now_fn() + timedelta(seconds=backoff_seconds)
                                latest.next_retry_at = next_retry_at
                                # Clear annotation session handles so the next
                                # attempt sends a full prompt rather than a
                                # context-free delta against a lost KV cache
                                # (e.g. after a vllm restart mid-annotation).
                                latest.metadata.pop("continuity_handle", None)
                                latest.metadata.pop("_ann_sent_feedback_ids", None)
                                try:
                                    event = transition_task(
                                        latest, TaskStatus.PENDING,
                                        actor="scheduler",
                                        reason=(
                                            f"worker bailed mid-annotation "
                                            f"({'permanent' if last_exception_was_permanent else 'transient'} "
                                            f"provider error, bail #{bails}"
                                            + (f", permanent #{permanent_bails}/{self.PERMANENT_BAIL_CAP}"
                                               if last_exception_was_permanent else "")
                                            + f"); holding {backoff_seconds}s before next claim"
                                        ),
                                        stage="recovery",
                                        metadata={"recovery": "worker_bail",
                                                  "previous_status": "annotating",
                                                  "worker_bail_count": bails,
                                                  "worker_permanent_bail_count": permanent_bails,
                                                  "permanent_error": last_exception_was_permanent,
                                                  "backoff_seconds": backoff_seconds},
                                    )
                                    self.store.save_task(latest)
                                    self.store.append_event(event)
                                except InvalidTransition:
                                    pass
                    except (FileNotFoundError, KeyError):
                        pass
                    busy_workers -= 1
                    completed += 1
                    if max_tasks is not None and completed >= max_tasks:
                        stop.set()

        async def observer() -> None:
            self._write_snapshot()
            # Settle window: give workers time to claim ANNOTATING / QC
            # orphans (via _try_claim_task's resume logic) before sweeping
            # any leftovers back to PENDING. Tasks the workers DO claim get
            # natural pipeline progression; tasks they DON'T (artifact
            # weirdness, lost races) get reset by the sweep.
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.config.resume_settle_seconds)
            except asyncio.TimeoutError:
                pass
            if not stop.is_set():
                self._delayed_sweep_unclaimed_orphans()
            self._write_snapshot()
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.config.snapshot_interval_seconds)
                except asyncio.TimeoutError:
                    pass
                # Periodic recovery: reclaim leases held by hung workers, then
                # sweep any orphaned tasks that ended up lease-less without
                # a worker picking them up.
                self._reap_stale_leases()
                self._delayed_sweep_unclaimed_orphans()
                # Provider health probe — cheap 1-token pings every
                # PROBE_INTERVAL_SECONDS to surface auth/balance issues
                # ahead of the queue. Internally rate-limited so calling
                # at every observer tick is safe.
                try:
                    await self._probe_providers()
                except Exception:  # noqa: BLE001 — never let probe failure tank observer
                    pass
                # Skip the snapshot/heartbeat write on a shutdown tick so the
                # heartbeat file is not re-created right before the process
                # exits (which would leave a fresh heartbeat behind and make
                # the health check show "healthy" for the staleness window).
                if stop.is_set():
                    break
                self._write_snapshot()
                if owns_singleton:
                    self._refresh_singleton()

        # Worker pool starts at the configured size and can grow at
        # runtime when the observer detects the yaml's
        # `max_concurrent_tasks` went up. Held in a list so the observer
        # can append new worker tasks (`worker_tasks.append(...)`); they're
        # NOT awaited by `gather()` below — only the initial cohort is —
        # so dynamically-added workers run as background tasks bound by
        # the same `stop` event.
        worker_tasks: list[asyncio.Task] = [
            asyncio.create_task(worker(i))
            for i in range(self._runtime_max_workers)
        ]
        # Pool ceiling — highest worker_idx ever spawned. The observer
        # uses this to know how many new tasks to create when scaling up.
        spawned_workers = self._runtime_max_workers

        async def reload_max_workers_from_yaml() -> None:
            """Re-read the LLM profiles yaml mtime; if changed, parse it
            and apply (a) the new ``max_concurrent_tasks`` and (b) the
            new ``targets`` mapping. Scale-up spawns extra worker tasks;
            scale-down is implicit (workers self-skip when their idx is
            over the new cap). Target swap takes effect on the next
            `client_factory(target)` call — workers in flight finish
            their current LLM call against the OLD target binding, then
            pick up the new one on the next claim."""
            nonlocal spawned_workers
            path = self._profiles_yaml_path
            if path is None:
                return
            try:
                mtime = path.stat().st_mtime
            except OSError:
                return
            if mtime == self._profiles_yaml_mtime:
                return
            self._profiles_yaml_mtime = mtime
            try:
                from annotation_pipeline_skill.llm.profiles import load_llm_registry
                reg = load_llm_registry(path)
            except Exception:  # noqa: BLE001 — never let reload tank observer
                return
            import sys
            # (a) max_concurrent_tasks
            new_max = getattr(reg, "max_concurrent_tasks", None)
            if isinstance(new_max, int) and new_max > 0 and new_max != self._runtime_max_workers:
                old = self._runtime_max_workers
                self._runtime_max_workers = new_max
                print(
                    f"[scheduler] hot-reload max_concurrent_tasks: {old} → {new_max}",
                    file=sys.stderr,
                    flush=True,
                )
                from annotation_pipeline_skill.runtime.alerts import append_alert
                append_alert(self.store.root, {
                    "ts": self._now_fn().isoformat(),
                    "kind": "config_reload",
                    "message": f"max_concurrent_tasks: {old} → {new_max}",
                })
                while spawned_workers < new_max:
                    worker_tasks.append(asyncio.create_task(worker(spawned_workers)))
                    spawned_workers += 1
            # (b) targets (only effective when scheduler was constructed
            # in registry-binding mode; legacy `client_factory=lambda...`
            # mode keeps targets frozen because the caller's closure can't
            # be reached from here)
            if self._registry is not None and self._client_builder is not None:
                old_targets = dict(getattr(self._registry, "targets", {}))
                new_targets = dict(getattr(reg, "targets", {}))
                if old_targets != new_targets:
                    changes = []
                    for k in sorted(set(old_targets) | set(new_targets)):
                        if old_targets.get(k) != new_targets.get(k):
                            changes.append(f"{k}: {old_targets.get(k)!r} → {new_targets.get(k)!r}")
                    print(
                        "[scheduler] hot-reload targets:\n  "
                        + "\n  ".join(changes),
                        file=sys.stderr,
                        flush=True,
                    )
                    from annotation_pipeline_skill.runtime.alerts import append_alert
                    append_alert(self.store.root, {
                        "ts": self._now_fn().isoformat(),
                        "kind": "config_reload",
                        "message": ", ".join(changes),
                    })
                self._registry = reg

        async def config_watcher() -> None:
            """Fast-tick watcher dedicated to hot-reloading
            `max_concurrent_tasks` from the yaml. Independent of the
            (slow, snapshot-cadence) observer so concurrency edits show
            up within seconds, not 30s+."""
            while not stop.is_set():
                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=self.CONFIG_RELOAD_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    pass
                if stop.is_set():
                    return
                try:
                    await reload_max_workers_from_yaml()
                except Exception:  # noqa: BLE001 — never let reload tank watcher
                    pass

        observer_task = asyncio.create_task(observer())
        config_watcher_task = asyncio.create_task(config_watcher())
        try:
            try:
                await asyncio.gather(*worker_tasks, observer_task, config_watcher_task)
            except asyncio.CancelledError:
                stop.set()
                await asyncio.gather(
                    *worker_tasks, observer_task, config_watcher_task, return_exceptions=True
                )
                raise
        finally:
            if owns_singleton:
                self._release_singleton()
            # Always clear the heartbeat on exit so the health check
            # immediately shows unhealthy rather than waiting for the
            # staleness window to expire.
            self.store.clear_runtime_heartbeat()
        return completed

    def run_until_idle(self, stage_target: str = "annotation", *, max_tasks: int | None = None) -> RuntimeSnapshot:
        """Synchronous helper: run the pool until PENDING is drained.

        Convenience for tests and the ``run-cycle`` / ``runtime once`` CLI
        commands. Equivalent to ``run_forever(stop_when_idle=True)`` plus a
        final snapshot write.
        """
        asyncio.run(self.run_forever(stage_target=stage_target, stop_when_idle=True, max_tasks=max_tasks))
        return self._write_snapshot()

    def _try_claim_task(self, stage_target: str) -> tuple[Task, RuntimeLease, ActiveRun] | None:
        """Pick the next runnable task and reserve it.

        Returns ``None`` when no task is runnable. Workers are all in the
        same asyncio event loop with a synchronous SQLite store, so this
        method does not need a lock — only one worker runs at a time
        between awaits.

        Claimable statuses:
          PENDING        — fresh tasks; worker runs the full pipeline
          QC (resume)    — tasks whose annotation is done; metadata flag
                           ``runtime_next_stage=qc`` directs the runtime to
                           skip back to QC. Set either by the auto pipeline
                           when transitioning ANNOTATING→QC, or by this
                           function as part of resume-on-restart.
          ANNOTATING     — orphaned mid-pipeline (after a runtime restart).
                           If the task has an annotation_result artifact, we
                           promote it back to QC with runtime_next_stage=qc
                           so the worker resumes from QC instead of re-running
                           annotation. If there's no annotation_result yet, we
                           reset to PENDING so a worker re-runs annotation.
          ARBITRATING    — human-dragged HR / REJECTED cards (re-arbitrate
                           flow) — the worker calls the arbiter on them.
        """
        candidates = self.store.list_tasks_by_status(
            {TaskStatus.PENDING, TaskStatus.QC, TaskStatus.ARBITRATING, TaskStatus.ANNOTATING}
        )
        # Re-prioritize so downstream stages drain first. Without this,
        # workers see PENDING candidates (typically thousands) before
        # ARBITRATING candidates (handfuls), so newly-injected arbitrating
        # tasks pile up indefinitely while PENDING keeps refilling the
        # in-flight pool. The fix: pick the most-advanced stage first so
        # task-progress flows toward terminal states instead of backlogging
        # at arbitration. Tie-break order matches workflow direction:
        #   ARBITRATING > QC (resume) > ANNOTATING > PENDING
        # The arbiter_cap below still limits parallel arbitration to
        # ARBITER_SLOT_FRACTION; with PENDING starvation prevented by that
        # cap (workers can always fall through to PENDING once arbiter slots
        # are full), this only changes pickup ORDER, not the slot quota.
        _STAGE_PRIORITY = {
            TaskStatus.ARBITRATING: 0,
            TaskStatus.QC: 1,
            TaskStatus.ANNOTATING: 2,
            TaskStatus.PENDING: 3,
        }
        candidates.sort(
            key=lambda t: (_STAGE_PRIORITY.get(t.status, 99), t.created_at)
        )
        # Skip tasks that another worker is already running. Without this,
        # the ANNOTATING resume branch below would bounce a live in-flight
        # task back to PENDING on every claim attempt, producing a flood of
        # spurious annotating→pending→annotating audit events and inflating
        # apparent throughput. Match what _delayed_sweep_unclaimed_orphans
        # already does for the same reason.
        leased = {l.task_id for l in self.store.list_runtime_leases()}
        active = {r.task_id for r in self.store.list_active_runs()}
        # Role quota: count how many ARBITRATING tasks are currently
        # leased so we can skip new arbitrating candidates once the cap
        # is hit. Falls through to PENDING/QC candidates, keeping
        # annotation work flowing even when the arbiter queue is deep.
        arbiter_cap = (
            max(1, int(self.config.max_concurrent_tasks * self.ARBITER_SLOT_FRACTION))
            if self.ARBITER_SLOT_FRACTION > 0
            else self.config.max_concurrent_tasks
        )
        in_flight_ids = leased | active
        arbiter_in_flight = 0
        if in_flight_ids and arbiter_cap < self.config.max_concurrent_tasks:
            arbiter_in_flight = sum(
                1 for t in candidates
                if t.task_id in in_flight_ids and t.status is TaskStatus.ARBITRATING
            )
        now_ts = self._now_fn()
        for candidate in candidates:
            if candidate.task_id in leased or candidate.task_id in active:
                continue
            if candidate.status is TaskStatus.QC and candidate.metadata.get("runtime_next_stage") != "qc":
                continue
            # Enforce arbiter slot cap — skip ARBITRATING candidates when
            # at quota; loop continues to PENDING/QC.
            if (
                candidate.status is TaskStatus.ARBITRATING
                and arbiter_in_flight >= arbiter_cap
            ):
                continue
            # Respect bail-backoff: tasks the worker bailed on get a
            # next_retry_at stamped on them so they don't immediately get
            # re-claimed and re-fail against the same broken upstream.
            if candidate.next_retry_at is not None and candidate.next_retry_at > now_ts:
                continue
            if candidate.status is TaskStatus.ANNOTATING:
                # Genuine restart-orphan path: inspect artifacts to choose entry stage.
                self._prepare_annotating_for_resume(candidate)
                # _prepare_annotating_for_resume may have transitioned the
                # task — reload to get current status.
                candidate = self.store.load_task(candidate.task_id)
            acquired_at = self._now_fn()
            lease = self._lease_for(candidate, acquired_at)
            if not self.store.save_runtime_lease(lease):
                continue
            run = self._active_run_for(candidate, stage_target, acquired_at, lease.lease_id)
            self.store.save_active_run(run)
            return candidate, lease, run
        return None

    def _prepare_annotating_for_resume(self, task: Task) -> None:
        """Decide whether an orphaned ANNOTATING task resumes from QC or
        restarts from annotation, based on which artifacts already exist.

        - Has annotation_result + no qc_result for the same attempt → set
          ``runtime_next_stage=qc`` and transition status to QC. The worker
          will pick the QC-only resume path in SubagentRuntime.
        - Otherwise → transition to PENDING so a worker re-runs annotation
          (and the prelabel-reuse fast path picks up any pre-existing
          annotation_result on attempt 0).
        """
        from annotation_pipeline_skill.core.transitions import InvalidTransition, transition_task

        # Human Review's request_changes route puts the task into ANNOTATING
        # explicitly to force a re-annotation. The artifact heuristic below
        # would otherwise see the (now-rejected) annotation_result and bounce
        # straight to QC — defeating the operator's intent. Honor the marker
        # and restart from annotation (PENDING).
        if task.metadata.get("hr_request_changes"):
            task.metadata.pop("hr_request_changes", None)
            try:
                event = transition_task(
                    task, TaskStatus.PENDING,
                    actor="scheduler",
                    reason="resume after HR request_changes: restart from annotation",
                    stage="recovery",
                    metadata={"resume": "hr_request_changes_to_pending"},
                )
            except InvalidTransition:
                return
            self.store.save_task(task)
            self.store.append_event(event)
            return

        artifacts = self.store.list_artifacts(task.task_id)
        # An annotation artifact exists AND no qc_result follows it in
        # insertion order → resume at QC. ``list_artifacts`` returns
        # artifacts in insertion (seq) order, so a positional walk
        # captures the temporal relationship without a dedicated seq field.
        last_annotation_idx = None
        for idx, art in enumerate(artifacts):
            if art.kind == "annotation_result":
                last_annotation_idx = idx
        resume_qc = False
        if last_annotation_idx is not None:
            seen_qc_after = any(
                a.kind == "qc_result"
                for a in artifacts[last_annotation_idx + 1:]
            )
            resume_qc = not seen_qc_after
        try:
            if resume_qc:
                task.metadata["runtime_next_stage"] = "qc"
                event = transition_task(
                    task, TaskStatus.QC,
                    actor="scheduler",
                    reason="resume on restart: annotation artifact already present, skipping to QC",
                    stage="recovery",
                    metadata={"resume": "annotating_to_qc"},
                )
            else:
                event = transition_task(
                    task, TaskStatus.PENDING,
                    actor="scheduler",
                    reason="resume on restart: no annotation artifact yet, restart from annotation",
                    stage="recovery",
                    metadata={"resume": "annotating_to_pending"},
                )
        except InvalidTransition:
            return
        self.store.save_task(task)
        self.store.append_event(event)

    def _write_snapshot(self) -> RuntimeSnapshot:
        now = self._now_fn()
        self.store.save_runtime_heartbeat(now)
        snapshot = build_runtime_snapshot(self.store, self.config, now=now)
        self.store.save_runtime_snapshot(snapshot)
        return snapshot

    # ----- Singleton-owner sentinel ------------------------------------------------
    # A continuous (run_forever, stop_when_idle=False) scheduler claims the
    # project's store by writing scheduler_owner.json with its pid, hostname,
    # start time, and a refreshed last_heartbeat_at. A second `runtime run`
    # invocation reads this file and refuses to start if the heartbeat is still
    # fresh. The threshold mirrors snapshot.py's stale-heartbeat convention
    # (2× snapshot_interval_seconds, floored at 120s) so a process whose
    # observer loop has died for two ticks is treated as gone.
    #
    # On-demand callers (`run_until_idle` from the dashboard "run once" button,
    # tests, smoke scripts) bypass acquisition: their race with a long-running
    # scheduler is already bounded by the SQLite lease IntegrityError.
    @property
    def _singleton_owner_path(self) -> Path:
        return self.store.root / "runtime" / "scheduler_owner.json"

    def _singleton_stale_threshold_seconds(self) -> int:
        return max(self.config.snapshot_interval_seconds * 2, 120)

    def _load_singleton_owner(self) -> dict | None:
        path = self._singleton_owner_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_singleton_owner(self, *, started_at: datetime, heartbeat_at: datetime) -> None:
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": started_at.isoformat(),
            "last_heartbeat_at": heartbeat_at.isoformat(),
        }
        path = self._singleton_owner_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def _acquire_singleton(self) -> None:
        existing = self._load_singleton_owner()
        if existing is not None:
            last_heartbeat_raw = existing.get("last_heartbeat_at")
            if isinstance(last_heartbeat_raw, str):
                try:
                    last_heartbeat = datetime.fromisoformat(last_heartbeat_raw)
                except ValueError:
                    last_heartbeat = None
            else:
                last_heartbeat = None
            if last_heartbeat is not None:
                age = (self._now_fn() - last_heartbeat).total_seconds()
                threshold = self._singleton_stale_threshold_seconds()
                if age < threshold:
                    raise SchedulerAlreadyRunningError(
                        f"another scheduler is already running on this project: "
                        f"pid={existing.get('pid')} host={existing.get('hostname')} "
                        f"heartbeat {age:.0f}s ago (threshold {threshold}s). "
                        f"If you believe it is dead, wait {threshold - age:.0f}s for "
                        f"the heartbeat to age out, or delete "
                        f"{self._singleton_owner_path}.",
                        owner=existing,
                    )
        # Either no prior owner, or its heartbeat is stale → overwrite.
        now = self._now_fn()
        self._write_singleton_owner(started_at=now, heartbeat_at=now)

    def _refresh_singleton(self) -> None:
        # Preserve started_at across heartbeats so operators can see uptime.
        existing = self._load_singleton_owner() or {}
        started_at_raw = existing.get("started_at")
        try:
            started_at = (
                datetime.fromisoformat(started_at_raw)
                if isinstance(started_at_raw, str)
                else self._now_fn()
            )
        except ValueError:
            started_at = self._now_fn()
        self._write_singleton_owner(started_at=started_at, heartbeat_at=self._now_fn())

    def _release_singleton(self) -> None:
        # Only release if WE own it; otherwise we'd clobber a different live
        # scheduler that might exist after a crash-during-acquire scenario.
        existing = self._load_singleton_owner()
        if existing and existing.get("pid") == os.getpid():
            try:
                self._singleton_owner_path.unlink()
            except FileNotFoundError:
                pass

    def _lease_for(self, task: Task, acquired_at: datetime) -> RuntimeLease:
        lease_id = f"lease-{uuid4().hex}"
        return RuntimeLease(
            lease_id=lease_id,
            task_id=task.task_id,
            stage="qc" if task.status is TaskStatus.QC and task.metadata.get("runtime_next_stage") == "qc" else "annotation",
            acquired_at=acquired_at,
            heartbeat_at=acquired_at,
            expires_at=acquired_at + timedelta(seconds=self.config.stale_after_seconds),
            owner="local-runtime-scheduler",
            metadata={"runtime": "local_file"},
        )

    def _active_run_for(self, task: Task, stage_target: str, started_at: datetime, lease_id: str) -> ActiveRun:
        run_stage = "qc" if task.status is TaskStatus.QC and task.metadata.get("runtime_next_stage") == "qc" else "annotation"
        return ActiveRun(
            run_id=f"run-{uuid4().hex}",
            task_id=task.task_id,
            stage=run_stage,
            attempt_id=f"{task.task_id}-attempt-{task.current_attempt + 1}",
            provider_target="qc" if run_stage == "qc" else stage_target,
            started_at=started_at,
            heartbeat_at=started_at,
            metadata={"lease_id": lease_id},
        )

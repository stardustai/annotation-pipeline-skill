"""Prompt construction for annotator and QC subagents.

Extracted from SubagentRuntime to allow independent testing and re-use.
The six methods extracted are:
  - build_conventions_block  (was _build_conventions_block)
  - build_annotation_prompt  (was _annotation_prompt)
  - delta_feedback_items     (was _delta_feedback_items)
  - snapshot_sent_feedback   (was _snapshot_sent_feedback)
  - build_qc_prompt          (was _qc_prompt)
  - slim_annotation_payload  (was _slim_annotation_payload)

Two helpers that they depend on are also brought along:
  - _artifact_context
  - _read_artifact_payload

These helpers only use the store object, so they transfer cleanly.
"""
from __future__ import annotations

import json
from typing import Any

from robust_json import loads as _robust_json_loads

from annotation_pipeline_skill.core.models import ArtifactRef, Task
from annotation_pipeline_skill.core.schema_validation import resolve_output_schema
from annotation_pipeline_skill.services.feedback_service import build_feedback_bundle
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _parse_llm_json(text: str) -> Any:
    """Thin wrapper around the robust JSON parser (same as in subagent_cycle)."""
    return _robust_json_loads(text)


class AnnotationPromptBuilder:
    """Builds annotation and QC prompts, including conventions blocks."""

    def __init__(
        self,
        store: SqliteStore,
        project_id: str,
        config: Any,
    ):
        self._store = store
        self._project_id = project_id
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_annotation_prompt(
        self, task: Task, *, continuation_handle: str | None = None
    ) -> str:
        """Build the per-task annotation prompt — the user-message body.

        Returns JSON; the schema is INTENTIONALLY omitted (it's
        task-agnostic and now lives in the system prompt so it's part of
        the stable prefix vLLM can cache). The system prompt is byte-
        stable across tasks of the same project; this user payload is
        the variable per-task portion.

        Key ordering still matters within this payload for same-task
        multi-turn calls — stable head (task source rows) first,
        per-turn mutating tail (prior_artifacts, feedback_bundle) last.
        sort_keys would put `feedback_bundle` (most volatile) at byte 0;
        relying on dict insertion order preserves the stable→volatile
        layout.
        """
        if continuation_handle is None:
            return json.dumps(
                {
                    # Stable per task (source rows, task_id, annotator id).
                    "task": self._task_payload(task),
                    # Mutating per turn — kept at the tail so the head stays
                    # bytes-identical across turns of the same task.
                    "prior_artifacts": self._artifact_context(task.task_id),
                    "feedback_bundle": build_feedback_bundle(self._store, task.task_id),
                },
            )
        # Continuation turn: only send unseen feedback items.
        return json.dumps(
            {"feedback_bundle": {"items": self.delta_feedback_items(task)}},
        )

    def build_qc_prompt(self, task: Task, annotation_artifact: ArtifactRef) -> str:
        """Build the QC prompt JSON — user-message body.

        output_schema is no longer in the JSON (moved to system prompt
        for prefix-cache locality, same reason as build_annotation_prompt).
        """
        return json.dumps(
            {
                "task": self._task_payload(task),
                "annotation_artifact": {
                    **annotation_artifact.to_dict(),
                    "payload": self.slim_annotation_payload(annotation_artifact),
                },
                "feedback_bundle": build_feedback_bundle(self._store, task.task_id),
            },
        )

    def build_conventions_block(self, task: Task) -> str | None:
        """Look up entity conventions for this project and build a prompt block.

        Returns a string block to inject into annotator/QC/arbiter instructions,
        or None if no matching conventions exist.
        """
        from annotation_pipeline_skill.services.entity_convention_service import (
            EntityConventionService,
        )
        from annotation_pipeline_skill.services.row_mask_service import (
            apply_masks_to_task,
        )
        try:
            # Filter masked rows BEFORE concatenating, so masked content
            # can't surface in the convention-matcher's substring search.
            mtask = apply_masks_to_task(self._store, task)
            payload = (
                mtask.source_ref.get("payload")
                if isinstance(mtask.source_ref, dict)
                else None
            )
            rows = payload.get("rows") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                return None
            combined = "\n".join(
                r.get("input", "")
                for r in rows
                if isinstance(r, dict) and isinstance(r.get("input"), str)
            )
        except Exception:  # noqa: BLE001 — never let prompt build fail the task
            return None
        if not combined.strip():
            return None
        svc = EntityConventionService(self._store)
        try:
            matches = svc.find_matches_in_text(task.pipeline_id, combined)
        except Exception:  # noqa: BLE001
            return None
        if not matches:
            return None
        lines = []
        for m in matches[:50]:  # cap at 50 to bound prompt size
            note_suffix = f"  (notes: {m.notes})" if m.notes else ""
            if m.entity_type == "not_an_entity":
                lines.append(
                    f"  - {m.span_original!r} → DO NOT TAG (this span is "
                    f"intentionally NOT an entity in this project)" + note_suffix
                )
            else:
                lines.append(
                    f"  - {m.span_original!r} → entities.{m.entity_type}" + note_suffix
                )
        return (
            "KNOWN ENTITY CONVENTIONS FOR THIS PROJECT (established by prior "
            "QC consensus, arbiter rulings, or human review — apply them so "
            "ambiguous spans get classified consistently across tasks):\n"
            + "\n".join(lines)
        )

    def delta_feedback_items(self, task: Task) -> list[dict]:
        """Return feedback items not yet seen by the agent (for continuation turns)."""
        sent_ids: set[str] = set(task.metadata.get("_ann_sent_feedback_ids", []))
        bundle = build_feedback_bundle(self._store, task.task_id)
        return [
            item
            for item in bundle.get("items", [])
            if item["feedback_id"] not in sent_ids
        ]

    def snapshot_sent_feedback(self, task: Task) -> None:
        """Record which feedback IDs have been sent to the agent."""
        bundle = build_feedback_bundle(self._store, task.task_id)
        task.metadata["_ann_sent_feedback_ids"] = [
            item["feedback_id"] for item in bundle.get("items", [])
        ]

    def slim_annotation_payload(self, artifact: ArtifactRef) -> Any:
        """Return only the parsed annotation rows, dropping bulky metadata.

        Drops ``raw_response`` and other provider-side metadata that
        downstream consumers (QC, arbiter) don't read. This significantly
        reduces QC prompt size on tasks with large LLM response wrappers.
        """
        raw = self._read_artifact_payload(artifact)
        if not isinstance(raw, dict):
            return raw
        text = raw.get("text")
        if not isinstance(text, str):
            return {k: v for k, v in raw.items() if k != "raw_response"}
        try:
            return _parse_llm_json(text)
        except (json.JSONDecodeError, ValueError):
            return {"text": text}

    # ------------------------------------------------------------------
    # Private helpers (duplicated from SubagentRuntime — only use self._store)
    # ------------------------------------------------------------------

    # Allowlist of task.metadata keys that are stable across the multi-turn
    # lifetime of a task and that the LLM might plausibly use. Every other
    # metadata key mutates per turn (bail counts, retry counters, continuity
    # handles, scheduler state, exception classes, ...) and would defeat
    # vLLM's prefix cache if we leaked it into the prompt. The annotator /
    # QC instructions do not read any of those mutating fields anyway, so
    # the whitelist is essentially "what an LLM could conceivably need to
    # know" — currently just qc_policy and prelabeled.
    _STABLE_METADATA_KEYS = frozenset({"qc_policy", "prelabeled"})

    def _task_payload(self, task: Task) -> dict[str, Any]:
        """Build the prompt input dict for a task, masking filtered rows.

        ``task.metadata`` is filtered to ``_STABLE_METADATA_KEYS`` — leaking
        the full metadata dict in would inject scheduler counters
        (worker_bail_count, arbiter_mechanical_retries, continuity_handle,
        _ann_sent_feedback_ids, ...) that mutate every turn, busting the
        prefix-cache prefix from the very first bytes of the task payload.
        """
        from annotation_pipeline_skill.services.row_mask_service import (
            apply_masks_to_task,
        )
        masked = apply_masks_to_task(self._store, task)
        sref = masked.source_ref
        stable_metadata = {
            k: task.metadata[k]
            for k in self._STABLE_METADATA_KEYS
            if k in task.metadata
        }
        return {
            "task_id": task.task_id,
            "source_ref": sref,
            "selected_annotator_id": task.selected_annotator_id,
            "metadata": stable_metadata,
        }

    def _artifact_context(
        self, task_id: str, *, per_kind_limit: int = 1
    ) -> list[dict[str, Any]]:
        """Return recent artifacts grouped by kind, slimmed for prompt context.

        Keeps only the most recent ``per_kind_limit`` artifacts per kind to
        avoid unbounded prompt growth on tasks with many retry loops.
        """
        by_kind: dict[str, list[ArtifactRef]] = {}
        for artifact in self._store.list_artifacts(task_id):
            by_kind.setdefault(artifact.kind, []).append(artifact)
        selected: list[ArtifactRef] = []
        for arts in by_kind.values():
            selected.extend(arts[-per_kind_limit:])
        results: list[dict[str, Any]] = []
        for artifact in selected:
            payload = self._read_artifact_payload(artifact)
            if isinstance(payload, dict):
                payload = {
                    k: v
                    for k, v in payload.items()
                    if k not in {"raw_response", "usage", "diagnostics", "task_id"}
                }
                if artifact.kind == "arbiter_result":
                    payload = {k: v for k, v in payload.items() if k != "items"}
                decision = payload.get("decision")
                if isinstance(decision, dict):
                    drop_decision_keys = {"raw_response"}
                    if artifact.kind == "arbiter_result":
                        drop_decision_keys.add("corrected_annotation")
                    if any(k in decision for k in drop_decision_keys):
                        payload = {
                            **payload,
                            "decision": {
                                k: v
                                for k, v in decision.items()
                                if k not in drop_decision_keys
                            },
                        }
            wrapper = {
                k: v
                for k, v in artifact.to_dict().items()
                if k not in {"path", "content_type"}
            }
            results.append({**wrapper, "payload": payload})
        return results

    def _read_artifact_payload(self, artifact: ArtifactRef) -> Any:
        """Read and JSON-decode an artifact from disk. Returns None if missing."""
        path = self._store.root / artifact.path
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

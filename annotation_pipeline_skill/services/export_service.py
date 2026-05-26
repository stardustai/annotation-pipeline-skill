from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from annotation_pipeline_skill.core.models import ArtifactRef, ExportManifest, OutboxRecord, Task
from annotation_pipeline_skill.core.states import OutboxKind, TaskStatus
from annotation_pipeline_skill.services.row_mask_service import RowMaskService, filter_masked_rows
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


REQUIRED_TRAINING_ROW_FIELDS = [
    "task_id",
    "pipeline_id",
    "source_ref",
    "modality",
    "annotation_requirements",
    "annotation",
    "annotation_artifact_id",
    "annotation_artifact_path",
]


class TrainingDataExportService:
    def __init__(self, store: SqliteStore):
        self.store = store

    def export_jsonl(
        self,
        *,
        project_id: str,
        output_dir: Path,
        export_id: str | None = None,
        enqueue_external_submit: bool = False,
    ) -> ExportManifest:
        export_id = export_id or "export-" + uuid.uuid4().hex[:12]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "training_data.jsonl"

        accepted_tasks = [
            task
            for task in self.store.list_tasks_by_pipeline(project_id)
            if task.status is TaskStatus.ACCEPTED
        ]

        # Bulk-fetch all row masks for accepted tasks so the export
        # silently omits masked (row_index, input/output) entries from
        # both source_ref payloads and annotation payloads.
        mask_svc = RowMaskService(self.store)
        masked_by_task = mask_svc.masked_indices_by_task(
            [t.task_id for t in accepted_tasks]
        )

        # Build rules version timeline so each row can reference the version
        # that was active when its annotation artifact was created.
        rules_timeline = self._rules_version_timeline()

        rows: list[dict[str, Any]] = []
        included: list[str] = []
        excluded: list[dict[str, Any]] = []
        row_errors: list[dict[str, Any]] = []
        artifact_ids: list[str] = []
        source_files = sorted(
            {
                str(task.source_ref.get("path"))
                for task in accepted_tasks
                if isinstance(task.source_ref.get("path"), str)
            }
        )

        for task in accepted_tasks:
            pick = self._final_answer_artifact(task)
            if pick is None:
                excluded.append({"task_id": task.task_id, "reason": "missing_annotation_result"})
                continue
            artifact, human_authored = pick
            annotation_payload = self._read_artifact_payload(artifact)
            if annotation_payload is None:
                excluded.append({"task_id": task.task_id, "reason": "missing_annotation_payload"})
                continue
            masked_indices = masked_by_task.get(task.task_id) or set()
            row = self._training_row(
                task, artifact, annotation_payload,
                human_authored=human_authored,
                masked_indices=masked_indices,
            )
            validation_errors = self._validate_training_row(row)
            if validation_errors:
                row_errors.append({"task_id": task.task_id, "errors": validation_errors})
                excluded.append(
                    {
                        "task_id": task.task_id,
                        "reason": "invalid_training_row",
                        "errors": validation_errors,
                    }
                )
                continue
            rules_info = self._rules_at(artifact.created_at, rules_timeline)
            if rules_info:
                ver_label, _ = rules_info
                self._patch_rules_version(row, ver_label)
            else:
                self._drop_stale_rules_path(row)
            rows.append(row)
            included.append(task.task_id)
            artifact_ids.append(artifact.artifact_id)
            if enqueue_external_submit and task.external_ref is not None:
                self._enqueue_submit(task, export_id=export_id, row=row)

        output_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

        manifest = ExportManifest.new(
            project_id=project_id,
            output_paths=[self._relative_output_path(output_path)],
            task_ids_included=included,
            task_ids_excluded=excluded,
            artifact_ids=artifact_ids,
            source_files=source_files,
            annotation_rules_hash=self._annotation_rules_hash(),
            schema_version="jsonl-training-v2",
            validator_version="local-export-v2",
            validation_summary={
                "accepted_tasks": len(accepted_tasks),
                "included": len(included),
                "excluded": len(excluded),
                "required_fields": REQUIRED_TRAINING_ROW_FIELDS,
                "row_errors": row_errors,
                "errors": excluded,
            },
            known_limitations=["text-first JSONL sink; multimodal preview artifacts are referenced, not rendered"],
            export_id=export_id,
        )
        self.store.save_export_manifest(manifest)
        self._write_readme(output_dir, manifest, rows)
        return manifest

    def _write_readme(
        self,
        output_dir: Path,
        manifest: ExportManifest,
        rows: list[dict[str, Any]],
    ) -> None:
        rules_label, rules_changelog = self._latest_rules_version()
        entity_types = self._schema_entity_types()
        human_count = sum(1 for r in rows if r.get("human_authored"))
        non_human_count = len(rows) - human_count
        export_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        vs = manifest.validation_summary or {}
        accepted = vs.get("accepted_tasks", "?")
        included = vs.get("included", len(rows))
        excluded_list = vs.get("errors") or vs.get("excluded") or []
        excluded = len(excluded_list) if isinstance(excluded_list, list) else vs.get("excluded", "?")
        source_files = manifest.source_files or []

        entity_rows = "\n".join(
            f"| `{t['name']}` | {t['description']} |"
            for t in entity_types
        ) if entity_types else "| *(schema not found)* | |"

        source_list = "\n".join(f"- `{f}`" for f in source_files) if source_files else "- *(not recorded)*"

        changelog_note = f" — {rules_changelog}" if rules_changelog else ""

        readme = f"""\
# Training Data Export — README

**Export ID:** `{manifest.export_id}`
**Project:** `{manifest.project_id}`
**Export date:** {export_date}
**Annotation rules version:** {rules_label}{changelog_note}
**Schema version:** {manifest.schema_version}

---

## 1. 数据概况

| 字段 | 值 |
|---|---|
| 文件 | `training_data.jsonl` |
| 总行数 | {len(rows)} |
| 人工标注行 | {human_count} |
| 非人工标注行 | {non_human_count} |
| Accepted tasks | {accepted} |
| Included | {included} |
| Excluded (validation errors) | {excluded} |

---

## 2. 标注 Schema

Entity types（来自 `output_schema.json`）：

| 类型 | 说明 |
|---|---|
{entity_rows}

JSON extraction fields（json_structures）：
`status` / `risk` / `goal` / `strategy` / `constraint` / `decision` / `task` / `preference` / `reason` / `technology`

> `json_structures.technology` 仅用于「技术名称是谓语结构主语」的短语，不用于裸名称提及。
> 裸 product name（如 YouTube、WhatsApp）只入 `entities.technology`。

---

## 3. 标注规则版本

当前激活版本：**{rules_label}**{changelog_note}

规则文档路径：`.annotation-pipeline/document_versions/`（通过系统 API 查看全部版本）

---

## 4. 来源数据文件

{source_list}

---

## 5. 训练前置条件

> **此 export 在 QC 通过前不得直接用于训练。**

训练放行检查清单：

- [ ] QC 准确率 ≥ 98%（当前如低于此阈值，见 repair_report）
- [ ] repair manifest 中确认错误的 QC 发现已删除
- [ ] consumer app（YouTube / WhatsApp / Telegram 等）类型为 `technology`，相关 wrong_type 发现已过滤
- [ ] 非人工标注行（{non_human_count} 条）已单独核查
- [ ] 来源文件路径为标注时本地路径，不影响数据内容

---

## 6. 关联文件

| 文件 | 说明 |
|---|---|
| `training_data.jsonl` | 训练数据主文件 |
| `.annotation-pipeline/output_schema.json` | Schema 结构定义 |
| `.annotation-pipeline/document_versions/` | 全部标注规则版本 |
"""
        (output_dir / "README.md").write_text(readme, encoding="utf-8")

    def _rules_doc_id(self) -> str | None:
        try:
            rows = self.store._conn.execute(
                "SELECT document_id, metadata_json FROM documents"
            ).fetchall()
        except Exception:
            return None
        for r in rows:
            try:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                meta = {}
            if isinstance(meta, dict) and meta.get("role") == "annotation_rules":
                return r["document_id"]
        return None

    def _rules_version_timeline(self) -> list[tuple[Any, str, str]]:
        """Return [(created_at_dt, version_label, content_path), ...] sorted ascending."""
        from datetime import datetime, timezone
        doc_id = self._rules_doc_id()
        if not doc_id:
            return []
        try:
            rows = self.store._conn.execute(
                "SELECT version, content_path, created_at FROM document_versions "
                "WHERE document_id = ? ORDER BY created_at ASC",
                (doc_id,),
            ).fetchall()
        except Exception:
            return []
        result = []
        for r in rows:
            try:
                dt = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            result.append((dt, r["version"], r["content_path"]))
        return result

    def _rules_at(
        self,
        artifact_dt: Any,
        timeline: list[tuple[Any, str, str]],
    ) -> tuple[str, str] | None:
        """Return (version_label, content_path) of the rules version active when artifact_dt occurred."""
        from datetime import timezone
        if not timeline:
            return None
        dt = artifact_dt
        if hasattr(dt, "tzinfo") and dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        active: tuple[str, str] | None = None
        for ver_dt, ver_label, ver_path in timeline:
            if ver_dt <= dt:
                active = (ver_label, ver_path)
            else:
                break
        return active

    def _latest_rules_version(self) -> tuple[str, str]:
        """Return (version_label, changelog) of the latest annotation-rules document version."""
        try:
            doc_rows = self.store._conn.execute(
                "SELECT document_id, metadata_json FROM documents"
            ).fetchall()
        except Exception:
            return "unknown", ""
        target_doc_id: str | None = None
        for r in doc_rows:
            try:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                meta = {}
            if isinstance(meta, dict) and meta.get("role") == "annotation_rules":
                target_doc_id = r["document_id"]
                break
        if not target_doc_id:
            return "unknown", ""
        try:
            ver_row = self.store._conn.execute(
                """
                SELECT version, changelog FROM document_versions
                WHERE document_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (target_doc_id,),
            ).fetchone()
        except Exception:
            return "unknown", ""
        if ver_row is None:
            return "unknown", ""
        return ver_row["version"], ver_row["changelog"] or ""

    def _schema_entity_types(self) -> list[dict[str, str]]:
        """Parse entity type names and descriptions from output_schema.json."""
        schema_path = self.store.root / "output_schema.json"
        if not schema_path.exists():
            return []
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            one_of = schema.get("$defs", {}).get("entityType", {}).get("oneOf", [])
            return [
                {"name": e["const"], "description": e.get("description", "")}
                for e in one_of
                if "const" in e
            ]
        except Exception:
            return []

    def _final_answer_artifact(self, task: Task) -> tuple[ArtifactRef, bool] | None:
        """Return (artifact, human_authored) for this task's final answer. Prefer human_review_answer."""
        artifacts = self.store.list_artifacts(task.task_id)
        human_answers = [a for a in artifacts if a.kind == "human_review_answer"]
        if human_answers:
            return (human_answers[-1], True)
        annotations = [a for a in artifacts if a.kind == "annotation_result"]
        if annotations:
            return (annotations[-1], False)
        return None

    def _read_artifact_payload(self, artifact: ArtifactRef) -> Any:
        path = self.store.root / artifact.path
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _training_row(
        self,
        task: Task,
        artifact: ArtifactRef,
        artifact_payload: Any,
        *,
        human_authored: bool,
        masked_indices: "set[int] | None" = None,
    ) -> dict[str, Any]:
        masked = masked_indices or set()

        if human_authored:
            annotation = artifact_payload.get("answer") if isinstance(artifact_payload, dict) else artifact_payload
        else:
            annotation = artifact_payload.get("text", artifact_payload) if isinstance(artifact_payload, dict) else artifact_payload

        # Filter masked rows from the annotation payload when it's a dict
        # with a ``rows`` list (e.g. structured annotation_result payloads).
        if masked and isinstance(annotation, dict):
            annotation = filter_masked_rows(annotation, masked)
        elif masked and isinstance(annotation, str):
            # annotation_result stores the annotation as a JSON string.
            try:
                parsed = json.loads(annotation)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
                filtered = filter_masked_rows(parsed, masked)
                if filtered is not parsed:
                    annotation = json.dumps(filtered, ensure_ascii=False)

        # Filter masked rows from source_ref.payload when it has rows.
        source_ref = task.source_ref
        if masked and isinstance(source_ref, dict):
            payload = source_ref.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                filtered_payload = filter_masked_rows(payload, masked)
                if filtered_payload is not payload:
                    source_ref = {**source_ref, "payload": filtered_payload}

        return {
            "task_id": task.task_id,
            "pipeline_id": task.pipeline_id,
            "source_ref": source_ref,
            "modality": task.modality,
            "annotation_requirements": task.annotation_requirements,
            "annotation": annotation,
            "annotation_artifact_id": artifact.artifact_id,
            "annotation_artifact_path": artifact.path,
            "human_authored": human_authored,
        }

    def _drop_stale_rules_path(self, row: dict[str, Any]) -> None:
        """Remove legacy rules_path key from annotation_guidance when no DB version is assignable."""
        try:
            row["source_ref"]["payload"]["annotation_guidance"].pop("rules_path", None)
        except (KeyError, TypeError, AttributeError):
            pass

    def _patch_rules_version(self, row: dict[str, Any], version_label: str) -> None:
        """Set annotation_guidance.rules_version to the DB version label active at annotation time."""
        sr = row.get("source_ref")
        if not isinstance(sr, dict):
            return
        payload = sr.get("payload")
        if not isinstance(payload, dict):
            return
        guidance = payload.get("annotation_guidance")
        if isinstance(guidance, dict):
            guidance.pop("rules_path", None)
            guidance["rules_version"] = version_label

    def _validate_training_row(self, row: dict[str, Any]) -> list[str]:
        errors = [
            f"missing_{field}"
            for field in REQUIRED_TRAINING_ROW_FIELDS
            if field not in row
        ]
        if not isinstance(row.get("task_id"), str) or not row.get("task_id"):
            errors.append("task_id_required")
        if not isinstance(row.get("pipeline_id"), str) or not row.get("pipeline_id"):
            errors.append("pipeline_id_required")
        if not isinstance(row.get("source_ref"), dict) or not row.get("source_ref"):
            errors.append("source_ref_required")
        if not isinstance(row.get("annotation_artifact_id"), str) or not row.get("annotation_artifact_id"):
            errors.append("annotation_artifact_id_required")
        if not isinstance(row.get("annotation_artifact_path"), str) or not row.get("annotation_artifact_path"):
            errors.append("annotation_artifact_path_required")

        annotation = row.get("annotation")
        if annotation in (None, "", [], {}):
            errors.append("annotation_required")
        elif isinstance(annotation, str):
            self._validate_annotation_json_string(annotation, errors)
        elif not isinstance(annotation, (dict, list)):
            errors.append("annotation_must_be_json_value")
        return errors

    def _validate_annotation_json_string(self, value: str, errors: list[str]) -> None:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            errors.append("annotation_string_must_be_json")
            return
        if parsed in (None, "", [], {}):
            errors.append("annotation_json_must_not_be_empty")

    def _enqueue_submit(self, task: Task, *, export_id: str, row: dict[str, Any]) -> None:
        record = OutboxRecord.new(
            task_id=task.task_id,
            kind=OutboxKind.SUBMIT,
            payload={
                "task_id": task.task_id,
                "external_ref": task.external_ref.to_dict() if task.external_ref else None,
                "export_id": export_id,
                "result": row,
            },
        )
        self.store.save_outbox(record)

    def _annotation_rules_hash(self) -> str | None:
        """Hash the latest annotation_rules version content from the DB."""
        doc_id = self._rules_doc_id()
        if not doc_id:
            return None
        try:
            ver_row = self.store._conn.execute(
                "SELECT content_path FROM document_versions WHERE document_id=? ORDER BY created_at DESC LIMIT 1",
                (doc_id,),
            ).fetchone()
        except Exception:
            return None
        if not ver_row or not ver_row["content_path"]:
            return None
        content_file = self.store.root / ver_row["content_path"]
        if not content_file.exists():
            return None
        return sha256(content_file.read_bytes()).hexdigest()

    def _relative_output_path(self, output_path: Path) -> str:
        try:
            return str(output_path.relative_to(self.store.root))
        except ValueError:
            return str(output_path)

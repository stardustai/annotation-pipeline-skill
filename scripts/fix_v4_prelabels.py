#!/usr/bin/env python3
"""
批量修复 v4_ner_phrase 中 PENDING 任务的 pre-label artifacts。

修复内容：
1. shared-type field placement — 多词 technology 从 entities 移到 json_structures
2. verbatim check — 用 try_align_to_verbatim 对齐；无法对齐的 span 直接删除
3. cross-type collision — 同一 span 出现多个 entity 类型，保留第一个，删其余
4. missing rows — 补充缺失 row（空 entities/json_structures）
5. entity → generic_entity — 清理遗留的 entity 类型 key

只处理 PENDING 状态任务的 pre-label artifact（provider=v3_migration 或 prelabeled）。
"""
from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from annotation_pipeline_skill.core.schema_validation import (
    auto_fix_safe_spans_in_place,
    find_verbatim_violations,
    find_cross_type_collisions,
    find_shared_type_field_violations,
)
from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.store.sqlite_store import SqliteStore

V4_ANNOTATION_DIR = REPO_ROOT / "projects" / "v4_ner_phrase" / ".annotation-pipeline"

VALID_ENTITY_TYPES = {
    "person", "organization", "project", "document", "time",
    "number", "event", "location", "technology", "generic_entity",
}


def _fix_entity_rename(payload: dict) -> int:
    """把残留的 entity key 改成 generic_entity，返回修改次数。"""
    count = 0
    for row in payload.get("rows", []):
        output = row.get("output", {})
        entities = output.get("entities", {})
        if "entity" in entities:
            val = entities.pop("entity")
            existing = entities.get("generic_entity", [])
            merged = list(dict.fromkeys(existing + val))  # 去重保序
            entities["generic_entity"] = merged
            count += 1
    return count


def _fix_unknown_entity_types(payload: dict) -> int:
    """删除 v4 schema 里不存在的 entity 类型，返回删除数。"""
    count = 0
    for row in payload.get("rows", []):
        entities = row.get("output", {}).get("entities", {})
        bad_keys = [k for k in list(entities.keys()) if k not in VALID_ENTITY_TYPES]
        for k in bad_keys:
            del entities[k]
            count += 1
    return count


def _fix_cross_type_collisions(payload: dict) -> int:
    """同 span 出现在多个 entity type 里，保留 first-seen，删其余。"""
    count = 0
    for row in payload.get("rows", []):
        entities = row.get("output", {}).get("entities", {})
        seen: set[str] = set()
        for type_name in list(entities.keys()):
            items = entities.get(type_name, [])
            if not isinstance(items, list):
                continue
            new_items = []
            for span in items:
                if span not in seen:
                    seen.add(span)
                    new_items.append(span)
                else:
                    count += 1
            entities[type_name] = new_items
    return count


def _fix_missing_rows(payload: dict, source_rows: list[dict]) -> int:
    """补全缺失的 row（用空 output），返回补充数量。"""
    existing_ids = {
        r["row_id"] for r in payload.get("rows", [])
        if isinstance(r, dict) and "row_id" in r
    }
    added = 0
    for src_row in source_rows:
        rid = src_row.get("row_id")
        ridx = src_row.get("row_index", 0)
        if rid and rid not in existing_ids:
            payload["rows"].append({
                "row_index": ridx,
                "row_id": rid,
                "output": {"entities": {}, "json_structures": {}},
            })
            added += 1
    # 按 row_index 排序
    if added:
        payload["rows"].sort(key=lambda r: r.get("row_index", 0))
    return added


def _fix_verbatim_drop(payload: dict, source_rows: list[dict]) -> int:
    """删除在原文中找不到的 span（try_align_to_verbatim 已经尝试过对齐）。"""
    from annotation_pipeline_skill.core.schema_validation import find_verbatim_violations
    violations = find_verbatim_violations(payload, source_rows)
    if not violations:
        return 0
    # 建立 (row_index, type, span) → True 的快速查找
    to_drop: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for v in violations:
        ridx = v.get("row_index", 0)
        fld = v.get("field", "entities")      # "entities" or "json_structures"
        typ = v.get("entity_type") or v.get("type", "")
        span = v.get("span", "")
        if typ and span:
            to_drop[ridx][f"{fld}.{typ}"].add(span)
    count = 0
    for row in payload.get("rows", []):
        ridx = row.get("row_index", 0)
        if ridx not in to_drop:
            continue
        output = row.get("output", {})
        for fld in ("entities", "json_structures"):
            type_dict = output.get(fld, {})
            for typ, items in type_dict.items():
                key = f"{fld}.{typ}"
                drops = to_drop[ridx].get(key, set())
                if drops and isinstance(items, list):
                    before = len(items)
                    type_dict[typ] = [s for s in items if s not in drops]
                    count += before - len(type_dict[typ])
    return count


def process_task(
    task: Task,
    art_path: str,
    store: SqliteStore,
    *,
    dry_run: bool = False,
) -> dict:
    full_path = store.root / art_path
    raw = json.loads(full_path.read_text(encoding="utf-8"))
    text = raw.get("text", "")
    try:
        payload = json.loads(text)
    except Exception:
        return {"error": "json_parse_failed"}

    source_rows = (
        task.source_ref.get("payload", {}).get("rows", [])
        if isinstance(task.source_ref, dict) else []
    )

    stats = {}

    # 1. entity → generic_entity
    n = _fix_entity_rename(payload)
    if n: stats["entity_rename"] = n

    # 2. 未知 entity type 清理
    n = _fix_unknown_entity_types(payload)
    if n: stats["unknown_type_drop"] = n

    # 3. auto_fix_safe_spans_in_place（verbatim 对齐 + shared-type 路由）
    n = auto_fix_safe_spans_in_place(task, payload)
    if n: stats["auto_fix"] = n

    # 4. 删除仍然无法对齐的 verbatim violations
    n = _fix_verbatim_drop(payload, source_rows)
    if n: stats["verbatim_drop"] = n

    # 5. cross-type collision
    n = _fix_cross_type_collisions(payload)
    if n: stats["cross_type_fix"] = n

    # 6. missing rows 补全
    n = _fix_missing_rows(payload, source_rows)
    if n: stats["missing_rows_added"] = n

    if not stats:
        return {"changed": False}

    if not dry_run:
        new_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        raw["text"] = new_text
        full_path.write_text(
            json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    return {"changed": True, **stats}


def main(dry_run: bool = False) -> None:
    store = SqliteStore.open(V4_ANNOTATION_DIR)

    # 只处理 PENDING 任务
    db = sqlite3.connect(f"file:{V4_ANNOTATION_DIR / 'db.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    cur.execute(
        "SELECT t.task_id, t.source_ref_json, ar.path "
        "FROM tasks t "
        "JOIN artifact_refs ar ON ar.task_id = t.task_id "
        "WHERE t.status = 'pending' "
        "  AND ar.kind = 'annotation_result' "
        "  AND (ar.metadata_json LIKE '%v3_migration%' OR ar.metadata_json LIKE '%prelabel%') "
        "  AND ar.seq = (SELECT MIN(seq) FROM artifact_refs ar2 "
        "                WHERE ar2.task_id = t.task_id AND ar2.kind = 'annotation_result') "
        "ORDER BY t.task_id"
    )
    rows = cur.fetchall()
    db.close()

    print(f"找到 {len(rows)} 个 PENDING 任务的 pre-label artifact")
    if dry_run:
        print("（DRY-RUN 模式，不写入文件）")

    total_stats: dict[str, int] = defaultdict(int)
    changed = 0
    errors = 0

    for i, row in enumerate(rows):
        task_id = row["task_id"]
        source_ref = json.loads(row["source_ref_json"])
        art_path = row["path"]

        # 构造轻量 Task 对象（只需要 task_id 和 source_ref）
        task = Task.from_dict({
            "task_id": task_id,
            "pipeline_id": "v4_ner_phrase",
            "source_ref": source_ref,
            "external_ref": None,
            "modality": "text",
            "annotation_requirements": {},
            "selected_annotator_id": None,
            "status": "pending",
            "current_attempt": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        })

        result = process_task(task, art_path, store, dry_run=dry_run)

        if "error" in result:
            errors += 1
            print(f"  ERROR {task_id}: {result['error']}")
        elif result.get("changed"):
            changed += 1
            for k, v in result.items():
                if k != "changed":
                    total_stats[k] += v

        if (i + 1) % 500 == 0:
            print(f"  进度 {i+1}/{len(rows)} ...")

    print(f"\n完成")
    print(f"  修改: {changed} / {len(rows)}")
    print(f"  错误: {errors}")
    print(f"  修复明细:")
    for k, v in sorted(total_stats.items()):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry_run)

"""Roll back row_masks whose true word-trigram Jaccard with the recorded
representative is below a threshold (default 0.05).

The row_dedup_service writes a `reason` like
``"near-duplicate of <rep_task>:<rep_idx> in cluster <cid>"`` for every
masked row. We parse the rep out of that reason, compute the *exact*
word-trigram Jaccard between the masked row's text and the rep's text,
and DELETE the mask if it falls below the threshold.

This catches false-positive masks caused by transitive linking in the
MinHash-LSH connected-components clusterer — a problem on short rows
where chain neighbours can have J=0.

Usage:
    .venv/bin/python scripts/rollback_low_jaccard_masks.py \
        --project-root projects/v3_initial_deployment \
        --threshold 0.05               # dry-run by default
    .venv/bin/python scripts/rollback_low_jaccard_masks.py \
        --project-root projects/v3_initial_deployment \
        --threshold 0.05 --apply
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from annotation_pipeline_skill.store.sqlite_store import SqliteStore

_REASON_PAT = re.compile(r"near-duplicate of (\S+):(\d+) in cluster (\S+)")


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    toks = text.split()
    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)}  # type: ignore[misc]


def _jaccard(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    s1, s2 = _trigrams(a), _trigrams(b)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True,
                    help="path to project dir (parent of .annotation-pipeline/)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="DELETE masks whose mask→rep Jaccard < threshold")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; otherwise dry-run only")
    args = ap.parse_args(argv)

    project_root = pathlib.Path(args.project_root)
    store = SqliteStore.open(project_root / ".annotation-pipeline")

    masks = store._conn.execute(
        "SELECT task_id, row_index, reason FROM row_masks"
    ).fetchall()
    print(f"total active masks: {len(masks)}")

    # Collect (masked, rep) pairs and the tasks we need text for
    parsed: list[tuple[str, int, str, int]] = []
    need_tasks: set[str] = set()
    for r in masks:
        m = _REASON_PAT.search(r["reason"] or "")
        if not m:
            continue
        rep_task, rep_idx = m.group(1), int(m.group(2))
        parsed.append((r["task_id"], r["row_index"], rep_task, rep_idx))
        need_tasks.add(r["task_id"])
        need_tasks.add(rep_task)
    print(f"parsed mask→rep pairs: {len(parsed)} (of {len(masks)})")

    # Bulk-load row texts
    text_cache: dict[tuple[str, int], str] = {}
    chunks = list(need_tasks)
    for i in range(0, len(chunks), 500):
        batch = chunks[i:i + 500]
        placeholders = ",".join("?" * len(batch))
        rows2 = store._conn.execute(
            f"SELECT task_id, source_ref_json FROM tasks "
            f"WHERE task_id IN ({placeholders})",
            batch,
        ).fetchall()
        for row in rows2:
            try:
                sr = json.loads(row["source_ref_json"])
            except Exception:
                continue
            for r2 in sr.get("payload", {}).get("rows", []):
                ri = r2.get("row_index")
                inp = r2.get("input")
                if isinstance(ri, int) and isinstance(inp, str):
                    text_cache[(row["task_id"], ri)] = inp

    # Identify low-Jaccard masks
    to_delete: list[tuple[str, int, float, str, int]] = []
    skipped_no_text = 0
    for mt, mi, rt, ri in parsed:
        j = _jaccard(text_cache.get((mt, mi)), text_cache.get((rt, ri)))
        if j is None:
            skipped_no_text += 1
            continue
        if j < args.threshold:
            to_delete.append((mt, mi, j, rt, ri))

    print(f"skipped (no text for either side): {skipped_no_text}")
    print(f"masks with mask→rep Jaccard < {args.threshold}: {len(to_delete)}")

    # Show distribution of what we'd delete
    if to_delete:
        zeros = sum(1 for _, _, j, _, _ in to_delete if j == 0.0)
        print(f"  of which J == 0.000 : {zeros}")
        print(f"  of which 0 < J < {args.threshold}: {len(to_delete) - zeros}")
        print()
        print("first 10 to-delete examples:")
        for mt, mi, j, rt, ri in to_delete[:10]:
            mtxt = (text_cache.get((mt, mi)) or "")[:70]
            rtxt = (text_cache.get((rt, ri)) or "")[:70]
            print(f"  J={j:.4f}  {mt[-6:]}:{mi} «{mtxt}»")
            print(f"           rep={rt[-6:]}:{ri} «{rtxt}»")

    if not args.apply:
        print("\n[dry-run] no deletes applied. Re-run with --apply to delete.")
        return 0

    if not to_delete:
        print("\nnothing to delete")
        return 0

    print(f"\n[apply] deleting {len(to_delete)} masks…")
    cur = store._conn.cursor()
    cur.executemany(
        "DELETE FROM row_masks WHERE task_id=? AND row_index=?",
        [(mt, mi) for mt, mi, _, _, _ in to_delete],
    )
    store._conn.commit()
    print(f"[apply] deleted {cur.rowcount} masks")
    remaining = store._conn.execute(
        "SELECT COUNT(*) AS n FROM row_masks"
    ).fetchone()["n"]
    print(f"row_masks remaining: {remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

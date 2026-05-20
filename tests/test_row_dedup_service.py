"""Tests for RowDedupService — scan, mask, cache state.

Uses minhash provider (no HTTP calls) and a real SqliteStore.
Seeds 6 tasks each with 3 rows; some rows are near-duplicates
across tasks via string templates.
"""
from __future__ import annotations

import pytest

from annotation_pipeline_skill.core.models import Task
from annotation_pipeline_skill.core.states import TaskStatus
from annotation_pipeline_skill.services.row_dedup_service import RowDedupService
from annotation_pipeline_skill.services.row_mask_service import RowMaskService
from annotation_pipeline_skill.similarity.profiles import SimilarityProfile
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


# ---------------------------------------------------------------------------
# Test profiles
# ---------------------------------------------------------------------------

_MINHASH_PROFILE = SimilarityProfile(
    name="MinHash",
    provider="minhash",
    model="minhash-w3-p64",
    shingle_size=3,
    num_perm=64,
)

_PROFILES = {"MinHash": _MINHASH_PROFILE}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_store(tmp_path) -> SqliteStore:
    return SqliteStore.open(tmp_path / "proj")


def _make_task(task_id: str, pipeline_id: str, rows: list[dict]) -> Task:
    task = Task.new(
        task_id=task_id,
        pipeline_id=pipeline_id,
        source_ref={
            "kind": "jsonl",
            "payload": {"rows": rows},
        },
    )
    task.status = TaskStatus.ACCEPTED
    return task


def _seed_tasks(store: SqliteStore, pipeline_id: str) -> list[Task]:
    """Seed 6 tasks, each with 3 rows.

    Rows 0 across all tasks share a near-identical template, creating a
    cluster of near-duplicates. Rows 1 and 2 are unique per task.
    """
    template = "The substation equipment report for unit {n} shows a critical fault in the transformer."
    tasks = []
    for i in range(6):
        rows = [
            {"row_index": 0, "input": template.format(n=i)},
            {"row_index": 1, "input": f"Unique row one content for task {i}: banana kiwi mango"},
            {"row_index": 2, "input": f"Unique row two content for task {i}: alpha beta gamma delta"},
        ]
        t = _make_task(f"task-{i:02d}", pipeline_id, rows)
        store.save_task(t)
        tasks.append(t)
    return tasks


def _make_service(store: SqliteStore) -> RowDedupService:
    return RowDedupService(store, _PROFILES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scan_rows_with_minhash_finds_template_groups(tmp_path):
    """scan_rows should find a cluster of near-duplicate template rows."""
    store = _open_store(tmp_path)
    _seed_tasks(store, "proj")
    svc = _make_service(store)

    result = svc.scan_rows(
        project_id="proj",
        profile_name="MinHash",
        jaccard_threshold=0.3,
    )

    assert result["row_count"] == 18  # 6 tasks * 3 rows
    assert result["task_count"] == 6
    assert isinstance(result["clusters"], list)
    assert result["params"]["profile"] == "MinHash"
    assert result["params"]["metric"] == "jaccard"

    # Should find at least one cluster (the template rows)
    assert len(result["clusters"]) >= 1

    # Largest cluster should have members from the template rows (row_index=0)
    largest = result["clusters"][0]
    assert len(largest["members"]) >= 2
    assert all("task_id" in m for m in largest["members"])
    assert all("row_index" in m for m in largest["members"])
    assert all("text_preview" in m for m in largest["members"])
    assert largest["method"] == "minhash"
    assert 0.0 <= largest["similarity"] <= 1.0


def test_scan_rows_includes_already_masked_rows_tagged(tmp_path):
    """Masked rows MUST appear in the scan output, tagged ``masked: true``.

    Why this matters: after an operator masks part of a cluster, the
    next ``Re-scan`` should still show those rows so the operator
    keeps context of "what I've already handled" — they just render
    with a red ``masked`` badge instead of an active checkbox. The
    only behavioural difference from an unmasked row is that masked
    rows are barred from being the cluster representative
    (``rep_exclude`` passed to MinHashLSHFinder).
    """
    store = _open_store(tmp_path)
    tasks = _seed_tasks(store, "proj")
    svc = _make_service(store)

    # Mask row 0 of all 6 tasks (the template rows that will form the cluster)
    mask_svc = RowMaskService(store)
    for t in tasks:
        mask_svc.apply(
            task_id=t.task_id,
            row_index=0,
            reason="pre-masked",
            masked_by="test",
        )

    result = svc.scan_rows(
        project_id="proj",
        profile_name="MinHash",
        jaccard_threshold=0.3,
    )

    # All 18 rows still scanned (6 tasks × 3 rows). Masked rows are
    # NOT excluded — they participate in clustering with a tag.
    assert result["row_count"] == 18

    # The template cluster still forms, and contains the 6 masked
    # row-0 entries — all tagged masked=True.
    masked_in_cluster = [
        m for c in result["clusters"]
        for m in c["members"]
        if m["row_index"] == 0
    ]
    assert len(masked_in_cluster) >= 1, (
        "template row-0 cluster should still appear in scan output"
    )
    # Every row-0 member that's there must be tagged masked=True
    assert all(m["masked"] is True for m in masked_in_cluster), (
        f"row-0 members not tagged masked: {masked_in_cluster}"
    )

    # And no masked row should be the rep (rep is conventionally the
    # lex-smallest member; the finder excludes masked from rep choice).
    for c in result["clusters"]:
        if not c["members"]:
            continue
        rep = min(c["members"], key=lambda m: (str(m["task_id"]), int(m["row_index"])))
        # If the cluster has any unmasked member, the rep must be unmasked
        if any(not m["masked"] for m in c["members"]):
            assert not rep.get("masked", False) or any(
                not m["masked"] for m in c["members"] if (str(m["task_id"]), int(m["row_index"])) <= (str(rep["task_id"]), int(rep["row_index"]))
            ), f"masked row chosen as cluster rep: {rep}"


def test_mask_duplicates_keeps_representative(tmp_path):
    """mask_duplicates keeps the representative (smallest task_id:row_index) unmasked."""
    store = _open_store(tmp_path)
    _seed_tasks(store, "proj")
    svc = _make_service(store)

    # Scan first to get clusters
    result = svc.scan_rows(
        project_id="proj",
        profile_name="MinHash",
        jaccard_threshold=0.3,
    )

    assert len(result["clusters"]) >= 1
    cluster = result["clusters"][0]
    members = cluster["members"]
    assert len(members) >= 2

    # Mask duplicates in the cluster
    mask_result = svc.mask_duplicates(
        project_id="proj",
        members=members,
        cluster_id=cluster["cluster_id"],
        similarity=cluster["similarity"],
        profile_name="MinHash",
        model=_MINHASH_PROFILE.model,
    )

    assert mask_result["masked"] == len(members) - 1  # all but representative
    assert mask_result["skipped"] == 0

    # Check: representative (smallest task_id, row_index) is NOT masked
    mask_svc = RowMaskService(store)
    sorted_members = sorted(members, key=lambda m: (str(m["task_id"]), int(m["row_index"])))
    rep = sorted_members[0]
    rep_indices = mask_svc.masked_indices_for_task(rep["task_id"])
    assert rep["row_index"] not in rep_indices

    # Check: all others ARE masked
    for m in sorted_members[1:]:
        indices = mask_svc.masked_indices_for_task(m["task_id"])
        assert m["row_index"] in indices


def test_mask_duplicates_idempotent(tmp_path):
    """Calling mask_duplicates twice returns masked=0 on the second call."""
    store = _open_store(tmp_path)
    _seed_tasks(store, "proj")
    svc = _make_service(store)

    result = svc.scan_rows(
        project_id="proj",
        profile_name="MinHash",
        jaccard_threshold=0.3,
    )
    assert len(result["clusters"]) >= 1
    cluster = result["clusters"][0]
    members = cluster["members"]
    assert len(members) >= 2

    # First call
    r1 = svc.mask_duplicates(
        project_id="proj",
        members=members,
        cluster_id=cluster["cluster_id"],
        similarity=cluster["similarity"],
        profile_name="MinHash",
        model=_MINHASH_PROFILE.model,
    )
    assert r1["masked"] == len(members) - 1

    # Second call — all already masked
    r2 = svc.mask_duplicates(
        project_id="proj",
        members=members,
        cluster_id=cluster["cluster_id"],
        similarity=cluster["similarity"],
        profile_name="MinHash",
        model=_MINHASH_PROFILE.model,
    )
    assert r2["masked"] == 0
    assert r2["skipped"] == len(members) - 1


def test_get_cache_state_stale_after_new_mask(tmp_path):
    """Cache should be stale after masking a row (content_hash changes)."""
    store = _open_store(tmp_path)
    tasks = _seed_tasks(store, "proj")
    svc = _make_service(store)

    # Scan to populate cache
    svc.scan_rows(
        project_id="proj",
        profile_name="MinHash",
        jaccard_threshold=0.3,
    )

    state_before = svc.get_cache_state(project_id="proj", profile_name="MinHash")
    assert state_before["cached"] is True
    assert state_before["stale"] is False

    # Apply a new mask outside of scan_rows — this changes the input set
    mask_svc = RowMaskService(store)
    mask_svc.apply(
        task_id=tasks[0].task_id,
        row_index=2,
        reason="manual",
        masked_by="test",
    )

    state_after = svc.get_cache_state(project_id="proj", profile_name="MinHash")
    assert state_after["cached"] is True
    assert state_after["stale"] is True
    assert state_after["cached_content_hash"] != state_after["current_content_hash"]


def test_scan_rows_cache_hit_on_second_run(tmp_path):
    """Second scan with same data should use cached embeddings."""
    store = _open_store(tmp_path)
    _seed_tasks(store, "proj")
    svc = _make_service(store)

    r1 = svc.scan_rows(project_id="proj", profile_name="MinHash", jaccard_threshold=0.3)
    assert r1["params"]["embedding_cache"]["hits"] == 0
    assert r1["params"]["embedding_cache"]["misses"] == 18

    r2 = svc.scan_rows(project_id="proj", profile_name="MinHash", jaccard_threshold=0.3)
    assert r2["params"]["embedding_cache"]["hits"] == 18
    assert r2["params"]["embedding_cache"]["misses"] == 0

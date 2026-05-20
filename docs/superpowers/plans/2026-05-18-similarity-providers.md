# Similarity Providers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two complementary near-duplicate / similarity-detection paths to the annotation pipeline — MinHash + LSH for fast byte-level near-duplicate detection, and a configurable embedding provider (initially Jina local HTTP server) for semantic clustering + 2D visualization — and wire both to a uniform "cluster → optional batch reject" workflow that reuses the recently-shipped `ACCEPTED → REJECTED` transition.

**Architecture:** Two phases. Phase 1 ships MinHash + LSH end-to-end (zero new HTTP deps; finds template batches in seconds). Phase 2 adds an `EmbeddingProvider` Protocol with a Jina HTTP client + a separate "semantic clusters" script that does UMAP → HDBSCAN and emits 2D coords for visualization. Both share a common text-extractor (concat `task.source_ref.payload.rows[*].input`) and a common cluster-report shape so the apply-reject path is provider-agnostic. Config follows the existing `llm_profiles.yaml` pattern: a `similarity_profiles.yaml` at workspace root, picked up by `--project-root` like everything else.

**Tech Stack:** Python 3.11+, datasketch (MinHash + LSH), httpx (Jina HTTP), numpy (embedding arrays), umap-learn + hdbscan (Phase 2 only), matplotlib (Phase 2 optional). SqliteStore for task access, existing `transition_task` for the reject path.

---

## File Structure

```
annotation_pipeline_skill/similarity/
├── __init__.py
├── extractors.py     # task → canonical text (concat row inputs); deterministic
├── clusters.py       # ClusterReport dataclass + JSON I/O + apply-reject helper
├── minhash.py        # MinHashLSHFinder: signatures + LSH bucketing + connected-components
├── profiles.py       # SimilarityProfile dataclass + similarity_profiles.yaml loader
└── embeddings.py     # EmbeddingProvider Protocol + JinaHTTPEmbeddingClient

scripts/
├── find_near_duplicates_minhash.py    # phase 1; dry-run default; --apply optional
└── find_semantic_clusters.py          # phase 2

tests/
├── test_similarity_extractors.py
├── test_similarity_minhash.py
├── test_similarity_clusters.py
├── test_similarity_profiles.py
└── test_similarity_embeddings.py
```

`similarity/` is a new top-level package alongside `services/` and `runtime/`. Each module has one job; the cluster report is the integration point so MinHash and embedding pipelines are interchangeable downstream.

---

## Phase 1 — MinHash + LSH

### Task 1: Canonical-text extractor

**Files:**
- Create: `annotation_pipeline_skill/similarity/__init__.py` (empty)
- Create: `annotation_pipeline_skill/similarity/extractors.py`
- Test: `tests/test_similarity_extractors.py`

- [ ] **Step 1: Write the failing test**

`tests/test_similarity_extractors.py`:
```python
import pytest

from annotation_pipeline_skill.similarity.extractors import canonical_task_text


def _task(rows):
    return type("T", (), {"source_ref": {"payload": {"rows": rows}}})()


def test_concatenates_row_inputs_with_newline():
    task = _task([
        {"row_index": 0, "input": "first row text"},
        {"row_index": 1, "input": "second row text"},
    ])
    assert canonical_task_text(task) == "first row text\nsecond row text"


def test_orders_by_row_index_not_list_position():
    task = _task([
        {"row_index": 5, "input": "later"},
        {"row_index": 1, "input": "earlier"},
    ])
    assert canonical_task_text(task) == "earlier\nlater"


def test_skips_non_string_inputs():
    task = _task([
        {"row_index": 0, "input": "ok"},
        {"row_index": 1, "input": None},
        {"row_index": 2, "input": "also ok"},
    ])
    assert canonical_task_text(task) == "ok\nalso ok"


def test_returns_empty_string_for_no_rows():
    assert canonical_task_text(_task([])) == ""
    assert canonical_task_text(type("T", (), {"source_ref": {}})()) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_similarity_extractors.py -q`
Expected: 4 failed with `ModuleNotFoundError: No module named 'annotation_pipeline_skill.similarity.extractors'`

- [ ] **Step 3: Write minimal implementation**

`annotation_pipeline_skill/similarity/__init__.py`:
```python
```

`annotation_pipeline_skill/similarity/extractors.py`:
```python
"""Extract a single canonical text per task for similarity comparison.

Concatenates row inputs ordered by row_index. Deterministic so the same
task always produces the same text — both MinHash signatures and
embeddings are sensitive to byte-level changes.
"""
from __future__ import annotations

from typing import Any


def canonical_task_text(task: Any) -> str:
    """Concatenate ``task.source_ref.payload.rows[*].input`` ordered by
    ``row_index``, joined with newlines. Returns ``""`` when the task
    has no parseable rows.
    """
    source_ref = getattr(task, "source_ref", None)
    if not isinstance(source_ref, dict):
        return ""
    payload = source_ref.get("payload")
    if not isinstance(payload, dict):
        return ""
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return ""
    pairs: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        idx = row.get("row_index") if isinstance(row.get("row_index"), int) else i
        text = row.get("input")
        if isinstance(text, str):
            pairs.append((idx, text))
    pairs.sort(key=lambda p: p[0])
    return "\n".join(text for _, text in pairs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_similarity_extractors.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/similarity/__init__.py \
        annotation_pipeline_skill/similarity/extractors.py \
        tests/test_similarity_extractors.py
git commit -m "feat(similarity): canonical_task_text extractor — deterministic per-task text for dedup/embedding"
```

---

### Task 2: ClusterReport + apply-reject helper

**Files:**
- Create: `annotation_pipeline_skill/similarity/clusters.py`
- Test: `tests/test_similarity_clusters.py`

The cluster report is the integration boundary — MinHash and embedding pipelines both emit it; the apply-reject path consumes it. Defining this first keeps the two providers honest about their output shape.

- [ ] **Step 1: Write the failing test**

`tests/test_similarity_clusters.py`:
```python
import json

import pytest

from annotation_pipeline_skill.similarity.clusters import (
    Cluster,
    ClusterReport,
    pick_representative,
)


def test_pick_representative_returns_smallest_task_id_by_default():
    cluster = Cluster(
        cluster_id="c0",
        task_ids=["v3-002", "v3-001", "v3-003"],
        method="minhash",
        similarity=0.95,
    )
    # Smallest task_id is the deterministic representative (oldest in a
    # zero-padded sequential ID scheme).
    assert pick_representative(cluster) == "v3-001"


def test_pick_representative_handles_single_member_cluster():
    cluster = Cluster(cluster_id="c0", task_ids=["solo"], method="minhash", similarity=1.0)
    assert pick_representative(cluster) == "solo"


def test_cluster_report_to_json_round_trip(tmp_path):
    report = ClusterReport(
        project_id="proj",
        method="minhash",
        params={"shingle_size": 5, "jaccard_threshold": 0.7},
        clusters=[
            Cluster(cluster_id="c0", task_ids=["a", "b"], method="minhash", similarity=0.92),
            Cluster(cluster_id="c1", task_ids=["c", "d", "e"], method="minhash", similarity=0.88),
        ],
    )
    path = tmp_path / "report.json"
    report.to_json_file(path)
    loaded = ClusterReport.from_json_file(path)
    assert loaded.project_id == "proj"
    assert loaded.method == "minhash"
    assert len(loaded.clusters) == 2
    assert loaded.clusters[0].task_ids == ["a", "b"]


def test_report_tasks_to_reject_excludes_representatives():
    report = ClusterReport(
        project_id="proj",
        method="minhash",
        params={},
        clusters=[
            Cluster(cluster_id="c0", task_ids=["t-001", "t-002", "t-003"],
                    method="minhash", similarity=0.95),
            Cluster(cluster_id="c1", task_ids=["solo"],
                    method="minhash", similarity=1.0),
        ],
    )
    # Singletons are never rejected.
    to_reject = report.tasks_to_reject()
    assert sorted(to_reject) == ["t-002", "t-003"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_similarity_clusters.py -q`
Expected: import error / 4 failed.

- [ ] **Step 3: Write minimal implementation**

`annotation_pipeline_skill/similarity/clusters.py`:
```python
"""Cluster report — shared output shape for MinHash and embedding pipelines.

A ``ClusterReport`` is the integration boundary: any provider that finds
groups of similar tasks emits one of these, and any consumer (the
apply-reject script, a UI panel, an audit log) reads them. Provider-
specific knobs (Jaccard threshold for MinHash, UMAP params for embedding)
go in ``params``; the cluster shape itself is identical.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    task_ids: list[str]
    method: str  # "minhash" or "embedding"
    similarity: float  # average pairwise similarity inside the cluster


@dataclass
class ClusterReport:
    project_id: str
    method: str
    params: dict[str, Any]
    clusters: list[Cluster] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "method": self.method,
            "params": self.params,
            "clusters": [asdict(c) for c in self.clusters],
        }

    def to_json_file(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: Path | str) -> "ClusterReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            project_id=data["project_id"],
            method=data["method"],
            params=data.get("params") or {},
            clusters=[Cluster(**c) for c in data.get("clusters", [])],
        )

    def tasks_to_reject(self) -> list[str]:
        """Every task in every cluster of size >= 2 except the chosen
        representative. Singleton clusters are excluded.
        """
        out: list[str] = []
        for c in self.clusters:
            if len(c.task_ids) < 2:
                continue
            rep = pick_representative(c)
            out.extend(tid for tid in c.task_ids if tid != rep)
        return out


def pick_representative(cluster: Cluster) -> str:
    """Return the cluster's canonical representative — the lexicographically
    smallest task_id. For zero-padded sequential IDs this is the
    earliest-imported task, which we prefer to keep so the audit trail
    on the surviving task points to the original batch landing.
    """
    return min(cluster.task_ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_similarity_clusters.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/similarity/clusters.py \
        tests/test_similarity_clusters.py
git commit -m "feat(similarity): Cluster + ClusterReport — shared shape for MinHash/embedding paths"
```

---

### Task 3: Add datasketch dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Inspect current dependencies**

Run: `grep -A 10 'dependencies =' pyproject.toml`
Expected: list of 5 existing deps (openai, pydantic, pyyaml, jsonschema, robust-json-parser).

- [ ] **Step 2: Add datasketch and numpy**

Edit `pyproject.toml`, add to the `dependencies` array:
```toml
dependencies = [
  "openai>=2.0",
  "pydantic>=2.0",
  "pyyaml>=6.0",
  "jsonschema>=4.0",
  "robust-json-parser>=0.1",
  "datasketch>=1.6",
  "numpy>=1.26",
]
```

- [ ] **Step 3: Install**

Run: `.venv/bin/pip install -e . 2>&1 | tail -5`
Expected: `Successfully installed datasketch-... numpy-...` (or "already satisfied").

- [ ] **Step 4: Verify importable**

Run: `.venv/bin/python -c "from datasketch import MinHash, MinHashLSH; import numpy; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "deps: datasketch + numpy for MinHash similarity provider"
```

---

### Task 4: MinHashLSHFinder

**Files:**
- Create: `annotation_pipeline_skill/similarity/minhash.py`
- Test: `tests/test_similarity_minhash.py`

- [ ] **Step 1: Write the failing test**

`tests/test_similarity_minhash.py`:
```python
import pytest

from annotation_pipeline_skill.similarity.minhash import (
    MinHashLSHFinder,
    shingle,
)


def test_shingle_produces_word_ngrams():
    out = shingle("the quick brown fox", n=2)
    assert "the quick" in out
    assert "quick brown" in out
    assert "brown fox" in out
    assert len(out) == 3


def test_shingle_lowercases_and_collapses_whitespace():
    a = shingle("The   Quick\nBrown", n=2)
    b = shingle("the quick brown", n=2)
    assert a == b


def test_finder_clusters_byte_level_near_duplicates():
    finder = MinHashLSHFinder(shingle_size=3, num_perm=128, jaccard_threshold=0.7)
    finder.add("t1", "As of 2024 substation KNO-001 reported breaker monitor priority low")
    finder.add("t2", "As of 2024 substation KNO-002 reported breaker monitor priority low")
    finder.add("t3", "As of 2024 substation KNO-003 reported breaker monitor priority low")
    finder.add("u1", "Weekly sync — Project Phoenix: status is on hold, NPS at 3x")
    clusters = finder.clusters()
    # Three near-duplicate substation reports should land in one cluster;
    # the unrelated Project Phoenix task is a singleton.
    assert len(clusters) >= 1
    cluster_with_t1 = next(c for c in clusters if "t1" in c.task_ids)
    assert set(cluster_with_t1.task_ids) == {"t1", "t2", "t3"}
    assert cluster_with_t1.method == "minhash"
    assert 0.7 <= cluster_with_t1.similarity <= 1.0


def test_finder_emits_singletons_when_requested():
    finder = MinHashLSHFinder(shingle_size=3, num_perm=128, jaccard_threshold=0.7)
    finder.add("alone", "a wholly unique sentence unlike anything else here")
    clusters = finder.clusters(include_singletons=True)
    assert any(c.task_ids == ["alone"] for c in clusters)


def test_finder_skips_empty_text():
    finder = MinHashLSHFinder(shingle_size=3, num_perm=128, jaccard_threshold=0.7)
    finder.add("empty", "")
    finder.add("real", "this has content for shingles to form")
    # Empty text contributes no shingles; finder should not crash and
    # should not group it with anything.
    clusters = finder.clusters(include_singletons=True)
    empty_clusters = [c for c in clusters if "empty" in c.task_ids]
    assert empty_clusters == [] or empty_clusters[0].task_ids == ["empty"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_similarity_minhash.py -q`
Expected: import error / failures.

- [ ] **Step 3: Write minimal implementation**

`annotation_pipeline_skill/similarity/minhash.py`:
```python
"""MinHash + LSH near-duplicate clustering for tasks.

Finds groups of tasks whose canonical text overlaps at the word-n-gram
level beyond a Jaccard threshold. Complexity is roughly O(N) thanks to
LSH bucketing — no pairwise comparison.

Use this for catching byte-level template-style duplicates (the
substation-equipment-report batch is the prototypical case). For
semantic / paraphrased similarity, see the embedding path in
``embeddings.py``.
"""
from __future__ import annotations

import re
from typing import Iterable

from datasketch import MinHash, MinHashLSH

from annotation_pipeline_skill.similarity.clusters import Cluster

_WHITESPACE_RE = re.compile(r"\s+")


def shingle(text: str, n: int = 5) -> set[str]:
    """Word-level n-gram shingle set. Lowercases and collapses runs of
    whitespace so trivially-different spacing doesn't perturb the
    fingerprint."""
    if not text:
        return set()
    normalized = _WHITESPACE_RE.sub(" ", text.lower()).strip()
    tokens = normalized.split(" ")
    if len(tokens) < n:
        return {normalized} if normalized else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class MinHashLSHFinder:
    """Incrementally add (task_id, text) pairs, then ask for clusters.

    Cluster discovery walks the LSH index: each task queries for its near
    neighbours, edges accumulate, connected components become clusters.
    """

    def __init__(
        self,
        *,
        shingle_size: int = 5,
        num_perm: int = 128,
        jaccard_threshold: float = 0.7,
    ):
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.jaccard_threshold = jaccard_threshold
        self._lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
        self._minhashes: dict[str, MinHash] = {}

    def add(self, task_id: str, text: str) -> None:
        shingles = shingle(text, n=self.shingle_size)
        m = MinHash(num_perm=self.num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        self._minhashes[task_id] = m
        # datasketch raises if the key is already present — defensive
        # remove first so re-adds during dev are non-fatal.
        try:
            self._lsh.remove(task_id)
        except KeyError:
            pass
        self._lsh.insert(task_id, m)

    def clusters(self, *, include_singletons: bool = False) -> list[Cluster]:
        # Build an undirected graph: edge between task A and task B if
        # they collide in LSH (likely Jaccard >= threshold).
        edges: dict[str, set[str]] = {tid: set() for tid in self._minhashes}
        for tid, m in self._minhashes.items():
            for neighbour in self._lsh.query(m):
                if neighbour == tid:
                    continue
                edges[tid].add(neighbour)
                edges[neighbour].add(tid)
        # Connected components via iterative BFS.
        visited: set[str] = set()
        components: list[list[str]] = []
        for start in self._minhashes:
            if start in visited:
                continue
            queue = [start]
            comp: list[str] = []
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                queue.extend(n for n in edges[node] if n not in visited)
            components.append(comp)
        out: list[Cluster] = []
        for i, comp in enumerate(sorted(components, key=len, reverse=True)):
            if len(comp) < 2 and not include_singletons:
                continue
            out.append(
                Cluster(
                    cluster_id=f"mh-{i}",
                    task_ids=sorted(comp),
                    method="minhash",
                    similarity=self._average_pairwise_similarity(comp),
                )
            )
        return out

    def _average_pairwise_similarity(self, task_ids: Iterable[str]) -> float:
        ids = list(task_ids)
        if len(ids) < 2:
            return 1.0
        total = 0.0
        n = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                total += self._minhashes[ids[i]].jaccard(self._minhashes[ids[j]])
                n += 1
        return total / n if n else 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_similarity_minhash.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/similarity/minhash.py \
        tests/test_similarity_minhash.py
git commit -m "feat(similarity): MinHashLSHFinder — O(N) word-ngram near-duplicate clustering"
```

---

### Task 5: scripts/find_near_duplicates_minhash.py

**Files:**
- Create: `scripts/find_near_duplicates_minhash.py`

This script walks every ACCEPTED task, builds the index, dumps a JSON cluster report, and optionally batch-rejects non-representatives. Mirrors the dry-run/--apply convention of `reject_template_tasks.py`.

- [ ] **Step 1: Create the script**

`scripts/find_near_duplicates_minhash.py`:
```python
"""Find near-duplicate ACCEPTED tasks via MinHash + LSH.

Pipeline: load every accepted task, extract canonical text, compute
MinHash signatures, build the LSH index at the requested Jaccard
threshold, and emit a ClusterReport JSON. With --apply, transition
every non-representative task in every cluster from ACCEPTED to
REJECTED (audit stage="similarity_dedup_minhash").

Dry-run is the default. Output JSON is written even in dry-run so the
operator can review clusters before deciding.

Usage:
    .venv/bin/python scripts/find_near_duplicates_minhash.py \\
        --project-root projects/v3_initial_deployment \\
        --shingle-size 5 --jaccard-threshold 0.7 \\
        --report-path /tmp/minhash-clusters.json

    # apply (after reviewing the report):
    .venv/bin/python scripts/find_near_duplicates_minhash.py \\
        --project-root projects/v3_initial_deployment --apply
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime, timezone

from annotation_pipeline_skill.core.models import TaskStatus
from annotation_pipeline_skill.core.transitions import (
    InvalidTransition,
    transition_task,
)
from annotation_pipeline_skill.similarity.clusters import (
    ClusterReport,
    pick_representative,
)
from annotation_pipeline_skill.similarity.extractors import canonical_task_text
from annotation_pipeline_skill.similarity.minhash import MinHashLSHFinder
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


REJECT_STAGE = "similarity_dedup_minhash"
REJECT_ACTOR = "operator"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--shingle-size", type=int, default=5)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--jaccard-threshold", type=float, default=0.7)
    ap.add_argument(
        "--report-path",
        default="/tmp/minhash-clusters.json",
        help="Where to write the ClusterReport JSON (always written, even in dry-run)",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="Transition all non-representative cluster members ACCEPTED → REJECTED",
    )
    args = ap.parse_args(argv)

    root = pathlib.Path(args.project_root) / ".annotation-pipeline"
    store = SqliteStore.open(root)
    row = store._conn.execute("SELECT pipeline_id FROM tasks LIMIT 1").fetchone()
    if row is None:
        print("no tasks in store", file=sys.stderr); return 1
    project_id = row["pipeline_id"]

    print(f"loading ACCEPTED tasks for project {project_id!r}…")
    finder = MinHashLSHFinder(
        shingle_size=args.shingle_size,
        num_perm=args.num_perm,
        jaccard_threshold=args.jaccard_threshold,
    )
    n_added = 0
    for t in store.list_tasks_by_pipeline(project_id):
        if t.status is not TaskStatus.ACCEPTED:
            continue
        text = canonical_task_text(t)
        if not text.strip():
            continue
        finder.add(t.task_id, text)
        n_added += 1
    print(f"indexed {n_added} ACCEPTED tasks")

    clusters = finder.clusters(include_singletons=False)
    n_dup = sum(len(c.task_ids) for c in clusters)
    print(f"found {len(clusters)} clusters covering {n_dup} tasks "
          f"(at jaccard ≥ {args.jaccard_threshold})")

    report = ClusterReport(
        project_id=project_id,
        method="minhash",
        params={
            "shingle_size": args.shingle_size,
            "num_perm": args.num_perm,
            "jaccard_threshold": args.jaccard_threshold,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        clusters=clusters,
    )
    report_path = pathlib.Path(args.report_path)
    report.to_json_file(report_path)
    print(f"report written → {report_path}")

    # Preview top 5 clusters
    print("\ntop clusters by size:")
    for c in clusters[:5]:
        rep = pick_representative(c)
        print(f"  cluster {c.cluster_id}  size={len(c.task_ids)}  sim={c.similarity:.3f}"
              f"  rep={rep}")
        sample = [t for t in c.task_ids if t != rep][:3]
        for tid in sample:
            print(f"    would-reject {tid}")

    if not args.apply:
        print("\n[dry-run] no transitions. Re-run with --apply to commit "
              "(after reviewing the report).")
        return 0

    to_reject = report.tasks_to_reject()
    print(f"\n[apply] transitioning {len(to_reject)} tasks ACCEPTED → REJECTED")
    moved = skipped = 0
    cluster_by_task = {tid: c for c in clusters for tid in c.task_ids}
    for tid in to_reject:
        try:
            t = store.load_task(tid)
        except (FileNotFoundError, KeyError):
            skipped += 1; continue
        if t.status is not TaskStatus.ACCEPTED:
            skipped += 1; continue
        c = cluster_by_task[tid]
        rep = pick_representative(c)
        try:
            ev = transition_task(
                t, TaskStatus.REJECTED,
                actor=REJECT_ACTOR,
                reason=(
                    f"MinHash 近重复检测 (jaccard ≥ {args.jaccard_threshold})；"
                    f"簇 {c.cluster_id}（{len(c.task_ids)} 个 task，相似度 "
                    f"{c.similarity:.2f}）保留代表 {rep}"
                ),
                stage=REJECT_STAGE,
                attempt_id=None,
                metadata={
                    "rejection_kind": "similarity_dedup_minhash",
                    "cluster_id": c.cluster_id,
                    "cluster_size": len(c.task_ids),
                    "cluster_similarity": c.similarity,
                    "representative_task_id": rep,
                    "jaccard_threshold": args.jaccard_threshold,
                    "shingle_size": args.shingle_size,
                    "previous_status": "accepted",
                    "reversible_via": "manual_drag to ARBITRATING or ACCEPTED",
                },
            )
            store.save_task(t)
            store.append_event(ev)
            moved += 1
        except InvalidTransition as exc:
            print(f"  skip {tid}: {exc}")
            skipped += 1
    print(f"[apply] moved={moved}  skipped={skipped}")
    print("\nNext step: scripts/rebootstrap_stats_merged.py --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: Run dry-run on the actual project**

Run: `.venv/bin/python scripts/find_near_duplicates_minhash.py --project-root projects/v3_initial_deployment --jaccard-threshold 0.7 2>&1 | tail -30`
Expected: prints the indexed task count, found clusters (likely 0 — the template batch is already rejected so the remaining accepted set should be diverse), writes `/tmp/minhash-clusters.json`, and ends with `[dry-run] no transitions`.

- [ ] **Step 3: Verify the report file is valid JSON**

Run: `.venv/bin/python -c "import json; d=json.load(open('/tmp/minhash-clusters.json')); print('clusters:', len(d['clusters']), 'method:', d['method'])"`
Expected: `clusters: <N> method: minhash`

- [ ] **Step 4: Commit**

```bash
git add scripts/find_near_duplicates_minhash.py
git commit -m "feat(similarity): scripts/find_near_duplicates_minhash.py — phase 1 dedup driver"
```

---

## Phase 2 — Embedding-based semantic clustering

### Task 6: SimilarityProfile + similarity_profiles.yaml loader

**Files:**
- Create: `annotation_pipeline_skill/similarity/profiles.py`
- Test: `tests/test_similarity_profiles.py`

Mirrors `annotation_pipeline_skill/llm/profiles.py`: profile dataclass + yaml loader that searches workspace-global then project-local. Keeps embedding-provider config out of code.

- [ ] **Step 1: Write the failing test**

`tests/test_similarity_profiles.py`:
```python
import pytest

from annotation_pipeline_skill.similarity.profiles import (
    SimilarityProfile,
    load_similarity_profiles,
)


def test_load_profiles_from_yaml(tmp_path):
    yaml_text = """
profiles:
  jina_small:
    provider: jina_http
    model: jina-embeddings-v3-small-en
    base_url: http://127.0.0.1:8001
    api_key: sk-local
    batch_size: 32
    max_tokens: 512
  random_baseline:
    provider: random
    model: random-128
    dim: 128
"""
    path = tmp_path / "similarity_profiles.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    profiles = load_similarity_profiles(path)
    assert "jina_small" in profiles
    assert profiles["jina_small"].provider == "jina_http"
    assert profiles["jina_small"].base_url == "http://127.0.0.1:8001"
    assert profiles["jina_small"].batch_size == 32
    assert profiles["random_baseline"].provider == "random"


def test_load_profiles_rejects_unknown_provider(tmp_path):
    yaml_text = """
profiles:
  bad:
    provider: not_a_real_provider
    model: x
"""
    path = tmp_path / "similarity_profiles.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown provider"):
        load_similarity_profiles(path)


def test_profile_resolve_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "from-env")
    p = SimilarityProfile(
        name="jina_small", provider="jina_http", model="x",
        base_url="http://x", api_key=None, api_key_env="JINA_API_KEY",
    )
    assert p.resolve_api_key() == "from-env"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_similarity_profiles.py -q`
Expected: import error / 3 failures.

- [ ] **Step 3: Write minimal implementation**

`annotation_pipeline_skill/similarity/profiles.py`:
```python
"""SimilarityProfile + yaml loader — config for embedding providers.

Mirrors annotation_pipeline_skill/llm/profiles.py. A workspace-global
``similarity_profiles.yaml`` or a project-local one defines named
profiles (jina_http, random baseline, future sentence-transformers
local, etc.). Scripts pass ``--profile <name>`` to pick one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ALLOWED_PROVIDERS = {"jina_http", "random"}


@dataclass(frozen=True)
class SimilarityProfile:
    name: str
    provider: str  # one of ALLOWED_PROVIDERS
    model: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    batch_size: int = 32
    max_tokens: int | None = None
    timeout_seconds: float = 60.0
    dim: int | None = None  # for the random-baseline provider

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


def load_similarity_profiles(path: Path | str) -> dict[str, SimilarityProfile]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profiles_raw = raw.get("profiles") or {}
    if not isinstance(profiles_raw, dict):
        raise ValueError("similarity_profiles.yaml must define a top-level 'profiles' map")
    out: dict[str, SimilarityProfile] = {}
    for name, body in profiles_raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"profile {name!r} must be a mapping")
        provider = body.get("provider")
        if provider not in ALLOWED_PROVIDERS:
            raise ValueError(
                f"profile {name!r}: unknown provider {provider!r}; "
                f"allowed: {sorted(ALLOWED_PROVIDERS)}"
            )
        out[name] = SimilarityProfile(
            name=name,
            provider=provider,
            model=str(body.get("model") or ""),
            base_url=body.get("base_url"),
            api_key=body.get("api_key"),
            api_key_env=body.get("api_key_env"),
            batch_size=int(body.get("batch_size") or 32),
            max_tokens=body.get("max_tokens"),
            timeout_seconds=float(body.get("timeout_seconds") or 60.0),
            dim=body.get("dim"),
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_similarity_profiles.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add annotation_pipeline_skill/similarity/profiles.py \
        tests/test_similarity_profiles.py
git commit -m "feat(similarity): SimilarityProfile + yaml loader (workspace/project resolution)"
```

---

### Task 7: EmbeddingProvider Protocol + Jina HTTP client

**Files:**
- Create: `annotation_pipeline_skill/similarity/embeddings.py`
- Test: `tests/test_similarity_embeddings.py`
- Modify: `pyproject.toml` (add httpx)

- [ ] **Step 1: Add httpx**

Edit `pyproject.toml` `dependencies`:
```toml
  "httpx>=0.27",
```

Run: `.venv/bin/pip install -e . 2>&1 | tail -3`
Expected: httpx installed (or already satisfied).

- [ ] **Step 2: Write the failing test**

`tests/test_similarity_embeddings.py`:
```python
import numpy as np
import pytest

from annotation_pipeline_skill.similarity.embeddings import (
    EmbeddingResult,
    JinaHTTPEmbeddingClient,
    RandomEmbeddingClient,
    build_embedding_client,
)
from annotation_pipeline_skill.similarity.profiles import SimilarityProfile


def test_random_client_returns_deterministic_vectors():
    profile = SimilarityProfile(
        name="rand", provider="random", model="r-128", dim=128,
    )
    client = build_embedding_client(profile)
    out = client.embed(["a", "b", "a"])
    assert isinstance(out, EmbeddingResult)
    assert out.vectors.shape == (3, 128)
    # Same text → same vector (deterministic seed by hash).
    np.testing.assert_allclose(out.vectors[0], out.vectors[2])


def test_jina_http_client_batches_and_returns_vectors(monkeypatch):
    # Mock httpx.Client.post to avoid hitting a real server.
    profile = SimilarityProfile(
        name="jina_small", provider="jina_http",
        model="jina-embeddings-v3-small-en",
        base_url="http://127.0.0.1:8001", api_key="sk-test",
        batch_size=2,
    )
    client = build_embedding_client(profile)
    calls = []

    class FakeResp:
        def __init__(self, vectors):
            self.vectors = vectors
        def raise_for_status(self): pass
        def json(self):
            return {"data": [{"embedding": v} for v in self.vectors]}

    def fake_post(url, json, headers, timeout):
        calls.append(json["input"])
        return FakeResp([[float(len(t)), 0.0, 1.0] for t in json["input"]])

    monkeypatch.setattr(client._http, "post", fake_post)
    out = client.embed(["aa", "b", "ccc", "dddd"])
    # batch_size=2 → exactly 2 HTTP calls
    assert len(calls) == 2
    assert calls[0] == ["aa", "b"]
    assert calls[1] == ["ccc", "dddd"]
    assert out.vectors.shape == (4, 3)
    np.testing.assert_array_equal(out.vectors[:, 0], [2.0, 1.0, 3.0, 4.0])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_similarity_embeddings.py -q`
Expected: import error.

- [ ] **Step 4: Write minimal implementation**

`annotation_pipeline_skill/similarity/embeddings.py`:
```python
"""EmbeddingProvider Protocol + concrete clients.

Two providers ship:

  - ``JinaHTTPEmbeddingClient`` — talks to a Jina-compatible HTTP server
    (any OpenAI-compatible /v1/embeddings endpoint also works). Used for
    the local Jina-embedding-v3-small server.
  - ``RandomEmbeddingClient`` — deterministic per-text random vectors,
    useful as a baseline / for tests when no real server is running.

``build_embedding_client(profile)`` is the factory: pick the right
client based on ``profile.provider``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from annotation_pipeline_skill.similarity.profiles import SimilarityProfile


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray  # shape (N, dim)
    model: str
    provider: str


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingResult: ...


class JinaHTTPEmbeddingClient:
    """OpenAI-compatible /v1/embeddings client. Tested against a local
    Jina embedding server but anything that speaks the OpenAI shape
    works.
    """

    def __init__(self, profile: SimilarityProfile):
        self.profile = profile
        if not profile.base_url:
            raise ValueError(f"profile {profile.name!r} needs base_url")
        self._http = httpx.Client(timeout=profile.timeout_seconds)
        self._url = profile.base_url.rstrip("/") + "/v1/embeddings"

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=np.zeros((0, 0), dtype=np.float32),
                model=self.profile.model, provider="jina_http",
            )
        headers = {"Content-Type": "application/json"}
        api_key = self.profile.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        all_vecs: list[list[float]] = []
        bs = max(1, self.profile.batch_size)
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            resp = self._http.post(
                self._url,
                json={"model": self.profile.model, "input": batch},
                headers=headers,
                timeout=self.profile.timeout_seconds,
            )
            resp.raise_for_status()
            payload = resp.json()
            for row in payload["data"]:
                all_vecs.append(row["embedding"])
        return EmbeddingResult(
            vectors=np.asarray(all_vecs, dtype=np.float32),
            model=self.profile.model, provider="jina_http",
        )

    def close(self) -> None:
        self._http.close()


class RandomEmbeddingClient:
    """Deterministic per-text random vectors. Seeded from a hash of the
    text so the same text always maps to the same vector — useful for
    unit tests, plumbing checks, and as a structural baseline."""

    def __init__(self, profile: SimilarityProfile):
        self.profile = profile
        self.dim = int(profile.dim or 128)

    def embed(self, texts: list[str]) -> EmbeddingResult:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(
                hashlib.sha256(t.encode("utf-8")).digest()[:4], "little",
            )
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self.dim).astype(np.float32)
        return EmbeddingResult(vectors=out, model=self.profile.model, provider="random")

    def close(self) -> None:
        return None


def build_embedding_client(profile: SimilarityProfile) -> EmbeddingProvider:
    if profile.provider == "jina_http":
        return JinaHTTPEmbeddingClient(profile)
    if profile.provider == "random":
        return RandomEmbeddingClient(profile)
    raise ValueError(f"unknown embedding provider: {profile.provider!r}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_similarity_embeddings.py -q`
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add annotation_pipeline_skill/similarity/embeddings.py \
        tests/test_similarity_embeddings.py \
        pyproject.toml
git commit -m "feat(similarity): EmbeddingProvider Protocol + Jina HTTP + random baseline"
```

---

### Task 8: Add UMAP + HDBSCAN dependencies

**Files:**
- Modify: `pyproject.toml`

UMAP for 2D projection, HDBSCAN for density-based clustering that doesn't need a fixed k.

- [ ] **Step 1: Add to pyproject.toml**

Edit `pyproject.toml` `dependencies`:
```toml
  "umap-learn>=0.5",
  "hdbscan>=0.8",
```

- [ ] **Step 2: Install + verify**

Run: `.venv/bin/pip install -e . 2>&1 | tail -3`
Expected: `Successfully installed umap-learn-... hdbscan-...`

Run: `.venv/bin/python -c "import umap, hdbscan; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: umap-learn + hdbscan for semantic-cluster pipeline"
```

---

### Task 9: scripts/find_semantic_clusters.py

**Files:**
- Create: `scripts/find_semantic_clusters.py`

End-to-end: load tasks → extract text → embed via configured provider → UMAP-2D → HDBSCAN → emit ClusterReport + 2D coords + optional PNG.

- [ ] **Step 1: Create the script**

`scripts/find_semantic_clusters.py`:
```python
"""Find semantic-similar ACCEPTED tasks via embedding + UMAP + HDBSCAN.

Pipeline:
  1. Load every ACCEPTED task, extract canonical text.
  2. Embed via the configured SimilarityProfile (default: jina_small).
  3. UMAP-project to 2D.
  4. HDBSCAN cluster.
  5. Emit ClusterReport JSON + per-task (x, y, cluster_id) coords.
  6. (Optional) PNG scatter plot.

With --apply, transition non-representative cluster members ACCEPTED →
REJECTED (audit stage="similarity_dedup_embedding"). Dry-run is the
default — always inspect the cluster report first; embedding clusters
are looser than MinHash clusters so false positives matter more.

Usage:
    .venv/bin/python scripts/find_semantic_clusters.py \\
        --project-root projects/v3_initial_deployment \\
        --profile jina_small \\
        --min-cluster-size 5 \\
        --report-path /tmp/embedding-clusters.json \\
        --coords-path /tmp/embedding-coords.json \\
        --plot-path /tmp/embedding-scatter.png
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

from annotation_pipeline_skill.core.models import TaskStatus
from annotation_pipeline_skill.core.transitions import (
    InvalidTransition,
    transition_task,
)
from annotation_pipeline_skill.similarity.clusters import (
    Cluster,
    ClusterReport,
    pick_representative,
)
from annotation_pipeline_skill.similarity.embeddings import build_embedding_client
from annotation_pipeline_skill.similarity.extractors import canonical_task_text
from annotation_pipeline_skill.similarity.profiles import load_similarity_profiles
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


REJECT_STAGE = "similarity_dedup_embedding"
REJECT_ACTOR = "operator"


def _resolve_profiles_path(workspace_root: pathlib.Path) -> pathlib.Path:
    # workspace-global > project-local
    ws = workspace_root / "similarity_profiles.yaml"
    if ws.exists():
        return ws
    raise FileNotFoundError(
        f"no similarity_profiles.yaml found at {ws} — create one with at "
        f"least one profile (see docs in similarity/profiles.py)"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--profile", required=True,
                    help="Profile name from similarity_profiles.yaml")
    ap.add_argument("--profiles-yaml", default=None,
                    help="Path to similarity_profiles.yaml; default: <workspace>/similarity_profiles.yaml")
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--min-cluster-size", type=int, default=5,
                    help="HDBSCAN min_cluster_size; smaller = more, looser clusters")
    ap.add_argument("--report-path", default="/tmp/embedding-clusters.json")
    ap.add_argument("--coords-path", default="/tmp/embedding-coords.json")
    ap.add_argument("--plot-path", default=None,
                    help="Optional path to write a PNG scatter plot")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    project_root = pathlib.Path(args.project_root)
    root = project_root / ".annotation-pipeline"
    store = SqliteStore.open(root)
    row = store._conn.execute("SELECT pipeline_id FROM tasks LIMIT 1").fetchone()
    if row is None:
        print("no tasks in store", file=sys.stderr); return 1
    project_id = row["pipeline_id"]

    profiles_path = pathlib.Path(args.profiles_yaml) if args.profiles_yaml \
        else _resolve_profiles_path(project_root.parent)
    profiles = load_similarity_profiles(profiles_path)
    if args.profile not in profiles:
        print(f"profile {args.profile!r} not in {profiles_path}; "
              f"available: {sorted(profiles)}", file=sys.stderr); return 1
    profile = profiles[args.profile]

    print(f"loading ACCEPTED tasks for project {project_id!r}…")
    task_ids: list[str] = []
    texts: list[str] = []
    for t in store.list_tasks_by_pipeline(project_id):
        if t.status is not TaskStatus.ACCEPTED:
            continue
        text = canonical_task_text(t)
        if not text.strip():
            continue
        task_ids.append(t.task_id)
        texts.append(text)
    print(f"  {len(task_ids)} tasks to embed")

    print(f"embedding via {profile.provider}:{profile.model} (batch={profile.batch_size})…")
    client = build_embedding_client(profile)
    emb = client.embed(texts)
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass
    print(f"  vectors: {emb.vectors.shape}")

    import umap, hdbscan
    print("UMAP-projecting to 2D…")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=42,
    )
    coords = reducer.fit_transform(emb.vectors)
    print(f"HDBSCAN clustering (min_cluster_size={args.min_cluster_size})…")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  found {n_clusters} clusters; {(labels == -1).sum()} tasks marked noise")

    # Build clusters
    cluster_members: dict[int, list[str]] = {}
    for tid, lbl in zip(task_ids, labels):
        if lbl == -1:
            continue
        cluster_members.setdefault(int(lbl), []).append(tid)
    clusters: list[Cluster] = []
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    for lbl, members in cluster_members.items():
        # Average pairwise cosine similarity inside the cluster
        idxs = [id_to_idx[m] for m in members]
        sims = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = emb.vectors[idxs[i]], emb.vectors[idxs[j]]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                sims.append(float(np.dot(a, b) / denom) if denom else 0.0)
        avg_sim = float(np.mean(sims)) if sims else 1.0
        clusters.append(
            Cluster(
                cluster_id=f"emb-{lbl}",
                task_ids=sorted(members),
                method="embedding",
                similarity=avg_sim,
            )
        )
    clusters.sort(key=lambda c: len(c.task_ids), reverse=True)

    report = ClusterReport(
        project_id=project_id,
        method="embedding",
        params={
            "profile": args.profile,
            "model": profile.model,
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "min_cluster_size": args.min_cluster_size,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        clusters=clusters,
    )
    report.to_json_file(args.report_path)
    print(f"report → {args.report_path}")

    # Coords output: one record per task with 2D position + cluster_id
    coords_out = [
        {"task_id": tid, "x": float(coords[i, 0]), "y": float(coords[i, 1]),
         "cluster_id": (f"emb-{int(labels[i])}" if labels[i] != -1 else None)}
        for i, tid in enumerate(task_ids)
    ]
    pathlib.Path(args.coords_path).write_text(
        json.dumps(coords_out, indent=2), encoding="utf-8",
    )
    print(f"coords → {args.coords_path}")

    if args.plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        # Noise = grey, clusters = colored
        noise_mask = labels == -1
        ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
                   c="lightgrey", s=4, alpha=0.5, label="noise")
        non_noise = ~noise_mask
        if non_noise.any():
            ax.scatter(coords[non_noise, 0], coords[non_noise, 1],
                       c=labels[non_noise], s=8, cmap="tab20", alpha=0.8)
        ax.set_title(f"{project_id} — {len(task_ids)} tasks, {n_clusters} clusters")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(args.plot_path, dpi=120)
        print(f"plot → {args.plot_path}")

    print("\ntop clusters by size:")
    for c in clusters[:5]:
        rep = pick_representative(c)
        print(f"  {c.cluster_id}  size={len(c.task_ids)}  cos={c.similarity:.3f}  rep={rep}")

    if not args.apply:
        print("\n[dry-run] no transitions. Review the report + plot first.")
        return 0

    to_reject = report.tasks_to_reject()
    print(f"\n[apply] {len(to_reject)} tasks → REJECTED")
    moved = skipped = 0
    cluster_by_task = {tid: c for c in clusters for tid in c.task_ids}
    for tid in to_reject:
        try:
            t = store.load_task(tid)
        except (FileNotFoundError, KeyError):
            skipped += 1; continue
        if t.status is not TaskStatus.ACCEPTED:
            skipped += 1; continue
        c = cluster_by_task[tid]
        rep = pick_representative(c)
        try:
            ev = transition_task(
                t, TaskStatus.REJECTED,
                actor=REJECT_ACTOR,
                reason=(
                    f"embedding 语义聚类 ({profile.model}, cos≈{c.similarity:.2f})；"
                    f"簇 {c.cluster_id}（{len(c.task_ids)} 个 task）保留代表 {rep}"
                ),
                stage=REJECT_STAGE,
                metadata={
                    "rejection_kind": "similarity_dedup_embedding",
                    "cluster_id": c.cluster_id,
                    "cluster_size": len(c.task_ids),
                    "cluster_similarity": c.similarity,
                    "representative_task_id": rep,
                    "embedding_profile": args.profile,
                    "embedding_model": profile.model,
                    "previous_status": "accepted",
                    "reversible_via": "manual_drag to ARBITRATING or ACCEPTED",
                },
            )
            store.save_task(t)
            store.append_event(ev)
            moved += 1
        except InvalidTransition as exc:
            print(f"  skip {tid}: {exc}")
            skipped += 1
    print(f"[apply] moved={moved}  skipped={skipped}")
    print("Next step: scripts/rebootstrap_stats_merged.py --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: Create the workspace similarity_profiles.yaml**

Create `projects/similarity_profiles.yaml`:
```yaml
profiles:
  jina_small:
    provider: jina_http
    model: jina-embeddings-v3-small-en
    base_url: http://127.0.0.1:8001   # adjust to the actual local Jina port
    api_key: ""                       # most local servers don't need one
    batch_size: 32
    max_tokens: 512
    timeout_seconds: 60
  random_baseline:
    provider: random
    model: random-128
    dim: 128
```

- [ ] **Step 3: Smoke-test with the random baseline (no live server needed)**

Run:
```bash
.venv/bin/python scripts/find_semantic_clusters.py \
    --project-root projects/v3_initial_deployment \
    --profile random_baseline \
    --min-cluster-size 10 \
    --report-path /tmp/emb-baseline-clusters.json \
    --coords-path /tmp/emb-baseline-coords.json 2>&1 | tail -15
```
Expected: prints task count, vector shape (N, 128), UMAP / HDBSCAN steps, cluster count (usually noisy — random vectors → many noise points). No crashes; both JSON files written. This confirms the pipeline is wired end-to-end without depending on a live embedding server.

- [ ] **Step 4: Run against the actual Jina server**

Run (after confirming the local Jina server is up on the port in the yaml):
```bash
.venv/bin/python scripts/find_semantic_clusters.py \
    --project-root projects/v3_initial_deployment \
    --profile jina_small \
    --min-cluster-size 5 \
    --report-path /tmp/emb-clusters.json \
    --coords-path /tmp/emb-coords.json \
    --plot-path /tmp/emb-scatter.png 2>&1 | tail -20
```
Expected: real cluster discovery + a scatter PNG. Inspect the report — clusters at high cosine similarity (≥ 0.85) that span semantically-related but not byte-identical tasks are what this pipeline is for.

- [ ] **Step 5: Commit**

```bash
git add scripts/find_semantic_clusters.py projects/similarity_profiles.yaml
git commit -m "feat(similarity): scripts/find_semantic_clusters.py + similarity_profiles.yaml (phase 2)"
```

---

## Self-Review

After implementing, run:

1. **Full similarity test suite:**
   ```bash
   .venv/bin/python -m pytest tests/test_similarity_*.py -q
   ```
   Expected: all green.

2. **Regression on the existing suite:**
   ```bash
   .venv/bin/python -m pytest tests/test_models_and_transitions.py \
       tests/test_local_runtime_scheduler.py \
       tests/test_schema_validation.py \
       tests/test_entity_statistics_service.py -q
   ```
   Expected: all green. Our changes are additive — these touch unrelated paths but should still pass.

3. **Spec checklist:**
   - ☐ `canonical_task_text` deterministic (same input → same output every call)
   - ☐ `ClusterReport` round-trips through JSON
   - ☐ `MinHashLSHFinder` finds the substation-template tasks if you re-accept them (sanity)
   - ☐ Random embedding client returns deterministic vectors keyed on text hash
   - ☐ Jina HTTP client batches per `profile.batch_size`
   - ☐ Both scripts write reports in dry-run; `--apply` is opt-in
   - ☐ Reject path uses the new stages `similarity_dedup_minhash` /
     `similarity_dedup_embedding` (queryable later)
   - ☐ Audit reason text is Chinese-prose, like the existing template
     cleanup commit message — operators read these in the UI
   - ☐ Representatives are picked deterministically (smallest task_id)
   - ☐ Singletons never get rejected

4. **No placeholders / TODOs**: grep your diffs for `TODO`, `FIXME`, `XXX`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-18-similarity-providers.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

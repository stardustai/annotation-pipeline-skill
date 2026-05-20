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
        except ValueError:
            pass
        self._lsh.insert(task_id, m)

    def clusters(
        self,
        *,
        include_singletons: bool = False,
        verify_against_rep: bool = True,
        rep_exclude: set[str] | None = None,
    ) -> list[Cluster]:
        """Return clusters from the accumulated MinHash index.

        With ``verify_against_rep=True`` (default), each connected
        component is *trimmed* so that every non-rep member has a
        direct MinHash Jaccard ≥ ``jaccard_threshold`` against the
        cluster's representative (lex-smallest key). This eliminates
        chain-linking false positives — pairs that landed in the same
        component only because they share a common intermediate node,
        not because they are directly similar.

        Why this matters: on short texts (1-bigram, 2-trigram rows),
        LSH can transit through "lucky" hash collisions and connect
        rows whose pairwise Jaccard is 0. Rep-anchored verification
        keeps the cluster "centered" on its rep with a guarantee
        ``J(member, rep) ≥ threshold`` for every kept member.

        Trade-off: J(a, b) for two non-rep members can still be below
        threshold (mathematically ≥ 2t − 1 for t = threshold). For
        most practical t (0.3–0.7) this is acceptable. All-pairs
        verification would be N² and prohibitive on large clusters.
        """
        # Build an undirected graph: edge between task A and task B if
        # they collide in LSH (likely Jaccard >= threshold).
        edges: dict[str, set[str]] = {tid: set() for tid in self._minhashes}
        # LSH is a CANDIDATE generator — false positives are part of the
        # design. Verify each candidate pair's actual MinHash Jaccard
        # before accepting the edge, otherwise clusters can include pairs
        # well below the documented threshold (observed: 0.27 in a
        # threshold=0.7 run).
        for tid, m in self._minhashes.items():
            for neighbour in self._lsh.query(m):
                if neighbour == tid or neighbour in edges[tid]:
                    continue
                if m.jaccard(self._minhashes[neighbour]) < self.jaccard_threshold:
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
            # Step 2: rep-anchored verification. Drop members whose
            # direct MinHash Jaccard with the rep is below threshold —
            # they got swept in via chain-linking through intermediate
            # nodes, not because they actually resemble the rep.
            #
            # ``rep_exclude`` lets the caller mark certain keys as
            # ineligible to be the rep (e.g. already-masked rows that
            # still belong in the cluster for display but shouldn't
            # anchor verification). We pick the lex-smallest key NOT in
            # ``rep_exclude``; if every member is excluded, fall back
            # to the lex-smallest of the whole component (so the
            # cluster doesn't disappear silently).
            if verify_against_rep and len(comp) >= 2:
                comp_sorted = sorted(comp)
                rep_key: str
                if rep_exclude:
                    eligible = [k for k in comp_sorted if k not in rep_exclude]
                    rep_key = eligible[0] if eligible else comp_sorted[0]
                else:
                    rep_key = comp_sorted[0]
                rep_mh = self._minhashes[rep_key]
                verified = [rep_key]
                for member in comp_sorted:
                    if member == rep_key:
                        continue
                    mh = self._minhashes[member]
                    if mh.jaccard(rep_mh) >= self.jaccard_threshold:
                        verified.append(member)
                comp = verified

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

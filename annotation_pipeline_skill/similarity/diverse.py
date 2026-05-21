"""Farthest-first diverse-example selection over context snippets.

Used by the annotation knowledge base MCP tool to surface up to k
representative snippets per (span, type) bucket: snippets that are
maximally dissimilar to each other, so an LLM agent sees the breadth
of past contexts rather than near-duplicates.
"""
from __future__ import annotations

from datasketch import MinHash

from annotation_pipeline_skill.similarity.minhash import shingle


_NUM_PERM = 64  # Lower than minhash.py default (128) — we operate on
                # short snippets and only do small pairwise comparisons,
                # so 64 is plenty and ~2x faster to build.


def _build_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=_NUM_PERM)
    for s in shingle(text, n=3):
        m.update(s.encode("utf-8"))
    return m


def select_diverse_examples(snippets: list[str], k: int = 3) -> list[str]:
    """Select up to k snippets that maximize pairwise dissimilarity.

    Algorithm: farthest-first traversal. Seed with the lexicographically
    smallest snippet (deterministic), then repeatedly add the snippet
    whose maximum Jaccard similarity to the already-selected set is
    smallest (i.e., the candidate farthest from its nearest already-selected neighbor).
    """
    # Deduplicate while preserving original ordering for tie-break stability.
    deduped: list[str] = []
    seen: set[str] = set()
    for s in snippets:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    if len(deduped) <= k:
        return deduped

    minhashes = [_build_minhash(s) for s in deduped]
    seed_idx = min(range(len(deduped)), key=lambda i: deduped[i])
    selected: list[int] = [seed_idx]

    while len(selected) < k:
        best_idx, best_distance = -1, -1.0
        for i in range(len(deduped)):
            if i in selected:
                continue
            # Distance to the SET = 1 - max(similarity to any selected).
            max_sim = max(minhashes[i].jaccard(minhashes[j]) for j in selected)
            distance = 1.0 - max_sim
            if distance > best_distance or (
                distance == best_distance and (best_idx == -1 or deduped[i] < deduped[best_idx])
            ):
                best_distance, best_idx = distance, i
        selected.append(best_idx)

    return [deduped[i] for i in selected]

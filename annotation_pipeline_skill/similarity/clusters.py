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

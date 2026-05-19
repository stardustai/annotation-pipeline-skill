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

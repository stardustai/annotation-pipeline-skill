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
    finder = MinHashLSHFinder(shingle_size=3, num_perm=128, jaccard_threshold=0.45)
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
    assert 0.4 <= cluster_with_t1.similarity <= 1.0


def test_finder_emits_singletons_when_requested():
    finder = MinHashLSHFinder(shingle_size=3, num_perm=128, jaccard_threshold=0.45)
    finder.add("alone", "a wholly unique sentence unlike anything else here")
    clusters = finder.clusters(include_singletons=True)
    assert any(c.task_ids == ["alone"] for c in clusters)


def test_finder_skips_empty_text():
    finder = MinHashLSHFinder(shingle_size=3, num_perm=128, jaccard_threshold=0.45)
    finder.add("empty", "")
    finder.add("real", "this has content for shingles to form")
    # Empty text contributes no shingles; finder should not crash and
    # should not group it with anything.
    clusters = finder.clusters(include_singletons=True)
    empty_clusters = [c for c in clusters if "empty" in c.task_ids]
    assert empty_clusters == [] or empty_clusters[0].task_ids == ["empty"]

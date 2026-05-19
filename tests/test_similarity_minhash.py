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
    # Test data mirrors the real-world substation-template shape: ~30 tokens
    # per task with only a handful varying between near-duplicates. At
    # shingle_size=3 this gives Jaccard well above 0.7 between any pair of
    # t1/t2/t3, exercising the production threshold honestly.
    finder = MinHashLSHFinder(shingle_size=3, num_perm=128, jaccard_threshold=0.7)
    base = (
        "As of 2024-09-19 Kano substation KNO-SS-002 reported equipment "
        "{eq} breaker with health score 91 30-day failure probability 0.1 "
        "remaining useful life 258 days recommended action monitor priority low"
    )
    finder.add("t1", base.format(eq="KNO-SS-002-BRE-4974"))
    finder.add("t2", base.format(eq="KNO-SS-002-BRE-4975"))
    finder.add("t3", base.format(eq="KNO-SS-002-BRE-4976"))
    finder.add("u1", "Weekly sync — Project Phoenix: status is on hold, NPS at 3x")
    clusters = finder.clusters()
    # Three near-duplicate substation reports land in one cluster; the
    # unrelated Project Phoenix task is a singleton (excluded by default).
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

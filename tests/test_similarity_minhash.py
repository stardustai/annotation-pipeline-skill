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


def test_finder_rep_verification_drops_chain_linked_outliers():
    """A chain A↔B↔C where A-C have ~0 direct Jaccard should not survive
    rep-anchored verification: only B (and the rep A) remain in the cluster.

    We craft three texts so that:
      • J(A,B) >= 0.30 — share a middle block
      • J(B,C) >= 0.30 — share a different middle block from B↔A
      • J(A,C) ≈ 0     — endpoints don't share trigrams

    Without verification, connected-components would put all three in one
    cluster (LSH adds A-B and B-C edges). With verify_against_rep=True, the
    rep (lex-smallest = "A") is the anchor; B passes (J(A,B)>=0.3), C fails
    (J(A,C)≈0) and is dropped.
    """
    # 20-word rows. A's head and B's head overlap (10 shared words ⇒ strong rep
    # Jaccard); B's tail and C's tail overlap (10 shared words ⇒ chain through B);
    # A and C share NO trigrams (head vs tail of B). So with verification we
    # expect only {A,B}; without, {A,B,C} via the chain.
    A = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo1 lima1 mike1 nov1 osc1 pap1 que1 rom1 sie1 tan1"
    B = "alpha bravo charlie delta echo foxtrot golf hotel india juliet uniform victor whiskey xray yankee zulu aaaa bbbb cccc dddd"
    C = "fff1 ggg1 hhh1 iii1 jjj1 kkk1 lll1 mmm1 nnn1 ooo1 uniform victor whiskey xray yankee zulu aaaa bbbb cccc dddd"

    finder = MinHashLSHFinder(shingle_size=3, num_perm=256, jaccard_threshold=0.20)
    finder.add("A", A); finder.add("B", B); finder.add("C", C)

    # With verification (default): the chain-linked C drops out of the cluster.
    verified = finder.clusters()
    assert len(verified) == 1
    assert set(verified[0].task_ids) == {"A", "B"}, (
        f"expected rep-anchored cluster to keep only A,B but got {verified[0].task_ids}"
    )

    # Without verification: legacy behavior keeps the full chain A,B,C.
    legacy = finder.clusters(verify_against_rep=False)
    assert len(legacy) == 1
    assert set(legacy[0].task_ids) == {"A", "B", "C"}


def test_finder_skips_empty_text():
    finder = MinHashLSHFinder(shingle_size=3, num_perm=128, jaccard_threshold=0.7)
    finder.add("empty", "")
    finder.add("real", "this has content for shingles to form")
    # Empty text contributes no shingles; finder should not crash and
    # should not group it with anything.
    clusters = finder.clusters(include_singletons=True)
    empty_clusters = [c for c in clusters if "empty" in c.task_ids]
    assert empty_clusters == [] or empty_clusters[0].task_ids == ["empty"]

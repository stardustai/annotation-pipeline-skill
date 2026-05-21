from annotation_pipeline_skill.similarity.diverse import select_diverse_examples


def test_returns_input_when_smaller_than_k():
    snippets = ["a", "b"]
    out = select_diverse_examples(snippets, k=3)
    assert out == snippets


def test_returns_k_when_input_larger():
    snippets = [
        "Apple customer support helped me yesterday",
        "Apple customer service was great today",
        "My iPad from Apple broke last week",
        "Apple announced new privacy policy for developers",
    ]
    out = select_diverse_examples(snippets, k=3)
    assert len(out) == 3
    # All returned snippets must be from the input.
    assert all(s in snippets for s in out)


def test_picks_dissimilar_pair_over_near_duplicates():
    snippets = [
        "Apple customer support helped me yesterday",
        "Apple customer support helped me yesterday afternoon",
        "Apple customer support helped me today morning",
        "My iPad from Apple broke last week badly",
    ]
    out = select_diverse_examples(snippets, k=2)
    # Greedy farthest-first should NOT return the two near-duplicates
    # from the top of the list as its 2-element answer.
    assert not (
        out[0].startswith("Apple customer support helped me yesterday")
        and out[1].startswith("Apple customer support helped me yesterday")
    )


def test_deterministic_for_same_input():
    snippets = [
        "Apple customer support helped me yesterday",
        "Apple customer service was great today",
        "My iPad from Apple broke last week",
        "Apple announced new privacy policy for developers",
    ]
    a = select_diverse_examples(snippets, k=3)
    b = select_diverse_examples(snippets, k=3)
    assert a == b


def test_deduplicates_identical_snippets():
    snippets = ["dup", "dup", "dup", "different one entirely"]
    out = select_diverse_examples(snippets, k=3)
    # After dedup only 2 distinct snippets exist.
    assert sorted(out) == sorted(["dup", "different one entirely"])

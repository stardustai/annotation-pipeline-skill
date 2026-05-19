import math
import pytest


def test_type_entropy_empty():
    from annotation_pipeline_skill.interfaces.api import _type_entropy
    assert _type_entropy({}) == 0.0


def test_type_entropy_single_type():
    from annotation_pipeline_skill.interfaces.api import _type_entropy
    assert _type_entropy({"organization": 10}) == pytest.approx(0.0)


def test_type_entropy_uniform_split():
    from annotation_pipeline_skill.interfaces.api import _type_entropy
    # 50/50 split → H = 1.0 bit
    result = _type_entropy({"organization": 5, "project": 5})
    assert result == pytest.approx(1.0)


def test_type_entropy_three_way():
    from annotation_pipeline_skill.interfaces.api import _type_entropy
    # 3-way equal → H = log2(3) ≈ 1.585
    result = _type_entropy({"a": 1, "b": 1, "c": 1})
    assert result == pytest.approx(math.log2(3), rel=1e-6)


def test_wordfreq_score_english():
    from annotation_pipeline_skill.interfaces.api import _wordfreq_score
    score = _wordfreq_score("very nice")
    assert score >= 4.0  # "very" and "nice" are extremely common


def test_wordfreq_score_chinese():
    from annotation_pipeline_skill.interfaces.api import _wordfreq_score
    # 系统 (system) is a very common Chinese word
    score = _wordfreq_score("系统")
    assert score > 0.0


def test_wordfreq_score_empty():
    from annotation_pipeline_skill.interfaces.api import _wordfreq_score
    assert _wordfreq_score("") == 0.0


def test_wordfreq_score_oov():
    from annotation_pipeline_skill.interfaces.api import _wordfreq_score
    # A truly rare/invented token returns 0 from zipf_frequency; average degrades gracefully.
    score = _wordfreq_score("xyzzyqwerty")
    assert score >= 0.0  # no crash, some score

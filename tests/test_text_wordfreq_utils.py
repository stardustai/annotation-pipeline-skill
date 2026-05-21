from annotation_pipeline_skill.text.wordfreq_utils import wordfreq_score


def test_wordfreq_score_high_for_generic_english_word():
    score = wordfreq_score("the")
    assert score > 7.0  # zipf 7+ for ultra-common words


def test_wordfreq_score_low_for_proper_noun():
    score = wordfreq_score("Substack")
    assert score < 4.5


def test_wordfreq_score_handles_cjk():
    score = wordfreq_score("苹果")
    assert score > 4.0  # 'apple' in Chinese is common


def test_wordfreq_score_empty_returns_zero():
    assert wordfreq_score("") == 0.0


def test_wordfreq_score_averages_multi_token_span():
    multi = wordfreq_score("the cat")
    the_score = wordfreq_score("the")
    cat_score = wordfreq_score("cat")
    assert abs(multi - (the_score + cat_score) / 2) < 0.01

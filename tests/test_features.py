import pytest

from shoppilot.ranking.features import brand_in_query, word_overlap


def test_word_overlap_counts_share_of_query_words():
    assert word_overlap("red running shoes", "Nike Running Shoes") == pytest.approx(2 / 3)


def test_word_overlap_empty_query():
    assert word_overlap("", "Nike Running Shoes") == 0.0


def test_brand_found_in_query():
    assert brand_in_query("nike running shoes", "Nike") == 1


def test_empty_brand_is_not_a_match():
    assert brand_in_query("running shoes", "") == 0

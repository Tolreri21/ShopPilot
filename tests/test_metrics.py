import pytest

from shoppilot.eval.metrics import ndcg_at_10


def test_ideal_order_is_one():
    assert ndcg_at_10(["E", "E", "S", "I"], [4, 3, 2, 1]) == pytest.approx(1.0)


def test_irrelevant_first_is_lower():
    assert ndcg_at_10(["I", "E", "E", "S"], [4, 3, 2, 1]) == pytest.approx(0.698, abs=0.001)


def test_all_irrelevant_is_zero():
    assert ndcg_at_10(["I", "I", "I"], [3, 2, 1]) == 0.0

import pandas as pd
import pytest

from shoppilot.eval.metrics import mean_ndcg, ndcg_at_10


def test_ideal_order_is_one():
    assert ndcg_at_10(["E", "E", "S", "I"], [4, 3, 2, 1]) == pytest.approx(1.0)


def test_irrelevant_first_is_lower():
    assert ndcg_at_10(["I", "E", "E", "S"], [4, 3, 2, 1]) == pytest.approx(0.698, abs=0.001)


def test_all_irrelevant_is_zero():
    assert ndcg_at_10(["I", "I", "I"], [3, 2, 1]) == 0.0


def test_mean_ndcg_averages_queries_and_skips_single_product():
    pairs = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2, 3],
            "esci_label": ["E", "I", "I", "E", "E"],
            "score": [2, 1, 2, 1, 1],
        }
    )
    expected = (1.0 + ndcg_at_10(["I", "E"], [2, 1])) / 2
    assert mean_ndcg(pairs, "score") == pytest.approx(expected)

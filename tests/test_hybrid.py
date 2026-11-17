import pandas as pd
import pytest

from shoppilot.search.hybrid import rrf


def test_rrf_uses_ranks_not_scores():
    pairs = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "bm25": [30.0, 20.0, 10.0],
            "dense": [0.1, 0.9, 0.5],
        }
    )
    score = rrf(pairs, ["bm25", "dense"])
    assert score[0] == pytest.approx(1 / 61 + 1 / 63)
    assert score.idxmax() == 1


def test_rrf_ranks_inside_each_query():
    pairs = pd.DataFrame({"query_id": [1, 2], "bm25": [100.0, 1.0]})
    score = rrf(pairs, ["bm25"])
    assert score[0] == score[1]

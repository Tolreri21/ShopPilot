import lightgbm as lgb
import pandas as pd

from shoppilot.ranking.train import expected_gain


def test_expected_gain_is_higher_for_exact_matches():
    features = pd.DataFrame({"x": [1.0, 1.0, 0.0, 0.0] * 10})
    labels = [0, 0, 3, 3] * 10
    model = lgb.LGBMClassifier(n_estimators=10, min_child_samples=1, verbose=-1)
    model.fit(features, labels)
    scores = expected_gain(model, pd.DataFrame({"x": [1.0, 0.0]}))
    assert 0 <= scores[1] < scores[0] <= 1

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from shoppilot.config import settings
from shoppilot.eval.metrics import GAINS, mean_ndcg
from shoppilot.ranking.features import FEATURES
from shoppilot.search.hybrid import rrf

LABELS = ["E", "S", "C", "I"]


def expected_gain(model: lgb.LGBMClassifier, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    gains = np.array([GAINS[LABELS[i]] for i in model.classes_])
    return probabilities @ gains


def main() -> None:
    train = pd.read_parquet(settings.data_dir / "processed/features_train.parquet")
    test = pd.read_parquet(settings.data_dir / "processed/features_test.parquet")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=settings.seed)
    fit_rows, valid_rows = next(splitter.split(train, groups=train["query_id"]))
    fit, valid = train.iloc[fit_rows], train.iloc[valid_rows]
    fit_y = fit["esci_label"].map(LABELS.index)
    valid_y = valid["esci_label"].map(LABELS.index)

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=settings.seed,
        verbose=-1,
    )
    model.fit(
        fit[FEATURES],
        fit_y,
        eval_X=valid[FEATURES],
        eval_y=valid_y,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    print(f"best iteration: {model.best_iteration_}")

    test["hybrid"] = rrf(test, ["bm25", "dense"])
    test["ranker"] = expected_gain(model, test[FEATURES])
    print(f"Hybrid RRF:         {mean_ndcg(test, 'hybrid'):.4f}")
    print(f"Hybrid + LightGBM:  {mean_ndcg(test, 'ranker'):.4f}")

    importance = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    print((importance / importance.sum()).sort_values(ascending=False).round(3))

    (settings.data_dir.parent / "artifacts").mkdir(exist_ok=True)
    model.booster_.save_model(settings.data_dir.parent / "artifacts/ranker.txt")


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.metrics import ndcg_score

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def ndcg_at_10(labels: list[str], scores: list[float]) -> float:
    gains = [GAINS[label] for label in labels]
    return ndcg_score([gains], [scores], k=10)


def mean_ndcg(pairs: pd.DataFrame, score_column: str) -> float:
    results = []
    for _, group in pairs.groupby("query_id"):
        if len(group) < 2:
            continue
        results.append(ndcg_at_10(group["esci_label"].tolist(), group[score_column].tolist()))
    return sum(results) / len(results)

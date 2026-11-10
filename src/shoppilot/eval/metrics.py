from sklearn.metrics import ndcg_score

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def ndcg_at_10(labels: list[str], scores: list[float]) -> float:
    gains = [GAINS[label] for label in labels]
    return ndcg_score([gains], [scores], k=10)

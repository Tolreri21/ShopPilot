import pandas as pd


def rrf(pairs: pd.DataFrame, columns: list[str], k: int = 60) -> pd.Series:
    score = pd.Series(0.0, index=pairs.index)
    for column in columns:
        rank = pairs.groupby("query_id")[column].rank(ascending=False, method="first")
        score += 1 / (k + rank)
    return score

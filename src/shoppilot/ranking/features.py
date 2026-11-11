import pandas as pd

from shoppilot.config import settings
from shoppilot.eval.search_eval import add_bm25, add_dense, load_pairs
from shoppilot.search.dense import INSTRUCTION

FEATURES = [
    "bm25",
    "dense",
    "bm25_rank",
    "dense_rank",
    "word_overlap",
    "brand_in_query",
    "title_words",
    "query_words",
]


def word_overlap(query: str, title: str) -> float:
    query_words = set(query.lower().split())
    title_words = set(title.lower().split())
    if not query_words:
        return 0.0
    return len(query_words & title_words) / len(query_words)


def brand_in_query(query: str, brand: str) -> int:
    brand = brand.lower().strip()
    return int(brand != "" and brand in query.lower())


def add_features(pairs: pd.DataFrame, products: pd.DataFrame) -> None:
    titles = products["product_title"].to_numpy()[pairs["position"]]
    brands = products["product_brand"].fillna("").to_numpy()[pairs["position"]]
    queries = pairs["query"].tolist()

    pairs["bm25_rank"] = pairs.groupby("query_id")["bm25"].rank(ascending=False)
    pairs["dense_rank"] = pairs.groupby("query_id")["dense"].rank(ascending=False)
    pairs["word_overlap"] = [word_overlap(q, t) for q, t in zip(queries, titles)]
    pairs["brand_in_query"] = [brand_in_query(q, b) for q, b in zip(queries, brands)]
    pairs["title_words"] = [len(t.split()) for t in titles]
    pairs["query_words"] = [len(q.split()) for q in queries]


def main() -> None:
    products = pd.read_parquet(settings.data_dir / "processed/products.parquet")
    for split in ["train", "test"]:
        pairs = load_pairs(products, split)
        add_bm25(pairs, products)
        add_dense(pairs, "dense", INSTRUCTION)
        add_features(pairs, products)
        pairs.to_parquet(settings.data_dir / f"processed/features_{split}.parquet")
        print(f"{split}: {len(pairs)} pairs, {pairs['query_id'].nunique()} queries")


if __name__ == "__main__":
    main()

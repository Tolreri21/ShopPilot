import numpy as np
import pandas as pd
from tqdm import tqdm

from shoppilot.config import settings
from shoppilot.eval.metrics import mean_ndcg
from shoppilot.search.bm25 import bm25_scores, build_index, product_texts
from shoppilot.search.dense import INSTRUCTION, encode, load_model
from shoppilot.search.hybrid import rrf


def load_pairs(products: pd.DataFrame, split: str) -> pd.DataFrame:
    queries = pd.read_parquet(settings.data_dir / "processed/queries.parquet")
    judgements = pd.read_parquet(settings.data_dir / "processed/judgements.parquet")
    split_queries = queries[queries["split"] == split]
    pairs = judgements.merge(split_queries[["query_id", "query"]], on="query_id")
    positions = pd.Series(range(len(products)), index=products["product_id"])
    pairs["position"] = pairs["product_id"].map(positions)
    return pairs.reset_index(drop=True)


def add_bm25(pairs: pd.DataFrame, products: pd.DataFrame) -> None:
    retriever = build_index(product_texts(products))
    pairs["bm25"] = 0.0
    for query, group in tqdm(pairs.groupby("query"), desc="bm25"):
        scores = bm25_scores(retriever, query)
        pairs.loc[group.index, "bm25"] = scores[group["position"]]


def add_dense(pairs: pd.DataFrame, column: str, prefix: str) -> None:
    product_vectors = np.load(settings.data_dir / "processed/product_vectors.npy")
    queries = pairs["query"].unique().tolist()
    query_vectors = encode(load_model(), [prefix + q for q in queries])
    rows = pairs["query"].map(pd.Series(range(len(queries)), index=queries))
    pairs[column] = (product_vectors[pairs["position"]] * query_vectors[rows]).sum(axis=1)


def main() -> None:
    products = pd.read_parquet(settings.data_dir / "processed/products.parquet")
    pairs = load_pairs(products, "test")

    rng = np.random.default_rng(settings.seed)
    pairs["random"] = rng.random(len(pairs))
    add_bm25(pairs, products)
    add_dense(pairs, "dense", "")
    add_dense(pairs, "dense_instruction", INSTRUCTION)
    pairs["hybrid"] = rrf(pairs, ["bm25", "dense_instruction"])

    systems = {
        "Random": "random",
        "BM25": "bm25",
        "Embeddings (bge-small)": "dense",
        "Embeddings + query instruction": "dense_instruction",
        "Hybrid RRF": "hybrid",
    }
    print(f"| System | nDCG@10 (test, {pairs['query_id'].nunique()} queries) |")
    print("|---|---|")
    for name, column in systems.items():
        print(f"| {name} | {mean_ndcg(pairs, column):.4f} |")

    pairs.to_parquet(settings.data_dir / "processed/test_scores.parquet")


if __name__ == "__main__":
    main()

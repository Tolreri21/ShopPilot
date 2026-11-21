import bm25s
import numpy as np
import pandas as pd

from shoppilot.config import settings


def product_texts(products: pd.DataFrame) -> list[str]:
    texts = (
        products["product_title"]
        + " "
        + products["product_brand"].fillna("")
        + " "
        + products["product_bullet_point"].fillna("")
    )
    return texts.tolist()


def build_index(texts: list[str]) -> bm25s.BM25:
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en"))
    return retriever


def bm25_scores(retriever: bm25s.BM25, query: str) -> np.ndarray:
    tokens = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)
    if not tokens[0]:
        return np.zeros(retriever.scores["num_docs"])
    return retriever.get_scores(tokens[0])


def main() -> None:
    products = pd.read_parquet(settings.data_dir / "processed/products.parquet")
    retriever = build_index(product_texts(products))
    retriever.save(settings.data_dir.parent / "artifacts/bm25")
    print(f"saved bm25 index for {len(products)} products")


if __name__ == "__main__":
    main()

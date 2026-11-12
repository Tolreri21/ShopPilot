import bm25s
import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import select

from shoppilot.config import settings
from shoppilot.db.models import Product
from shoppilot.db.session import Session
from shoppilot.eval.metrics import GAINS
from shoppilot.ranking.features import FEATURES, add_features
from shoppilot.ranking.train import LABELS
from shoppilot.search.bm25 import bm25_scores
from shoppilot.search.dense import INSTRUCTION, encode, load_model

ARTIFACTS = settings.data_dir.parent / "artifacts"
CANDIDATES = 100


def to_dict(product: Product, score: float | None = None) -> dict:
    return {
        "product_id": product.product_id,
        "title": product.title,
        "brand": product.brand,
        "bullets": product.bullets,
        "price": product.price,
        "in_stock": product.in_stock,
        "score": score,
    }


class Searcher:
    def __init__(self) -> None:
        self.ranker = lgb.Booster(model_file=ARTIFACTS / "ranker.txt")
        self.retriever = bm25s.BM25.load(ARTIFACTS / "bm25")
        self.products = pd.read_parquet(
            settings.data_dir / "processed/products.parquet",
            columns=["product_id", "product_title", "product_brand"],
        )
        self.positions = pd.Series(range(len(self.products)), index=self.products["product_id"])
        self.model = load_model()
        self.gains = np.array([GAINS[label] for label in LABELS])

    def search(
        self, query: str, max_price: float | None = None, in_stock: bool = False, limit: int = 10
    ) -> list[dict]:
        bm25 = bm25_scores(self.retriever, query)
        bm25_ids = self.products["product_id"].to_numpy()[np.argsort(-bm25)[:CANDIDATES]]
        vector = encode(self.model, [INSTRUCTION + query])[0]

        filters = []
        if max_price is not None:
            filters.append(Product.price <= max_price)
        if in_stock:
            filters.append(Product.in_stock)
        distance = Product.embedding.cosine_distance(vector)

        with Session() as session:
            dense_ids = session.scalars(
                select(Product.product_id).where(*filters).order_by(distance).limit(CANDIDATES)
            ).all()
            candidate_ids = set(bm25_ids) | set(dense_ids)
            rows = session.execute(
                select(Product, (1 - distance).label("dense")).where(
                    Product.product_id.in_(candidate_ids), *filters
                )
            ).all()

        if not rows:
            return []
        products = {product.product_id: product for product, _ in rows}
        pairs = pd.DataFrame(
            {
                "query_id": 0,
                "query": query,
                "product_id": [product.product_id for product, _ in rows],
                "dense": [dense for _, dense in rows],
            }
        )
        pairs["position"] = pairs["product_id"].map(self.positions)
        pairs["bm25"] = bm25[pairs["position"]]
        add_features(pairs, self.products)
        pairs["score"] = self.ranker.predict(pairs[FEATURES]) @ self.gains

        top = pairs.sort_values("score", ascending=False).head(limit)
        return [
            to_dict(products[pid], round(s, 4)) for pid, s in zip(top["product_id"], top["score"])
        ]

    def get_product(self, product_id: str) -> dict | None:
        with Session() as session:
            product = session.get(Product, product_id)
            return None if product is None else to_dict(product)

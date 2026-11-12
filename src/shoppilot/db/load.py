import numpy as np
import pandas as pd
from sqlalchemy import delete, insert
from tqdm import tqdm

from shoppilot.config import settings
from shoppilot.db.models import Product
from shoppilot.db.session import Session

BATCH_SIZE = 5000


def product_rows(products: pd.DataFrame, store: pd.DataFrame, vectors: np.ndarray) -> list[dict]:
    data = products.merge(store, on="product_id", how="left", validate="one_to_one")
    rows = []
    for i, row in enumerate(data.itertuples()):
        rows.append(
            {
                "product_id": row.product_id,
                "title": row.product_title,
                "brand": None if pd.isna(row.product_brand) else row.product_brand,
                "bullets": None if pd.isna(row.product_bullet_point) else row.product_bullet_point,
                "price": float(row.price),
                "price_source": row.price_source,
                "in_stock": bool(row.in_stock),
                "embedding": vectors[i],
            }
        )
    return rows


def main() -> None:
    products = pd.read_parquet(settings.data_dir / "processed/products.parquet")
    store = pd.read_parquet(settings.data_dir / "processed/store.parquet")
    vectors = np.load(settings.data_dir / "processed/product_vectors.npy")
    rows = product_rows(products, store, vectors)

    with Session() as session:
        session.execute(delete(Product))
        for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="products"):
            session.execute(insert(Product), rows[start : start + BATCH_SIZE])
        session.commit()
    print(f"loaded {len(rows)} products")


if __name__ == "__main__":
    main()

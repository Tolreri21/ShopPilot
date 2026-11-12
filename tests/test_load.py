import numpy as np
import pandas as pd

from shoppilot.db.load import product_rows


def test_product_rows_join_store_and_vectors():
    products = pd.DataFrame(
        {
            "product_id": ["A", "B"],
            "product_title": ["Phone case", "Rice cooker"],
            "product_brand": ["Spigen", None],
            "product_bullet_point": [None, "5 cups"],
        }
    )
    store = pd.DataFrame(
        {
            "product_id": ["B", "A"],
            "price": [45.0, 9.99],
            "price_source": ["knn", "knn"],
            "in_stock": [False, True],
        }
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    rows = product_rows(products, store, vectors)
    assert rows[0]["price"] == 9.99
    assert rows[1]["brand"] is None
    assert rows[1]["in_stock"] is False
    assert rows[1]["embedding"].tolist() == [0.0, 1.0]

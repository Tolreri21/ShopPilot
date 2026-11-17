import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from shoppilot.config import settings
from shoppilot.data.prices import load_known_prices
from shoppilot.data.simulate import PRODUCT_GROUPS, assign_tier
from shoppilot.search.dense import encode, load_model

NEIGHBORS = 10


def rules_price(title: str) -> float:
    low, high = PRODUCT_GROUPS[assign_tier(title)]["price_range"]
    return (low + high) / 2


def main() -> None:
    known = load_known_prices()
    vectors = encode(load_model(), known["title"].tolist())
    log_prices = np.log(known["price"].to_numpy())

    train_x, test_x, train_y, test_y, _, test_titles = train_test_split(
        vectors, log_prices, known["title"], test_size=0.2, random_state=settings.seed
    )

    print(f"known prices: {len(known)}, hold-out: {len(test_y)}, MAE in log(price)")
    median_pred = np.full(len(test_y), np.median(train_y))
    print(f"median: {mean_absolute_error(test_y, median_pred):.3f}")
    rules_pred = np.log(test_titles.map(rules_price))
    print(f"rules:  {mean_absolute_error(test_y, rules_pred):.3f}")
    for k in [1, 5, 10, 20]:
        knn = KNeighborsRegressor(n_neighbors=k, metric="cosine").fit(train_x, train_y)
        print(f"kNN k={k}: {mean_absolute_error(test_y, knn.predict(test_x)):.3f}")

    knn = KNeighborsRegressor(n_neighbors=NEIGHBORS, metric="cosine").fit(vectors, log_prices)
    product_vectors = np.load(settings.data_dir / "processed/product_vectors.npy")
    store = pd.read_parquet(settings.data_dir / "processed/store.parquet")
    store["price"] = np.exp(knn.predict(product_vectors)).round(2)
    store["price_source"] = "knn"
    store.to_parquet(settings.data_dir / "processed/store.parquet")
    print(f"median store price: ${store['price'].median():.2f}")


if __name__ == "__main__":
    main()

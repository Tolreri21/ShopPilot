import pandas as pd
import pytest

from shoppilot.config import settings
from shoppilot.data import build_subset

N_TRAIN = 6100
N_TEST = 2100


@pytest.fixture
def subset(tmp_path, monkeypatch):
    rows = []
    for query_id in range(N_TRAIN + N_TEST):
        split = "train" if query_id < N_TRAIN else "test"
        for product_id in (f"P{query_id}", f"P{query_id + 1}"):
            rows.append((query_id, f"query {query_id}", product_id, "us", "E", 1, split))
    rows.append((0, "query 0", "PX", "us", "E", 0, "train"))
    rows.append((0, "query 0", "PJ", "jp", "E", 1, "train"))

    examples = pd.DataFrame(
        rows,
        columns=[
            "query_id",
            "query",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "split",
        ],
    )
    all_product_ids = [f"P{i}" for i in range(N_TRAIN + N_TEST + 1)] + ["PX", "PJ"]
    products = pd.DataFrame(
        {
            "product_id": all_product_ids,
            "product_title": "title",
            "product_bullet_point": "bullet",
            "product_brand": "brand",
            "product_locale": "us",
        }
    )
    es_product = products[products["product_id"] == "P0"].assign(product_locale="es")
    products = pd.concat([products, es_product], ignore_index=True)

    raw = tmp_path / "raw"
    raw.mkdir()
    examples.to_parquet(raw / "shopping_queries_dataset_examples.parquet")
    products.to_parquet(raw / "shopping_queries_dataset_products.parquet")

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    build_subset.main()

    processed = tmp_path / "processed"
    return {
        "queries": pd.read_parquet(processed / "queries.parquet"),
        "judgements": pd.read_parquet(processed / "judgements.parquet"),
        "products": pd.read_parquet(processed / "products.parquet"),
    }


def test_no_duplicates(subset):
    assert not subset["queries"].duplicated().any()
    assert not subset["judgements"].duplicated().any()
    assert not subset["products"]["product_id"].duplicated().any()


def test_all_products_are_in_judgements(subset):
    product_ids = set(subset["products"]["product_id"])
    judgement_ids = set(subset["judgements"]["product_id"])

    assert product_ids == judgement_ids


def test_filters_are_applied(subset):
    judgement_ids = set(subset["judgements"]["product_id"])

    assert "PX" not in judgement_ids
    assert "PJ" not in judgement_ids


def test_train_and_test_do_not_overlap(subset):
    queries = subset["queries"]
    train_ids = set(queries[queries["split"] == "train"]["query_id"])
    test_ids = set(queries[queries["split"] == "test"]["query_id"])

    assert len(train_ids) == 6000
    assert len(test_ids) == 2000
    assert train_ids.isdisjoint(test_ids)

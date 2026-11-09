import pandas as pd

from shoppilot.config import settings


def main():
    examples = pd.read_parquet(
        settings.data_dir / "raw/shopping_queries_dataset_examples.parquet",
        filters=[("product_locale", "==", "us"), ("small_version", "==", 1)],
    )
    train_examples = examples[examples["split"] == "train"]
    test_examples = examples[examples["split"] == "test"]

    train_ids = (
        pd.Series(train_examples["query_id"].unique())
        .sample(n=6000, random_state=settings.seed)
        .tolist()
    )
    test_ids = (
        pd.Series(test_examples["query_id"].unique())
        .sample(n=2000, random_state=settings.seed)
        .tolist()
    )

    selected_ids = train_ids + test_ids
    queries = examples[examples["query_id"].isin(selected_ids)][
        ["query_id", "query", "split"]
    ].drop_duplicates()
    judgements = examples[examples["query_id"].isin(selected_ids)][
        ["query_id", "product_id", "esci_label"]
    ]
    product_ids = judgements["product_id"].unique().tolist()

    products = pd.read_parquet(
        settings.data_dir / "raw/shopping_queries_dataset_products.parquet",
        columns=["product_id", "product_title", "product_bullet_point", "product_brand"],
        filters=[("product_id", "in", product_ids), ("product_locale", "==", "us")],
    )
    (settings.data_dir / "processed").mkdir(parents=True, exist_ok=True)

    products.to_parquet(settings.data_dir / "processed/products.parquet")
    judgements.to_parquet(settings.data_dir / "processed/judgements.parquet")
    queries.to_parquet(settings.data_dir / "processed/queries.parquet")

    print(f"queries: {len(queries)}, judgments: {len(judgements)}, products: {len(products)}")


if __name__ == "__main__":
    main()

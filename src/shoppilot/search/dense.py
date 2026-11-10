import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from shoppilot.config import settings

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts, batch_size=128, normalize_embeddings=True, show_progress_bar=len(texts) > 1000
    )


def main() -> None:
    products = pd.read_parquet(settings.data_dir / "processed/products.parquet")
    model = load_model()
    vectors = encode(model, products["product_title"].tolist())
    np.save(settings.data_dir / "processed/product_vectors.npy", vectors)
    print(f"saved {vectors.shape}")


if __name__ == "__main__":
    main()

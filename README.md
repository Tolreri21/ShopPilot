# ShopPilot

Agentic shopping assistant: hybrid retrieval (BM25 + pgvector), a learning-to-rank re-ranker and a
tool-using LLM agent with human-in-the-loop checkout. Every component is evaluated on the
[Amazon ESCI](https://github.com/amazon-science/esci-data) relevance labels, extended with prices,
ratings and reviews from [ESCI-S](https://github.com/shuttie/esci-s).

> Work in progress.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make setup   # Python 3.12 venv + dependencies
make up      # Postgres 16 + pgvector
make data    # download ESCI + ESCI-S into data/raw (~1.2 GB)
make test    # run tests (the DB test needs `make up`)
```

Settings come from environment variables or `.env` (see `.env.example`).

## Data

Both datasets are Apache-2.0. Raw files live in `data/raw/` and are not committed.

## Search results

Task 1 protocol: for each test query only the judged products are ranked. Gains E=1, S=0.1,
C=0.01, I=0. Dev subset: 2000 test queries, 40k judged pairs, seed 42.

| System | nDCG@10 |
|---|---|
| Random | 0.578 |
| BM25 (title + brand + bullets) | 0.695 |
| Embeddings (bge-small, title) | 0.724 |
| Embeddings + query instruction | 0.728 |
| Hybrid RRF (BM25 + embeddings) | 0.725 |

```bash
make embed   # product title embeddings -> data/processed/product_vectors.npy
make eval    # prints the table above
```

## Prices

ESCI has no prices. The 852 US products with real prices from the ESCI-S sample are used to
predict a price for every product with kNN regression on title embeddings (log price).

| Model | MAE in log(price), 20% hold-out |
|---|---|
| Keyword rules | 0.904 |
| Median of all prices | 0.720 |
| kNN, k=10 | 0.681 |

Prices are a simulation for the demo shop, not real Amazon prices.

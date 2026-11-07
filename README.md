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

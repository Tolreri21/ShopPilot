# ShopPilot

Agentic shopping assistant: hybrid retrieval (BM25 + pgvector), a learning-to-rank re-ranker and a
tool-using LLM agent with human-in-the-loop checkout. Every component is evaluated on the
[Amazon ESCI](https://github.com/amazon-science/esci-data) relevance labels, extended with prices,
ratings and reviews from [ESCI-S](https://github.com/shuttie/esci-s).

![ShopPilot demo](docs/demo.gif)

The chat page in action: search with a price filter, add to cart, FAQ, checkout by the user
([mp4 with sound](docs/demo.mp4)). Products and prices come from a real search run; the agent
replies in the clip are scripted.

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
| Hybrid + LightGBM re-ranker | 0.728 |

```bash
make embed   # product title embeddings -> data/processed/product_vectors.npy
make eval    # prints the table above
```

### LightGBM re-ranker

A `LGBMClassifier` predicts the ESCI label of a (query, product) pair from 8 features: BM25 score
and rank, embedding cosine and rank, share of query words in the title, brand in query, title
length and query length. Products are sorted by expected gain
`P(E) + 0.1 * P(S) + 0.01 * P(C)`. It is trained on 6000 train queries, with early stopping on a
20% validation split grouped by query, and evaluated on the 2000 test queries.

Top features by gain: embedding cosine (48%), word overlap (12%), query length (11%).

```bash
make features   # data/processed/features_{train,test}.parquet
make train      # prints nDCG@10 and saves artifacts/ranker.txt
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

## API

Postgres 16 + pgvector, schema managed with Alembic (tables + HNSW index on embeddings).

`GET /search?q=knife set&max_price=50&in_stock=true&limit=10`:
1. BM25 top 100 from an in-memory `bm25s` index;
2. pgvector top 100 by cosine distance, with the price and stock filters in SQL;
3. the union of candidates is re-ranked by the LightGBM model.

`GET /products/{product_id}` returns one product.

```bash
make up        # Postgres
make index     # artifacts/bm25
make load      # alembic upgrade head + load 147k products with embeddings
make api       # http://localhost:8000/docs
```

### Docker

The API also runs in a container (CPU-only torch, ~2.5 GB image). `data/` and `artifacts/` are
mounted read-only, so run `make subset embed prices features train index` and `make load faq` first.

```bash
make docker-api   # builds the image, runs migrations, serves http://localhost:8000
```

## Agent

A LangChain `create_agent` (LangGraph ReAct loop) on Gemini with 5 tools: `search_products`
(the search above), `product_details`, `add_to_cart`, `view_cart` and `shop_faq`. Chat history
is kept per cart with an in-memory LangGraph checkpointer.

The agent cannot place orders. Checkout is a separate endpoint the customer calls after looking
at the cart, so a person always confirms the payment step.

| Endpoint | |
|---|---|
| `POST /carts` | new cart, returns `cart_id` |
| `POST /chat` | `{"cart_id": 1, "message": "..."}`, returns the agent answer |
| `GET /carts/{cart_id}` | items and total |
| `POST /carts/{cart_id}/checkout` | creates an order |

`GET /` is a small chat page: chat on the left, the cart on the right and a Checkout button, so
the order is always placed by the person, not by the agent.

### FAQ retrieval

`shop_faq` searches 15 shop policy entries (shipping, returns, payment...) with pgvector. Tested on
20 paraphrased customer questions:

| FAQ text | Query instruction | Hit@1 | Hit@3 |
|---|---|---|---|
| question | no | 0.85 | 0.90 |
| question | yes | 0.80 | 0.95 |
| question + answer | no | 0.80 | 1.00 |
| question + answer | yes | 0.95 | 1.00 |

```bash
cp .env.example .env   # add GEMINI_API_KEY from https://aistudio.google.com/apikey
make faq               # embed and load the faq
make faq-eval          # prints the table above
make api
```

### Agent evaluation

20 single-turn scenarios in `src/shoppilot/agent/scenarios.json`: catalog search with price and
stock filters, product details, cart actions, shop policies and 6 messages where the agent should
not call any tool (greetings, off-topic, "pay for me", prompt injection). Each scenario runs
with a new cart.

- **tool accuracy**: the first tool call is the expected one (or no call when none is expected);
- **argument accuracy**: the expected arguments are passed, e.g. `max_price=40` for "under $40";
- **grounded answers**: every product id in the answer came from the user message or a tool
  output, so the agent did not invent products.

```bash
make agent-eval   # needs GEMINI_API_KEY, ~4 min because of free tier rate limits
```

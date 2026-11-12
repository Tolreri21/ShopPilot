.PHONY: setup lint format test up down data subset embed eval prices features train index migrate load api

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

up:
	docker compose up -d --wait

down:
	docker compose down

data:
	uv run python -m shoppilot.data.download

subset:
	uv run python -m shoppilot.data.build_subset
	uv run python -m shoppilot.data.simulate

embed:
	uv run python -m shoppilot.search.dense

eval:
	uv run python -m shoppilot.eval.search_eval

prices:
	uv run python -m shoppilot.data.knn_price

features:
	uv run python -m shoppilot.ranking.features

train:
	uv run python -m shoppilot.ranking.train

index:
	uv run python -m shoppilot.search.bm25

migrate:
	uv run alembic upgrade head

load: migrate
	uv run python -m shoppilot.db.load

api:
	uv run uvicorn shoppilot.api.main:app --reload

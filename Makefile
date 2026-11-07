.PHONY: setup lint format test up down data

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

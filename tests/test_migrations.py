import pytest
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from alembic import command
from shoppilot.db.session import engine


def test_migrations_create_all_tables():
    try:
        engine.connect().close()
    except OperationalError:
        pytest.skip("Postgres is not running, start it with `make up`")

    command.upgrade(Config("alembic.ini"), "head")
    tables = set(inspect(engine).get_table_names())
    assert {"products", "carts", "cart_items", "orders", "faq"} <= tables
    indexes = [index["name"] for index in inspect(engine).get_indexes("products")]
    assert "products_embedding_idx" in indexes

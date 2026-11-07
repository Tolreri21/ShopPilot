import psycopg
import pytest

from shoppilot.config import settings


def test_pgvector_is_installed():
    try:
        conn = psycopg.connect(settings.database_url, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running, start it with `make up`")

    with conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        distance = conn.execute("SELECT '[1, 0]'::vector <-> '[0, 0]'::vector").fetchone()[0]

    assert distance == 1.0

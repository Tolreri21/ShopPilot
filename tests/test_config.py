from shoppilot.config import Settings


def test_settings_read_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db:5433/test")

    assert Settings().database_url == "postgresql://user:pass@db:5433/test"

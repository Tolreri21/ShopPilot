from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://shoppilot:shoppilot@localhost:5432/shoppilot"
    data_dir: Path = Path("data")
    seed: int = 42


settings = Settings()

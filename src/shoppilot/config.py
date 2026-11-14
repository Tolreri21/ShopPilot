from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://shoppilot:shoppilot@localhost:5432/shoppilot"
    data_dir: Path = Path("data")
    seed: int = 42
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"


settings = Settings()

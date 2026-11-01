import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "examate-rag"
    environment: str = "local"  # local | ci | production
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    )
    primary_model: str = "gemini-3.8-flash"
    secondary_model: str = "gemini-3.1-flash-lite"
    database_url: str = "sqlite:///gemini_usage.db"
    llm_timeout_seconds: float = 15.0
    llm_max_retries: int = 2
    usd_to_inr_rate: float = 84.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

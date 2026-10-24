from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
import os 
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "examate-rag"
    environment: str = "local"  # local | ci | production
    gemini_api_key:str = str(os.getenv("GOOGLE_API_KEY"))
    primary_model:str = "gemini-3.8-flash"
    secondary_model:str = "gemini-3.1-flash-lite"

@lru_cache
def get_settings() -> Settings:
    return Settings()

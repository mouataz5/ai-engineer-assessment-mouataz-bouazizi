from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment / .env.

    Keys are optional so the app and the test suite can start without
    credentials; a missing key only fails when that client is actually used.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"

    superhero_api_token: str | None = None
    superhero_base_url: str = "https://superheroapi.com/api"

    http_timeout: float = 5.0
    llm_timeout: float = 20.0
    top_k: int = 3

    db_path: str = "data/movies.db"
    dataset_csv: str = "data/movies.csv"


@lru_cache
def get_settings() -> Settings:
    return Settings()

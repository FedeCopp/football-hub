"""
config.py
Configurazione centralizzata letta dal file .env
Supporta: Groq (gratis), OpenAI, Ollama locale
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database — Neon PostgreSQL (cloud gratuito) o locale
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/footballhub"
    REDIS_URL: str = "redis://localhost:6379/0"

    # API Keys calcio
    FOOTBALL_DATA_API_KEY: str = ""
    API_FOOTBALL_KEY: str = ""
    ODDS_API_KEY: str = ""

    # LLM — Groq (gratis) ha priorità, poi OpenAI, poi Ollama locale
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-70b-versatile"

    OPENAI_API_KEY: str = ""

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # Scraping
    SCRAPER_USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    SCRAPER_DELAY: float = 2.0
    PROXY_URL: str = ""

    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False
    SECRET_KEY: str = "change-this-in-production"

    @property
    def llm_provider(self) -> str:
        if self.GROQ_API_KEY:
            return "groq"
        if self.OPENAI_API_KEY:
            return "openai"
        return "ollama"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

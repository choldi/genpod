"""Configuration settings for lightTTS using pydantic-settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PORT: int = 8000
    DEVICE: str = "cpu"
    MODELS_PATH: str = "./data/models"
    VOICES_PATH: str = "./data/voices"
    
    # Logging configuration
    LOGLEVEL: str = "INFO"
    LOGDEST: str = "console"  # console, file, both
    LOGPATH: str = "./logs/lighttts.log"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = Settings()

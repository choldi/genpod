"""Configuration settings for lightTTS using pydantic-settings."""

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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

"""Dependency injection for FastAPI."""

from functools import lru_cache
from core.config import settings
from core.lighttts.engine import LightTTSEngine


@lru_cache
def get_settings():
    """Get cached settings instance."""
    return settings


# Singleton instance of LightTTSEngine
_lighttts_engine: LightTTSEngine | None = None


def get_lighttts_engine() -> LightTTSEngine:
    """Dependency to get LightTTSEngine singleton instance."""
    global _lighttts_engine
    if _lighttts_engine is None:
        s = get_settings()
        _lighttts_engine = LightTTSEngine(
            models_path=s.models_path,
            voices_path=s.voices_path,
            device=s.device,
        )
    return _lighttts_engine

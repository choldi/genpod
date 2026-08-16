"""Dependency injection for FastAPI."""

from functools import lru_cache
from core.config import settings
from core.lighttts.engine import LightTTSEngine
from core.exceptions import ModelLoadError
import logging

logger = logging.getLogger(__name__)


@lru_cache
def get_settings():
    """Get cached settings instance."""
    return settings


# Singleton instance of LightTTSEngine
_lighttts_engine: LightTTSEngine | None = None
_engine_init_error: Exception | None = None


def get_lighttts_engine() -> LightTTSEngine:
    """Dependency to get LightTTSEngine singleton instance."""
    global _lighttts_engine, _engine_init_error
    
    # If we already have an engine, return it
    if _lighttts_engine is not None:
        return _lighttts_engine
    
    # If we had a previous initialization error, re-raise it
    if _engine_init_error is not None:
        logger.error("Returning previous engine initialization error")
        raise _engine_init_error
    
    # Try to initialize the engine
    try:
        s = get_settings()
        logger.info(f"Initializing LightTTSEngine with models_path={s.MODELS_PATH}, voices_path={s.VOICES_PATH}, device={s.DEVICE}")
        _lighttts_engine = LightTTSEngine(
            models_path=s.MODELS_PATH,
            voices_path=s.VOICES_PATH,
            device=s.DEVICE,
        )
        logger.info("LightTTSEngine initialized successfully")
        return _lighttts_engine
    except Exception as e:
        _engine_init_error = e
        logger.error(f"Failed to initialize LightTTSEngine: {e}", exc_info=True)
        # Re-raise as ModelLoadError for consistent error handling
        if isinstance(e, ModelLoadError):
            raise
        raise ModelLoadError(f"Engine initialization failed: {e}") from e

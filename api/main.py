"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_lighttts_engine, get_settings
from api.routes import tts, clone, voices
from core.exceptions import (
    VoiceNotFoundError,
    ModelLoadError,
    SynthesisError,
    CloningError,
    AudioTooShortError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting up LightTTS API...")
    settings = get_settings()
    logger.info(f"Device: {settings.device}")
    logger.info(f"Models path: {settings.models_path}")
    logger.info(f"Voices path: {settings.voices_path}")
    yield
    logger.info("Shutting down LightTTS API...")


app = FastAPI(
    title="LightTTS API",
    description="Text-to-Speech API powered by CosyVoice 2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LightTTS API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    settings = get_settings()
    try:
        engine = get_lighttts_engine()
        model_loaded = engine is not None
    except Exception:
        model_loaded = False

    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "device": settings.device,
    }


@app.exception_handler(VoiceNotFoundError)
async def voice_not_found_handler(request: Request, exc: VoiceNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "VoiceNotFoundError", "message": str(exc)},
    )


@app.exception_handler(ModelLoadError)
async def model_load_handler(request: Request, exc: ModelLoadError):
    return JSONResponse(
        status_code=503,
        content={"error": "ModelLoadError", "message": str(exc)},
    )


@app.exception_handler(SynthesisError)
async def synthesis_handler(request: Request, exc: SynthesisError):
    return JSONResponse(
        status_code=500,
        content={"error": "SynthesisError", "message": str(exc)},
    )


@app.exception_handler(CloningError)
async def cloning_handler(request: Request, exc: CloningError):
    return JSONResponse(
        status_code=500,
        content={"error": "CloningError", "message": str(exc)},
    )


@app.exception_handler(AudioTooShortError)
async def audio_too_short_handler(request: Request, exc: AudioTooShortError):
    return JSONResponse(
        status_code=400,
        content={"error": "AudioTooShortError", "message": str(exc)},
    )


app.include_router(voices.router, prefix="/api/v1", tags=["voices"])
app.include_router(tts.router, prefix="/api/v1", tags=["tts"])
app.include_router(clone.router, prefix="/api/v1", tags=["clone"])

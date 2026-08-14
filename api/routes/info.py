"""API information and documentation endpoint."""
from pathlib import Path
from fastapi import APIRouter
from api.schemas import APIInfoResponse

router = APIRouter()

try:
    from core.lighttts.engine import EMOTION_PRESETS
    _EMOTION_TAGS = list(EMOTION_PRESETS.keys())
except ImportError:
    _EMOTION_TAGS = [
        "<happy>", "<sad>", "<serious>", "<whisper>",
        "<angry>", "<narrative>", "<slow>", "<fast>", "<neutral>",
    ]

_ENDPOINTS = {
    "POST /api/v1/tts": "Generate speech from text. Supports emotion tags, speed/pitch control, and streaming.",
    "POST /api/v1/clone": "Clone a voice from reference audio. Requires audio file, transcript, and voice name.",
    "GET /api/v1/voices": "List all available voices (base and cloned).",
    "GET /api/v1/info": "This endpoint. API documentation and usage information.",
    "GET /api/v1/help": "Alias for /info.",
    "GET /api/v1/usage": "Alias for /info.",
    "GET /health": "Health check.",
}

_LANGUAGES = ["en", "es", "ca", "zh", "ja", "ko"]


def _load_readme() -> str:
    """Load README.md content from project root."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "README.md",
        Path("/app/README.md"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                continue
    return ""


def _build_info() -> APIInfoResponse:
    return APIInfoResponse(
        name="GenPod AI Voice Service",
        version="1.0.0",
        description=(
            "Production-ready Text-to-Speech service with voice cloning "
            "capabilities, powered by CosyVoice 2. Supports emotion tags "
            "for prosodic variation, dual mode (fast/studio), and streaming."
        ),
        endpoints=_ENDPOINTS,
        emotion_tags=_EMOTION_TAGS,
        supported_languages=_LANGUAGES,
        readme=_load_readme(),
    )


@router.get("/info", response_model=APIInfoResponse)
async def get_api_info():
    """Get comprehensive API documentation and usage information."""
    return _build_info()


@router.get("/help", response_model=APIInfoResponse)
async def get_help():
    """Alias for /info endpoint."""
    return _build_info()


@router.get("/usage", response_model=APIInfoResponse)
async def get_usage():
    """Alias for /info endpoint."""
    return _build_info()


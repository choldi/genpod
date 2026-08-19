"""Voice cloning routes with enhanced error handling."""

import os
import tempfile
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request

from api.dependencies import get_lighttts_engine
from core.lighttts.engine import LightTTSEngine
from core.exceptions import (
    CloningError, 
    AudioTooShortError,
    ModelLoadError,
    GPUOutOfMemoryError,
    ValidationError,
    TTSException
)
from core.logger import get_logger, set_correlation_id, set_request_context, clear_log_context

router = APIRouter()
logger = get_logger(__name__)


@router.post("/clone")
async def clone_voice(
    http_request: Request,
    audio: UploadFile = File(..., description="Audio file for voice cloning"),
    voice_name: str = Form(..., min_length=1, max_length=100),
    transcript: str = Form(..., min_length=1),
    language: str = Form(default="en"),
    model: str = Form(default="cosyvoice2"),
    engine: LightTTSEngine = Depends(get_lighttts_engine),
):
    """Clone a voice from an audio sample with comprehensive error handling."""
    # Set up request context for logging
    correlation_id = set_correlation_id()
    set_request_context({
        "endpoint": "/clone",
        "method": "POST",
        "client_ip": http_request.client.host if http_request.client else "unknown",
        "voice_name": voice_name,
        "language": language,
        "model": model
    })
    
    tmp_path = None
    try:
        # Validate file type
        allowed_types = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/flac"]
        if audio.content_type not in allowed_types:
            logger.warning(f"Invalid audio file type: {audio.content_type}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format. Allowed: {', '.join(allowed_types)}"
            )
        
        # Validate file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        content = await audio.read()
        if len(content) > max_size:
            logger.warning(f"Audio file too large: {len(content)} bytes")
            raise HTTPException(status_code=400, detail="Audio file too large (max 50MB)")
        
        # 1. Guardar el archivo subido en un temporal
        suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"Cloning voice '{voice_name}' from {tmp_path}")

        # Validate model parameter
        supported_models = engine.SUPPORTED_MODELS
        # Handle model selection logic:
        # 1. If model is not provided or empty, use first model alphabetically
        # 2. If model provided but not supported, return error with list of supported models
        if not model or model.strip() == "":
            model = sorted(supported_models)[0]
            logger.info(f"No model provided for cloning, using default model: {model}")
        elif model not in supported_models:
            logger.error(f"Unsupported model for cloning: {model}. Supported: {supported_models}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {model}. Supported models: {sorted(supported_models)}"
            )

        # 2. Llamar al motor (esto devuelve un string: el voice_id)
        voice_id = engine.clone_voice(
            audio_path=tmp_path,
            transcript=transcript,
            voice_name=voice_name,
            language=language,
            model=model,
        )

        # 3. Devolver la respuesta correcta
        logger.info(f"Voice cloned successfully: {voice_id}")
        return {
            "message": "Voice cloned successfully",
            "voice_id": voice_id,
            "model": model
        }

    except AudioTooShortError as e:
        logger.warning(f"Audio too short: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ModelLoadError as e:
        logger.error(f"Model load error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except GPUOutOfMemoryError as e:
        logger.error(f"GPU OOM during cloning: {e}")
        raise HTTPException(status_code=503, detail="GPU out of memory. Try again later.")
    except CloningError as e:
        logger.error(f"Cloning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except TTSException as e:
        logger.error(f"TTS error: {e.message}", extra={"error_details": e.to_dict()})
        raise _map_tts_exception(e)
    except Exception as e:
        logger.error(f"Unexpected error during cloning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process reference audio: {str(e)}")
    finally:
        # 4. Limpiar el archivo temporal
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {tmp_path}: {e}")
        clear_log_context()


def _map_tts_exception(e: TTSException) -> HTTPException:
    """Map TTS exceptions to appropriate HTTP responses."""
    status_codes = {
        "VOICE_NOT_FOUND": 404,
        "AUDIO_TOO_SHORT": 400,
        "MODEL_LOAD_ERROR": 503,
        "SYNTHESIS_ERROR": 500,
        "CLONING_ERROR": 500,
        "INVALID_TRANSCRIPT": 400,
        "MODEL_NOT_AVAILABLE": 400,
        "GPU_OOM": 503,
        "CONFIGURATION_ERROR": 500,
        "VALIDATION_ERROR": 400,
    }
    
    status_code = status_codes.get(e.error_code, 500)
    
    headers = {}
    if e.recoverable and e.retry_after:
        headers["Retry-After"] = str(e.retry_after)
    
    return HTTPException(
        status_code=status_code,
        detail=e.to_dict(),
        headers=headers
    )

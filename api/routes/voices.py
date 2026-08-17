"""Voice management routes with dynamic voice listing."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.dependencies import get_lighttts_engine
from api.schemas import VoiceListResponse, VoiceInfo
from core.lighttts.engine import LightTTSEngine
from core.exceptions import TTSException
from core.logger import get_logger, set_correlation_id, set_request_context, clear_log_context

router = APIRouter()
logger = get_logger(__name__)


@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(
    http_request: Request,
    model: str = Query(None, description="Filter voices by model type (cosyvoice2, cosyvoice3, voxcpm)"),
    engine: LightTTSEngine = Depends(get_lighttts_engine),
):
    """List all available voices with optional model filtering."""
    correlation_id = set_correlation_id()
    set_request_context({
        "endpoint": "/voices",
        "method": "GET",
        "client_ip": http_request.client.host if http_request.client else "unknown",
        "model_filter": model
    })
    
    try:
        logger.debug(f"Listing all voices (model filter: {model})")
        voices_data = engine.list_voices(model_type=model)
        voice_infos = [
            VoiceInfo(
                voice_id=v.get("voice_id", ""),
                name=v.get("name", ""),
                language=v.get("language", "en"),
                description=v.get("description"),
                metadata=v.get("metadata"),
                is_cloned=v.get("is_cloned", False),
            )
            for v in voices_data
        ]
        logger.info(f"Returning {len(voice_infos)} voices")
        return VoiceListResponse(voices=voice_infos, total=len(voice_infos))
    except TTSException as e:
        logger.error(f"TTS error listing voices: {e.message}", extra={"error_details": e.to_dict()})
        raise _map_tts_exception(e)
    except Exception as e:
        logger.error(f"Error listing voices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_log_context()


@router.delete("/voices/{voice_id}")
async def delete_voice(
    voice_id: str,
    http_request: Request,
    engine: LightTTSEngine = Depends(get_lighttts_engine)
):
    """Delete a cloned voice by ID."""
    correlation_id = set_correlation_id()
    set_request_context({
        "endpoint": "/voices/{voice_id}",
        "method": "DELETE",
        "client_ip": http_request.client.host if http_request.client else "unknown",
        "voice_id": voice_id
    })
    
    try:
        logger.info(f"Deleting voice: {voice_id}")
        # Check if voice exists and is cloned
        voice_info = engine._voice_registry.get_voice_info(voice_id)
        
        if not voice_info.get("is_cloned", False):
            logger.warning(f"Attempted to delete base voice: {voice_id}")
            raise HTTPException(status_code=400, detail="Cannot delete base voices")
        
        # Delete the voice
        result = engine.delete_voice(voice_id)
        if result:
            logger.info(f"Voice '{voice_id}' deleted successfully")
            return {"message": f"Voice '{voice_id}' deleted successfully"}
        else:
            logger.warning(f"Voice '{voice_id}' not found")
            raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
                
    except HTTPException:
        raise
    except TTSException as e:
        logger.error(f"TTS error deleting voice: {e.message}", extra={"error_details": e.to_dict()})
        raise _map_tts_exception(e)
    except Exception as e:
        logger.error(f"Error deleting voice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_log_context()


@router.get("/models")
async def list_models(
    http_request: Request,
    engine: LightTTSEngine = Depends(get_lighttts_engine),
):
    """List available models and their status."""
    correlation_id = set_correlation_id()
    set_request_context({
        "endpoint": "/models",
        "method": "GET",
        "client_ip": http_request.client.host if http_request.client else "unknown"
    })
    
    try:
        logger.debug("Listing available models")
        models = engine.get_available_models()
        return {
            "models": models,
            "default_model": engine.default_model,
            "supported_models": engine.SUPPORTED_MODELS
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
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

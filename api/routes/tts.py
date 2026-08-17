"""Text-to-Speech routes with enhanced error handling."""

import io
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.dependencies import get_lighttts_engine
from api.schemas import TTSRequest
from core.lighttts.engine import LightTTSEngine
from core.exceptions import (
    VoiceNotFoundError, 
    SynthesisError, 
    ModelNotAvailableError,
    GPUOutOfMemoryError,
    ValidationError,
    TTSException
)
from core.logger import get_logger, set_correlation_id, set_request_context, clear_log_context

router = APIRouter()
logger = get_logger(__name__)


@router.post("/tts")
async def synthesize(
    request: TTSRequest,
    http_request: Request,
    engine: LightTTSEngine = Depends(get_lighttts_engine),
):
    """Synthesize speech from text with dual mode support and comprehensive error handling."""
    # Set up request context for logging
    correlation_id = set_correlation_id()
    set_request_context({
        "endpoint": "/tts",
        "method": "POST",
        "client_ip": http_request.client.host if http_request.client else "unknown"
    })
    
    try:
        # --- LOG DE PARÁMETROS RECIBIDOS ---
        logger.info(
            f"TTS Request received: text='{request.text[:50]}...' (len={len(request.text)}), "
            f"voice_id='{request.voice_id}', language='{request.language}', "
            f"stream={request.stream}, mode='{request.mode}', speed={request.speed}, "
            f"pitch={request.pitch}, emotion='{request.emotion}', emotion_tags={request.emotion_tags}, "
            f"chunk_size={request.chunk_size}, model='{request.model}'",
            extra={"correlation_id": correlation_id}
        )
        
        # Validate model parameter
        supported_models = engine.SUPPORTED_MODELS
        if request.model not in supported_models:
            logger.error(f"Unsupported model: {request.model}. Supported: {supported_models}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {request.model}. Supported models: {supported_models}"
            )
        
        # --- MODE LOGIC ---
        is_fast_mode = getattr(request, "mode", "studio") == "fast"
        
        # Get parameters with safe defaults
        speed = getattr(request, "speed", None)
        pitch = getattr(request, "pitch", None)
        emotion = getattr(request, "emotion", "neutral")
        emotion_tags = getattr(request, "emotion_tags", False)
        chunk_size = getattr(request, "chunk_size", None)
        
        # Determine language safely (prioritize 'lang', fallback to 'language')
        lang_val = getattr(request, "language", "en")
        
        if is_fast_mode:
            logger.info(f"Mode: FAST | Lang: {lang_val} | Optimizing for low latency")
            # Ajuste más suave para evitar pérdida de claridad
            if speed is None: speed = 1.05 
            if pitch is None: pitch = 1.0
        else:
            logger.info(f"Mode: STUDIO | Lang: {lang_val} | Optimizing for quality")
            # Ritmo natural para podcasts
            if speed is None: speed = 0.95
            if pitch is None: pitch = 1.0
            
        # Respetar el parámetro stream de la petición
        stream = request.stream
        logger.info(f"Final parameters: speed={speed}, pitch={pitch}, stream={stream}, chunk_size={chunk_size}, model={request.model}")
        # --- END MODE LOGIC ---

        # CRITICAL: Ensure we use the EXACT text from the request
        input_text = request.text
        if not input_text or len(input_text.strip()) == 0:
            logger.warning("Empty text received in request")
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Handle chunk_size logic:
        # - chunk_size > 0: split by that character count
        # - chunk_size <= 0: no splitting (single chunk)
        # - chunk_size is None: use default (5000 characters)
        effective_chunk_size = None
        if chunk_size is not None:
            if chunk_size > 0:
                effective_chunk_size = chunk_size
                logger.info(f"Custom chunk size: {effective_chunk_size} characters")
            else:
                effective_chunk_size = 0  # 0 means no chunking
                logger.info("Chunking disabled (chunk_size <= 0)")
        else:
            effective_chunk_size = 5000  # Default chunk size
            logger.info(f"Using default chunk size: {effective_chunk_size} characters")

        if stream:
            logger.info("Starting streaming response")
            def audio_generator():
                try:
                    logger.debug(f"Streaming synthesis: '{input_text[:50]}...' | chunk_size={effective_chunk_size} | model={request.model}")
                    for chunk in engine.synthesize(
                        text=input_text,
                        voice_id=request.voice_id,
                        lang=lang_val,
                        speed=speed,
                        pitch=pitch,
                        emotion=emotion,
                        stream=True,
                        emotion_tags=emotion_tags,
                        chunk_size=effective_chunk_size,
                        model=request.model,
                    ):
                        yield chunk
                except TTSException as e:
                    logger.error(f"Streaming TTS error: {e.message}", extra={"error_details": e.to_dict()})
                    # Can't raise HTTPException in generator, log and stop
                    return
                except Exception as e:
                    logger.error(f"Streaming error: {e}", exc_info=True)
                    return

            return StreamingResponse(
                audio_generator(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=speech_{request.voice_id}.wav",
                    "X-Correlation-ID": correlation_id
                },
            )
        else:
            logger.info("Starting non-streaming response")
            logger.debug(f"Full synthesis: '{input_text[:50]}...'")
            audio_data = b""
            try:
                for chunk in engine.synthesize(
                    text=input_text,
                    voice_id=request.voice_id,
                    lang=lang_val,
                    speed=speed,
                    pitch=pitch,
                    emotion=emotion,
                    stream=False,
                    emotion_tags=emotion_tags,
                    chunk_size=effective_chunk_size,
                    model=request.model,
                ):
                    audio_data += chunk

                logger.info(f"Non-streaming synthesis completed, total bytes: {len(audio_data)}")
                return StreamingResponse(
                    io.BytesIO(audio_data),
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"attachment; filename=speech_{request.voice_id}.wav",
                        "X-Correlation-ID": correlation_id
                    },
                )
            except TTSException as e:
                logger.error(f"Synthesis TTS error: {e.message}", extra={"error_details": e.to_dict()})
                raise _map_tts_exception(e)
            except Exception as e:
                logger.error(f"Unexpected error in TTS: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
                
    except VoiceNotFoundError as e:
        logger.error(f"Voice not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ModelNotAvailableError as e:
        logger.error(f"Model not available: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except GPUOutOfMemoryError as e:
        logger.error(f"GPU OOM: {e}")
        raise HTTPException(status_code=503, detail="GPU out of memory. Try again later or use a smaller model.")
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except SynthesisError as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in TTS: {e}", exc_info=True)
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
    
    # Add retry-after header for recoverable errors
    headers = {}
    if e.recoverable and e.retry_after:
        headers["Retry-After"] = str(e.retry_after)
    
    return HTTPException(
        status_code=status_code,
        detail=e.to_dict(),
        headers=headers
    )

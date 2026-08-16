"""Text-to-Speech routes."""

import io
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import get_lighttts_engine
from api.schemas import TTSRequest
from core.lighttts.engine import LightTTSEngine
from core.exceptions import VoiceNotFoundError, SynthesisError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/tts")
async def synthesize(
    request: TTSRequest,
    engine: LightTTSEngine = Depends(get_lighttts_engine),
):
    """Synthesize speech from text with dual mode support."""
    try:
        # --- LOG DE PARÁMETROS RECIBIDOS ---
        logger.info(
            f"TTS Request: text_len={len(request.text)}, voice_id='{request.voice_id}', "
            f"language='{request.language}', stream={request.stream}, mode='{request.mode}', "
            f"speed={request.speed}, pitch={request.pitch}, emotion='{request.emotion}', "
            f"emotion_tags={request.emotion_tags}, chunk_size={request.chunk_size}"
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
            stream = True
        else:
            logger.info(f"Mode: STUDIO | Lang: {lang_val} | Optimizing for quality")
            # Ritmo natural para podcasts
            if speed is None: speed = 0.95
            if pitch is None: pitch = 1.0
            stream = True
        # --- END MODE LOGIC ---

        # CRITICAL: Ensure we use the EXACT text from the request
        input_text = request.text
        if not input_text or len(input_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if stream:
            def audio_generator():
                try:
                    logger.debug(f"Streaming synthesis: '{input_text[:50]}...' | chunk_size={chunk_size}")
                    for chunk in engine.synthesize(
                        text=input_text,
                        voice_id=request.voice_id,
                        lang=lang_val,
                        speed=speed,
                        pitch=pitch,
                        emotion=emotion,
                        stream=True,
                        emotion_tags=emotion_tags,
                        chunk_size=chunk_size,
                    ):
                        yield chunk
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    raise

            return StreamingResponse(
                audio_generator(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=speech_{request.voice_id}.wav"
                },
            )
        else:
            logger.debug(f"Full synthesis: '{input_text[:50]}...'")
            audio_data = b""
            for chunk in engine.synthesize(
                text=input_text,
                voice_id=request.voice_id,
                lang=lang_val,
                speed=speed,
                pitch=pitch,
                emotion=emotion,
                stream=False,
                emotion_tags=emotion_tags,
                chunk_size=chunk_size,
            ):
                audio_data += chunk

            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=speech_{request.voice_id}.wav"
                },
            )
    except VoiceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SynthesisError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in TTS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


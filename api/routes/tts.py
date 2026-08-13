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
    """Synthesize speech from text."""
    try:
        # --- MODE LOGIC ---
        is_fast_mode = getattr(request, "mode", "studio") == "fast"
        
        speed = getattr(request, "speed", None)
        pitch = getattr(request, "pitch", None)
        emotion = getattr(request, "emotion", "neutral")
        
        if is_fast_mode:
            logger.info("Mode: FAST | Optimizing for low latency")
            if speed is None: speed = 1.15
            if pitch is None: pitch = 1.0
            stream = True
        else:
            logger.info("Mode: STUDIO | Optimizing for quality")
            if speed is None: speed = 0.95
            if pitch is None: pitch = 1.0
            stream = True
        # --- END MODE LOGIC ---

        # Determine language safely (supports both 'lang' and 'language' from schema)
        lang_val = getattr(request, "lang", getattr(request, "language", "en"))

        if stream:
            def audio_generator():
                try:
                    for chunk in engine.synthesize(
                        text=request.text,
                        voice_id=request.voice_id,
                        lang=lang_val,
                        speed=speed,
                        pitch=pitch,
                        emotion=emotion,
                        stream=True,
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
            audio_data = b""
            for chunk in engine.synthesize(
                text=request.text,
                voice_id=request.voice_id,
                lang=lang_val,
                speed=speed,
                pitch=pitch,
                emotion=emotion,
                stream=False,
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
        logger.error(f"Unexpected error in TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

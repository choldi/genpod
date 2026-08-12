import logging
logger = logging.getLogger(__name__)
"""Text-to-Speech routes."""

import io
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import get_lighttts_engine
from api.schemas import TTSRequest
from core.lighttts.engine import LightTTSEngine
from core.exceptions import VoiceNotFoundError, SynthesisError

router = APIRouter()


@router.post("/tts")
async def synthesize(
    request: TTSRequest,
    engine: LightTTSEngine = Depends(get_lighttts_engine),
):
    """Synthesize speech from text."""
    try:
        if request.stream:
            def audio_generator():
                try:
                    for chunk in engine.synthesize(
                        text=request.text,
                        voice_id=request.voice_id,
                        lang=request.language,
                        stream=True,
                    ):
                        yield chunk
                except Exception as e:
                    logging.error(f"Streaming error: {e}")
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
                lang=request.language,
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
        raise HTTPException(status_code=500, detail=str(e))

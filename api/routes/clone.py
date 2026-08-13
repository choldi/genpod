"""Voice cloning routes."""

import os
import tempfile
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from api.dependencies import get_lighttts_engine
from core.lighttts.engine import LightTTSEngine
from core.exceptions import CloningError, AudioTooShortError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/clone")
async def clone_voice(
    audio: UploadFile = File(..., description="Audio file for voice cloning"),
    voice_name: str = Form(..., min_length=1, max_length=100),
    transcript: str = Form(..., min_length=1),
    language: str = Form(default="es"),
    engine: LightTTSEngine = Depends(get_lighttts_engine),
):
    """Clone a voice from an audio sample."""
    tmp_path = None
    try:
        # 1. Guardar el archivo subido en un temporal
        suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"Cloning voice '{voice_name}' from {tmp_path}")

        # 2. Llamar al motor (esto devuelve un string: el voice_id)
        voice_id = engine.clone_voice(
            audio_path=tmp_path,
            transcript=transcript,
            voice_name=voice_name,
            language=language,
        )

        # 3. Devolver la respuesta correcta (voice_id es un string, no un dict)
        return {
            "message": "Voice cloned successfully",
            "voice_id": voice_id
        }

    except AudioTooShortError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except CloningError as e:
        logger.error(f"Cloning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during cloning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process reference audio: {str(e)}")
    finally:
        # 4. Limpiar el archivo temporal
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

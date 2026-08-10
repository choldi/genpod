"""Voice cloning routes."""

import os
import tempfile
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_lighttts_engine
from api.schemas import CloneResponse
from core.lighttts.engine import LightTTSEngine
from core.exceptions import CloningError, AudioTooShortError

router = APIRouter()


@router.post("/clone", response_model=CloneResponse)
async def clone_voice(
    audio: UploadFile = File(..., description="Audio file for voice cloning"),
    voice_name: str = Form(..., min_length=1, max_length=100),
    transcript: str = Form(..., min_length=1),
    language: str = Form(default="en"),
    engine: LightTTSEngine = Depends(get_lighttts_engine),
):
    """Clone a voice from an audio sample."""
    tmp_path = None
    try:
        suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        result = engine.clone_voice(
            audio_path=tmp_path,
            transcript=transcript,
            voice_name=voice_name,
            language=language,
        )

        return CloneResponse(
            voice_id=result.get("voice_id", ""),
            voice_name=voice_name,
            message="Voice cloned successfully",
        )
    except AudioTooShortError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except CloningError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

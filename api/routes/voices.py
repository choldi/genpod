"""Voice management routes."""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_lighttts_engine
from api.schemas import VoiceListResponse, VoiceInfo
from core.lighttts.engine import LightTTSEngine

router = APIRouter()


@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(engine: LightTTSEngine = Depends(get_lighttts_engine)):
    """List all available voices."""
    try:
        voices_data = engine.list_voices()
        voice_infos = [
            VoiceInfo(
                voice_id=v.get("voice_id", ""),
                name=v.get("name", ""),
                language=v.get("language", "en"),
                description=v.get("description"),
                metadata=v.get("metadata"),
            )
            for v in voices_data
        ]
        return VoiceListResponse(voices=voice_infos, total=len(voice_infos))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

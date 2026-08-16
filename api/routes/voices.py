"""Voice management routes."""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_lighttts_engine
from api.schemas import VoiceListResponse, VoiceInfo
from core.lighttts.engine import LightTTSEngine
from core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(engine: LightTTSEngine = Depends(get_lighttts_engine)):
    """List all available voices."""
    try:
        logger.debug("Listing all voices")
        voices_data = engine.list_voices()
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
    except Exception as e:
        logger.error(f"Error listing voices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/voices/{voice_id}")
async def delete_voice(
    voice_id: str,
    engine: LightTTSEngine = Depends(get_lighttts_engine)
):
    """Delete a cloned voice by ID."""
    try:
        logger.info(f"Deleting voice: {voice_id}")
        # Llamamos al método del motor para borrar la voz
        # Asumimos que el motor tiene un método 'delete_voice' o similar
        if hasattr(engine, 'delete_voice'):
            engine.delete_voice(voice_id)
            logger.info(f"Voice '{voice_id}' deleted successfully")
            return {"message": f"Voice '{voice_id}' deleted successfully"}
        else:
            # Fallback si el motor no tiene el método directo:
            # Intentamos borrarlo del directorio de voces manualmente
            import os
            voice_path = os.path.join(engine.voices_path, voice_id)
            if os.path.exists(voice_path):
                import shutil
                shutil.rmtree(voice_path)
                logger.info(f"Voice '{voice_id}' deleted from disk")
                return {"message": f"Voice '{voice_id}' deleted from disk"}
            else:
                logger.warning(f"Voice '{voice_id}' not found")
                raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
                
    except Exception as e:
        logger.error(f"Error deleting voice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

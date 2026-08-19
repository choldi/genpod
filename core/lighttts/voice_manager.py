"""Voice Manager - Handles reading/writing voice metadata with enhanced error handling."""

import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from core.exceptions import VoiceNotFoundError, ValidationError
from core.logger import get_logger

logger = get_logger(__name__)


class VoiceManager:
    """Manages voice profiles and metadata with atomic operations."""

    def __init__(self, voices_path: str):
        """Initialize voice manager.

        Args:
            voices_path: Path to voices directory.
        """
        self.voices_path = Path(voices_path)
        self.voices_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"VoiceManager initialized with path: {voices_path}")

    def save_voice_metadata(self, voice_id: str, metadata: Dict[str, Any]) -> None:
        """Save voice metadata to JSON file atomically.

        Args:
            voice_id: Unique voice identifier.
            metadata: Voice metadata dictionary.

        Raises:
            ValidationError: If voice_id or metadata is invalid.
        """
        if not voice_id or not isinstance(voice_id, str):
            raise ValidationError("voice_id", voice_id, "Voice ID must be a non-empty string")
        
        if not metadata or not isinstance(metadata, dict):
            raise ValidationError("metadata", metadata, "Metadata must be a dictionary")

        # Verificar si el modelo solicitado existe
        model = metadata.get('model')
        if model and model != 'cosyvoice2':
            try:
                self._check_model_exists(model)
            except FileNotFoundError as e:
                logger.error(f"Modelo '{model}' no encontrado: {e}")
                raise VoiceNotFoundError(f"Modelo '{model}' no encontrado")

        # Guardar la metadata
        with open(self.voices_path / f"{voice_id}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    def _check_model_exists(self, model: str) -> None:
        """Verificar si el modelo existe en el directorio de modelos."""
        model_path = self.voices_path / f"{model}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo '{model}' no encontrado")

    def list_voices(self) -> List[Dict[str, Any]]:
        """List all voice metadata files with error resilience.

        Returns:
            List of voice metadata dictionaries.
        """
        voices = []
        errors = 0
        
        for json_file in self.voices_path.glob("*.json"):
            # Skip temporary files
            if json_file.name.endswith(".tmp"):
                continue
                
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    
                # Validate basic structure
                if isinstance(metadata, dict) and "voice_id" in metadata:
                    voices.append(metadata)
                else:
                    logger.warning(f"Skipping invalid voice file {json_file}: missing voice_id")
                    errors += 1
                    
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping invalid voice file {json_file}: {e}")
                errors += 1
                continue
        
        if errors > 0:
            logger.warning(f"Encountered {errors} errors while listing voices")
        
        logger.debug(f"Listed {len(voices)} voices ({errors} errors)")
        return voices

    def delete_voice(self, voice_id: str) -> bool:
        """Delete voice profile and metadata with cleanup.

        Args:
            voice_id: Unique voice identifier.

        Returns:
            True if deleted, False if not found.
        """
        if not voice_id:
            logger.warning("Attempted to delete voice with empty ID")
            return False
        
        metadata_path = self.voices_path / f"{voice_id}.json"
        audio_path = self.voices_path / f"{voice_id}.wav"
        deleted = False
        errors = []
        
        # Delete metadata file
        if metadata_path.exists():
            try:
                metadata_path.unlink()
                deleted = True
                logger.debug(f"Deleted metadata for voice: {voice_id}")
            except Exception as e:
                errors.append(f"metadata: {e}")
                logger.error(f"Failed to delete metadata for {voice_id}: {e}", exc_info=True)
        
        # Delete audio file
        if audio_path.exists():
            try:
                audio_path.unlink()
                deleted = True
                logger.debug(f"Deleted audio for voice: {voice_id}")
            except Exception as e:
                errors.append(f"audio: {e}")
                logger.error(f"Failed to delete audio for {voice_id}: {e}", exc_info=True)
        
        # Clean up any temporary files
        for tmp_file in self.voices_path.glob(f"{voice_id}.*.tmp"):
            try:
                tmp_file.unlink()
            except Exception:
                pass
        
        if deleted:
            logger.info(f"Voice deleted: {voice_id}")
        else:
            logger.warning(f"Voice not found for deletion: {voice_id}")
        
        if errors:
            logger.error(f"Errors during voice deletion for {voice_id}: {errors}")
        
        return deleted

    def voice_exists(self, voice_id: str) -> bool:
        """Check if voice exists."""
        metadata_path = self.voices_path / f"{voice_id}.json"
        return metadata_path.exists()

    def get_voice_audio_path(self, voice_id: str) -> Optional[Path]:
        """Get path to voice audio file if it exists."""
        audio_path = self.voices_path / f"{voice_id}.wav"
        return audio_path if audio_path.exists() else None

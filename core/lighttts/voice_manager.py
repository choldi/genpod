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
            raise ValidationError("metadata", metadata, "Metadata must be a non-empty dictionary")
        
        # Add timestamp if not present
        if "created_at" not in metadata:
            metadata["created_at"] = time.time()
        
        metadata["updated_at"] = time.time()
        metadata["voice_id"] = voice_id
        
        metadata_path = self.voices_path / f"{voice_id}.json"
        temp_path = self.voices_path / f"{voice_id}.json.tmp"
        
        try:
            # Write to temporary file first (atomic operation)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Atomic rename
            temp_path.replace(metadata_path)
            logger.debug(f"Saved voice metadata for: {voice_id}")
        except Exception as e:
            # Cleanup temp file on error
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            logger.error(f"Failed to save voice metadata for {voice_id}: {e}", exc_info=True)
            raise

    def load_voice_metadata(self, voice_id: str) -> Dict[str, Any]:
        """Load voice metadata from JSON file.

        Args:
            voice_id: Unique voice identifier.

        Returns:
            Voice metadata dictionary.

        Raises:
            VoiceNotFoundError: If voice metadata not found or corrupted.
        """
        if not voice_id:
            raise VoiceNotFoundError("empty")
        
        metadata_path = self.voices_path / f"{voice_id}.json"
        if not metadata_path.exists():
            logger.warning(f"Voice metadata not found: {voice_id}")
            raise VoiceNotFoundError(voice_id)
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Validate required fields
            if not isinstance(metadata, dict):
                raise VoiceNotFoundError(f"Voice '{voice_id}' has invalid metadata format")
            
            logger.debug(f"Loaded voice metadata for: {voice_id}")
            return metadata
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in voice metadata for {voice_id}: {e}", exc_info=True)
            raise VoiceNotFoundError(f"Voice '{voice_id}' has corrupted metadata")
        except Exception as e:
            logger.error(f"Failed to load voice metadata for {voice_id}: {e}", exc_info=True)
            raise VoiceNotFoundError(f"Voice '{voice_id}' metadata load failed: {e}")

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

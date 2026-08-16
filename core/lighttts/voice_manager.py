"""Voice Manager - Handles reading/writing voice metadata."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from core.exceptions import VoiceNotFoundError
from core.logger import get_logger

logger = get_logger(__name__)


class VoiceManager:
    """Manages voice profiles and metadata."""

    def __init__(self, voices_path: str):
        """Initialize voice manager.

        Args:
            voices_path: Path to voices directory.
        """
        self.voices_path = Path(voices_path)
        self.voices_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"VoiceManager initialized with path: {voices_path}")

    def save_voice_metadata(self, voice_id: str, metadata: Dict[str, Any]) -> None:
        """Save voice metadata to JSON file.

        Args:
            voice_id: Unique voice identifier.
            metadata: Voice metadata dictionary.
        """
        # Add timestamp if not present
        if "created_at" not in metadata:
            metadata["created_at"] = time.time()
        
        metadata_path = self.voices_path / f"{voice_id}.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved voice metadata for: {voice_id}")
        except Exception as e:
            logger.error(f"Failed to save voice metadata for {voice_id}: {e}", exc_info=True)
            raise

    def load_voice_metadata(self, voice_id: str) -> Dict[str, Any]:
        """Load voice metadata from JSON file.

        Args:
            voice_id: Unique voice identifier.

        Returns:
            Voice metadata dictionary.

        Raises:
            VoiceNotFoundError: If voice metadata not found.
        """
        metadata_path = self.voices_path / f"{voice_id}.json"
        if not metadata_path.exists():
            logger.warning(f"Voice metadata not found: {voice_id}")
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.debug(f"Loaded voice metadata for: {voice_id}")
            return metadata
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in voice metadata for {voice_id}: {e}", exc_info=True)
            raise VoiceNotFoundError(f"Voice '{voice_id}' has corrupted metadata")
        except Exception as e:
            logger.error(f"Failed to load voice metadata for {voice_id}: {e}", exc_info=True)
            raise

    def list_voices(self) -> List[Dict[str, Any]]:
        """List all voice metadata files.

        Returns:
            List of voice metadata dictionaries.
        """
        voices = []
        for json_file in self.voices_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    voices.append(metadata)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping invalid voice file {json_file}: {e}")
                continue
        logger.debug(f"Listed {len(voices)} voices")
        return voices

    def delete_voice(self, voice_id: str) -> bool:
        """Delete voice profile and metadata.

        Args:
            voice_id: Unique voice identifier.

        Returns:
            True if deleted, False if not found.
        """
        metadata_path = self.voices_path / f"{voice_id}.json"
        audio_path = self.voices_path / f"{voice_id}.wav"
        deleted = False
        if metadata_path.exists():
            try:
                metadata_path.unlink()
                deleted = True
                logger.debug(f"Deleted metadata for voice: {voice_id}")
            except Exception as e:
                logger.error(f"Failed to delete metadata for {voice_id}: {e}", exc_info=True)
        if audio_path.exists():
            try:
                audio_path.unlink()
                deleted = True
                logger.debug(f"Deleted audio for voice: {voice_id}")
            except Exception as e:
                logger.error(f"Failed to delete audio for {voice_id}: {e}", exc_info=True)
        if deleted:
            logger.info(f"Voice deleted: {voice_id}")
        else:
            logger.warning(f"Voice not found for deletion: {voice_id}")
        return deleted

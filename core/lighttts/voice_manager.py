"""Voice Manager - Handles reading/writing voice metadata."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from core.exceptions import VoiceNotFoundError


class VoiceManager:
    """Manages voice profiles and metadata."""

    def __init__(self, voices_path: str):
        """Initialize voice manager.

        Args:
            voices_path: Path to voices directory.
        """
        self.voices_path = Path(voices_path)
        self.voices_path.mkdir(parents=True, exist_ok=True)

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
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

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
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found")
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
            except (json.JSONDecodeError, OSError):
                continue
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
            metadata_path.unlink()
            deleted = True
        if audio_path.exists():
            audio_path.unlink()
            deleted = True
        return deleted

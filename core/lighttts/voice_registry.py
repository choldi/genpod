"""Voice Registry - Manages voice listing and metadata with model association."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from core.lighttts.voice_manager import VoiceManager
from core.logger import get_logger

logger = get_logger(__name__)


class VoiceRegistry:
    """Manages voice listing and metadata with model association."""

    def __init__(self, voices_path: str, voice_manager: VoiceManager) -> None:
        """Initialize the voice registry.
        
        Args:
            voices_path: Path to voices directory
            voice_manager: VoiceManager instance
        """
        self.voices_path = Path(voices_path)
        self._voice_manager = voice_manager
        self._voice_cache: Optional[List[Dict[str, Any]]] = None
        self._base_speakers_cache: Optional[Dict[str, str]] = None
        self._cloned_voices_cache: Optional[Set[str]] = None
        self._voice_model_map: Dict[str, str] = {}  # voice_id -> model_type

    def list_voices(self) -> List[Dict[str, Any]]:
        """Scan and return all available voices (base and cloned) with model info."""
        if self._voice_cache is not None:
            return self._voice_cache

        voices = []
        
        # Load base speakers from each model type
        base_speakers = self._get_base_speakers()
        for model_type, speakers in base_speakers.items():
            for spk_id, spk_name in speakers.items():
                voice_id = f"{model_type}:{spk_id}"
                voices.append({
                    "voice_id": voice_id,
                    "name": spk_name,
                    "language": "multi",
                    "gender": "unknown",
                    "is_cloned": False,
                    "sample_rate": 24000,
                    "model": model_type,
                    "base_speaker_id": spk_id,
                })
                self._voice_model_map[voice_id] = model_type

        # Load cloned voices
        cloned_voices = self._get_cloned_voices()
        for voice_id, metadata in cloned_voices.items():
            model_type = metadata.get("model", "cosyvoice2")  # Default for backward compatibility
            voices.append({
                "voice_id": voice_id,
                "name": metadata.get("name", voice_id),
                "language": metadata.get("language", "en"),
                "gender": metadata.get("gender", "unknown"),
                "is_cloned": True,
                "sample_rate": metadata.get("sample_rate", 24000),
                "model": model_type,
                "transcript": metadata.get("transcript", ""),
                "created_at": metadata.get("created_at", ""),
            })
            self._voice_model_map[voice_id] = model_type

        self._voice_cache = voices
        logger.debug(f"Voice registry loaded {len(voices)} voices")
        return voices

    def _get_base_speakers(self) -> Dict[str, Dict[str, str]]:
        """Get base speakers for all available models."""
        if self._base_speakers_cache is not None:
            return self._base_speakers_cache

        # This would ideally come from model metadata
        # For now, we use known speaker lists per model type
        base_speakers = {
            "cosyvoice2": {
                "speaker_0": "CosyVoice2 Female 1",
                "speaker_1": "CosyVoice2 Male 1",
                "speaker_2": "CosyVoice2 Female 2",
                "speaker_3": "CosyVoice2 Male 2",
            },
            "cosyvoice3": {
                "speaker_0": "CosyVoice3 Female 1",
                "speaker_1": "CosyVoice3 Male 1",
                "speaker_2": "CosyVoice3 Female 2",
                "speaker_3": "CosyVoice3 Male 2",
            },
            "voxcpm": {
                "speaker_0": "VoxCPM Speaker 1",
                "speaker_1": "VoxCPM Speaker 2",
            },
        }
        
        self._base_speakers_cache = base_speakers
        return base_speakers

    def _get_cloned_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get all cloned voices with metadata."""
        if self._cloned_voices_cache is not None:
            return self._cloned_voices_cache

        cloned = {}
        for metadata_file in self.voices_path.glob("*.json"):
            voice_id = metadata_file.stem
            try:
                metadata = self._voice_manager.load_voice_metadata(voice_id)
                cloned[voice_id] = metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata for {voice_id}: {e}")
        
        self._cloned_voices_cache = cloned
        return cloned

    def is_cloned_voice(self, voice_id: str) -> bool:
        """Check if a voice is a cloned voice."""
        voices = self.list_voices()
        for voice in voices:
            if voice["voice_id"] == voice_id:
                return voice["is_cloned"]
        raise ValueError(f"Voice not found: {voice_id}")

    def get_base_speaker_id(self, voice_id: str) -> str:
        """Get the base speaker ID for a base voice."""
        voices = self.list_voices()
        for voice in voices:
            if voice["voice_id"] == voice_id:
                if voice["is_cloned"]:
                    raise ValueError(f"Voice {voice_id} is cloned, not a base voice")
                return voice.get("base_speaker_id", voice_id.split(":")[-1])
        raise ValueError(f"Base voice not found: {voice_id}")

    def get_voice_model(self, voice_id: str) -> str:
        """Get the model type associated with a voice."""
        if voice_id in self._voice_model_map:
            return self._voice_model_map[voice_id]
        
        # Refresh cache and try again
        self.list_voices()
        if voice_id in self._voice_model_map:
            return self._voice_model_map[voice_id]
            
        raise ValueError(f"Voice not found: {voice_id}")

    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """Get full voice information."""
        voices = self.list_voices()
        for voice in voices:
            if voice["voice_id"] == voice_id:
                return voice
        raise ValueError(f"Voice not found: {voice_id}")

    def invalidate_cache(self) -> None:
        """Invalidate internal caches."""
        self._voice_cache = None
        self._base_speakers_cache = None
        self._cloned_voices_cache = None
        self._voice_model_map.clear()
        logger.debug("Voice registry cache invalidated")

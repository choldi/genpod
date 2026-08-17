"""Voice Registry - Dynamic voice listing with model metadata querying."""

import logging
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from core.lighttts.voice_manager import VoiceManager
from core.logger import get_logger
from core.exceptions import ModelLoadError

logger = get_logger(__name__)


@dataclass
class ModelSpeakerInfo:
    """Information about a model's speaker."""
    model_type: str
    speaker_id: str
    name: str
    language: str = "multi"
    gender: str = "unknown"
    sample_rate: int = 24000


class VoiceRegistry:
    """Manages voice listing and metadata with dynamic model association."""

    def __init__(self, voices_path: str, voice_manager: VoiceManager) -> None:
        """Initialize the voice registry.
        
        Args:
            voices_path: Path to voices directory
            voice_manager: VoiceManager instance
        """
        self.voices_path = Path(voices_path)
        self._voice_manager = voice_manager
        self._voice_cache: Optional[List[Dict[str, Any]]] = None
        self._base_speakers_cache: Optional[Dict[str, List[ModelSpeakerInfo]]] = None
        self._cloned_voices_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._voice_model_map: Dict[str, str] = {}  # voice_id -> model_type
        self._model_metadata_cache: Dict[str, Dict[str, Any]] = {}

    def list_voices(self, model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Scan and return all available voices (base and cloned) with model info.
        
        Args:
            model_filter: Optional model type to filter voices (cosyvoice2, cosyvoice3, voxcpm)
            
        Returns:
            List of voice dictionaries
        """
        if self._voice_cache is not None and model_filter is None:
            return self._voice_cache

        voices = []
        
        # Load base speakers from each model type dynamically
        base_speakers = self._get_base_speakers()
        for model_type, speakers in base_speakers.items():
            if model_filter and model_type != model_filter:
                continue
                
            for speaker in speakers:
                voice_id = f"{model_type}:{speaker.speaker_id}"
                voices.append({
                    "voice_id": voice_id,
                    "name": speaker.name,
                    "language": speaker.language,
                    "gender": speaker.gender,
                    "is_cloned": False,
                    "sample_rate": speaker.sample_rate,
                    "model": model_type,
                    "base_speaker_id": speaker.speaker_id,
                })
                self._voice_model_map[voice_id] = model_type

        # Load cloned voices
        cloned_voices = self._get_cloned_voices()
        for voice_id, metadata in cloned_voices.items():
            model_type = metadata.get("model", "cosyvoice2")
            if model_filter and model_type != model_filter:
                continue
                
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
        logger.debug(f"Voice registry loaded {len(voices)} voices (filter: {model_filter})")
        return voices

    def _get_base_speakers(self) -> Dict[str, List[ModelSpeakerInfo]]:
        """Get base speakers for all available models by querying model metadata."""
        if self._base_speakers_cache is not None:
            return self._base_speakers_cache

        base_speakers = {}
        
        # Try to load speaker info from model metadata files
        for model_type in ["cosyvoice2", "cosyvoice3", "voxcpm"]:
            speakers = self._load_model_speakers(model_type)
            if speakers:
                base_speakers[model_type] = speakers
                logger.info(f"Loaded {len(speakers)} speakers for {model_type} from metadata")
            else:
                # Fallback to default speakers if metadata not available
                base_speakers[model_type] = self._get_default_speakers(model_type)
                logger.warning(f"Using default speakers for {model_type} (metadata not found)")

        self._base_speakers_cache = base_speakers
        return base_speakers

    def _load_model_speakers(self, model_type: str) -> Optional[List[ModelSpeakerInfo]]:
        """Load speaker information from model metadata files."""
        try:
            # Check for metadata file in model directory
            from core.config import settings
            models_path = Path(settings.MODELS_PATH)
            
            model_dirs = {
                "cosyvoice2": "CosyVoice2-0.5B",
                "cosyvoice3": "CosyVoice3-0.5B",
                "voxcpm": "VoxCPM"
            }
            
            model_dir = models_path / model_dirs.get(model_type, "")
            if not model_dir.exists():
                return None
            
            # Look for speaker metadata files
            metadata_files = list(model_dir.glob("speakers*.json")) + \
                           list(model_dir.glob("voice_config*.json")) + \
                           list(model_dir.glob("*.json"))
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Parse speaker information based on model type
                    speakers = self._parse_speaker_metadata(model_type, data)
                    if speakers:
                        return speakers
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Failed to parse {metadata_file}: {e}")
                    continue
            
            # Try to query model directly if it has a CLI or API
            speakers = self._query_model_speakers(model_type, model_dir)
            if speakers:
                return speakers
                
        except Exception as e:
            logger.debug(f"Could not load speakers for {model_type} from metadata: {e}")
        
        return None

    def _parse_speaker_metadata(self, model_type: str, data: Dict[str, Any]) -> Optional[List[ModelSpeakerInfo]]:
        """Parse speaker metadata from model configuration."""
        speakers = []
        
        try:
            if model_type in ["cosyvoice2", "cosyvoice3"]:
                # CosyVoice format: {"speakers": {"speaker_0": {"name": "...", "gender": "..."}}}
                if "speakers" in data:
                    for spk_id, spk_info in data["speakers"].items():
                        speakers.append(ModelSpeakerInfo(
                            model_type=model_type,
                            speaker_id=spk_id,
                            name=spk_info.get("name", f"{model_type} {spk_id}"),
                            language=spk_info.get("language", "multi"),
                            gender=spk_info.get("gender", "unknown"),
                            sample_rate=spk_info.get("sample_rate", 24000)
                        ))
                elif "spk2info" in data:  # Alternative format
                    for spk_id, spk_info in data["spk2info"].items():
                        speakers.append(ModelSpeakerInfo(
                            model_type=model_type,
                            speaker_id=spk_id,
                            name=spk_info.get("name", f"{model_type} {spk_id}"),
                            language=spk_info.get("language", "multi"),
                            gender=spk_info.get("gender", "unknown"),
                            sample_rate=spk_info.get("sample_rate", 24000)
                        ))
                        
            elif model_type == "voxcpm":
                # VoxCPM format: {"voices": [{"id": "...", "name": "..."}]}
                if "voices" in data:
                    for voice in data["voices"]:
                        speakers.append(ModelSpeakerInfo(
                            model_type=model_type,
                            speaker_id=voice.get("id", voice.get("speaker_id", "")),
                            name=voice.get("name", f"VoxCPM {voice.get('id', '')}"),
                            language=voice.get("language", "multi"),
                            gender=voice.get("gender", "unknown"),
                            sample_rate=voice.get("sample_rate", 24000)
                        ))
                        
        except Exception as e:
            logger.debug(f"Error parsing speaker metadata for {model_type}: {e}")
            return None
        
        return speakers if speakers else None

    def _query_model_speakers(self, model_type: str, model_dir: Path) -> Optional[List[ModelSpeakerInfo]]:
        """Try to query model directly for speaker list."""
        try:
            # This would require the model to be loaded, which we avoid during registry init
            # Could be implemented as a lazy-load when model is actually loaded
            return None
        except Exception:
            return None

    def _get_default_speakers(self, model_type: str) -> List[ModelSpeakerInfo]:
        """Get default speaker configurations for each model type."""
        defaults = {
            "cosyvoice2": [
                ModelSpeakerInfo("cosyvoice2", "speaker_0", "CosyVoice2 Female 1", "multi", "female"),
                ModelSpeakerInfo("cosyvoice2", "speaker_1", "CosyVoice2 Male 1", "multi", "male"),
                ModelSpeakerInfo("cosyvoice2", "speaker_2", "CosyVoice2 Female 2", "multi", "female"),
                ModelSpeakerInfo("cosyvoice2", "speaker_3", "CosyVoice2 Male 2", "multi", "male"),
            ],
            "cosyvoice3": [
                ModelSpeakerInfo("cosyvoice3", "speaker_0", "CosyVoice3 Female 1", "multi", "female"),
                ModelSpeakerInfo("cosyvoice3", "speaker_1", "CosyVoice3 Male 1", "multi", "male"),
                ModelSpeakerInfo("cosyvoice3", "speaker_2", "CosyVoice3 Female 2", "multi", "female"),
                ModelSpeakerInfo("cosyvoice3", "speaker_3", "CosyVoice3 Male 2", "multi", "male"),
            ],
            "voxcpm": [
                ModelSpeakerInfo("voxcpm", "speaker_0", "VoxCPM Speaker 1 (Female)", "multi", "female"),
                ModelSpeakerInfo("voxcpm", "speaker_1", "VoxCPM Speaker 2 (Male)", "multi", "male"),
                ModelSpeakerInfo("voxcpm", "speaker_2", "VoxCPM Speaker 3 (Female)", "multi", "female"),
                ModelSpeakerInfo("voxcpm", "speaker_3", "VoxCPM Speaker 4 (Male)", "multi", "male"),
            ],
        }
        return defaults.get(model_type, [])

    def _get_cloned_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get all cloned voices with metadata."""
        if self._cloned_voices_cache is not None:
            return self._cloned_voices_cache

        cloned = {}
        for metadata_file in self.voices_path.glob("*.json"):
            if metadata_file.name.endswith(".tmp"):
                continue
                
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

    def get_available_models(self) -> List[str]:
        """Get list of models that have available voices."""
        voices = self.list_voices()
        models = set(voice.get("model") for voice in voices if voice.get("model"))
        return sorted(list(models))

    def invalidate_cache(self) -> None:
        """Invalidate internal caches."""
        self._voice_cache = None
        self._base_speakers_cache = None
        self._cloned_voices_cache = None
        self._voice_model_map.clear()
        logger.debug("Voice registry cache invalidated")

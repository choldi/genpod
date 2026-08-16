"""Voice registry for listing and managing voices."""

import logging
from typing import List, Dict, Any

from core.lighttts.voice_manager import VoiceManager

logger = logging.getLogger(__name__)

# Mapping from simple API aliases to actual CosyVoice speaker IDs
VOICE_ALIAS_MAP = {
    "zh_female": "中文女",
    "zh_male": "中文男",
    "en_female": "英文女",
    "en_male": "英文男",
    "ja_female": "日本語女",
    "ko_female": "한국어女",
}


class VoiceRegistry:
    """Manages voice listing and metadata."""

    def __init__(self, voices_path: str, voice_manager: VoiceManager) -> None:
        self.voices_path = voices_path
        self._voice_manager = voice_manager

    def list_voices(self) -> List[Dict[str, Any]]:
        """Scan and return all available voices (base and cloned)."""
        voices: List[Dict[str, Any]] = []

        # Cloned voices
        for voice in self._voice_manager.list_voices():
            voices.append({
                "voice_id": voice.get("voice_id", ""),
                "name": voice.get("name", "Unknown"),
                "language": voice.get("language", "en"),
                "is_cloned": True,
                "sample_rate": voice.get("sample_rate", 24000),
            })

        # Base voices
        known_base_voices = [
            ("zh_female", "zh", "Base Chinese Female"),
            ("zh_male", "zh", "Base Chinese Male"),
            ("en_female", "en", "Base English Female"),
            ("en_male", "en", "Base English Male"),
            ("ja_female", "ja", "Base Japanese Female"),
            ("ko_female", "ko", "Base Korean Female"),
        ]

        for alias, lang, name in known_base_voices:
            voices.append({
                "voice_id": alias,
                "name": name,
                "language": lang,
                "is_cloned": False,
                "sample_rate": 24000,
            })

        return voices

    def is_cloned_voice(self, voice_id: str) -> bool:
        """Check if a voice ID corresponds to a cloned voice."""
        try:
            meta = self._voice_manager.load_voice_metadata(voice_id)
            return bool(meta and meta.get("is_cloned"))
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def get_base_speaker_id(self, voice_id: str) -> str:
        """Get actual CosyVoice speaker ID for base voice alias."""
        return VOICE_ALIAS_MAP.get(voice_id, voice_id)

    def get_cloned_voice_path(self, voice_id: str) -> str:
        """Get reference audio path for cloned voice."""
        from pathlib import Path
        return str(Path(self.voices_path) / f"{voice_id}.wav")

    def load_cloned_metadata(self, voice_id: str) -> Dict[str, Any]:
        """Load metadata for cloned voice."""
        return self._voice_manager.load_voice_metadata(voice_id)

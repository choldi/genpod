"""LightTTSEngine - Main wrapper for CosyVoice 2/3 model operations."""

import logging
import re
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional
import torch
import torchaudio

from core.config import settings
from core.exceptions import (
    VoiceNotFoundError,
    ModelLoadError,
    SynthesisError,
    CloningError,
    AudioTooShortError,
)
from core.lighttts.voice_manager import VoiceManager
from core.lighttts.model_loader import ModelLoader
from core.lighttts.voice_registry import VoiceRegistry
from core.lighttts.base_synthesizer import BaseSynthesizer
from core.lighttts.cloned_synthesizer import ClonedSynthesizer
from core.lighttts.audio_utils import tensor_to_wav_bytes

logger = logging.getLogger(__name__)


class LightTTSEngine:
    """Main wrapper class for CosyVoice 2/3 operations."""

    def __init__(self, models_path: str, voices_path: str, device: str = "cpu") -> None:
        """Initialize the CosyVoice model."""
        self.models_path = Path(models_path)
        self.voices_path = Path(voices_path)
        self.device = self._resolve_device(device)

        self._voice_manager = VoiceManager(str(self.voices_path))
        self._model_loader = ModelLoader(str(self.models_path), self.device)
        self._voice_registry = VoiceRegistry(str(self.voices_path), self._voice_manager)

        # Load model and initialize synthesizers
        model, model_version, load_wav = self._model_loader.load()
        self._model = model
        self._model_version = model_version
        self._load_wav = load_wav

        self._base_synthesizer = BaseSynthesizer(model, model_version)
        self._cloned_synthesizer = ClonedSynthesizer(
            model, model_version, str(self.voices_path), self._voice_manager
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve the actual device to use based on availability."""
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return device

    def list_voices(self) -> List[Dict[str, Any]]:
        """Scan and return all available voices (base and cloned)."""
        return self._voice_registry.list_voices()

    def synthesize(
        self,
        text: str,
        voice_id: str,
        lang: str = "en",
        stream: bool = True,
        speed: float = 1.0,
        pitch: float = 1.0,
        emotion: str = "neutral",
        emotion_tags: bool = False,
    ) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text and voice.

        Note: emotion and emotion_tags parameters are kept for API compatibility
        but are no longer processed - the underlying model handles prosody.
        """
        if not self._model:
            raise SynthesisError("Model not loaded")

        is_cloned = self._voice_registry.is_cloned_voice(voice_id)

        if is_cloned:
            yield from self._cloned_synthesizer.synthesize(
                text=text,
                voice_id=voice_id,
                speed=speed,
                pitch=pitch,
                language=lang,
                stream=stream,
            )
        else:
            actual_spk_id = self._voice_registry.get_base_speaker_id(voice_id)
            yield from self._base_synthesizer.synthesize(
                text=text,
                spk_id=actual_spk_id,
                speed=speed,
                pitch=pitch,
                stream=stream,
            )

    def clone_voice(
        self,
        audio_path: str,
        transcript: str,
        voice_name: str,
        language: str = "en",
    ) -> str:
        """Clone a voice from reference audio and transcript."""
        if not self._model:
            raise CloningError("Model not loaded")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise CloningError(f"Reference audio not found: {audio_path}")

        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            if duration < 3.0:
                raise AudioTooShortError(
                    f"Reference audio too short: {duration:.1f}s. Minimum 3 seconds required."
                )
            if duration > 30.0:
                logger.warning(
                    f"Reference audio is long ({duration:.1f}s). "
                    "Consider using a shorter clip for better results."
                )
        except AudioTooShortError:
            raise
        except Exception as e:
            raise CloningError(f"Failed to validate audio: {e}") from e

        # Sanitize voice_id
        voice_id = re.sub(r'[^\w\-]', '_', voice_name)[:32]

        dest_audio = self.voices_path / f"{voice_id}.wav"
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                waveform = resampler(waveform)
            torchaudio.save(str(dest_audio), waveform, 24000)
        except Exception as e:
            raise CloningError(f"Failed to process reference audio: {e}") from e

        metadata = {
            "voice_id": voice_id,
            "name": voice_name,
            "language": language,
            "gender": "unknown",
            "is_cloned": True,
            "sample_rate": 24000,
            "transcript": transcript,
            "created_at": str(torch.tensor(0).numpy()),
        }
        self._voice_manager.save_voice_metadata(voice_id, metadata)
        return voice_id

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice."""
        return self._voice_manager.delete_voice(voice_id)

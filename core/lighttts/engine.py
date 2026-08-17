"""LightTTSEngine - Main wrapper for CosyVoice 2/3 model operations."""

import logging
import re
import time
from datetime import datetime
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
from core.logger import get_logger

logger = get_logger(__name__)


class LightTTSEngine:
    """Main wrapper class for CosyVoice 2/3 operations."""

    def __init__(self, models_path: str, voices_path: str, device: str = "cpu") -> None:
        """Initialize the CosyVoice model."""
        logger.info("Initializing LightTTSEngine...")
        self.models_path = Path(models_path)
        self.voices_path = Path(voices_path)
        self.device = self._resolve_device(device)
        logger.info(f"Using device: {self.device}")

        # Validate paths exist
        self._validate_paths()

        self._voice_manager = VoiceManager(str(self.voices_path))
        self._model_loader = ModelLoader(str(self.models_path), self.device)
        self._voice_registry = VoiceRegistry(str(self.voices_path), self._voice_manager)

        # Load model and initialize synthesizers
        logger.info("Loading model...")
        try:
            model, model_version, load_wav = self._model_loader.load()
            self._model = model
            self._model_version = model_version
            self._load_wav = load_wav
            logger.info(f"Model loaded successfully (version: {model_version})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise ModelLoadError(f"Model loading failed: {e}") from e

        try:
            self._base_synthesizer = BaseSynthesizer(model, model_version)
            self._cloned_synthesizer = ClonedSynthesizer(
                model, model_version, str(self.voices_path), self._voice_manager
            )
            logger.info("Synthesizers initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize synthesizers: {e}", exc_info=True)
            raise ModelLoadError(f"Synthesizer initialization failed: {e}") from e

    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.models_path.exists():
            logger.error(f"Models path does not exist: {self.models_path}")
            raise ModelLoadError(f"Models path does not exist: {self.models_path}")
        
        if not self.voices_path.exists():
            logger.warning(f"Voices path does not exist, creating: {self.voices_path}")
            self.voices_path.mkdir(parents=True, exist_ok=True)
        
        # Check for model files (basic validation)
        model_files = list(self.models_path.glob("*.pt")) + list(self.models_path.glob("*.bin")) + list(self.models_path.glob("*.safetensors"))
        if not model_files:
            logger.warning(f"No model files found in {self.models_path}. Expected .pt, .bin, or .safetensors files.")

    def _resolve_device(self, device: str) -> str:
        """Resolve the actual device to use based on availability."""
        logger.debug(f"Resolving device. Requested: {device}")
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return device

    def list_voices(self) -> List[Dict[str, Any]]:
        """Scan and return all available voices (base and cloned)."""
        logger.info("Listing voices...")
        voices = self._voice_registry.list_voices()
        logger.info(f"Found {len(voices)} voices.")
        return voices

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
        chunk_size: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text and voice.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            lang: Language code
            stream: Whether to stream chunks
            speed: Speech speed factor
            pitch: Pitch factor
            emotion: Emotion style (kept for API compatibility)
            emotion_tags: Whether to use emotion tags (kept for API compatibility)
            chunk_size: Chunking strategy:
                - None (default): Sentence-based chunking (current behavior)
                - <= 0: No chunking, process entire text at once
                - > 0: Custom chunk size in characters
            model: Model identifier (kept for API compatibility, not used)

        Note: emotion and emotion_tags parameters are kept for API compatibility
        but are no longer processed - the underlying model handles prosody.
        The model parameter is also kept for API compatibility but is not used.
        """
        logger.info(
            f"Engine synthesize called: voice_id={voice_id}, text_len={len(text)}, "
            f"stream={stream}, speed={speed}, pitch={pitch}, chunk_size={chunk_size}"
        )
        
        if not self._model:
            logger.error("Model not loaded")
            raise SynthesisError("Model not loaded")

        try:
            is_cloned = self._voice_registry.is_cloned_voice(voice_id)
            logger.debug(f"Voice type: {'cloned' if is_cloned else 'base'}")
        except Exception as e:
            logger.error(f"Error checking voice type: {e}", exc_info=True)
            raise SynthesisError(f"Voice registry error: {e}") from e

        try:
            if is_cloned:
                logger.debug("Using cloned synthesizer")
                synthesizer = self._cloned_synthesizer
                synth_args = {
                    "text": text,
                    "voice_id": voice_id,
                    "speed": speed,
                    "pitch": pitch,
                    "language": lang,
                    "stream": stream,
                    "chunk_size": chunk_size,
                }
            else:
                logger.debug("Using base synthesizer")
                actual_spk_id = self._voice_registry.get_base_speaker_id(voice_id)
                logger.debug(f"Base speaker ID: {actual_spk_id}")
                synthesizer = self._base_synthesizer
                synth_args = {
                    "text": text,
                    "spk_id": actual_spk_id,
                    "speed": speed,
                    "pitch": pitch,
                    "stream": stream,
                    "chunk_size": chunk_size,
                }

            logger.debug("Starting synthesizer iteration")
            chunk_count = 0
            start_time = time.time()
            
            for chunk in synthesizer.synthesize(**synth_args):
                chunk_count += 1
                if chunk_count == 1:
                    logger.info(f"First chunk generated after {time.time() - start_time:.3f}s")
                yield chunk
                
            logger.info(f"Synthesis completed: {chunk_count} chunks generated in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            raise SynthesisError(f"Synthesis failed: {e}") from e

    def clone_voice(
        self,
        audio_path: str,
        transcript: str,
        voice_name: str,
        language: str = "en",
    ) -> str:
        """Clone a voice from reference audio and transcript."""
        logger.info(f"Cloning voice: name={voice_name}, audio={audio_path}, language={language}")
        
        if not self._model:
            logger.error("Model not loaded for cloning")
            raise CloningError("Model not loaded")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Reference audio not found: {audio_path}")
            raise CloningError(f"Reference audio not found: {audio_path}")

        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            logger.debug(f"Audio duration: {duration:.2f}s, sample_rate: {info.sample_rate}")
            
            if duration < 3.0:
                logger.error(f"Reference audio too short: {duration:.1f}s")
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
            logger.error(f"Failed to validate audio: {e}", exc_info=True)
            raise CloningError(f"Failed to validate audio: {e}") from e

        # Sanitize voice_id
        voice_id = re.sub(r'[^\w\-]', '_', voice_name)[:32]
        logger.debug(f"Sanitized voice_id: {voice_id}")

        dest_audio = self.voices_path / f"{voice_id}.wav"
        try:
            logger.debug("Loading and resampling audio")
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                waveform = resampler(waveform)
            torchaudio.save(str(dest_audio), waveform, 24000)
            logger.debug(f"Saved reference audio to {dest_audio}")
        except Exception as e:
            logger.error(f"Failed to process reference audio: {e}", exc_info=True)
            raise CloningError(f"Failed to process reference audio: {e}") from e

        metadata = {
            "voice_id": voice_id,
            "name": voice_name,
            "language": language,
            "gender": "unknown",
            "is_cloned": True,
            "sample_rate": 24000,
            "transcript": transcript,
            "created_at": datetime.utcnow().isoformat(),
        }
        try:
            self._voice_manager.save_voice_metadata(voice_id, metadata)
            logger.info(f"Voice cloned successfully: {voice_id}")
        except Exception as e:
            logger.error(f"Failed to save voice metadata: {e}", exc_info=True)
            raise CloningError(f"Failed to save voice metadata: {e}") from e
            
        return voice_id

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice."""
        logger.info(f"Deleting voice: {voice_id}")
        return self._voice_manager.delete_voice(voice_id)

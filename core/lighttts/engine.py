"""LightTTSEngine - Main wrapper for CosyVoice 2/3 and VoxCPM model operations with enhanced error handling."""

import logging
import re
import time
import uuid
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
    ModelNotAvailableError,
    GPUOutOfMemoryError,
    ValidationError,
)
from core.lighttts.voice_manager import VoiceManager
from core.lighttts.model_loader import ModelLoader
from core.lighttts.voice_registry import VoiceRegistry
from core.lighttts.base_synthesizer import BaseSynthesizer
from core.lighttts.cloned_synthesizer import ClonedSynthesizer
from core.lighttts.audio_utils import tensor_to_wav_bytes
from core.logger import get_logger, set_correlation_id, set_request_context, clear_log_context

logger = get_logger(__name__)


class LightTTSEngine:
    """Main wrapper class for CosyVoice 2/3 and VoxCPM operations with robust error handling."""

    DEFAULT_MODEL = "cosyvoice2"
    SUPPORTED_MODELS = ["cosyvoice2", "cosyvoice3", "voxcpm"]

    def __init__(
        self, 
        models_path: str, 
        voices_path: str, 
        device: str = "cpu",
        default_model: Optional[str] = None
    ) -> None:
        """Initialize the TTS engine with multi-model support.
        
        Args:
            models_path: Path to models directory
            voices_path: Path to voices directory
            device: Device to use (cpu, cuda, mps)
            default_model: Default model type to use
        """
        logger.info("Initializing LightTTSEngine...")
        self.models_path = Path(models_path)
        self.voices_path = Path(voices_path)
        self.device = self._resolve_device(device)
        self.default_model = default_model or self.DEFAULT_MODEL
        logger.info(f"Using device: {self.device}, default model: {self.default_model}")

        # Validate paths exist
        self._validate_paths()

        # Initialize components
        self._voice_manager = VoiceManager(str(self.voices_path))
        self._model_loader = ModelLoader(str(self.models_path), self.device)
        self._voice_registry = VoiceRegistry(str(self.voices_path), self._voice_manager)

        # Model and synthesizer caches
        self._models: Dict[str, Any] = {}
        self._model_versions: Dict[str, str] = {}
        self._load_wav_funcs: Dict[str, Any] = {}
        self._base_synthesizers: Dict[str, BaseSynthesizer] = {}
        self._cloned_synthesizers: Dict[str, ClonedSynthesizer] = {}

        # Load default model
        self._ensure_model_loaded(self.default_model)
        logger.info("LightTTSEngine initialized successfully")

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

    def _ensure_model_loaded(self, model_type: str) -> None:
        """Ensure a model is loaded and synthesizers are initialized."""
        if model_type in self._models:
            return

        if model_type not in self.SUPPORTED_MODELS:
            raise ModelNotAvailableError(model_type, self.SUPPORTED_MODELS)

        logger.info(f"Loading model: {model_type}")
        try:
            model, model_version, load_wav = self._model_loader.load(model_type)
            self._models[model_type] = model
            self._model_versions[model_type] = model_version
            self._load_wav_funcs[model_type] = load_wav

            # Initialize synthesizers for this model
            self._base_synthesizers[model_type] = BaseSynthesizer(
                model, model_version, model_type
            )
            self._cloned_synthesizers[model_type] = ClonedSynthesizer(
                model, model_version, str(self.voices_path), self._voice_manager, model_type
            )
            logger.info(f"Model {model_type} loaded and synthesizers initialized")
        except GPUOutOfMemoryError as e:
            logger.error(f"GPU OOM loading {model_type}: {e}")
            raise
        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}", exc_info=True)
            raise ModelLoadError(f"Model {model_type} loading failed: {e}") from e

    def get_available_models(self) -> Dict[str, Dict]:
        """Get information about available models."""
        return self._model_loader.get_available_models()

    def list_voices(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Scan and return all available voices, optionally filtered by model."""
        logger.info(f"Listing voices (model filter: {model_type})")
        try:
            voices = self._voice_registry.list_voices(model_filter=model_type)
            logger.info(f"Found {len(voices)} voices.")
            return voices
        except Exception as e:
            logger.error(f"Error listing voices: {e}", exc_info=True)
            raise SynthesisError(f"Failed to list voices: {e}", voice_id="unknown") from e

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
        """Generate audio chunks for the given text and voice with comprehensive error handling.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier (format: "model_type:speaker_id" for base voices)
            lang: Language code
            stream: Whether to stream chunks
            speed: Speech speed factor
            pitch: Pitch factor
            emotion: Emotion style (kept for API compatibility)
            emotion_tags: Whether to use emotion tags (kept for API compatibility)
            chunk_size: Chunking strategy:
                - None (default): Sentence-based chunking
                - <= 0: No chunking, process entire text at once
                - > 0: Custom chunk size in characters
            model: Model identifier (cosyvoice2, cosyvoice3, voxcpm). 
                   If not provided, inferred from voice_id or uses default.

        Note: emotion and emotion_tags parameters are kept for API compatibility
        but are no longer processed - the underlying model handles prosody.
        """
        # Set up request context for logging
        correlation_id = set_correlation_id()
        set_request_context({
            "voice_id": voice_id,
            "text_length": len(text),
            "lang": lang,
            "stream": stream,
            "speed": speed,
            "pitch": pitch,
            "model": model
        })
        
        logger.info(
            f"Engine synthesize called: voice_id={voice_id}, text_len={len(text)}, "
            f"stream={stream}, speed={speed}, pitch={pitch}, chunk_size={chunk_size}, model={model}",
            extra={"correlation_id": correlation_id}
        )
        
        # Validate inputs
        if not text or not text.strip():
            raise ValidationError("text", text, "Text cannot be empty")
        
        if not voice_id:
            raise ValidationError("voice_id", voice_id, "Voice ID cannot be empty")
        
        # Determine model type
        if model is None:
            try:
                model = self._voice_registry.get_voice_model(voice_id)
                logger.debug(f"Inferred model from voice: {model}")
            except ValueError:
                model = self.default_model
                logger.debug(f"Using default model: {model}")
        else:
            logger.debug(f"Using explicit model: {model}")

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ModelNotAvailableError(model, self.SUPPORTED_MODELS)

        # Ensure model is loaded
        try:
            self._ensure_model_loaded(model)
        except GPUOutOfMemoryError as e:
            logger.error(f"GPU OOM during model loading: {e}")
            raise SynthesisError(f"GPU out of memory: {e}", voice_id=voice_id) from e
        except ModelLoadError as e:
            logger.error(f"Model load error: {e}")
            raise SynthesisError(f"Model unavailable: {e}", voice_id=voice_id) from e

        try:
            is_cloned = self._voice_registry.is_cloned_voice(voice_id)
            logger.debug(f"Voice type: {'cloned' if is_cloned else 'base'}")
        except Exception as e:
            logger.error(f"Error checking voice type: {e}", exc_info=True)
            raise SynthesisError(f"Voice registry error: {e}", voice_id=voice_id) from e

        try:
            if is_cloned:
                logger.debug("Using cloned synthesizer")
                synthesizer = self._cloned_synthesizers[model]
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
                synthesizer = self._base_synthesizers[model]
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
            
        except GPUOutOfMemoryError as e:
            logger.error(f"GPU OOM during synthesis: {e}")
            # Try to recover by clearing cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise SynthesisError(f"GPU out of memory during synthesis: {e}", voice_id=voice_id) from e
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            raise SynthesisError(f"Synthesis failed: {e}", voice_id=voice_id) from e
        finally:
            clear_log_context()

    def clone_voice(
        self,
        audio_path: str,
        transcript: str,
        voice_name: str,
        language: str = "en",
        model: Optional[str] = None,
    ) -> str:
        """Clone a voice from reference audio and transcript with enhanced validation.
        
        Args:
            audio_path: Path to reference audio file
            transcript: Transcript of the reference audio
            voice_name: Name for the new voice
            language: Language code
            model: Model to associate with this voice (default: current default model)
            
        Returns:
            Voice ID of the cloned voice
        """
        correlation_id = set_correlation_id()
        set_request_context({
            "voice_name": voice_name,
            "audio_path": audio_path,
            "language": language,
            "model": model
        })
        
        target_model = model or self.default_model
        logger.info(f"Cloning voice: name={voice_name}, audio={audio_path}, language={language}, model={target_model}")
        
        # Validate inputs
        if not voice_name or not voice_name.strip():
            raise ValidationError("voice_name", voice_name, "Voice name cannot be empty")
        
        if not transcript or not transcript.strip():
            raise ValidationError("transcript", transcript, "Transcript cannot be empty")
        
        if target_model not in self.SUPPORTED_MODELS:
            raise ModelNotAvailableError(target_model, self.SUPPORTED_MODELS)

        # Ensure model is loaded
        try:
            self._ensure_model_loaded(target_model)
        except GPUOutOfMemoryError as e:
            logger.error(f"GPU OOM during model loading for cloning: {e}")
            raise CloningError(f"GPU out of memory: {e}", voice_name=voice_name) from e
        except ModelLoadError as e:
            logger.error(f"Model load error for cloning: {e}")
            raise CloningError(f"Model unavailable: {e}", voice_name=voice_name) from e

        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Reference audio not found: {audio_path}")
            raise CloningError(f"Reference audio not found: {audio_path}", voice_name=voice_name)

        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            logger.debug(f"Audio duration: {duration:.2f}s, sample_rate: {info.sample_rate}")
            
            if duration < 3.0:
                logger.error(f"Reference audio too short: {duration:.1f}s")
                raise AudioTooShortError(duration)
            if duration > 30.0:
                logger.warning(
                    f"Reference audio is long ({duration:.1f}s). "
                    "Consider using a shorter clip for better results."
                )
        except AudioTooShortError:
            raise
        except Exception as e:
            logger.error(f"Failed to validate audio: {e}", exc_info=True)
            raise CloningError(f"Failed to validate audio: {e}", voice_name=voice_name) from e

        # Sanitize voice_id
        voice_id = re.sub(r'[^\w\-]', '_', voice_name)[:32]
        logger.debug(f"Sanitized voice_id: {voice_id}")

        # Check for duplicate
        if self._voice_manager.voice_exists(voice_id):
            raise CloningError(f"Voice with ID '{voice_id}' already exists", voice_name=voice_name)

        dest_audio = self.voices_path / f"{voice_id}.wav"
        try:
            logger.debug("Loading and resampling audio")
            waveform, sr = torchaudio.load(str(audio_path))
            target_sr = 24000  # Default sample rate
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            torchaudio.save(str(dest_audio), waveform, target_sr)
            logger.debug(f"Saved reference audio to {dest_audio}")
        except Exception as e:
            logger.error(f"Failed to process reference audio: {e}", exc_info=True)
            # Cleanup on failure
            if dest_audio.exists():
                dest_audio.unlink(missing_ok=True)
            raise CloningError(f"Failed to process reference audio: {e}", voice_name=voice_name) from e

        metadata = {
            "voice_id": voice_id,
            "name": voice_name,
            "language": language,
            "gender": "unknown",
            "is_cloned": True,
            "sample_rate": target_sr,
            "transcript": transcript,
            "created_at": datetime.utcnow().isoformat(),
            "model": target_model,
        }
        try:
            self._voice_manager.save_voice_metadata(voice_id, metadata)
            self._voice_registry.invalidate_cache()  # Refresh registry
            logger.info(f"Voice cloned successfully: {voice_id} (model: {target_model})")
        except Exception as e:
            logger.error(f"Failed to save voice metadata: {e}", exc_info=True)
            # Cleanup on failure
            if dest_audio.exists():
                dest_audio.unlink(missing_ok=True)
            raise CloningError(f"Failed to save voice metadata: {e}", voice_name=voice_name) from e
        finally:
            clear_log_context()
            
        return voice_id

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice."""
        logger.info(f"Deleting voice: {voice_id}")
        try:
            result = self._voice_manager.delete_voice(voice_id)
            if result:
                self._voice_registry.invalidate_cache()
            return result
        except Exception as e:
            logger.error(f"Error deleting voice {voice_id}: {e}", exc_info=True)
            raise

    def unload_model(self, model_type: str) -> bool:
        """Unload a specific model from memory."""
        if model_type in self._models:
            del self._models[model_type]
            del self._model_versions[model_type]
            del self._load_wav_funcs[model_type]
            del self._base_synthesizers[model_type]
            del self._cloned_synthesizers[model_type]
            self._model_loader.unload_model(model_type)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_type}")
            return True
        return False

    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        if model_type not in self._models:
            raise ValueError(f"Model not loaded: {model_type}")
        return {
            "model_type": model_type,
            "version": self._model_versions[model_type],
            "device": self.device,
        }

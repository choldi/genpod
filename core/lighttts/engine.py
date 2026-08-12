"""LightTTSEngine - Wrapper for CosyVoice 2 model."""

import os
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any
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


class LightTTSEngine:
    """Main wrapper class for CosyVoice 2 operations."""

    def __init__(self, models_path: str, voices_path: str, device: str = "cpu"):
        """Initialize the CosyVoice 2 model.

        Args:
            models_path: Path to base model weights.
            voices_path: Path to cloned voice profiles.
            device: Device to run inference on ('cpu', 'cuda', 'mps').
        """
        self.models_path = Path(models_path)
        self.voices_path = Path(voices_path)
        self.device = self._resolve_device(device)
        self._model = None
        self._voice_manager = VoiceManager(str(self.voices_path))
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Resolve the actual device to use based on availability."""
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS requested but not available, falling back to CPU")
            return "cpu"
        return device

    def _load_model(self) -> None:
        """Load the CosyVoice 2 model."""
        try:
            # Import CosyVoice 2 - this will fail if not installed
            from cosyvoice.cli.cosyvoice import CosyVoice2
            from cosyvoice.utils.file_utils import load_wav

            # Store imports for later use
            self._CosyVoice2 = CosyVoice2
            self._load_wav = load_wav

            # Model path should contain the CosyVoice 2 weights
            model_dir = self.models_path / "CosyVoice2-0.5B"
            if not model_dir.exists():
                # Try alternative naming
                model_dirs = list(self.models_path.glob("CosyVoice*"))
                if model_dirs:
                    model_dir = model_dirs[0]
                else:
                    raise ModelLoadError(
                        f"CosyVoice 2 model not found in {self.models_path}. "
                        "Please download the model weights first."
                    )

            print(f"Loading CosyVoice 2 model from {model_dir} on {self.device}...")
            self._model = CosyVoice2(str(model_dir), load_jit=False, fp16=(self.device == "cuda"))
            print("Model loaded successfully!")

        except ImportError as e:
            raise ModelLoadError(
                "CosyVoice 2 library not installed. "
                "Please install it with: pip install -U git+https://github.com/FunAudioLLM/CosyVoice.git"
            ) from e
        except Exception as e:
            raise ModelLoadError(f"Failed to load CosyVoice 2 model: {e}") from e

    def list_voices(self) -> List[Dict[str, Any]]:
        """Scan and return all available voices (base and cloned)."""
        voices = []

        # Add cloned voices from voice manager
        cloned_voices = self._voice_manager.list_voices()
        for voice in cloned_voices:
            voices.append({
                "voice_id": voice.get("voice_id", ""),
                "name": voice.get("name", "Unknown"),
                "language": voice.get("language", "en"),
                "gender": voice.get("gender"),
                "is_cloned": True,
                "sample_rate": voice.get("sample_rate", 24000),
            })

        # Add base voices from CosyVoice 2 (pre-trained speakers)
        if self._model and hasattr(self._model, 'list_available_spks'):
            try:
                base_spks = self._model.list_available_spks()
                for spk in base_spks:
                    voices.append({
                        "voice_id": f"base_{spk}",
                        "name": f"Base {spk.capitalize()}",
                        "language": "en",
                        "gender": "unknown",
                        "is_cloned": False,
                        "sample_rate": 24000,
                    })
            except Exception:
                pass  # Base voice listing optional

        return voices

    def synthesize(
        self, text: str, voice_id: str, lang: str = "en", stream: bool = True
    ) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text and voice.

        Args:
            text: Text to synthesize.
            voice_id: Voice identifier (base_* for pre-trained, or cloned voice_id).
            lang: Language code.
            stream: Whether to yield chunks or return full audio.

        Yields:
            Audio chunks as bytes (WAV format).
        """
        if not self._model:
            raise SynthesisError("Model not loaded")

        try:
            # Determine if it's a base voice or cloned voice
            if voice_id.startswith("base_"):
                spk_id = voice_id.replace("base_", "")
                yield from self._synthesize_base(text, spk_id, stream)
            else:
                yield from self._synthesize_cloned(text, voice_id, stream)

        except Exception as e:
            raise SynthesisError(f"Synthesis failed: {e}") from e

    def _synthesize_base(
        self, text: str, spk_id: str, stream: bool
    ) -> Generator[bytes, None, None]:
        """Synthesize using a base (pre-trained) speaker."""
        # CosyVoice 2 uses inference_sft for base speakers
        # This generates the full audio at once, we'll chunk it for streaming
        try:
            # Generate speech
            output = self._model.inference_sft(text, spk_id, stream=False)
            
            # output is a dict with 'tts_speech' tensor
            speech = output['tts_speech']
            
            # Convert to bytes (WAV format)
            import io
            buffer = io.BytesIO()
            torchaudio.save(buffer, speech.unsqueeze(0).cpu(), 24000, format="wav")
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            if stream:
                # Yield in chunks (e.g., 4096 bytes)
                chunk_size = 4096
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]
            else:
                yield audio_bytes

        except Exception as e:
            raise SynthesisError(f"Base voice synthesis failed: {e}") from e

    def _synthesize_cloned(
        self, text: str, voice_id: str, stream: bool
    ) -> Generator[bytes, None, None]:
        """Synthesize using a cloned voice."""
        # Load voice metadata to get reference audio path
        metadata = self._voice_manager.load_voice_metadata(voice_id)
        ref_audio_path = self.voices_path / f"{voice_id}.wav"
        
        if not ref_audio_path.exists():
            raise VoiceNotFoundError(f"Reference audio for voice '{voice_id}' not found")

        try:
            # Load reference audio
            ref_speech = self._load_wav(str(ref_audio_path), 24000)
            ref_speech = ref_speech.to(self.device)
            
            # CosyVoice 2 uses inference_zero_shot for cloned voices
            # It needs the reference speech and the prompt text (transcript)
            prompt_text = metadata.get("transcript", "")
            
            output = self._model.inference_zero_shot(
                text, prompt_text, ref_speech.unsqueeze(0), stream=False
            )
            
            speech = output['tts_speech']
            
            # Convert to bytes
            import io
            buffer = io.BytesIO()
            torchaudio.save(buffer, speech.unsqueeze(0).cpu(), 24000, format="wav")
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            if stream:
                chunk_size = 4096
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]
            else:
                yield audio_bytes

        except Exception as e:
            raise SynthesisError(f"Cloned voice synthesis failed: {e}") from e

    def clone_voice(
        self, audio_path: str, transcript: str, voice_name: str, language: str = "en"
    ) -> str:
        """Clone a voice from reference audio and transcript.

        Args:
            audio_path: Path to reference audio file.
            transcript: Transcript of the reference audio.
            voice_name: Name for the new voice.
            language: Language code.

        Returns:
            New voice_id.
        """
        if not self._model:
            raise CloningError("Model not loaded")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise CloningError(f"Reference audio not found: {audio_path}")

        # Validate audio duration (CosyVoice needs at least a few seconds)
        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            if duration < 3.0:
                raise AudioTooShortError(
                    f"Reference audio too short: {duration:.1f}s. Minimum 3 seconds required."
                )
            if duration > 30.0:
                print(f"Warning: Reference audio is long ({duration:.1f}s). "
                      "Consider using a shorter clip for better results.")
        except AudioTooShortError:
            raise
        except Exception as e:
            raise CloningError(f"Failed to validate audio: {e}") from e

        # Generate unique voice_id
        voice_id = f"cloned_{uuid.uuid4().hex[:12]}"
        
        # Copy reference audio to voices directory
        dest_audio = self.voices_path / f"{voice_id}.wav"
        try:
            # Resample to 24kHz if needed
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                waveform = resampler(waveform)
            torchaudio.save(str(dest_audio), waveform, 24000)
        except Exception as e:
            raise CloningError(f"Failed to process reference audio: {e}") from e

        # Save metadata
        metadata = {
            "voice_id": voice_id,
            "name": voice_name,
            "language": language,
            "gender": "unknown",  # Could be detected or user-provided
            "is_cloned": True,
            "sample_rate": 24000,
            "transcript": transcript,
            "created_at": str(torch.tensor(0).numpy()),  # placeholder for timestamp
        }
        
        self._voice_manager.save_voice_metadata(voice_id, metadata)
        
        return voice_id

"""Cloned Synthesizer for zero-shot voice cloning."""

import logging
import torch
from pathlib import Path
from typing import Generator, Optional, Any, Dict

from core.lighttts.voice_manager import VoiceManager
from core.lighttts.audio_utils import (
    tensor_to_wav_bytes,
    apply_pitch_shift,
    crossfade_chunks,
    validate_audio_length,
)
from core.exceptions import VoiceNotFoundError
from core.logger import get_logger

logger = get_logger(__name__)


class ClonedSynthesizer:
    """Handles synthesis with cloned voices using zero-shot inference."""

    def __init__(
        self,
        model: Any,
        model_version: str,
        voices_path: str,
        voice_manager: VoiceManager,
        model_type: str = "cosyvoice2",
    ) -> None:
        """Initialize the cloned synthesizer.
        
        Args:
            model: Loaded model instance
            model_version: Model version string
            voices_path: Path to voices directory
            voice_manager: VoiceManager instance
            model_type: Type of model (cosyvoice2, cosyvoice3, voxcpm)
        """
        self.model = model
        self.model_version = model_version
        self.voices_path = Path(voices_path)
        self._voice_manager = voice_manager
        self.model_type = model_type
        logger.debug(f"ClonedSynthesizer initialized for {model_type} ({model_version})")

    def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float,
        pitch: float,
        language: str,
        stream: bool,
        chunk_size: Optional[int] = None,
    ) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text and cloned voice.
        
        Args:
            text: Text to synthesize
            voice_id: Cloned voice identifier
            speed: Speech speed factor
            pitch: Pitch factor
            language: Language code
            stream: Whether to stream chunks
            chunk_size: Chunking strategy
            
        Yields:
            Audio chunks as WAV bytes
        """
        logger.debug(f"Cloned synthesis: voice_id={voice_id}, text_len={len(text)}, "
                    f"speed={speed}, pitch={pitch}, language={language}, chunk_size={chunk_size}")

        # Load voice metadata and reference audio
        try:
            metadata = self._voice_manager.load_voice_metadata(voice_id)
            ref_audio_path = self.voices_path / f"{voice_id}.wav"
            
            if not ref_audio_path.exists():
                raise VoiceNotFoundError(f"Reference audio not found for voice: {voice_id}")
                
        except Exception as e:
            logger.error(f"Failed to load voice {voice_id}: {e}", exc_info=True)
            raise VoiceNotFoundError(f"Voice not found: {voice_id}") from e

        try:
            # Prepare text chunks
            text_chunks = self._prepare_text_chunks(text, chunk_size)
            logger.debug(f"Text split into {len(text_chunks)} chunks")

            for i, chunk_text in enumerate(text_chunks):
                logger.debug(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk_text[:50]}...")
                
                # Call model-specific synthesis
                if self.model_type.startswith("cosyvoice"):
                    speech = self._synthesize_cosyvoice(chunk_text, ref_audio_path, metadata, speed, language)
                elif self.model_type == "voxcpm":
                    speech = self._synthesize_voxcpm(chunk_text, ref_audio_path, metadata, speed, language)
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")

                # Apply pitch shift if needed
                if pitch != 1.0:
                    speech = apply_pitch_shift(speech, pitch)

                # Validate audio length
                if not validate_audio_length(speech):
                    logger.warning("Generated audio too short, skipping")
                    continue

                # Convert to WAV bytes
                wav_bytes = tensor_to_wav_bytes(speech)
                yield wav_bytes

        except Exception as e:
            logger.error(f"Cloned synthesis failed: {e}", exc_info=True)
            raise

    def _prepare_text_chunks(self, text: str, chunk_size: Optional[int]) -> list:
        """Prepare text chunks based on chunking strategy."""
        if chunk_size is None:
            # Default: sentence-based chunking
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s for s in sentences if s]
        elif chunk_size <= 0:
            # No chunking
            return [text]
        else:
            # Custom character-based chunking
            chunks = []
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if i + chunk_size < len(text):
                    last_punct = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
                    if last_punct > chunk_size * 0.5:
                        chunk = chunk[:last_punct + 1]
                chunks.append(chunk.strip())
            return [c for c in chunks if c]

    def _synthesize_cosyvoice(
        self, 
        text: str, 
        ref_audio_path: Path, 
        metadata: Dict[str, Any],
        speed: float,
        language: str
    ) -> torch.Tensor:
        """Synthesize using CosyVoice model with zero-shot cloning."""
        # Load reference audio
        from core.lighttts.model_loader import load_wav  # This would be the load_wav from model
        # Actually, we need to get the load_wav function from the model loader
        # This is a placeholder - actual implementation depends on CosyVoice API
        
        # CosyVoice zero-shot inference typically requires:
        # 1. Reference audio (prompt speech)
        # 2. Prompt text (transcript of reference audio)
        # 3. Target text
        
        prompt_text = metadata.get("transcript", "")
        if not prompt_text:
            logger.warning("No transcript in metadata, zero-shot quality may suffer")
        
        # Placeholder - actual implementation depends on CosyVoice version
        raise NotImplementedError("CosyVoice cloned synthesis not fully implemented")

    def _synthesize_voxcpm(
        self, 
        text: str, 
        ref_audio_path: Path, 
        metadata: Dict[str, Any],
        speed: float,
        language: str
    ) -> torch.Tensor:
        """Synthesize using VoxCPM model with zero-shot cloning."""
        try:
            # Load reference audio using torchaudio (VoxCPM typically uses 24kHz)
            import torchaudio
            prompt_speech, sr = torchaudio.load(str(ref_audio_path))
            
            # Resample to 24kHz if needed
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                prompt_speech = resampler(prompt_speech)
            
            # Get prompt text from metadata
            prompt_text = metadata.get("transcript", "")
            if not prompt_text:
                logger.warning("No transcript in metadata, zero-shot quality may suffer")
            
            # Prepare synthesis parameters for VoxCPM zero-shot
            # VoxCPM zero-shot typically requires: prompt_speech, prompt_text, target_text
            synthesis_kwargs = {
                "text": text,
                "prompt_speech": prompt_speech,
                "prompt_text": prompt_text,
                "speed": speed,
            }
            
            # Add language if supported
            if language:
                synthesis_kwargs["language"] = language
            
            # Call VoxCPM zero-shot inference
            if hasattr(self.model, 'generate_zero_shot'):
                speech = self.model.generate_zero_shot(**synthesis_kwargs)
            elif hasattr(self.model, 'zero_shot_inference'):
                speech = self.model.zero_shot_inference(**synthesis_kwargs)
            elif hasattr(self.model, 'clone_voice'):
                speech = self.model.clone_voice(**synthesis_kwargs)
            elif hasattr(self.model, 'inference'):
                # Some models use a unified inference method
                speech = self.model.inference(**synthesis_kwargs)
            else:
                # Try direct call
                speech = self.model(**synthesis_kwargs)
            
            # Ensure output is torch.Tensor
            if not isinstance(speech, torch.Tensor):
                speech = torch.tensor(speech)
            
            # Ensure correct shape: [1, samples] or [samples]
            if speech.dim() == 1:
                speech = speech.unsqueeze(0)
            elif speech.dim() > 2:
                speech = speech.squeeze(0)
                if speech.dim() > 1:
                    speech = speech[0:1]  # Take first channel
            
            logger.debug(f"VoxCPM zero-shot synthesis successful: shape={speech.shape}")
            return speech
            
        except Exception as e:
            logger.error(f"VoxCPM zero-shot synthesis failed: {e}", exc_info=True)
            raise RuntimeError(f"VoxCPM zero-shot synthesis error: {e}") from e

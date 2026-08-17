"""Base Synthesizer for pre-trained voices."""

import logging
import torch
from typing import Generator, Optional, List, Any

from core.lighttts.audio_utils import (
    tensor_to_wav_bytes,
    apply_pitch_shift,
    crossfade_chunks,
    validate_audio_length,
)
from core.logger import get_logger

logger = get_logger(__name__)


class BaseSynthesizer:
    """Handles synthesis with base (pre-trained) voices."""

    def __init__(self, model: Any, model_version: str, model_type: str = "cosyvoice2") -> None:
        """Initialize the base synthesizer.
        
        Args:
            model: Loaded model instance
            model_version: Model version string
            model_type: Type of model (cosyvoice2, cosyvoice3, voxcpm)
        """
        self.model = model
        self.model_version = model_version
        self.model_type = model_type
        logger.debug(f"BaseSynthesizer initialized for {model_type} ({model_version})")

    def synthesize(
        self,
        text: str,
        spk_id: str,
        speed: float,
        pitch: float,
        stream: bool,
        chunk_size: Optional[int] = None,
    ) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text and speaker.
        
        Args:
            text: Text to synthesize
            spk_id: Speaker ID (base speaker)
            speed: Speech speed factor
            pitch: Pitch factor
            stream: Whether to stream chunks
            chunk_size: Chunking strategy
            
        Yields:
            Audio chunks as WAV bytes
        """
        logger.debug(f"Base synthesis: spk_id={spk_id}, text_len={len(text)}, "
                    f"speed={speed}, pitch={pitch}, chunk_size={chunk_size}")

        try:
            # Prepare text chunks based on chunk_size
            text_chunks = self._prepare_text_chunks(text, chunk_size)
            logger.debug(f"Text split into {len(text_chunks)} chunks")

            for i, chunk_text in enumerate(text_chunks):
                logger.debug(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk_text[:50]}...")
                
                # Call model-specific synthesis
                if self.model_type.startswith("cosyvoice"):
                    speech = self._synthesize_cosyvoice(chunk_text, spk_id, speed)
                elif self.model_type == "voxcpm":
                    speech = self._synthesize_voxcpm(chunk_text, spk_id, speed)
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
            logger.error(f"Base synthesis failed: {e}", exc_info=True)
            raise

    def _prepare_text_chunks(self, text: str, chunk_size: Optional[int]) -> List[str]:
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
                # Try to break at sentence boundary
                if i + chunk_size < len(text):
                    last_punct = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
                    if last_punct > chunk_size * 0.5:  # Only if not too short
                        chunk = chunk[:last_punct + 1]
                chunks.append(chunk.strip())
            return [c for c in chunks if c]

    def _synthesize_cosyvoice(self, text: str, spk_id: str, speed: float) -> torch.Tensor:
        """Synthesize using CosyVoice model."""
        # CosyVoice expects specific speaker format
        if not spk_id.startswith("speaker_"):
            spk_id = f"speaker_{spk_id}"
        
        # Call model inference
        # Note: CosyVoice API may vary between versions
        if hasattr(self.model, 'inference_zero_shot'):
            # CosyVoice 2/3 zero-shot inference
            # For base speakers, we might need a different approach
            # This is a simplified version - actual implementation depends on CosyVoice API
            pass
        
        # Placeholder - actual implementation depends on CosyVoice version
        # This should be replaced with actual model.call
        raise NotImplementedError("CosyVoice synthesis not fully implemented")

    def _synthesize_voxcpm(self, text: str, spk_id: str, speed: float) -> torch.Tensor:
        """Synthesize using VoxCPM model."""
        # Placeholder for VoxCPM synthesis
        # Will be implemented in Phase 2
        raise NotImplementedError("VoxCPM synthesis not yet implemented (Phase 2)")

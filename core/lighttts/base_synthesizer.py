"""Base voice synthesis for LightTTS."""

import logging
import re
from typing import Generator, List, Optional
import torch
import torchaudio

from core.exceptions import SynthesisError, VoiceNotFoundError
from core.lighttts.audio_utils import (
    tensor_to_wav_bytes,
    apply_pitch_shift,
    crossfade_chunks,
    validate_audio_length,
)

logger = logging.getLogger(__name__)


class BaseSynthesizer:
    """Handles synthesis with base (pre-trained) voices."""

    def __init__(self, model: object, model_version: str) -> None:
        self._model = model
        self._model_version = model_version

    def synthesize(
        self,
        text: str,
        spk_id: str,
        speed: float,
        pitch: float,
        stream: bool,
        chunk_size: Optional[int] = None,
    ) -> Generator[bytes, None, None]:
        """Generate audio for base voice with configurable chunking.

        Args:
            text: Text to synthesize
            spk_id: Speaker ID
            speed: Speed factor
            pitch: Pitch factor
            stream: Whether to stream chunks
            chunk_size: Chunking strategy:
                - None (default): Sentence-based chunking
                - <= 0: No chunking, process entire text
                - > 0: Custom chunk size in characters
        """
        try:
            # Determine chunking strategy
            if chunk_size is None:
                # Default: sentence-based chunking
                audio_bytes = self._synthesize_sentence_chunked(text, spk_id, speed, pitch)
            elif chunk_size <= 0:
                # No chunking: process entire text
                audio_bytes = self._synthesize_full_text(text, spk_id, speed, pitch)
            else:
                # Custom chunk size
                audio_bytes = self._synthesize_custom_chunked(text, spk_id, speed, pitch, chunk_size)

            if stream:
                chunk_size_bytes = 4096
                for i in range(0, len(audio_bytes), chunk_size_bytes):
                    yield audio_bytes[i:i + chunk_size_bytes]
            else:
                yield audio_bytes

        except VoiceNotFoundError:
            raise
        except Exception as e:
            raise SynthesisError(f"Base voice synthesis failed: {e}") from e

    def _synthesize_full_text(
        self, text: str, spk_id: str, speed: float, pitch: float
    ) -> bytes:
        """Synthesize entire text in a single inference call."""
        try:
            target_text = self._ensure_punctuation(text)
            logger.debug(f"Full-text base synthesis (v{self._model_version}): '{target_text[:60]}...'")

            if self._model_version == "v3" and hasattr(self._model, 'inference_sft'):
                output_generator = self._model.inference_sft(target_text, spk_id, stream=False)
            else:
                output_generator = self._model.inference_sft(target_text, spk_id, stream=False)

            all_speech_chunks: List[torch.Tensor] = []
            for out_dict in output_generator:
                speech = out_dict['tts_speech']
                if speech.dim() == 3:
                    speech = speech.squeeze(0)
                if speech.dim() == 1:
                    speech = speech.unsqueeze(0)
                all_speech_chunks.append(speech.cpu())

            if not all_speech_chunks:
                raise SynthesisError("Model did not generate any audio for base voice (full-text)")

            speech = torch.cat(all_speech_chunks, dim=-1)
            logger.debug(f"Full-text base audio: {speech.shape[-1] / 24000:.2f}s")

            return self._post_process(speech, speed, pitch)

        except Exception as e:
            logger.error(f"Full-text synthesis failed: {e}")
            raise SynthesisError(f"Full-text synthesis failed: {e}") from e

    def _synthesize_sentence_chunked(
        self, text: str, spk_id: str, speed: float, pitch: float
    ) -> bytes:
        """Synthesize long text by splitting into sentence chunks with crossfade (default behavior)."""
        text_chunks = self._split_text_into_chunks(text)
        logger.debug(f"Base voice: text split into {len(text_chunks)} sentence chunks")

        all_speech_chunks: List[torch.Tensor] = []

        for idx, chunk in enumerate(text_chunks):
            logger.debug(f"Synthesizing base chunk {idx + 1}/{len(text_chunks)}: '{chunk[:60]}...'")

            chunk = self._ensure_punctuation(chunk)

            try:
                if self._model_version == "v3" and hasattr(self._model, 'inference_sft'):
                    output_generator = self._model.inference_sft(chunk, spk_id, stream=False)
                else:
                    output_generator = self._model.inference_sft(chunk, spk_id, stream=False)

                chunk_speech_list: List[torch.Tensor] = []
                for out_dict in output_generator:
                    speech = out_dict['tts_speech']
                    if speech.dim() == 3:
                        speech = speech.squeeze(0)
                    if speech.dim() == 1:
                        speech = speech.unsqueeze(0)
                    chunk_speech_list.append(speech.cpu())

                if chunk_speech_list:
                    chunk_speech = torch.cat(chunk_speech_list, dim=-1)
                    if validate_audio_length(chunk_speech):
                        all_speech_chunks.append(chunk_speech)
                        logger.debug(f"Base chunk {idx + 1} generated: {chunk_speech.shape[-1] / 24000:.2f}s")
                    else:
                        logger.warning(f"Base chunk {idx + 1} did not generate valid audio")

            except Exception as e:
                logger.error(f"Failed to synthesize base chunk {idx + 1}: {e}")
                raise SynthesisError(f"Chunk {idx + 1} synthesis failed: {e}") from e

        if not all_speech_chunks:
            raise SynthesisError("Model did not generate any audio for base voice (sentence-chunked)")

        # Use crossfade for smooth transitions
        speech = crossfade_chunks(all_speech_chunks)
        logger.debug(f"Final base audio (crossfaded): {speech.shape[-1] / 24000:.2f}s")

        return self._post_process(speech, speed, pitch)

    def _synthesize_custom_chunked(
        self, text: str, spk_id: str, speed: float, pitch: float, chunk_size: int
    ) -> bytes:
        """Synthesize long text by splitting into custom-sized chunks with crossfade."""
        text_chunks = self._split_text_by_size(text, chunk_size)
        logger.debug(f"Base voice: text split into {len(text_chunks)} custom chunks (~{chunk_size} chars each)")

        all_speech_chunks: List[torch.Tensor] = []

        for idx, chunk in enumerate(text_chunks):
            logger.debug(f"Synthesizing base chunk {idx + 1}/{len(text_chunks)}: '{chunk[:60]}...'")

            chunk = self._ensure_punctuation(chunk)

            try:
                if self._model_version == "v3" and hasattr(self._model, 'inference_sft'):
                    output_generator = self._model.inference_sft(chunk, spk_id, stream=False)
                else:
                    output_generator = self._model.inference_sft(chunk, spk_id, stream=False)

                chunk_speech_list: List[torch.Tensor] = []
                for out_dict in output_generator:
                    speech = out_dict['tts_speech']
                    if speech.dim() == 3:
                        speech = speech.squeeze(0)
                    if speech.dim() == 1:
                        speech = speech.unsqueeze(0)
                    chunk_speech_list.append(speech.cpu())

                if chunk_speech_list:
                    chunk_speech = torch.cat(chunk_speech_list, dim=-1)
                    if validate_audio_length(chunk_speech):
                        all_speech_chunks.append(chunk_speech)
                        logger.debug(f"Base chunk {idx + 1} generated: {chunk_speech.shape[-1] / 24000:.2f}s")
                    else:
                        logger.warning(f"Base chunk {idx + 1} did not generate valid audio")

            except Exception as e:
                logger.error(f"Failed to synthesize base chunk {idx + 1}: {e}")
                raise SynthesisError(f"Chunk {idx + 1} synthesis failed: {e}") from e

        if not all_speech_chunks:
            raise SynthesisError("Model did not generate any audio for base voice (custom-chunked)")

        # Use crossfade for smooth transitions
        speech = crossfade_chunks(all_speech_chunks)
        logger.debug(f"Final base audio (crossfaded): {speech.shape[-1] / 24000:.2f}s")

        return self._post_process(speech, speed, pitch)

    def _post_process(self, speech: torch.Tensor, speed: float, pitch: float) -> bytes:
        """Apply speed and pitch adjustments."""
        if speed != 1.0:
            speech = torchaudio.functional.resample(speech, int(24000 * speed), 24000)
        if pitch != 1.0:
            speech = apply_pitch_shift(speech, pitch)
        return tensor_to_wav_bytes(speech)

    def _ensure_punctuation(self, text: str) -> str:
        """Ensure text ends with punctuation."""
        text = text.strip()
        if not any(text.endswith(p) for p in [".", "!", "?", "。", "！", "？"]):
            text += "."
        return text

    def _split_text_into_chunks(self, text: str, max_chars: int = 5000) -> List[str]:
        """Split text into chunks based on sentence boundaries (default behavior)."""
        sentences = re.split(r'(?<=[.!?。！？])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text]

        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _split_text_by_size(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size characters."""
        if chunk_size <= 0:
            return [text]

        words = text.split()
        chunks: List[str] = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

"""Base voice synthesis for LightTTS."""

import logging
from typing import Generator, List
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
    ) -> Generator[bytes, None, None]:
        """Generate audio for base voice."""
        try:
            # For CosyVoice 3, use streaming inference for better prosody continuity
            if self._model_version == "v3" and hasattr(self._model, 'inference_sft_stream'):
                audio_bytes = self._synthesize_streaming(text, spk_id, speed, pitch)
            # For v2, try single-shot first (handles long texts well)
            elif self._model_version == "v2":
                audio_bytes = self._synthesize_single_shot(text, spk_id, speed, pitch)
            # For v1 or fallback, use chunked synthesis
            else:
                audio_bytes = self._synthesize_chunked(text, spk_id, speed, pitch)

            if stream:
                chunk_size = 4096
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]
            else:
                yield audio_bytes

        except VoiceNotFoundError:
            raise
        except Exception as e:
            raise SynthesisError(f"Base voice synthesis failed: {e}") from e

    def _synthesize_streaming(
        self, text: str, spk_id: str, speed: float, pitch: float
    ) -> bytes:
        """Synthesize using CosyVoice 3 streaming inference."""
        try:
            target_text = self._ensure_punctuation(text)
            logger.debug(f"Streaming synthesis (v3): '{target_text[:60]}...'")

            output_generator = self._model.inference_sft(target_text, spk_id, stream=True)

            all_speech_chunks: List[torch.Tensor] = []
            for out_dict in output_generator:
                speech = out_dict['tts_speech']
                if speech.dim() == 3:
                    speech = speech.squeeze(0)
                if speech.dim() == 1:
                    speech = speech.unsqueeze(0)
                all_speech_chunks.append(speech.cpu())

            if not all_speech_chunks:
                raise SynthesisError("Model did not generate any audio for base voice (streaming)")

            speech = torch.cat(all_speech_chunks, dim=-1)
            logger.debug(f"Streaming base audio: {speech.shape[-1] / 24000:.2f}s")

            return self._post_process(speech, speed, pitch)

        except Exception as e:
            logger.warning(f"Streaming synthesis failed, falling back to single-shot: {e}")
            return self._synthesize_single_shot(text, spk_id, speed, pitch)

    def _synthesize_single_shot(
        self, text: str, spk_id: str, speed: float, pitch: float
    ) -> bytes:
        """Synthesize base voice in a single shot for the entire text."""
        try:
            target_text = self._ensure_punctuation(text)
            logger.debug(f"Single-shot base synthesis: '{target_text[:60]}...'")

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
                raise SynthesisError("Model did not generate any audio for base voice (single-shot)")

            speech = torch.cat(all_speech_chunks, dim=-1)
            logger.debug(f"Single-shot base audio: {speech.shape[-1] / 24000:.2f}s")

            return self._post_process(speech, speed, pitch)

        except Exception as e:
            logger.warning(f"Single-shot synthesis failed, falling back to chunked: {e}")
            return self._synthesize_chunked(text, spk_id, speed, pitch)

    def _synthesize_chunked(
        self, text: str, spk_id: str, speed: float, pitch: float
    ) -> bytes:
        """Synthesize long text by chunks with crossfade (v1/v2 fallback)."""
        text_chunks = self._split_text_into_chunks(text)
        logger.debug(f"Base voice: text split into {len(text_chunks)} chunks")

        all_speech_chunks: List[torch.Tensor] = []

        for idx, chunk in enumerate(text_chunks):
            logger.debug(f"Synthesizing base chunk {idx + 1}/{len(text_chunks)}: '{chunk[:60]}...'")

            chunk = self._ensure_punctuation(chunk)

            output_generator = self._model.inference_sft(chunk, spk_id, stream=False)

            chunk_speech_list: List[torch.Tensor] = []
            for out_dict in output_generator:
                speech = out_dict['tts_speech']
                if speech.dim() == 3:
                    speech = speech.squeeze(0)
                chunk_speech_list.append(speech.cpu())

            if chunk_speech_list:
                chunk_speech = torch.cat(chunk_speech_list, dim=-1)
                if validate_audio_length(chunk_speech):
                    all_speech_chunks.append(chunk_speech)
                    logger.debug(f"Base chunk {idx + 1} generated: {chunk_speech.shape[-1] / 24000:.2f}s")
                else:
                    logger.warning(f"Base chunk {idx + 1} did not generate valid audio")

        if not all_speech_chunks:
            raise SynthesisError("Model did not generate any audio for base voice")

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
        """Split text into chunks based on sentence boundaries."""
        sentences = __import__('re').split(r'(?<=[.!?。！？])\s*', text)
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

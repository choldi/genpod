"""Cloned voice synthesis for LightTTS."""

import logging
import re
from pathlib import Path
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

# Prefix required for CosyVoice 3 zero-shot prompt to prevent hallucination
ZERO_SHOT_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
# Maximum length for prompt text to avoid model reproducing it
MAX_PROMPT_CHARS = 100
# Maximum characters per text chunk for synthesis (used when chunk_size=None)
MAX_CHUNK_CHARS = 5000


class ClonedSynthesizer:
    """Handles synthesis with cloned voices using zero-shot inference."""

    def __init__(self, model: object, model_version: str, voices_path: str, voice_manager) -> None:
        self._model = model
        self._model_version = model_version
        self._voices_path = Path(voices_path)
        self._voice_manager = voice_manager

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
        """Generate audio for cloned voice with configurable chunking.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            speed: Speed factor
            pitch: Pitch factor
            language: Language code
            stream: Whether to stream chunks
            chunk_size: Chunking strategy:
                - None (default): Sentence-based chunking with hallucination prevention
                - <= 0: No chunking, process entire text at once
                - > 0: Custom chunk size in characters
        """
        metadata = self._voice_manager.load_voice_metadata(voice_id)
        ref_audio_path = self._voices_path / f"{voice_id}.wav"

        if not ref_audio_path.exists():
            raise VoiceNotFoundError(f"Reference audio for voice '{voice_id}' not found")

        try:
            # Prepare prompt text ONCE from metadata
            prompt_text = self._prepare_prompt_text(metadata.get("transcript", ""))

            # Ensure target text ends with punctuation
            target_text = text.strip()
            if not any(target_text.endswith(p) for p in [".", "!", "?", "。", "！", "？"]):
                target_text += "."

            # Determine chunking strategy
            if chunk_size is None:
                # Default: sentence-based chunking with hallucination prevention
                logger.debug(f"Starting sentence-chunked zero-shot synthesis ({self._model_version}): '{target_text[:60]}...'")
                audio_bytes = self._synthesize_sentence_chunked(
                    target_text, prompt_text, str(ref_audio_path), speed, pitch
                )
            elif chunk_size <= 0:
                # No chunking: process entire text
                logger.debug(f"Starting full-text zero-shot synthesis ({self._model_version}): '{target_text[:60]}...'")
                audio_bytes = self._synthesize_full_text(
                    target_text, prompt_text, str(ref_audio_path), speed, pitch
                )
            else:
                # Custom chunk size
                logger.debug(f"Starting custom-chunked zero-shot synthesis ({self._model_version}, chunk_size={chunk_size}): '{target_text[:60]}...'")
                audio_bytes = self._synthesize_custom_chunked(
                    target_text, prompt_text, str(ref_audio_path), speed, pitch, chunk_size
                )

            if stream:
                chunk_size_bytes = 4096
                for i in range(0, len(audio_bytes), chunk_size_bytes):
                    yield audio_bytes[i:i + chunk_size_bytes]
            else:
                yield audio_bytes

        except VoiceNotFoundError:
            raise
        except SynthesisError:
            raise
        except Exception as e:
            logger.error(f"Cloned voice synthesis failed: {e}")
            raise SynthesisError(f"Cloned voice synthesis failed: {e}") from e

    def _synthesize_full_text(
        self,
        target_text: str,
        prompt_text: str,
        ref_audio_path: str,
        speed: float,
        pitch: float,
    ) -> bytes:
        """Synthesize entire text in a single zero-shot inference call."""
        try:
            logger.debug(f"Full-text zero-shot inference for text length: {len(target_text)}")

            output_generator = self._model.inference_zero_shot(
                target_text, prompt_text, ref_audio_path, stream=False
            )

            all_speech_chunks: List[torch.Tensor] = []
            for out_dict in output_generator:
                speech = out_dict['tts_speech']
                if speech.dim() == 3:
                    speech = speech.squeeze(0)
                if speech.dim() == 1:
                    speech = speech.unsqueeze(0)
                all_speech_chunks.append(speech.cpu())

            if not all_speech_chunks:
                raise SynthesisError("Model did not generate any audio for cloned voice (full-text)")

            speech = torch.cat(all_speech_chunks, dim=-1)
            logger.debug(f"Full-text cloned audio: {speech.shape[-1] / 24000:.2f}s")

            return self._post_process(speech, speed, pitch)

        except Exception as e:
            logger.error(f"Full-text zero-shot synthesis failed: {e}")
            raise SynthesisError(f"Full-text synthesis failed: {e}") from e

    def _synthesize_sentence_chunked(
        self,
        target_text: str,
        prompt_text: str,
        ref_audio_path: str,
        speed: float,
        pitch: float,
    ) -> bytes:
        """Synthesize long text by splitting into sentence chunks with crossfade (default behavior)."""
        text_chunks = self._split_text_into_chunks(target_text)
        logger.debug(f"Cloned voice: text split into {len(text_chunks)} sentence chunks")

        all_speech_chunks: List[torch.Tensor] = []

        for idx, chunk in enumerate(text_chunks):
            logger.debug(f"Synthesizing cloned chunk {idx + 1}/{len(text_chunks)}: '{chunk[:60]}...'")

            try:
                output_generator = self._model.inference_zero_shot(
                    chunk, prompt_text, ref_audio_path, stream=False
                )

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
                        logger.debug(f"Cloned chunk {idx + 1} generated: {chunk_speech.shape[-1] / 24000:.2f}s")
                    else:
                        logger.warning(f"Cloned chunk {idx + 1} did not generate valid audio")
                else:
                    logger.warning(f"Cloned chunk {idx + 1} returned empty generator")

            except Exception as e:
                logger.error(f"Failed to synthesize cloned chunk {idx + 1}: {e}")
                raise SynthesisError(f"Chunk {idx + 1} synthesis failed: {e}") from e

        if not all_speech_chunks:
            raise SynthesisError("Model did not generate any audio for cloned voice (sentence-chunked)")

        # Use crossfade for smooth transitions between chunks
        speech = crossfade_chunks(all_speech_chunks)
        logger.debug(f"Final cloned audio (crossfaded): {speech.shape[-1] / 24000:.2f}s")

        return self._post_process(speech, speed, pitch)

    def _synthesize_custom_chunked(
        self,
        target_text: str,
        prompt_text: str,
        ref_audio_path: str,
        speed: float,
        pitch: float,
        chunk_size: int,
    ) -> bytes:
        """Synthesize long text by splitting into custom-sized chunks with crossfade."""
        text_chunks = self._split_text_by_size(target_text, chunk_size)
        logger.debug(f"Cloned voice: text split into {len(text_chunks)} custom chunks (~{chunk_size} chars each)")

        all_speech_chunks: List[torch.Tensor] = []

        for idx, chunk in enumerate(text_chunks):
            logger.debug(f"Synthesizing cloned chunk {idx + 1}/{len(text_chunks)}: '{chunk[:60]}...'")

            try:
                output_generator = self._model.inference_zero_shot(
                    chunk, prompt_text, ref_audio_path, stream=False
                )

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
                        logger.debug(f"Cloned chunk {idx + 1} generated: {chunk_speech.shape[-1] / 24000:.2f}s")
                    else:
                        logger.warning(f"Cloned chunk {idx + 1} did not generate valid audio")
                else:
                    logger.warning(f"Cloned chunk {idx + 1} returned empty generator")

            except Exception as e:
                logger.error(f"Failed to synthesize cloned chunk {idx + 1}: {e}")
                raise SynthesisError(f"Chunk {idx + 1} synthesis failed: {e}") from e

        if not all_speech_chunks:
            raise SynthesisError("Model did not generate any audio for cloned voice (custom-chunked)")

        # Use crossfade for smooth transitions between chunks
        speech = crossfade_chunks(all_speech_chunks)
        logger.debug(f"Final cloned audio (crossfaded): {speech.shape[-1] / 24000:.2f}s")

        return self._post_process(speech, speed, pitch)

    def _post_process(self, speech: torch.Tensor, speed: float, pitch: float) -> bytes:
        """Apply speed and pitch adjustments."""
        if speed != 1.0:
            speech = torchaudio.functional.resample(speech, int(24000 * speed), 24000)
        if pitch != 1.0:
            speech = apply_pitch_shift(speech, pitch)
        return tensor_to_wav_bytes(speech)

    def _prepare_prompt_text(self, transcript: str) -> str:
        """Prepare the prompt text for zero-shot inference with required prefix."""
        prompt = transcript.strip()

        # Truncate to avoid model reproducing the prompt
        if len(prompt) > MAX_PROMPT_CHARS:
            truncated = prompt[:MAX_PROMPT_CHARS]
            last_punct = max(
                truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'),
                truncated.rfind('。'), truncated.rfind('！'), truncated.rfind('？')
            )
            if last_punct > MAX_PROMPT_CHARS * 0.5:
                prompt = truncated[:last_punct + 1]
            else:
                prompt = truncated

        # Ensure ends with punctuation
        if not any(prompt.endswith(p) for p in [".", "!", "?", "。", "！", "？"]):
            prompt += "."

        # CRITICAL FIX: Add required prefix for CosyVoice 3 zero-shot
        prompt = ZERO_SHOT_PROMPT_PREFIX + prompt + "<|endofprompt|>"

        return prompt

    def _split_text_into_chunks(self, text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
        """Split text into chunks based on sentence boundaries (default behavior)."""
        # Split by punctuation followed by whitespace or end of string
        sentences = re.split(r'(?<=[.!?。！？])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text]

        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence exceeds max_chars and we have content, finalize current chunk
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

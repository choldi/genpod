"""Cloned voice synthesis for LightTTS."""

import logging
import re
from pathlib import Path
from typing import Generator, List
import torch
import torchaudio

from core.exceptions import SynthesisError, VoiceNotFoundError
from core.lighttts.audio_utils import (
    tensor_to_wav_bytes,
    apply_pitch_shift,
    validate_audio_length,
)

logger = logging.getLogger(__name__)

# Prefix required for CosyVoice 3 zero-shot prompt to prevent hallucination
ZERO_SHOT_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
# Maximum length for prompt text to avoid model reproducing it
MAX_PROMPT_CHARS = 100
# Minimum text length to trigger chunking (CosyVoice 2/3 handles long texts well)
MIN_CHUNK_THRESHOLD = 50000


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
    ) -> Generator[bytes, None, None]:
        """Generate audio for cloned voice."""
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

            # STRATEGY 1: Streaming zero-shot inference (BEST for all text lengths)
            if hasattr(self._model, 'inference_zero_shot'):
                try:
                    logger.debug(f"Attempting streaming zero-shot synthesis ({self._model_version}): '{target_text[:60]}...'")
                    audio_bytes = self._synthesize_streaming(
                        target_text, prompt_text, str(ref_audio_path), speed, pitch
                    )
                except Exception as e:
                    logger.warning(f"Streaming zero-shot failed: {e}, trying single-shot")
                    audio_bytes = self._synthesize_single_shot(
                        target_text, prompt_text, str(ref_audio_path), speed, pitch
                    )
            else:
                audio_bytes = self._synthesize_single_shot(
                    target_text, prompt_text, str(ref_audio_path), speed, pitch
                )

            if stream:
                chunk_size = 4096
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]
            else:
                yield audio_bytes

        except VoiceNotFoundError:
            raise
        except SynthesisError:
            raise
        except Exception as e:
            logger.error(f"Cloned voice synthesis failed: {e}")
            raise SynthesisError(f"Cloned voice synthesis failed: {e}") from e

    def _synthesize_streaming(
        self,
        target_text: str,
        prompt_text: str,
        ref_audio_path: str,
        speed: float,
        pitch: float,
    ) -> bytes:
        """Synthesize cloned voice using streaming zero-shot inference."""
        try:
            logger.debug(f"Streaming zero-shot synthesis ({self._model_version}): '{target_text[:60]}...'")

            output_generator = self._model.inference_zero_shot(
                target_text, prompt_text, ref_audio_path, stream=True
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
                raise SynthesisError("Model did not generate any audio for cloned voice (streaming)")

            speech = torch.cat(all_speech_chunks, dim=-1)
            logger.debug(f"Streaming cloned audio: {speech.shape[-1] / 24000:.2f}s")

            return self._post_process(speech, speed, pitch)

        except Exception as e:
            logger.warning(f"Streaming zero-shot failed: {e}")
            raise

    def _synthesize_single_shot(
        self,
        target_text: str,
        prompt_text: str,
        ref_audio_path: str,
        speed: float,
        pitch: float,
    ) -> bytes:
        """Synthesize cloned voice in a single shot for the entire text."""
        try:
            logger.debug(f"Single-shot zero-shot synthesis: '{target_text[:60]}...'")

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
                raise SynthesisError("Model did not generate any audio for cloned voice (single-shot)")

            speech = torch.cat(all_speech_chunks, dim=-1)
            logger.debug(f"Single-shot cloned audio: {speech.shape[-1] / 24000:.2f}s")

            return self._post_process(speech, speed, pitch)

        except Exception as e:
            logger.warning(f"Single-shot synthesis failed: {e}")
            raise

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

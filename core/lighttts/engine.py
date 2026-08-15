"""LightTTSEngine - Wrapper for CosyVoice 2 model."""

import os
import re
import logging
import tempfile
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
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

logger = logging.getLogger(__name__)

# Mapping from simple API aliases to actual CosyVoice speaker IDs
VOICE_ALIAS_MAP = {
    "zh_female": "中文女",
    "zh_male": "中文男",
    "en_female": "英文女",
    "en_male": "英文男",
    "ja_female": "日本語女",
    "ko_female": "한국어女",
}

# Emotion presets: tag -> (speed, pitch) - used for v1/v2
EMOTION_PRESETS = {
    "<happy>":      {"speed": 1.10, "pitch": 1.10},
    "<sad>":        {"speed": 0.90, "pitch": 0.92},
    "<serious>":    {"speed": 0.95, "pitch": 0.95},
    "<whisper>":    {"speed": 0.85, "pitch": 0.88},
    "<angry>":      {"speed": 1.15, "pitch": 1.15},
    "<narrative>":  {"speed": 0.95, "pitch": 1.00},
    "<slow>":       {"speed": 0.80, "pitch": 1.00},
    "<fast>":       {"speed": 1.20, "pitch": 1.00},
    "<neutral>":    {"speed": 1.00, "pitch": 1.00},
}

# CosyVoice 3 native emotion/style tags
COSYVOICE3_STYLE_TAGS = {
    "happy": "[laughter]",
    "sad": "[cry]",
    "serious": "[serious]",
    "whisper": "[whisper]",
    "angry": "[angry]",
    "narrative": "[narrative]",
    "slow": "[slow]",
    "fast": "[fast]",
    "neutral": "",
}

# Maximum length for prompt text to avoid model reproducing it
MAX_PROMPT_CHARS = 50
# Maximum length for each synthesis chunk (increased for better context)
MAX_CHUNK_CHARS = 300
# Minimum valid audio length in samples (0.1s at 24kHz)
MIN_AUDIO_SAMPLES = 2400


class LightTTSEngine:
    """Main wrapper class for CosyVoice 2 operations."""

    def __init__(self, models_path: str, voices_path: str, device: str = "cpu") -> None:
        """Initialize the CosyVoice 2 model."""
        self.models_path = Path(models_path)
        self.voices_path = Path(voices_path)
        self.device = self._resolve_device(device)
        self._model: Optional[CosyVoice | CosyVoice2] = None
        self._model_version: str = "unknown"  # "v1", "v2", "v3"
        self._voice_manager = VoiceManager(str(self.voices_path))
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Resolve the actual device to use based on availability."""
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return device

    def _load_model(self) -> None:
        """Load the CosyVoice model (supports v1, v2, and v3)."""
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3
            from cosyvoice.utils.file_utils import load_wav

            self._load_wav = load_wav

            model_dir = self._detect_model_dir()
            logger.info(f"Loading model from {model_dir} on {self.device}...")

            if (model_dir / "cosyvoice3.yaml").exists():
                logger.info("Detected CosyVoice 3 model")
                self._model = CosyVoice3(str(model_dir), fp16=(self.device == "cuda"))
                self._model_version = "v3"
            elif (model_dir / "cosyvoice2.yaml").exists():
                logger.info("Detected CosyVoice 2 model")
                self._model = CosyVoice2(str(model_dir), load_jit=False, fp16=(self.device == "cuda"))
                self._model_version = "v2"
            else:
                logger.info("Detected CosyVoice v1 (SFT) model")
                self._model = CosyVoice(str(model_dir), load_jit=False, fp16=(self.device == "cuda"))
                self._model_version = "v1"

            logger.info(f"Model loaded successfully (version: {self._model_version})")

        except ImportError as e:
            raise ModelLoadError(
                "CosyVoice library not installed or outdated. "
                "Please ensure the latest version is cloned and PYTHONPATH is set correctly."
            ) from e
        except Exception as e:
            raise ModelLoadError(f"Failed to load CosyVoice model: {e}") from e

    def _detect_model_dir(self) -> Path:
        """Detect which CosyVoice model version is available."""
        if (self.models_path / "CosyVoice3-0.5B" / "cosyvoice3.yaml").exists():
            return self.models_path / "CosyVoice3-0.5B"
        if (self.models_path / "CosyVoice2-0.5B" / "cosyvoice2.yaml").exists():
            return self.models_path / "CosyVoice2-0.5B"
        if (self.models_path / "CosyVoice-300M-SFT" / "cosyvoice.yaml").exists():
            return self.models_path / "CosyVoice-300M-SFT"

        model_dirs = list(self.models_path.glob("CosyVoice*"))
        if model_dirs:
            return model_dirs[0]

        raise ModelLoadError(
            f"No CosyVoice model found in {self.models_path}. "
            "Please download the model weights first."
        )

    def list_voices(self) -> List[Dict[str, Any]]:
        """Scan and return all available voices (base and cloned)."""
        voices: List[Dict[str, Any]] = []

        for voice in self._voice_manager.list_voices():
            voices.append({
                "voice_id": voice.get("voice_id", ""),
                "name": voice.get("name", "Unknown"),
                "language": voice.get("language", "en"),
                "is_cloned": True,
                "sample_rate": voice.get("sample_rate", 24000),
            })

        known_base_voices = [
            ("zh_female", "zh", "Base Chinese Female"),
            ("zh_male", "zh", "Base Chinese Male"),
            ("en_female", "en", "Base English Female"),
            ("en_male", "en", "Base English Male"),
            ("ja_female", "ja", "Base Japanese Female"),
            ("ko_female", "ko", "Base Korean Female"),
        ]

        for alias, lang, name in known_base_voices:
            voices.append({
                "voice_id": alias,
                "name": name,
                "language": lang,
                "is_cloned": False,
                "sample_rate": 24000,
            })

        return voices

    def _parse_emotion_tags(self, text: str) -> List[Dict[str, Any]]:
        """Parse emotion tags from text and return segments with their presets."""
        segments: List[Dict[str, Any]] = []
        current_emotion = "<neutral>"

        pattern = r'(<(?:' + '|'.join(tag.strip('<>') for tag in EMOTION_PRESETS.keys()) + r')>)'
        parts = re.split(pattern, text)

        for part in parts:
            if not part.strip():
                continue

            if part in EMOTION_PRESETS:
                current_emotion = part
            else:
                preset = EMOTION_PRESETS.get(current_emotion, EMOTION_PRESETS["<neutral>"])
                segments.append({
                    "text": part.strip(),
                    "speed": preset["speed"],
                    "pitch": preset["pitch"],
                    "emotion": current_emotion,
                })

        return segments if segments else [{"text": text, "speed": 1.0, "pitch": 1.0, "emotion": "<neutral>"}]

    def _apply_style_tags_for_v3(self, text: str, emotion: str) -> str:
        """Apply CosyVoice 3 native style tags to text."""
        if self._model_version != "v3":
            return text
        
        style_tag = COSYVOICE3_STYLE_TAGS.get(emotion.strip("<>"), "")
        if style_tag:
            return f"{style_tag}{text}"
        return text

    def _apply_pitch_shift(self, waveform: torch.Tensor, pitch_factor: float) -> torch.Tensor:
        """Apply pitch shifting to audio waveform."""
        if pitch_factor == 1.0:
            return waveform

        effects = [
            ["pitch", str(pitch_factor * 100)],
            ["rate", "24000"],
        ]

        try:
            shifted, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, 24000, effects
            )
            return shifted
        except Exception:
            logger.warning("Pitch shift failed, returning original waveform")
            return waveform

    def _is_cloned_voice(self, voice_id: str) -> bool:
        """Check if a voice ID corresponds to a cloned voice."""
        try:
            meta = self._voice_manager.load_voice_metadata(voice_id)
            return bool(meta and meta.get("is_cloned"))
        except FileNotFoundError:
            return False
        except Exception:
            return False

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
        """Generate audio chunks for the given text and voice."""
        if not self._model:
            raise SynthesisError("Model not loaded")

        try:
            is_cloned = self._is_cloned_voice(voice_id)

            if is_cloned:
                ref_audio_path = self.voices_path / f"{voice_id}.wav"
                if not ref_audio_path.exists():
                    raise VoiceNotFoundError(f"Reference audio for voice '{voice_id}' not found")

            if emotion_tags:
                segments = self._parse_emotion_tags(text)
            else:
                segments = [{"text": text, "speed": speed, "pitch": pitch, "emotion": emotion}]

            for segment in segments:
                seg_text = segment["text"]
                seg_speed = segment["speed"]
                seg_pitch = segment["pitch"]
                seg_emotion = segment.get("emotion", "neutral")

                if not seg_text:
                    continue

                # Apply CosyVoice 3 native tags if applicable
                if self._model_version == "v3":
                    seg_text = self._apply_style_tags_for_v3(seg_text, seg_emotion)
                    # For v3, we don't need post-processing speed/pitch if using native tags
                    seg_speed = 1.0
                    seg_pitch = 1.0

                if is_cloned:
                    audio_bytes = self._synthesize_cloned_segment(
                        seg_text, voice_id, seg_speed, seg_pitch, lang
                    )
                else:
                    actual_spk_id = VOICE_ALIAS_MAP.get(voice_id, voice_id)
                    audio_bytes = self._synthesize_base_segment(
                        seg_text, actual_spk_id, seg_speed, seg_pitch
                    )

                if stream:
                    chunk_size = 4096
                    for i in range(0, len(audio_bytes), chunk_size):
                        yield audio_bytes[i:i + chunk_size]
                else:
                    yield audio_bytes

        except VoiceNotFoundError:
            raise
        except Exception as e:
            raise SynthesisError(f"Synthesis failed: {e}") from e

    def _tensor_to_wav_bytes(self, speech: torch.Tensor) -> bytes:
        """Convert a speech tensor to WAV bytes."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            torchaudio.save(tmp_path, speech, 24000, format="wav")
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _split_text_into_chunks(self, text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
        """Split text into chunks based on sentence boundaries with overlap for continuity."""
        # Split by sentence-ending punctuation followed by whitespace or end of string
        sentences = re.split(r'(?<=[.!?。！？])\s*', text)
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed max_chars and we have content, finalize current chunk
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap: last sentence of previous chunk for context
                # But only if the previous chunk had more than one sentence
                last_sentence = current_chunk.strip().split()[-1] if current_chunk.strip() else ""
                current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _prepare_prompt_text(self, transcript: str) -> str:
        """Prepare the prompt text for zero-shot inference.

        Truncates the transcript to avoid model reproducing it,
        and appends the end-of-prompt token.
        """
        prompt = transcript.strip()

        # Truncate to avoid model reproducing the prompt
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS]

        # Ensure ends with punctuation
        if not any(prompt.endswith(p) for p in [".", "!", "?", "。", "！", "？"]):
            prompt += "."

        # Append end-of-prompt token
        prompt += "<|endofprompt|>"

        return prompt

    def _synthesize_base_segment(
        self, text: str, spk_id: str, speed: float, pitch: float
    ) -> bytes:
        """Synthesize a single segment using base voice and return bytes."""
        try:
            # For CosyVoice 3, use streaming inference for better prosody continuity
            if self._model_version == "v3" and hasattr(self._model, 'inference_sft_stream'):
                return self._synthesize_base_segment_streaming(text, spk_id, speed, pitch)
            
            # For v1/v2, use chunked synthesis with overlap for continuity
            return self._synthesize_base_segment_chunked(text, spk_id, speed, pitch)

        except Exception as e:
            raise SynthesisError(f"Base voice synthesis failed: {e}") from e

    def _synthesize_base_segment_streaming(
        self, text: str, spk_id: str, speed: float, pitch: float
    ) -> bytes:
        """Synthesize using CosyVoice 3 streaming inference for better continuity."""
        try:
            # Ensure text ends with punctuation
            target_text = text.strip()
            if not any(target_text.endswith(p) for p in [".", "!", "?", "。", "！", "？"]):
                target_text += "."

            logger.debug(f"Streaming synthesis (v3): '{target_text[:60]}...'")

            # Use streaming inference if available
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

            # Apply speed/pitch if needed (for v3, native tags handle most cases)
            if speed != 1.0:
                speech = torchaudio.functional.resample(speech, int(24000 * speed), 24000)
            if pitch != 1.0:
                speech = self._apply_pitch_shift(speech, pitch)

            return self._tensor_to_wav_bytes(speech)

        except Exception as e:
            logger.warning(f"Streaming synthesis failed, falling back to chunked: {e}")
            return self._synthesize_base_segment_chunked(text, spk_id, speed, pitch)

    def _synthesize_base_segment_chunked(
        self, text: str, spk_id: str, speed: float, pitch: float
    ) -> bytes:
        """Synthesize long text by chunks with overlap for prosody continuity (v1/v2)."""
        # Split long text into chunks
        text_chunks = self._split_text_into_chunks(text)
        logger.debug(f"Base voice: text split into {len(text_chunks)} chunks")

        all_speech_chunks: List[torch.Tensor] = []

        for idx, chunk in enumerate(text_chunks):
            logger.debug(f"Synthesizing base chunk {idx + 1}/{len(text_chunks)}: '{chunk[:60]}...'")

            # Ensure chunk ends with punctuation
            if not any(chunk.endswith(p) for p in [".", "!", "?", "。", "！", "？"]):
                chunk += "."

            output_generator = self._model.inference_sft(chunk, spk_id, stream=False)

            chunk_speech_list: List[torch.Tensor] = []
            for out_dict in output_generator:
                speech = out_dict['tts_speech']
                if speech.dim() == 3:
                    speech = speech.squeeze(0)
                chunk_speech_list.append(speech.cpu())

            if chunk_speech_list:
                chunk_speech = torch.cat(chunk_speech_list, dim=-1)
                if chunk_speech.shape[-1] >= MIN_AUDIO_SAMPLES:
                    all_speech_chunks.append(chunk_speech)
                    logger.debug(f"Base chunk {idx + 1} generated: {chunk_speech.shape[-1] / 24000:.2f}s")
                else:
                    logger.warning(f"Base chunk {idx + 1} did not generate valid audio")

        if not all_speech_chunks:
            raise SynthesisError("Model did not generate any audio for base voice")

        speech = torch.cat(all_speech_chunks, dim=-1)
        logger.debug(f"Final base audio: {speech.shape[-1] / 24000:.2f}s")

        if speed != 1.0:
            speech = torchaudio.functional.resample(speech, int(24000 * speed), 24000)

        if pitch != 1.0:
            speech = self._apply_pitch_shift(speech, pitch)

        return self._tensor_to_wav_bytes(speech)

    def _synthesize_cloned_segment(
        self,
        text: str,
        voice_id: str,
        speed: float,
        pitch: float,
        language: str = "es",
    ) -> bytes:
        """Synthesize a single segment using cloned voice and return bytes."""
        metadata = self._voice_manager.load_voice_metadata(voice_id)
        ref_audio_path = self.voices_path / f"{voice_id}.wav"

        if not ref_audio_path.exists():
            raise VoiceNotFoundError(f"Reference audio for voice '{voice_id}' not found")

        try:
            # Prepare prompt text ONCE from metadata (transcript of reference audio)
            prompt_text = self._prepare_prompt_text(metadata.get("transcript", ""))

            # Ensure target text ends with punctuation
            target_text = text.strip()
            if not any(target_text.endswith(p) for p in [".", "!", "?", "。", "！", "？"]):
                target_text += "."

            # Try streaming zero-shot inference for ALL model versions if available
            if hasattr(self._model, 'inference_zero_shot') and callable(getattr(self._model, 'inference_zero_shot', None)):
                # Check if streaming is supported by trying stream=True
                return self._synthesize_cloned_segment_streaming(
                    target_text, prompt_text, str(ref_audio_path), speed, pitch
                )

            # Fallback to chunked synthesis only if streaming not available
            return self._synthesize_cloned_segment_chunked(
                target_text, prompt_text, str(ref_audio_path), speed, pitch
            )

        except VoiceNotFoundError:
            raise
        except SynthesisError:
            raise
        except Exception as e:
            logger.error(f"Cloned voice synthesis failed: {e}")
            raise SynthesisError(f"Cloned voice synthesis failed: {e}") from e

    def _synthesize_cloned_segment_streaming(
        self,
        target_text: str,
        prompt_text: str,
        ref_audio_path: str,
        speed: float,
        pitch: float,
    ) -> bytes:
        """Synthesize cloned voice using streaming zero-shot inference (all versions)."""
        try:
            logger.debug(f"Streaming zero-shot synthesis ({self._model_version}): '{target_text[:60]}...'")

            # Try streaming first
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

            if speed != 1.0:
                speech = torchaudio.functional.resample(speech, int(24000 * speed), 24000)
            if pitch != 1.0:
                speech = self._apply_pitch_shift(speech, pitch)

            return self._tensor_to_wav_bytes(speech)

        except Exception as e:
            logger.warning(f"Streaming zero-shot failed, falling back to single-shot: {e}")
            # Fallback to single-shot for the entire text (not chunked)
            return self._synthesize_cloned_segment_single_shot(
                target_text, prompt_text, ref_audio_path, speed, pitch
            )

    def _synthesize_cloned_segment_single_shot(
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

            if speed != 1.0:
                speech = torchaudio.functional.resample(speech, int(24000 * speed), 24000)
            if pitch != 1.0:
                speech = self._apply_pitch_shift(speech, pitch)

            return self._tensor_to_wav_bytes(speech)

        except Exception as e:
            logger.warning(f"Single-shot synthesis failed, falling back to chunked: {e}")
            return self._synthesize_cloned_segment_chunked(
                target_text, prompt_text, ref_audio_path, speed, pitch
            )

    def _synthesize_cloned_segment_chunked(
        self,
        target_text: str,
        prompt_text: str,
        ref_audio_path: str,
        speed: float,
        pitch: float,
    ) -> bytes:
        """Synthesize cloned voice by chunks (last resort for v1/v2)."""
        # Split into manageable chunks with larger size for better context
        text_chunks = self._split_text_into_chunks(target_text, max_chars=MAX_CHUNK_CHARS)
        logger.debug(f"Cloned voice: text split into {len(text_chunks)} chunks (fallback)")

        all_speech_chunks: List[torch.Tensor] = []

        for idx, chunk in enumerate(text_chunks):
            logger.debug(f"Synthesizing cloned chunk {idx + 1}/{len(text_chunks)}: '{chunk[:60]}...'")

            # Ensure chunk ends with punctuation
            if not any(chunk.endswith(p) for p in [".", "!", "?", "。", "！", "？"]):
                chunk += "."

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
                if chunk_speech.shape[-1] >= MIN_AUDIO_SAMPLES:
                    all_speech_chunks.append(chunk_speech)
                    logger.debug(f"Cloned chunk {idx + 1} generated: {chunk_speech.shape[-1] / 24000:.2f}s")
                else:
                    logger.warning(f"Cloned chunk {idx + 1} did not generate valid audio")

        if not all_speech_chunks:
            raise SynthesisError("Model did not generate any audio for cloned voice")

        speech = torch.cat(all_speech_chunks, dim=-1)
        logger.debug(f"Final cloned audio: {speech.shape[-1] / 24000:.2f}s")

        if speed != 1.0:
            speech = torchaudio.functional.resample(speech, int(24000 * speed), 24000)

        if pitch != 1.0:
            speech = self._apply_pitch_shift(speech, pitch)

        return self._tensor_to_wav_bytes(speech)

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

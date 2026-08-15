"""LightTTSEngine - Wrapper for CosyVoice 2 model."""

import os
import re
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
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

# Mapping from simple API aliases to actual CosyVoice speaker IDs
VOICE_ALIAS_MAP = {
    "zh_female": "中文女",
    "zh_male": "中文男",
    "en_female": "英文女",
    "en_male": "英文男",
    "ja_female": "日本語女",
    "ko_female": "한국어女",
}

# Emotion presets: tag -> (speed, pitch)
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


class LightTTSEngine:
    """Main wrapper class for CosyVoice 2 operations."""

    def __init__(self, models_path: str, voices_path: str, device: str = "cpu"):
        """Initialize the CosyVoice 2 model."""
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
        """Load the CosyVoice model (supports v1, v2, and v3)."""
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3
            from cosyvoice.utils.file_utils import load_wav

            self._load_wav = load_wav

            # Detectar automáticamente la versión del modelo por el archivo yaml presente
            model_dir = None
            if (self.models_path / "CosyVoice3-0.5B" / "cosyvoice3.yaml").exists():
                model_dir = self.models_path / "CosyVoice3-0.5B"
            elif (self.models_path / "CosyVoice2-0.5B" / "cosyvoice2.yaml").exists():
                model_dir = self.models_path / "CosyVoice2-0.5B"
            elif (self.models_path / "CosyVoice-300M-SFT" / "cosyvoice.yaml").exists():
                model_dir = self.models_path / "CosyVoice-300M-SFT"
            else:
                # Fallback: buscar cualquier carpeta que empiece por CosyVoice
                model_dirs = list(self.models_path.glob("CosyVoice*"))
                if model_dirs:
                    model_dir = model_dirs[0]
                else:
                    raise ModelLoadError(
                        f"No CosyVoice model found in {self.models_path}. "
                        "Please download the model weights first."
                    )

            print(f"Loading model from {model_dir} on {self.device}...")
            
            # Inicializar la clase correcta según la versión detectada
            if (model_dir / "cosyvoice3.yaml").exists():
                print("✅ Detected CosyVoice 3 model (Optimized for multilingual in-the-wild synthesis)")
                # CosyVoice3 no usa el parámetro load_jit
                self._model = CosyVoice3(str(model_dir), fp16=(self.device == "cuda"))
            elif (model_dir / "cosyvoice2.yaml").exists():
                print("✅ Detected CosyVoice 2 model")
                self._model = CosyVoice2(str(model_dir), load_jit=False, fp16=(self.device == "cuda"))
            else:
                print("✅ Detected CosyVoice v1 (SFT) model")
                self._model = CosyVoice(str(model_dir), load_jit=False, fp16=(self.device == "cuda"))
            
            print("Model loaded successfully!")

        except ImportError as e:
            raise ModelLoadError(
                "CosyVoice library not installed or outdated. "
                "Please ensure the latest version is cloned and PYTHONPATH is set correctly."
            ) from e
        except Exception as e:
            raise ModelLoadError(f"Failed to load CosyVoice model: {e}") from e

    def list_voices(self) -> List[Dict[str, Any]]:
        """Scan and return all available voices (base and cloned)."""
        voices = []

        cloned_voices = self._voice_manager.list_voices()
        for voice in cloned_voices:
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
        segments = []
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
                })
        
        return segments if segments else [{"text": text, "speed": 1.0, "pitch": 1.0}]

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
            return waveform

    def synthesize(
        self, text: str, voice_id: str, lang: str = "en", stream: bool = True,
        speed: float = 1.0, pitch: float = 1.0, emotion: str = "neutral",
        emotion_tags: bool = False
    ) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text and voice."""
        if not self._model:
            raise SynthesisError("Model not loaded")
        
        try:
            is_cloned = False
            try:
                meta = self._voice_manager.load_voice_metadata(voice_id)
                if meta and meta.get("is_cloned"):
                    is_cloned = True
                    ref_audio_path = self.voices_path / f"{voice_id}.wav"
                    if not ref_audio_path.exists():
                        raise VoiceNotFoundError(f"Reference audio for voice '{voice_id}' not found")
            except FileNotFoundError:
                pass
            except VoiceNotFoundError:
                raise
            except Exception:
                pass

            if emotion_tags:
                segments = self._parse_emotion_tags(text)
            else:
                segments = [{"text": text, "speed": speed, "pitch": pitch}]
            
            for segment in segments:
                seg_text = segment["text"]
                seg_speed = segment["speed"]
                seg_pitch = segment["pitch"]
                
                if not seg_text:
                    continue
                
                if is_cloned:
                    audio_bytes = self._synthesize_cloned_segment(seg_text, voice_id, seg_speed, seg_pitch, lang)
                else:
                    actual_spk_id = VOICE_ALIAS_MAP.get(voice_id, voice_id)
                    audio_bytes = self._synthesize_base_segment(seg_text, actual_spk_id, seg_speed, seg_pitch)
                
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

    def _synthesize_base_segment(
        self, text: str, spk_id: str, speed: float, pitch: float
    ) -> bytes:
        """Synthesize a single segment using base voice and return bytes."""
        try:
            output_generator = self._model.inference_sft(text, spk_id, stream=False)
            
            for out_dict in output_generator:
                speech = out_dict['tts_speech']
                if speech.dim() == 3:
                    speech = speech.squeeze(0)
                
                if speed != 1.0:
                    speech = torchaudio.functional.resample(speech, int(24000 * speed), 24000)
                
                if pitch != 1.0:
                    speech = self._apply_pitch_shift(speech, pitch)
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    torchaudio.save(tmp_path, speech.cpu(), 24000, format="wav")
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                return audio_bytes
            
            return b""
        except Exception as e:
            raise SynthesisError(f"Base voice synthesis failed: {e}") from e

    def _synthesize_cloned_segment(
        self, text: str, voice_id: str, speed: float, pitch: float, language: str = "es"
    ) -> bytes:
        """Synthesize a single segment using cloned voice and return bytes."""
        metadata = self._voice_manager.load_voice_metadata(voice_id)
        ref_audio_path = self.voices_path / f"{voice_id}.wav"
        
        if not ref_audio_path.exists():
            raise VoiceNotFoundError(f"Reference audio for voice '{voice_id}' not found")

        try:
            prompt_text = metadata.get("transcript", "")
            
            # 🔍 LOGS DE DEPURACIÓN
            print(f"\n{'='*60}")
            print(f"🔍 SYNTHESIS DEBUG")
            print(f"{'='*60}")
            print(f"Voice ID: {voice_id}")
            print(f"Target text: '{text}'")
            print(f"Prompt text: '{prompt_text}'")
            print(f"Reference audio: {ref_audio_path}")
            
            # Verificar formato del audio de referencia
            try:
                info = torchaudio.info(str(ref_audio_path))
                print(f"Audio info: {info.sample_rate}Hz, {info.num_channels} channels, {info.num_frames/info.sample_rate:.2f}s")
            except Exception as e:
                print(f"⚠️  Error reading audio info: {e}")
            
            print(f"{'='*60}\n")
            
            output_generator = self._model.inference_zero_shot(
                text, prompt_text, str(ref_audio_path), stream=False
            )
            
            for out_dict in output_generator:
                speech = out_dict['tts_speech']
                
                print(f"🔍 Generated speech shape: {speech.shape}")
                print(f"🔍 Generated speech dtype: {speech.dtype}")
                print(f"🔍 Generated speech device: {speech.device}")
                print(f"🔍 Generated speech min/max: {speech.min().item():.4f} / {speech.max().item():.4f}")
                
                # Asegurar formato correcto [canales, muestras]
                if speech.dim() == 3:
                    speech = speech.squeeze(0)
                if speech.dim() == 1:
                    speech = speech.unsqueeze(0)
                
                print(f"🔍 After reshape: {speech.shape}")
                
                # Si el audio es demasiado corto, el vocoder fallará
                if speech.shape[-1] < 1000:
                    error_msg = (
                        f"El modelo generó un audio inválido (longitud: {speech.shape[-1]} muestras). "
                        f"Esto indica que CosyVoice 3 no pudo procesar la combinación de texto y audio de referencia.\n"
                        f"Posibles causas:\n"
                        f"1. La transcripción no coincide exactamente con el audio\n"
                        f"2. El audio de referencia tiene problemas de formato\n"
                        f"3. El texto contiene caracteres que el tokenizador no puede procesar\n"
                        f"4. El audio es demasiado corto o muy largo"
                    )
                    print(f"❌ ERROR: {error_msg}")
                    raise SynthesisError(error_msg)

                if speed != 1.0:
                    speech = torchaudio.functional.resample(speech, int(24000 * speed), 24000)
                
                if pitch != 1.0:
                    speech = self._apply_pitch_shift(speech, pitch)
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    torchaudio.save(tmp_path, speech.cpu(), 24000, format="wav")
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                print(f"✅ Synthesis successful: {len(audio_bytes)} bytes")
                return audio_bytes
            
            print(f"❌ No output generated from model")
            return b""
        except VoiceNotFoundError:
            raise
        except SynthesisError:
            raise
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            raise SynthesisError(f"Cloned voice synthesis failed: {e}") from e

    def clone_voice(
        self, audio_path: str, transcript: str, voice_name: str, language: str = "en"
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
                print(f"Warning: Reference audio is long ({duration:.1f}s). "
                      "Consider using a shorter clip for better results.")
        except AudioTooShortError:
            raise
        except Exception as e:
            raise CloningError(f"Failed to validate audio: {e}") from e
        
        voice_id = voice_name
        voice_id_clean = re.sub(r'[^\w\-]', '_', voice_id)
        voice_id = voice_id_clean[:32]
        
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


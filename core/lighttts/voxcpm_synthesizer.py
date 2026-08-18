"""VoxCPM Synthesizer for TTS synthesis with streaming and emotion control."""

import logging
import torch
import torchaudio
from pathlib import Path
from typing import Generator, Optional, Any, Dict, List, Union

from core.lighttts.base_synthesizer import BaseSynthesizer
from core.lighttts.voice_manager import VoiceManager
from core.lighttts.audio_utils import (
    tensor_to_wav_bytes,
    apply_pitch_shift,
    validate_audio_length,
)
from core.exceptions import VoiceNotFoundError, SynthesisError
from core.logger import get_logger

logger = get_logger(__name__)


class VoxCPMSynthesizer(BaseSynthesizer):
    """Synthesizer for VoxCPM models with streaming and emotion control support."""
    
    SUPPORTED_EMOTIONS = [
        "neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised",
        "calm", "excited", "whispering", "shouting", "laughing", "crying"
    ]
    
    SUPPORTED_EMOTION_TAGS = [
        "laugh", "breath", "cough", "sneeze", "sigh", "yawn", "gasp", "moan",
        "hmm", "uh", "ah", "oh", "wow", "hey", "huh", "eh", "hm", "mm"
    ]

    def __init__(
        self,
        model: Any,
        model_version: str,
        voices_path: str,
        voice_manager: VoiceManager,
        model_type: str = "voxcpm",
        device: str = "cuda",
        **kwargs
    ) -> None:
        """Initialize the VoxCPM synthesizer.
        
        Args:
            model: Loaded VoxCPM model instance
            model_version: Model version string ("0.5b" or "1b")
            voices_path: Path to voices directory
            voice_manager: VoiceManager instance
            model_type: Type of model ("voxcpm", "voxcpm-0.5b", "voxcpm-1b")
            device: Device to run inference on
            **kwargs: Additional arguments
        """
        super().__init__(model, model_version, voices_path, voice_manager, model_type, **kwargs)
        self.device = device
        self.model_version = model_version
        self._init_voxcpm_components()
        logger.info(f"VoxCPMSynthesizer initialized: version={model_version}, device={device}")

    def _init_voxcpm_components(self):
        """Initialize VoxCPM-specific components: CampPlus, SpeechTokenizer, etc."""
        try:
            # These would be initialized from the model or loaded separately
            # For now, we assume the model has these components accessible
            self.campplus = getattr(self.model, 'campplus', None)
            self.speech_tokenizer = getattr(self.model, 'speech_tokenizer', None)
            self.flow = getattr(self.model, 'flow', None)
            self.llm = getattr(self.model, 'llm', None)
            
            # Default sampling rate for VoxCPM
            self.sample_rate = 24000
            
            logger.debug("VoxCPM components initialized")
        except Exception as e:
            logger.warning(f"Could not initialize all VoxCPM components: {e}")

    def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "es",
        stream: bool = False,
        speed: float = 1.0,
        pitch: float = 1.0,
        emotion: str = "neutral",
        emotion_tags: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> Union[Generator[bytes, None, None], bytes]:
        """Main synthesis method with streaming support.
        
        Args:
            text: Text to synthesize
            voice_id: Voice identifier (format: "voxcpm:speaker_id" or just "speaker_id")
            language: Language code
            stream: Whether to stream chunks
            speed: Speech speed factor
            pitch: Pitch factor
            emotion: Emotion label
            emotion_tags: List of emotion tags for granular control
            chunk_size: Chunking strategy for long text
            **kwargs: Additional VoxCPM-specific parameters
            
        Returns:
            Generator of audio chunks (streaming) or complete audio bytes
        """
        logger.debug(f"VoxCPM synthesis: voice_id={voice_id}, text_len={len(text)}, "
                    f"stream={stream}, speed={speed}, pitch={pitch}, emotion={emotion}, "
                    f"emotion_tags={emotion_tags}, chunk_size={chunk_size}")

        # Parse voice_id to extract speaker_id
        speaker_id = self._parse_voice_id(voice_id)
        
        # Validate emotion
        if emotion not in self.SUPPORTED_EMOTIONS:
            logger.warning(f"Unsupported emotion: {emotion}, using neutral")
            emotion = "neutral"
        
        # Validate emotion tags
        if emotion_tags:
            invalid_tags = [tag for tag in emotion_tags if tag not in self.SUPPORTED_EMOTION_TAGS]
            if invalid_tags:
                logger.warning(f"Unsupported emotion tags: {invalid_tags}, ignoring")
                emotion_tags = [tag for tag in emotion_tags if tag in self.SUPPORTED_EMOTION_TAGS]

        # Prepare text chunks
        text_chunks = self._prepare_text_chunks(text, chunk_size)
        logger.debug(f"Text split into {len(text_chunks)} chunks")

        if stream:
            return self._synthesize_streaming(
                text_chunks, speaker_id, language, speed, pitch, emotion, emotion_tags, **kwargs
            )
        else:
            return self._synthesize_batch(
                text_chunks, speaker_id, language, speed, pitch, emotion, emotion_tags, **kwargs
            )

    def _synthesize_streaming(
        self,
        text_chunks: List[str],
        speaker_id: str,
        language: str,
        speed: float,
        pitch: float,
        emotion: str,
        emotion_tags: Optional[List[str]],
        **kwargs
    ) -> Generator[bytes, None, None]:
        """Streaming synthesis - yields audio chunks as they're generated."""
        try:
            for i, chunk_text in enumerate(text_chunks):
                logger.debug(f"Streaming chunk {i+1}/{len(text_chunks)}: {chunk_text[:50]}...")
                
                # Generate speech for this chunk
                speech = self._generate_speech(
                    chunk_text, speaker_id, language, speed, emotion, emotion_tags, **kwargs
                )
                
                # Apply pitch shift if needed
                if pitch != 1.0:
                    speech = apply_pitch_shift(speech, pitch)
                
                # Validate audio length
                if not validate_audio_length(speech):
                    logger.warning("Generated audio too short, skipping")
                    continue
                
                # Convert to WAV bytes and yield
                wav_bytes = tensor_to_wav_bytes(speech, sample_rate=self.sample_rate)
                yield wav_bytes
                
        except Exception as e:
            logger.error(f"VoxCPM streaming synthesis failed: {e}", exc_info=True)
            raise SynthesisError(f"Streaming synthesis error: {e}") from e

    def _synthesize_batch(
        self,
        text_chunks: List[str],
        speaker_id: str,
        language: str,
        speed: float,
        pitch: float,
        emotion: str,
        emotion_tags: Optional[List[str]],
        **kwargs
    ) -> bytes:
        """Batch synthesis - generates complete audio and returns as bytes."""
        try:
            all_speech = []
            
            for i, chunk_text in enumerate(text_chunks):
                logger.debug(f"Batch chunk {i+1}/{len(text_chunks)}: {chunk_text[:50]}...")
                
                speech = self._generate_speech(
                    chunk_text, speaker_id, language, speed, emotion, emotion_tags, **kwargs
                )
                
                if pitch != 1.0:
                    speech = apply_pitch_shift(speech, pitch)
                
                if validate_audio_length(speech):
                    all_speech.append(speech)
            
            if not all_speech:
                raise SynthesisError("No valid audio generated")
            
            # Concatenate all chunks
            if len(all_speech) > 1:
                # Simple concatenation - could be improved with crossfade
                combined_speech = torch.cat(all_speech, dim=-1)
            else:
                combined_speech = all_speech[0]
            
            return tensor_to_wav_bytes(combined_speech, sample_rate=self.sample_rate)
            
        except Exception as e:
            logger.error(f"VoxCPM batch synthesis failed: {e}", exc_info=True)
            raise SynthesisError(f"Batch synthesis error: {e}") from e

    def _generate_speech(
        self,
        text: str,
        speaker_id: str,
        language: str,
        speed: float,
        emotion: str,
        emotion_tags: Optional[List[str]],
        **kwargs
    ) -> torch.Tensor:
        """Generate speech using VoxCPM model."""
        try:
            # Get speaker embedding
            speaker_embedding = self._get_speaker_embedding(speaker_id)
            
            # Prepare generation parameters
            generation_kwargs = {
                "text": text,
                "speaker_embedding": speaker_embedding,
                "speed": speed,
                "language": language,
                "emotion": emotion,
            }
            
            # Add emotion tags if provided
            if emotion_tags:
                generation_kwargs["emotion_tags"] = emotion_tags
            
            # Add VoxCPM-specific parameters from kwargs
            voxcpm_params = {
                "top_k": kwargs.get("voxcpm_top_k", 50),
                "temperature": kwargs.get("voxcpm_temperature", 0.7),
                "repetition_penalty": kwargs.get("voxcpm_repetition_penalty", 1.1),
                "max_new_tokens": kwargs.get("voxcpm_max_new_tokens", 2048),
            }
            generation_kwargs.update(voxcpm_params)
            
            # Call model inference
            if hasattr(self.model, 'generate'):
                speech = self.model.generate(**generation_kwargs)
            elif hasattr(self.model, 'inference'):
                speech = self.model.inference(**generation_kwargs)
            elif hasattr(self.model, 'forward'):
                speech = self.model(**generation_kwargs)
            else:
                # Try direct call
                speech = self.model(**generation_kwargs)
            
            # Ensure output is torch.Tensor with correct shape
            if not isinstance(speech, torch.Tensor):
                speech = torch.tensor(speech)
            
            if speech.dim() == 1:
                speech = speech.unsqueeze(0)
            elif speech.dim() > 2:
                speech = speech.squeeze(0)
                if speech.dim() > 1:
                    speech = speech[0:1]  # Take first channel
            
            logger.debug(f"VoxCPM generation successful: shape={speech.shape}")
            return speech
            
        except Exception as e:
            logger.error(f"VoxCPM speech generation failed: {e}", exc_info=True)
            raise SynthesisError(f"Speech generation error: {e}") from e

    def _get_speaker_embedding(self, speaker_id: str) -> torch.Tensor:
        """Get speaker embedding for the given speaker_id."""
        # Try to load from voice manager (for cloned voices)
        try:
            metadata = self._voice_manager.load_voice_metadata(speaker_id)
            if "speaker_embedding" in metadata:
                embedding = metadata["speaker_embedding"]
                if isinstance(embedding, list):
                    return torch.tensor(embedding, device=self.device)
                elif isinstance(embedding, torch.Tensor):
                    return embedding.to(self.device)
        except Exception:
            pass
        
        # For built-in speakers, try to get from model
        if hasattr(self.model, 'get_speaker_embedding'):
            return self.model.get_speaker_embedding(speaker_id)
        
        # Fallback: return zero embedding (will use random/default speaker)
        logger.warning(f"No speaker embedding found for {speaker_id}, using default")
        return torch.zeros(192, device=self.device)  # CampPlus embedding dim

    def _parse_voice_id(self, voice_id: str) -> str:
        """Parse voice_id to extract speaker_id.
        
        Supports formats:
        - "voxcpm:speaker_id" -> "speaker_id"
        - "speaker_id" -> "speaker_id"
        """
        if voice_id.startswith("voxcpm:"):
            return voice_id.split(":", 1)[1]
        return voice_id

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
                if i + chunk_size < len(text):
                    last_punct = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
                    if last_punct > chunk_size * 0.5:
                        chunk = chunk[:last_punct + 1]
                chunks.append(chunk.strip())
            return [c for c in chunks if c]

    def clone_voice(
        self,
        audio_path: str,
        voice_name: str,
        transcript: str,
        language: str = "es",
        **kwargs
    ) -> Dict[str, Any]:
        """Clone a voice using VoxCPM zero-shot capability.
        
        Args:
            audio_path: Path to reference audio file
            voice_name: Name for the new voice
            transcript: Transcript of the reference audio
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with voice_id and metadata
        """
        logger.info(f"Cloning voice '{voice_name}' with VoxCPM from {audio_path}")
        
        try:
            # Load reference audio
            prompt_speech, sr = torchaudio.load(audio_path)
            
            # Resample to 24kHz if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                prompt_speech = resampler(prompt_speech)
            
            # Ensure mono
            if prompt_speech.shape[0] > 1:
                prompt_speech = prompt_speech.mean(dim=0, keepdim=True)
            
            # Extract speaker embedding using CampPlus
            if self.campplus is not None:
                speaker_embedding = self.campplus(prompt_speech)
            elif hasattr(self.model, 'extract_speaker_embedding'):
                speaker_embedding = self.model.extract_speaker_embedding(prompt_speech)
            else:
                # Fallback: use model's encoder if available
                speaker_embedding = self._extract_speaker_embedding_fallback(prompt_speech)
            
            # Tokenize audio using SpeechTokenizer
            if self.speech_tokenizer is not None:
                speech_tokens = self.speech_tokenizer.encode(prompt_speech)
            elif hasattr(self.model, 'tokenize_audio'):
                speech_tokens = self.model.tokenize_audio(prompt_speech)
            else:
                speech_tokens = None
            
            # Generate voice_id
            import uuid
            voice_id = f"voxcpm:{voice_name}_{uuid.uuid4().hex[:8]}"
            speaker_id = voice_id.split(":", 1)[1]
            
            # Save reference audio
            ref_audio_path = self.voices_path / f"{speaker_id}.wav"
            torchaudio.save(str(ref_audio_path), prompt_speech.cpu(), self.sample_rate)
            
            # Prepare metadata
            metadata = {
                "voice_id": voice_id,
                "name": voice_name,
                "language": language,
                "transcript": transcript,
                "model_type": "voxcpm",
                "model_version": self.model_version,
                "speaker_embedding": speaker_embedding.cpu().tolist() if isinstance(speaker_embedding, torch.Tensor) else speaker_embedding,
                "speech_tokens": speech_tokens.cpu().tolist() if speech_tokens is not None and isinstance(speech_tokens, torch.Tensor) else speech_tokens,
                "sample_rate": self.sample_rate,
                "is_cloned": True,
            }
            
            # Save metadata via voice manager
            self._voice_manager.save_voice_metadata(speaker_id, metadata)
            
            # Register in voice registry
            self._voice_manager.register_voice(voice_id, metadata)
            
            logger.info(f"Voice cloned successfully: {voice_id}")
            return {
                "voice_id": voice_id,
                "voice_name": voice_name,
                "message": "Voice cloned successfully with VoxCPM"
            }
            
        except Exception as e:
            logger.error(f"VoxCPM voice cloning failed: {e}", exc_info=True)
            raise SynthesisError(f"Voice cloning error: {e}") from e

    def _extract_speaker_embedding_fallback(self, audio: torch.Tensor) -> torch.Tensor:
        """Fallback method to extract speaker embedding."""
        # This is a placeholder - in practice, you'd use a proper speaker encoder
        # For now, return a zero embedding
        return torch.zeros(192, device=self.device)

    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return self.SUPPORTED_EMOTIONS.copy()

    def get_supported_emotion_tags(self) -> List[str]:
        """Get list of supported emotion tags."""
        return self.SUPPORTED_EMOTION_TAGS.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "supported_emotions": self.SUPPORTED_EMOTIONS,
            "supported_emotion_tags": self.SUPPORTED_EMOTION_TAGS,
        }

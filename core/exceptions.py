"""Custom exceptions for the TTS service with enhanced error categorization."""

import traceback
from typing import Optional, Dict, Any
from datetime import datetime


class TTSException(Exception):
    """Base exception for all TTS service errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "TTS_ERROR",
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.utcnow().isoformat()
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp
        }


class VoiceNotFoundError(TTSException):
    """Raised when a requested voice is not found."""
    
    def __init__(self, voice_id: str, available_voices: Optional[list] = None):
        super().__init__(
            message=f"Voice '{voice_id}' not found",
            error_code="VOICE_NOT_FOUND",
            details={"voice_id": voice_id, "available_voices": available_voices or []},
            recoverable=False
        )


class AudioTooShortError(TTSException):
    """Raised when reference audio for cloning is too short."""
    
    def __init__(self, duration: float, minimum: float = 3.0):
        super().__init__(
            message=f"Reference audio too short: {duration:.1f}s. Minimum {minimum}s required.",
            error_code="AUDIO_TOO_SHORT",
            details={"duration": duration, "minimum_required": minimum},
            recoverable=False
        )


class ModelLoadError(TTSException):
    """Raised when the TTS model fails to load."""
    
    def __init__(self, model_type: str, reason: str, recoverable: bool = True):
        super().__init__(
            message=f"Failed to load model '{model_type}': {reason}",
            error_code="MODEL_LOAD_ERROR",
            details={"model_type": model_type, "reason": reason},
            recoverable=recoverable,
            retry_after=30
        )


class SynthesisError(TTSException):
    """Raised when audio synthesis fails."""
    
    def __init__(self, reason: str, voice_id: str, recoverable: bool = True):
        super().__init__(
            message=f"Synthesis failed for voice '{voice_id}': {reason}",
            error_code="SYNTHESIS_ERROR",
            details={"voice_id": voice_id, "reason": reason},
            recoverable=recoverable,
            retry_after=5
        )


class CloningError(TTSException):
    """Raised when voice cloning process fails."""
    
    def __init__(self, reason: str, voice_name: str, recoverable: bool = False):
        super().__init__(
            message=f"Voice cloning failed for '{voice_name}': {reason}",
            error_code="CLONING_ERROR",
            details={"voice_name": voice_name, "reason": reason},
            recoverable=recoverable
        )


class InvalidTranscriptError(TTSException):
    """Raised when the transcript doesn't match the audio."""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid transcript: {reason}",
            error_code="INVALID_TRANSCRIPT",
            details={"reason": reason},
            recoverable=False
        )


class ModelNotAvailableError(TTSException):
    """Raised when requested model is not available."""
    
    def __init__(self, model_type: str, available_models: list):
        super().__init__(
            message=f"Model '{model_type}' not available. Available: {available_models}",
            error_code="MODEL_NOT_AVAILABLE",
            details={"model_type": model_type, "available_models": available_models},
            recoverable=False
        )


class GPUOutOfMemoryError(TTSException):
    """Raised when GPU runs out of memory during inference."""
    
    def __init__(self, model_type: str, requested_memory: Optional[str] = None):
        super().__init__(
            message=f"GPU out of memory while loading/running model '{model_type}'",
            error_code="GPU_OOM",
            details={"model_type": model_type, "requested_memory": requested_memory},
            recoverable=True,
            retry_after=60
        )


class ConfigurationError(TTSException):
    """Raised when there's a configuration issue."""
    
    def __init__(self, parameter: str, reason: str):
        super().__init__(
            message=f"Configuration error for '{parameter}': {reason}",
            error_code="CONFIGURATION_ERROR",
            details={"parameter": parameter, "reason": reason},
            recoverable=False
        )


class ValidationError(TTSException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            message=f"Validation failed for '{field}': {reason}",
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": str(value), "reason": reason},
            recoverable=False
        )

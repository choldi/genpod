"""Pydantic schemas for API request/response models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class VoiceInfo(BaseModel):
    """Information about a single voice."""
    voice_id: str = Field(..., description="Unique identifier for the voice")
    name: str = Field(..., description="Human-readable name of the voice")
    language: str = Field(default="en", description="Language code")
    description: Optional[str] = Field(default=None, description="Voice description")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    is_cloned: bool = False


class VoiceListResponse(BaseModel):
    """Response model for listing voices."""
    voices: List[VoiceInfo] = Field(default_factory=list, description="List of available voices")
    total: int = Field(..., description="Total number of voices")


class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: str = Field(..., description="Voice identifier to use")
    language: str = Field(default="en", description="Language code")
    stream: bool = Field(default=True, description="Whether to stream the audio")
    mode: str = Field(default="studio", description="Synthesis mode: 'fast' or 'studio'")
    speed: Optional[float] = Field(default=None, ge=0.5, le=2.0, description="Speed multiplier (0.5-2.0)")
    pitch: Optional[float] = Field(default=None, ge=0.5, le=2.0, description="Pitch multiplier (0.5-2.0)")
    emotion: str = Field(default="neutral", description="Emotion (reserved for future use)")
    emotion_tags: bool = Field(default=False, description="Enable emotion tag parsing in text")
    chunk_size: Optional[int] = Field(default=None, ge=0, description="Chunk size for streaming synthesis: >0 splits by characters, 0 no splitting, None uses default (5000 characters)")


class TTSResponse(BaseModel):
    """Response model for non-streaming TTS."""
    audio_url: Optional[str] = Field(default=None, description="URL to the generated audio")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    sample_rate: Optional[int] = Field(default=None, description="Audio sample rate")


class CloneRequest(BaseModel):
    """Request model for voice cloning."""
    voice_name: str = Field(..., min_length=1, max_length=100, description="Name for the new voice")
    transcript: str = Field(..., min_length=1, description="Transcript of the audio")
    language: str = Field(default="en", description="Language code")


class CloneResponse(BaseModel):
    """Response model for voice cloning."""
    voice_id: str = Field(..., description="ID of the newly created voice")
    voice_name: str = Field(..., description="Name of the newly created voice")
    message: str = Field(default="Voice cloned successfully", description="Status message")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok", description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(default=False, description="Whether the model is loaded")
    device: str = Field(..., description="Device being used")


class APIInfoResponse(BaseModel):
    """Response model for API info/help/usage endpoint."""
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="Service description")
    endpoints: Dict[str, str] = Field(default_factory=dict, description="Available endpoints with descriptions")
    emotion_tags: List[str] = Field(default_factory=list, description="Available emotion tags for TTS")
    supported_languages: List[str] = Field(default_factory=list, description="Supported language codes")
    readme: str = Field(default="", description="Full README documentation in Markdown")

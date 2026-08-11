"""Custom exceptions for the TTS service."""


class VoiceNotFoundError(Exception):
    """Raised when a requested voice is not found."""
    pass


class AudioTooShortError(Exception):
    """Raised when reference audio for cloning is too short."""
    pass


class ModelLoadError(Exception):
    """Raised when the TTS model fails to load."""
    pass


class SynthesisError(Exception):
    """Raised when audio synthesis fails."""
    pass


class CloningError(Exception):
    """Raised when voice cloning process fails."""
    pass


class InvalidTranscriptError(Exception):
    """Raised when the transcript doesn't match the audio."""
    pass

"""Audio processing utilities for LightTTS."""

import os
import tempfile
import logging
from typing import List
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Crossfade duration between chunks (seconds)
CROSSFADE_DURATION = 0.15
CROSSFADE_SAMPLES = int(24000 * CROSSFADE_DURATION)
# Minimum valid audio length in samples (0.1s at 24kHz)
MIN_AUDIO_SAMPLES = 2400


def tensor_to_wav_bytes(speech: torch.Tensor) -> bytes:
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


def apply_pitch_shift(waveform: torch.Tensor, pitch_factor: float) -> torch.Tensor:
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


def crossfade_chunks(chunks: List[torch.Tensor], crossfade_samples: int = CROSSFADE_SAMPLES) -> torch.Tensor:
    """Crossfade a list of audio chunks for smooth transitions."""
    if not chunks:
        return torch.zeros(1, 0)
    if len(chunks) == 1:
        return chunks[0]

    # Ensure all chunks are 2D [1, samples]
    processed_chunks = []
    for chunk in chunks:
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        elif chunk.dim() == 3:
            chunk = chunk.squeeze(0)
        processed_chunks.append(chunk)

    result = processed_chunks[0]
    for i in range(1, len(processed_chunks)):
        prev = result
        curr = processed_chunks[i]

        # Determine overlap length
        overlap = min(crossfade_samples, prev.shape[-1], curr.shape[-1])
        if overlap <= 0:
            result = torch.cat([prev, curr], dim=-1)
            continue

        # Create crossfade curves
        fade_out = torch.linspace(1.0, 0.0, overlap)
        fade_in = torch.linspace(0.0, 1.0, overlap)

        # Apply crossfade
        prev_end = prev[:, -overlap:] * fade_out
        curr_start = curr[:, :overlap] * fade_in
        crossfaded = prev_end + curr_start

        # Concatenate
        result = torch.cat([
            prev[:, :-overlap],
            crossfaded,
            curr[:, overlap:]
        ], dim=-1)

    return result


def validate_audio_length(speech: torch.Tensor) -> bool:
    """Check if audio has minimum valid length."""
    return speech.shape[-1] >= MIN_AUDIO_SAMPLES

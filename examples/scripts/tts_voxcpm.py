#!/usr/bin/env python3
"""
Example script for TTS synthesis using VoxCPM model.
Demonstrates both streaming and download modes, studio and fast modes.
"""

import requests
import argparse
import json
import os
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TTS_ENDPOINT = f"{API_BASE_URL}/tts"

def tts_voxcpm(
    text: str,
    voice_id: str,
    language: str = "es",
    stream: bool = False,
    mode: str = "studio",  # "studio" or "fast"
    speed: float = 1.0,
    pitch: float = 1.0,
    emotion: str = "neutral",
    emotion_tags: list = None,
    chunk_size: int = 100,
    output_file: str = None
):
    """
    Synthesize speech using VoxCPM model.
    
    Args:
        text: Text to synthesize
        voice_id: ID of the voice to use
        language: Language code
        stream: Whether to stream the response
        mode: Synthesis mode ("studio" for quality, "fast" for speed)
        speed: Speech speed factor
        pitch: Pitch factor
        emotion: Emotion label
        emotion_tags: List of emotion tags (if provided, enables emotion tagging)
        chunk_size: Chunk size for streaming
        output_file: Output file path (for download mode)
    """
    # Convert emotion_tags to boolean: True if tags are provided, False otherwise
    use_emotion_tags = bool(emotion_tags)
    
    payload = {
        "text": text,
        "voice_id": voice_id,
        "language": language,
        "stream": stream,
        "mode": mode,
        "speed": speed,
        "pitch": pitch,
        "emotion": emotion,
        "emotion_tags": use_emotion_tags,
        "chunk_size": chunk_size
    }
    
    print(f"Synthesizing with VoxCPM ({mode} mode)...")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Voice ID: {voice_id}")
    print(f"Streaming: {stream}")
    print(f"Speed: {speed}, Pitch: {pitch}")
    print(f"Emotion: {emotion}")
    print(f"Use emotion tags: {use_emotion_tags}")
    
    try:
        if stream:
            # Streaming response
            response = requests.post(TTS_ENDPOINT, json=payload, stream=True)
            response.raise_for_status()
            
            if output_file:
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"\n✅ Streaming audio saved to: {output_file}")
            else:
                # Play or process chunks in real-time
                print("\n🔊 Streaming audio chunks (first 5 chunks):")
                for i, chunk in enumerate(response.iter_content(chunk_size=8192)):
                    if chunk:
                        print(f"  Chunk {i+1}: {len(chunk)} bytes")
                        if i >= 4:
                            print("  ...")
                            break
        else:
            # Download mode - single response
            response = requests.post(TTS_ENDPOINT, json=payload)
            response.raise_for_status()
            
            # Check content type to determine how to handle response
            content_type = response.headers.get('Content-Type', '')
            
            if 'application/json' in content_type:
                # JSON response with metadata (and possibly base64 audio)
                result = response.json()
                print("\n✅ Synthesis completed!")
                print(f"Audio URL: {result.get('audio_url')}")
                print(f"Duration: {result.get('duration')}s")
                print(f"Sample rate: {result.get('sample_rate')}Hz")
                
                # If audio_url is a local path or base64, handle accordingly
                if output_file and result.get('audio_base64'):
                    import base64
                    audio_data = base64.b64decode(result['audio_base64'])
                    with open(output_file, 'wb') as f:
                        f.write(audio_data)
                    print(f"Audio saved to: {output_file}")
                
                return result
            else:
                # Binary audio response (WAV, MP3, etc.)
                print("\n✅ Synthesis completed! Received binary audio data.")
                print(f"Content-Type: {content_type}")
                print(f"Content-Length: {len(response.content)} bytes")
                
                if output_file:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    print(f"Audio saved to: {output_file}")
                
                # Return a mock result for consistency
                return {
                    "audio_url": output_file,
                    "duration": None,
                    "sample_rate": None,
                    "size_bytes": len(response.content)
                }
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error during synthesis: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response headers: {dict(e.response.headers)}")
            print(f"Response text: {e.response.text[:500]}")
        raise

def demo_all_modes(voice_id: str, text: str = "Hola, esto es una prueba de síntesis de voz con VoxCPM."):
    """Demonstrate all TTS modes with VoxCPM."""
    print("=" * 60)
    print("VOXCPM TTS DEMO - ALL MODES")
    print("=" * 60)
    
    modes = [
        {"mode": "studio", "stream": False, "desc": "Studio quality, download"},
        {"mode": "fast", "stream": False, "desc": "Fast mode, download"},
        {"mode": "studio", "stream": True, "desc": "Studio quality, streaming"},
        {"mode": "fast", "stream": True, "desc": "Fast mode, streaming"},
    ]
    
    for i, config in enumerate(modes):
        print(f"\n--- Demo {i+1}: {config['desc']} ---")
        output_file = f"voxcpm_{config['mode']}_{'stream' if config['stream'] else 'download'}.wav"
        
        tts_voxcpm(
            text=text,
            voice_id=voice_id,
            mode=config['mode'],
            stream=config['stream'],
            output_file=output_file if not config['stream'] else None
        )
        time.sleep(1)  # Brief pause between demos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS synthesis with VoxCPM")
    parser.add_argument("--text", default="Hola, esto es una prueba de síntesis de voz con VoxCPM.", help="Text to synthesize")
    parser.add_argument("--voice-id", required=True, help="Voice ID to use")
    parser.add_argument("--language", default="es", help="Language code (default: es)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--mode", choices=["studio", "fast"], default="studio", help="Synthesis mode")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed factor")
    parser.add_argument("--pitch", type=float, default=1.0, help="Pitch factor")
    parser.add_argument("--emotion", default="neutral", help="Emotion label")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--demo-all", action="store_true", help="Run all modes demo")
    parser.add_argument("--url", default=API_BASE_URL, help=f"API base URL (default: {API_BASE_URL})")
    
    args = parser.parse_args()
    API_BASE_URL = args.url
    TTS_ENDPOINT = f"{API_BASE_URL}/tts"
    
    if args.demo_all:
        demo_all_modes(args.voice_id, args.text)
    else:
        tts_voxcpm(
            text=args.text,
            voice_id=args.voice_id,
            language=args.language,
            stream=args.stream,
            mode=args.mode,
            speed=args.speed,
            pitch=args.pitch,
            emotion=args.emotion,
            output_file=args.output
        )

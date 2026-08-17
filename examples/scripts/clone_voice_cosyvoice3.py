#!/usr/bin/env python3
"""
Example script for voice cloning using CosyVoice3 model.
This script demonstrates how to clone a voice using the /clone endpoint.
"""

import requests
import argparse
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change if your API runs elsewhere
CLONE_ENDPOINT = f"{API_BASE_URL}/clone"

def clone_voice_cosyvoice3(audio_path: str, voice_name: str, transcript: str, language: str = "es"):
    """
    Clone a voice using CosyVoice3 model.
    
    Args:
        audio_path: Path to the reference audio file (WAV, MP3, etc.)
        voice_name: Name for the cloned voice
        transcript: Transcript of the reference audio
        language: Language code (default: "es" for Spanish)
    """
    # Validate audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Prepare the multipart form data
    files = {
        'audio': (os.path.basename(audio_path), open(audio_path, 'rb'), 'audio/wav')
    }
    data = {
        'voice_name': voice_name,
        'transcript': transcript,
        'language': language
    }
    
    print(f"Cloning voice '{voice_name}' using CosyVoice3...")
    print(f"Audio file: {audio_path}")
    print(f"Transcript: {transcript}")
    print(f"Language: {language}")
    
    try:
        response = requests.post(CLONE_ENDPOINT, files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
        print("\n✅ Voice cloned successfully!")
        print(f"Voice ID: {result.get('voice_id')}")
        print(f"Message: {result.get('message')}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error cloning voice: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        raise
    finally:
        files['audio'][1].close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone a voice using CosyVoice3")
    parser.add_argument("--audio", required=True, help="Path to reference audio file")
    parser.add_argument("--name", required=True, help="Name for the cloned voice")
    parser.add_argument("--transcript", required=True, help="Transcript of the reference audio")
    parser.add_argument("--language", default="es", help="Language code (default: es)")
    parser.add_argument("--url", default=API_BASE_URL, help=f"API base URL (default: {API_BASE_URL})")
    
    args = parser.parse_args()
    API_BASE_URL = args.url
    CLONE_ENDPOINT = f"{API_BASE_URL}/clone"
    
    clone_voice_cosyvoice3(args.audio, args.name, args.transcript, args.language)

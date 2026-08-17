#!/usr/bin/env python3
"""
Example script for listing available voices.
"""

import requests
import argparse
import json

# Configuration
API_BASE_URL = "http://localhost:8000"
VOICES_ENDPOINT = f"{API_BASE_URL}/voices"

def list_voices():
    """List all available voices."""
    print("Fetching available voices...")
    
    try:
        response = requests.get(VOICES_ENDPOINT)
        response.raise_for_status()
        
        result = response.json()
        voices = result.get('voices', [])
        
        print(f"\n✅ Found {len(voices)} voice(s):")
        print("-" * 80)
        
        for voice in voices:
            print(f"Voice ID: {voice.get('voice_id')}")
            print(f"  Name: {voice.get('name')}")
            print(f"  Language: {voice.get('language')}")
            print(f"  Model: {voice.get('model', 'N/A')}")
            print(f"  Created: {voice.get('created_at', 'N/A')}")
            print(f"  Duration: {voice.get('duration', 'N/A')}s")
            print(f"  Sample Rate: {voice.get('sample_rate', 'N/A')}Hz")
            print()
            
        return voices
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error fetching voices: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List available voices")
    parser.add_argument("--url", default=API_BASE_URL, help=f"API base URL (default: {API_BASE_URL})")
    
    args = parser.parse_args()
    API_BASE_URL = args.url
    VOICES_ENDPOINT = f"{API_BASE_URL}/voices"
    
    list_voices()

#!/usr/bin/env python3
"""
Example script for deleting a voice.
"""

import requests
import argparse

# Configuration
API_BASE_URL = "http://localhost:8000"

def delete_voice(voice_id: str):
    """Delete a voice by ID."""
    endpoint = f"{API_BASE_URL}/voices/{voice_id}"
    
    print(f"Deleting voice: {voice_id}")
    
    try:
        response = requests.delete(endpoint)
        response.raise_for_status()
        
        result = response.json()
        print(f"\n✅ {result.get('message', 'Voice deleted successfully')}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error deleting voice: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a voice")
    parser.add_argument("--voice-id", required=True, help="Voice ID to delete")
    parser.add_argument("--url", default=API_BASE_URL, help=f"API base URL (default: {API_BASE_URL})")
    
    args = parser.parse_args()
    API_BASE_URL = args.url
    
    delete_voice(args.voice_id)

GenPod AI Voice Service
Production-ready Text-to-Speech (TTS) and Voice Cloning API powered by CosyVoice 2. Designed for high-quality audio generation, podcast creation, and low-latency chatbot integrations.
FEATURES

    High-Quality TTS: Natural-sounding speech synthesis with dual modes (fast for chatbots, studio for podcasts).
    Zero-Shot Voice Cloning: Clone any voice from a 3-30 second audio sample.
    Emotion Tags: Control prosody dynamically using inline tags like happy, sad, whisper, etc.
    Multi-Language Support: English, Spanish, Catalan, Chinese, Japanese, and Korean.
    Streaming Responses: Real-time audio chunk streaming for low-latency applications.
    Docker Ready: Easy deployment with dedicated CPU and GPU profiles.

QUICK START
Prerequisites:

    Docker and Docker Compose
    NVIDIA GPU with CUDA support (recommended) or CPU fallback
    At least 8GB RAM (16GB+ recommended for GPU inference)

Installation:

    Clone the repository:
    git clone your-repository-url
    cd genpod
    Configure environment variables:
    cp .env.example .env
    Build and start the service:
    For GPU (Recommended):
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
    For CPU only:
    docker compose up -d

Note: The first request will take 30-60 seconds as the model loads into memory (lazy loading).
API ENDPOINTS

    Text-to-Speech: POST /api/v1/tts

Generate speech from text. Supports streaming and emotion tags.
Request Body (JSON):
{
  "text": "<happy>Hello everyone!</happy> <serious>Let's discuss this important topic.</serious>",
  "voice_id": "en_female",
  "language": "en",
  "mode": "studio",
  "emotion_tags": true
}
Available Emotion Tags:
<happy>, <sad>, <serious>, <whisper>, <angry>, <narrative>, <slow>, <fast>, <neutral>
Example (cURL):
curl -X POST "http://localhost:8000/api/v1/tts
" 
  -H "Content-Type: application/json" 
  -d '{
    "text": "<happy>Hello everyone!</happy> <serious>Today we have an important topic.</serious>",
    "voice_id": "en_female",
    "language": "en",
    "mode": "studio",
    "emotion_tags": true
  }' 
  --output output.wav

    Voice Cloning: POST /api/v1/clone

Clone a voice from a reference audio sample.
Parameters (Form Data):

    audio: WAV/MP3 file (3-30 seconds, mono, 16kHz recommended)
    voice_name: Human-readable name for the new voice
    transcript: Exact text spoken in the audio
    language: Language code (e.g., en, es, ca)

Example (cURL):
curl -X POST "http://localhost:8000/api/v1/clone
" 
  -F "audio=@reference.wav" 
  -F "voice_name=my_custom_voice" 
  -F "transcript=This is a sample of my voice for cloning." 
  -F "language=en"

    List Voices: GET /api/v1/voices

Retrieve all available base and cloned voices.
Example (cURL):
curl http://localhost:8000/api/v1/voices
 | jq

    API Information: GET /api/v1/info

Get comprehensive API documentation, available endpoints, and supported features.
Aliases: /api/v1/help, /api/v1/usage
Example (cURL):
curl http://localhost:8000/api/v1/info
 | jq
CONFIGURATION
Edit the .env file to customize the service:
DEVICE=cuda                    Options: cuda, cpu, mps
MODELS_PATH=/app/data/models   Path to model weights inside container
VOICES_PATH=/app/data/voices   Path to cloned voice profiles inside container
PORT=8000                      API port
TROUBLESHOOTING
Model Not Loading / 503 Service Unavailable:

    Ensure you have sufficient GPU memory (run nvidia-smi).
    Check container logs: docker compose logs -f lighttts-api
    Wait 30-60 seconds after the first request for lazy loading to complete.

Disk Space Issues:
Docker build cache can grow large. Clean it safely with:
docker system prune -a --volumes
Voice Cloning Fails:

    Ensure the reference audio is between 3 and 30 seconds.
    Verify the transcript matches the audio exactly.
    Check directory permissions: sudo chmod -R 777 ./data/voices

ARCHITECTURE
Client (cURL / Web / App)
       |
       v
FastAPI (api/main.py, api/routes/)
       |
       v
LightTTSEngine (core/lighttts/engine.py)
       |-- VoiceManager (Metadata & File I/O)
       +-- CosyVoice 2 / v1 SFT (Model Inference)
LICENSE
This project utilizes the CosyVoice model, which is licensed under the Apache 2.0 License. Please review the original repository for specific model usage terms.
SUPPORT
For issues, feature requests, or contributions, please open an issue in the project repository.


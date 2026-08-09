# Architecture.md

## Tech Stack
- **Core Engine:** CosyVoice 2 (wrapped via custom `lightTTS` module).
- **Backend API:** FastAPI, Uvicorn, Pydantic V2.
- **Client CLI:** Python standard library (`argparse`, `httpx` for async HTTP requests).
- **Deployment:** Docker, Docker Compose (with NVIDIA Container Toolkit support for GPU).

## Directory Structure
```text
.
├── .env.example             # Environment variables template
├── docker-compose.yml       # Orchestration (CPU/GPU profiles)
├── Dockerfile               # Multi-stage build for the API
├── requirements.txt         # Python dependencies
├── api/                     # FastAPI Backend
│   ├── __init__.py
│   ├── main.py              # FastAPI app initialization
│   ├── routes/              # API endpoints (tts.py, clone.py, voices.py)
│   ├── schemas.py           # Pydantic request/response models
│   └── dependencies.py      # Dependency injection (e.g., getting the lightTTS instance)
├── cli/                     # Command Line Interface Client
│   ├── __init__.py
│   └── client.py            # argparse setup and httpx calls to the API
├── core/                    # Core Business Logic & AI Engine
│   ├── __init__.py
│   ├── exceptions.py        # Custom exceptions
│   ├── config.py            # pydantic-settings configuration
│   └── lighttts/            # The CosyVoice 2 Wrapper
│       ├── __init__.py
│       ├── engine.py        # Main wrapper class (load model, tts, clone)
│       └── voice_manager.py # Handles saving/loading cloned voice profiles
├── docker/                  # Docker specific scripts
│   └── entrypoint.sh        # Startup script (downloads default models/voices)
└── data/                    # Mounted Docker volumes (gitignored)
    ├── models/              # CosyVoice 2 base weights
    └── voices/              # Cloned voice profiles (wav + json metadata)

Component Interaction Flow
1. Text-to-Speech (TTS) Flow

    CLI sends POST /api/v1/tts with {text, voice_id, lang}.
    API validates input via Pydantic.
    API calls lightTTS.engine.synthesize().
    Engine loads the specific voice profile (base or cloned) and generates audio chunks.
    API streams the audio chunks back to the CLI using StreamingResponse.
    CLI saves the stream to the local output_path.

2. Voice Cloning Flow

    CLI sends POST /api/v1/clone with multipart/form-data (audio file, text, new_voice_name).
    API saves the uploaded audio to a temporary directory.
    API calls lightTTS.engine.clone_voice().
    Engine extracts speaker embeddings using CosyVoice 2, saves the profile to data/voices/, and returns the new voice_id.
    API cleans up the temp file and returns {voice_id, status} to the CLI.

Hardware Configuration (Docker)

    The docker-compose.yml will use Docker Compose profiles.
    docker compose --profile gpu up: Mounts /dev/dsh and uses the nvidia runtime.
    docker compose --profile cpu up: Standard execution, forces PyTorch to use CPU.

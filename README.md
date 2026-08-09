# lightTTS - CosyVoice 2 TTS & Voice Cloning Service

A production-ready Text-to-Speech and Voice Cloning service powered by **CosyVoice 2**, exposed via a FastAPI backend with a Python CLI client.

## Features

- 🎙️ **High-Quality TTS**: CosyVoice 2 (0.5B) for natural speech synthesis
- 🎭 **Voice Cloning**: Clone voices from 3+ seconds of reference audio
- 🌊 **Streaming Audio**: Real-time audio streaming via HTTP
- 🐳 **Docker Ready**: CPU and GPU profiles with NVIDIA support
- 🔧 **Simple CLI**: Easy-to-use command line interface
- ⚡ **FastAPI Backend**: Async, high-performance API with automatic docs

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (for GPU profile)

### CPU Version

```bash
# Clone and navigate
git clone <repo-url>
cd lighttts

# Start the service
docker compose --profile cpu up --build
```

### GPU Version

```bash
# Requires NVIDIA Container Toolkit
docker compose --profile gpu up --build
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## CLI Usage

Install the CLI client:
```bash
pip install -e ./cli
# Or run directly: python -m cli.client ...
```

### List Available Voices
```bash
lighttts list
```

### Text-to-Speech
```bash
lighttts tts "Hello, this is a test!" --voice base_female --output output.wav
```

### Voice Cloning
```bash
lighttts clone --audio reference.wav --transcript "This is the reference text." --name "My Voice"
```

### Use Cloned Voice
```bash
lighttts tts "Speaking with my cloned voice!" --voice cloned_xxx --output cloned.wav
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/voices` | List all available voices |
| POST | `/api/v1/tts` | Text-to-Speech (streaming) |
| POST | `/api/v1/clone` | Clone a new voice |

## Configuration

Environment variables (`.env`):

```bash
HOST=0.0.0.0
PORT=8000
DEVICE=cpu          # cpu, cuda, mps
MODELS_PATH=/app/data/models
VOICES_PATH=/app/data/voices
LOG_LEVEL=info
```

## Project Structure

```
.
├── api/                 # FastAPI Backend
│   ├── main.py
│   ├── routes/
│   ├── schemas.py
│   └── dependencies.py
├── cli/                 # CLI Client
│   └── client.py
├── core/                # Core Logic
│   ├── config.py
│   ├── exceptions.py
│   └── lighttts/        # CosyVoice 2 Wrapper
│       ├── engine.py
│       └── voice_manager.py
├── docker/              # Docker Scripts
│   └── entrypoint.sh
├── data/                # Mounted Volumes (gitignored)
│   ├── models/
│   └── voices/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install -U git+https://github.com/FunAudioLLM/CosyVoice.git

# Run API locally
uvicorn api.main:app --reload

# Run CLI
python -m cli.client --help
```

## License

MIT License - See LICENSE file for details.

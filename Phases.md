# Phases.md

This document tracks the implementation progress of the CosyVoice 2 TTS & Cloning service. 
**Current Phase:** [x] Phase 5

## Phase 1: Project Scaffolding & Docker Base
- [x] Create the directory structure as defined in `Architecture.md`.
- [x] Create `requirements.txt` (fastapi, uvicorn, httpx, pydantic-settings, torch, torchaudio).
- [x] Create `.env.example` with default configurations (PORT, DEVICE, MODELS_PATH, VOICES_PATH).
- [x] Create `core/config.py` using `pydantic-settings` to load the `.env`.
- [x] Create the base `Dockerfile` (Python 3.10-slim, install system deps for audio, copy requirements).
- [x] Create `docker-compose.yml` with `cpu` and `gpu` profiles, defining volumes for `data/models` and `data/voices`.
- [x] Create `docker/entrypoint.sh` to handle initial setup.

## Phase 2: Core Engine (lightTTS Wrapper)
- [x] Create `core/exceptions.py` with custom errors.
- [x] Implement `core/lighttts/engine.py`:
  - [x] Class `LightTTSEngine`.
  - [x] `__init__`: Initialize CosyVoice 2 model (handle CPU/GPU device placement).
  - [x] `list_voices()`: Scan `data/voices/` and default models.
  - [x] `synthesize(text, voice_id, lang)`: Generator yielding audio chunks.
  - [x] `clone_voice(audio_path, transcript, voice_name)`: Extract embeddings and save profile.
- [x] Implement `core/lighttts/voice_manager.py` to handle reading/writing voice metadata (JSON).

## Phase 3: FastAPI Backend
- [x] Create `api/schemas.py` with Pydantic models for TTS and Clone requests/responses.
- [x] Create `api/dependencies.py` to initialize and yield the `LightTTSEngine` singleton.
- [x] Implement `api/routes/voices.py`: `GET /api/v1/voices`.
- [x] Implement `api/routes/tts.py`: `POST /api/v1/tts` (returning `StreamingResponse`).
- [x] Implement `api/routes/clone.py`: `POST /api/v1/clone` (handling `UploadFile`).
- [x] Create `api/main.py`, register routes, and add CORS/exception handlers.

## Phase 4: CLI Client
- [x] Create `cli/client.py`.
- [x] Implement `argparse` with subcommands: `list`, `tts`, `clone`.
- [x] Implement `httpx` async client to communicate with the FastAPI backend.
- [x] Handle file streaming in the CLI for the `tts` command (save to disk).
- [x] Handle file uploading in the CLI for the `clone` command.

## Phase 5: Integration, Default Voices & Testing
- [x] Update `docker/entrypoint.sh` to automatically download CosyVoice 2 model (includes base speakers) if not exists in `data/models`.
- [x] Test the full Docker build process for both CPU and GPU profiles.
- [x] End-to-end test: Start container -> CLI list voices -> CLI TTS -> CLI Clone -> CLI TTS with cloned voice.
- [x] Add a `README.md` with final usage instructions.

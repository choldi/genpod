
***

### 3. `Phases.md`
*Este es el plan de ejecución. Aider leerá esto para saber en qué paso está. Cuando termines una fase, simplemente marcas el checkbox y le dices a Aider: "Pasa a la siguiente fase".*

```markdown
# Phases.md

This document tracks the implementation progress of the CosyVoice 2 TTS & Cloning service. 
**Current Phase:** [ ] Phase 1

## Phase 1: Project Scaffolding & Docker Base
- [ ] Create the directory structure as defined in `Architecture.md`.
- [ ] Create `requirements.txt` (fastapi, uvicorn, httpx, pydantic-settings, torch, torchaudio).
- [ ] Create `.env.example` with default configurations (PORT, DEVICE, MODELS_PATH, VOICES_PATH).
- [ ] Create `core/config.py` using `pydantic-settings` to load the `.env`.
- [ ] Create the base `Dockerfile` (Python 3.10-slim, install system deps for audio, copy requirements).
- [ ] Create `docker-compose.yml` with `cpu` and `gpu` profiles, defining volumes for `data/models` and `data/voices`.
- [ ] Create `docker/entrypoint.sh` to handle initial setup.

## Phase 2: Core Engine (lightTTS Wrapper)
- [ ] Create `core/exceptions.py` with custom errors.
- [ ] Implement `core/lighttts/engine.py`:
  - [ ] Class `LightTTSEngine`.
  - [ ] `__init__`: Initialize CosyVoice 2 model (handle CPU/GPU device placement).
  - [ ] `list_voices()`: Scan `data/voices/` and default models.
  - [ ] `synthesize(text, voice_id, lang)`: Generator yielding audio chunks.
  - [ ] `clone_voice(audio_path, transcript, voice_name)`: Extract embeddings and save profile.
- [ ] Implement `core/lighttts/voice_manager.py` to handle reading/writing voice metadata (JSON).

## Phase 3: FastAPI Backend
- [ ] Create `api/schemas.py` with Pydantic models for TTS and Clone requests/responses.
- [ ] Create `api/dependencies.py` to initialize and yield the `LightTTSEngine` singleton.
- [ ] Implement `api/routes/voices.py`: `GET /api/v1/voices`.
- [ ] Implement `api/routes/tts.py`: `POST /api/v1/tts` (returning `StreamingResponse`).
- [ ] Implement `api/routes/clone.py`: `POST /api/v1/clone` (handling `UploadFile`).
- [ ] Create `api/main.py`, register routes, and add CORS/exception handlers.

## Phase 4: CLI Client
- [ ] Create `cli/client.py`.
- [ ] Implement `argparse` with subcommands: `list`, `tts`, `clone`.
- [ ] Implement `httpx` async client to communicate with the FastAPI backend.
- [ ] Handle file streaming in the CLI for the `tts` command (save to disk).
- [ ] Handle file uploading in the CLI for the `clone` command.

## Phase 5: Integration, Default Voices & Testing
- [ ] Update `docker/entrypoint.sh` to automatically download 2 default English voices (1 male, 1 female) for CosyVoice 2 if they don't exist in `data/models`.
- [ ] Test the full Docker build process for both CPU and GPU profiles.
- [ ] End-to-end test: Start container -> CLI list voices -> CLI TTS -> CLI Clone -> CLI TTS with cloned voice.
- [ ] Add a `README.md` with final usage instructions.

# Phases.md

This document tracks the implementation progress of the CosyVoice 2 TTS & Cloning service.

**Current Phase: [x] Phase 7**

---

## Phase 1: Project Scaffolding & Docker Base
- [x] Create the directory structure as defined in `Architecture.md`.
- [x] Create `requirements.txt` (fastapi, uvicorn, httpx, pydantic-settings, torch, torchaudio).
- [x] Create `.env.example` with default configurations (PORT, DEVICE, MODELS_PATH, VOICES_PATH).
- [x] Create `core/config.py` using `pydantic-settings` to load the `.env`.
- [x] Create the base `Dockerfile` (Python 3.10-slim, install system deps for audio, copy requirements).
- [x] Create `docker-compose.yml` with `cpu` and `gpu` profiles, defining volumes for `data/models` and `data/voices`.
- [x] Create `docker/entrypoint.sh` to handle initial setup.

> **Nota:** Se corrigió la estructura a archivos de override (`docker-compose.yml` + `docker-compose.gpu.yml`).

---

## Phase 2: Core Engine (lightTTS Wrapper)
- [x] Create `core/exceptions.py` with custom errors.
- [x] Implement `core/lighttts/engine.py`:
  - [x] Class `LightTTSEngine`.
  - [x] `__init__`: Initialize CosyVoice 2 model (handle CPU/GPU device placement).
  - [x] `list_voices()`: Scan `data/voices/` and default models.
  - [x] `synthesize(text, voice_id, lang)`: Generator yielding audio chunks.
  - [x] `clone_voice(audio_path, transcript, voice_name)`: Extract embeddings and save profile.
- [x] Implement `core/lighttts/voice_manager.py` to handle reading/writing voice metadata (JSON).

---

## Phase 3: FastAPI Backend
- [x] Create `api/schemas.py` with Pydantic models for TTS and Clone requests/responses.
- [x] Create `api/dependencies.py` to initialize and yield the `LightTTSEngine` singleton.
- [x] Implement `api/routes/voices.py`: `GET /api/v1/voices`.
- [x] Implement `api/routes/tts.py`: `POST /api/v1/tts` (returning `StreamingResponse`).
- [x] Implement `api/routes/clone.py`: `POST /api/v1/clone` (handling `UploadFile`).
- [x] Create `api/main.py`, register routes, and add CORS/exception handlers.

---

## Phase 4: CLI Client
- [x] Create `cli/client.py`.
- [x] Implement `argparse` with subcommands: `list`, `tts`, `clone`.
- [x] Implement `httpx` async client to communicate with the FastAPI backend.
- [x] Handle file streaming in the CLI for the `tts` command (save to disk).
- [x] Handle file uploading in the CLI for the `clone` command.

---

## Phase 5: Integration, Default Voices & Testing
- [x] Update `docker/entrypoint.sh` to automatically download CosyVoice 2 model if not exists in `data/models`.
- [x] Test the full Docker build process for both CPU and GPU profiles.
- [x] End-to-end test: Start container → CLI list voices → CLI TTS → CLI Clone → CLI TTS with cloned voice.
- [x] Add a `README.md` with final usage instructions.

---

## Phase 6: Bug Fixes & Audio Pipeline Corrections
- [x] **Fix tensor dimension error (3D → 2D):** Corregido `Expected 2D Tensor, got 3D` en `torchaudio.save`. Se añadió `.squeeze(0)` cuando el tensor de salida tiene 3 dimensiones `[1, canales, muestras]`.
- [x] **Fix torchaudio.save argument order:** Corregido el orden de argumentos `(ruta, tensor, sample_rate)`. Se reemplazó `io.BytesIO` por `tempfile.NamedTemporaryFile` para evitar el error `Invalid file: tensor(...)`.
- [x] **Fix inference_zero_shot input type:** CosyVoice v1 SFT espera la **ruta del archivo (string)** como tercer argumento de `inference_zero_shot`, no un tensor de PyTorch. Se cambió de `ref_speech.to(self.device)` a `str(ref_audio_path)`.
- [x] **Fix positional arguments for CosyVoice v1:** `inference_zero_shot()` no acepta keyword arguments (`text=`, `prompt_text=`). Se corrigió a llamada posicional: `(text, prompt_text, ref_audio_path, False)`.
- [x] **Fix audio format for cloning:** El audio de referencia debe ser **Mono, 16kHz** para `inference_zero_shot`. Se añadió conversión automática con `torchaudio.transforms.Resample`.
- [x] **Fix clone endpoint parameters:** Los parámetros correctos son `voice_name` (no `name`), `language` (no `lang`), y `transcript` es **obligatorio** (`Form(..., min_length=1)`).
- [x] **Fix clone endpoint response:** `engine.clone_voice()` devuelve un `str` (voice_id), no un `dict`. Se corrigió `api/routes/clone.py` para evitar `'str' object has no attribute 'get'`.
- [x] **Fix Docker volume permissions:** El directorio `data/voices` montado como volumen no tenía permisos de escritura para `appuser`. Solución: `sudo chmod -R 777 ./data/voices ./data/models` en el host.
- [x] **Fix Matplotlib/HuggingFace cache error:** El contenedor no podía escribir en `/home/appuser/.cache` ni crear directorios temporales. Se añadieron variables de entorno `MPLCONFIGDIR=/app/tmp/matplotlib` y `TRANSFORMERS_CACHE=/app/tmp/huggingface` en `docker-compose.yml`.
- [x] **Fix disk space exhaustion:** La acumulación de imágenes Docker antiguas (~256 GB) y caché de build (~145 GB) llenó el disco. Solución: `docker system prune -a --volumes`.

---

## Phase 7: Feature Enhancements & Voice Management
- [x] **Dual mode support (fast vs studio):** Añadido parámetro `mode` al endpoint `/tts`.
  - `mode: "fast"` → Velocidad 1.05x, optimizado para WhatsApp/chatbots (baja latencia).
  - `mode: "studio"` → Velocidad 0.95x, optimizado para podcasts/audiolibros (calidad natural).
  - Campo `mode: str = "studio"` añadido a `TTSRequest` en `api/schemas.py`.
  - Lógica de modo implementada en `api/routes/tts.py`.
- [x] **Speed, pitch, emotion parameters:** Añadidos parámetros de control de prosodia al endpoint `/tts` y propagados a `engine.synthesize()`.
- [x] **Custom voice naming (no UUIDs):** Modificado `clone_voice()` para usar `voice_name` como `voice_id` en lugar de generar UUIDs aleatorios (`cloned_xxxxx`). Los archivos se guardan como `{voice_name}.wav` y `{voice_name}.json`.
- [x] **is_cloned attribute in API response:** Añadido campo `is_cloned: bool` al modelo `VoiceInfo` en `api/schemas.py` y al endpoint `GET /api/v1/voices`. Permite filtrar voces clonadas vs base.
- [x] **Voice deletion:** Implementado borrado de voces clonadas eliminando archivos `.wav` y `.json` del directorio `data/voices/`.
- [x] **Improved error handling (404 vs 500):** Las excepciones `VoiceNotFoundError` ahora se propagan correctamente hasta la API, devolviendo `HTTP 404` con JSON limpio en lugar de `HTTP 500`.
- [x] **Voice listing and cleanup script:** Creado script bash `reset_and_clone.sh` que lista voces clonadas vía API, las borra del servidor y crea una nueva voz en un solo comando.
- [x] **Multi-language support verified:** Sistema probado con español (`es`), catalán (`ca`), inglés (`en`) y chino (`zh`).

---

## Current Status: 🟢 PRODUCTION READY

All core features implemented, tested, and debugged. System is ready for:
- WhatsApp integration (fast mode)
- Podcast and audiobook generation (studio mode)
- Voice cloning with custom names and multi-language support
- Voice management (list, create, delete)

## Known Limitations
- Model loading takes 30-60 seconds on first request (lazy loading).
- Docker image size ~18 GB (PyTorch + CUDA + CosyVoice dependencies).
- Zero-shot voice cloning quality depends heavily on reference audio quality and transcript accuracy.
- CosyVoice v1 SFT has limited native language support; non-Asian/European languages may sound accented.

## Future Enhancements (Optional)
- [ ] Model quantization (INT8/FP16) for reduced memory footprint.
- [ ] Batch processing endpoint for multiple texts.
- [ ] WebSocket support for real-time streaming.
- [ ] Voice quality scoring and automatic reference audio validation.
- [ ] DELETE endpoint in API (`DELETE /api/v1/voices/{voice_id}`) to avoid SSH-based cleanup.
- [ ] Authentication and rate limiting middleware.
- [ ] Docker image optimization (multi-stage build, `.dockerignore`).
- [ ] Implement tags in text to express emotions

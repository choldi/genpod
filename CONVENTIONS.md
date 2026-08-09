# CONVENTIONS.md

## 1. General Philosophy & Aider Rules
- **Core Principles:** Strictly follow DRY (Don't Repeat Yourself) and KISS (Keep It Simple, Stupid). Write clean, readable, and maintainable code.
- **Aider Workflow:** 
  - Always read existing code and context before making changes.
  - Make minimal changes; do not rewrite entire files unless explicitly asked.
  - Do not invent libraries or CosyVoice 2 API methods. If unsure, check docs or ask.
  - Provide a very brief summary of changes after editing.

## 2. Python Code Style & Formatting
- **Version:** Python 3.10+.
- **Formatting & Linting:** Follow PEP 8. Use `black` for formatting and `ruff` for linting. Execute them before every commit.
- **Complexity:** Enforce a reasonable cyclomatic complexity limit (max 10) using `ruff` or `radon`. Refactor if it exceeds this.
- **Type Hinting:** Strict static typing is mandatory for all function signatures, class attributes, and return types. Use built-in generics (e.g., `list[str]`) or `typing`.
- **Naming Conventions:**
  - `snake_case` for functions, variables, and modules.
  - `PascalCase` for classes.
  - `UPPER_SNAKE_CASE` for constants and environment variables.
  - Use highly descriptive names for variables and functions.
- **Docstrings:** Use Google-style docstrings for all public functions, classes, and modules.

## 3. Testing & Quality Assurance
- **Framework:** Use `pytest` as the standard testing framework.
- **Coverage:** Write unit tests for all business logic. Maintain **>80% test coverage** for critical code paths (especially `core/lighttts/` and API routes). Use `pytest-cov` to measure.
- **Mocking:** Mock external I/O, heavy AI model inferences, and file system operations during unit tests to keep them fast and isolated.

## 4. Error Handling, Logging & Security
- **Exceptions:** Never use bare `except:` or `except Exception:`. Catch specific exceptions. Create custom domain exceptions in `core/exceptions.py` (e.g., `VoiceNotFoundError`, `AudioTooShortError`).
- **Logging:** 
  - Use the standard `logging` module. Never use `print()` for application logic.
  - Use appropriate levels: `DEBUG` (dev details), `INFO` (workflow events), `WARNING` (recoverable issues), `ERROR` (failures).
  - **NEVER log sensitive information** (API keys, tokens, user PII).
  - Configure logging levels and formats strictly via environment variables.
- **Security & Validation:**
  - **Never hardcode secrets.** Use `pydantic-settings` to load configurations from `.env`.
  - **Validate all user inputs** rigorously using Pydantic V2 schemas before processing.

## 5. FastAPI & API Design
- **Async/Await:** Use `async def` for all route handlers and I/O bound operations.
- **Pydantic:** Use Pydantic V2 models for all request bodies, query parameters, and response schemas.
- **File Handling:** Use `UploadFile` for incoming audio. Use `StreamingResponse` for returning generated audio to prevent RAM exhaustion.

## 6. Docker & Environment
- **Environment Variables:** Never hardcode paths, ports, or model names.
- **Security:** The Docker container must run as a non-root user.

## 7. The "lightTTS" Wrapper Architecture
- **Isolation:** All direct interactions with the underlying CosyVoice 2 code must be strictly encapsulated within the `core/lighttts/` module. The API (`api/`) and CLI (`cli/`) must NEVER import CosyVoice or PyTorch directly; they must only interact with the `lightTTS` wrapper interfaces.


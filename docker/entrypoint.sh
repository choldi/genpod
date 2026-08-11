#!/bin/bash
set -e

# lightTTS Docker Entrypoint Script
# Creates necessary directories and starts the application

echo "=== lightTTS Entrypoint Started ==="
echo "Device: ${DEVICE:-cpu}"
echo "Models Path: ${MODELS_PATH:-/app/data/models}"
echo "Voices Path: ${VOICES_PATH:-/app/data/voices}"

# Create directories if they don't exist
mkdir -p "${MODELS_PATH:-/app/data/models}"
mkdir -p "${VOICES_PATH:-/app/data/voices}"

echo "=== lightTTS Entrypoint Complete ==="
echo "Starting application..."

# Execute the main command
exec "$@"

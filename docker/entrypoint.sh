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

# Workaround: transformers puede buscar nvcc al iniciar
# Crear dummy nvcc si no existe (evita instalar cuda-nvcc-12-1 de 1.5GB)
if ! command -v nvcc &> /dev/null; then
    echo "Creating dummy nvcc..."
    cat > /usr/local/bin/nvcc << 'EOF'
#!/bin/bash
echo "nvcc dummy - CUDA compilation not available in runtime image"
exit 1
EOF
    chmod +x /usr/local/bin/nvcc
fi

# Asegurar que TRITON_CACHE_DIR existe
mkdir -p "${TRITON_CACHE_DIR:-/tmp/.triton}"

echo "=== lightTTS Entrypoint Complete ==="
echo "Starting application..."

# Execute the main command
exec "$@"

#!/bin/bash
set -e

# lightTTS Docker Entrypoint Script
# Handles initial setup: downloading default models if needed

echo "=== lightTTS Entrypoint Started ==="
echo "Device: ${DEVICE:-cpu}"
echo "Models Path: ${MODELS_PATH:-/app/data/models}"
echo "Voices Path: ${VOICES_PATH:-/app/data/voices}"

# Create directories if they don't exist
mkdir -p "${MODELS_PATH:-/app/data/models}"
mkdir -p "${VOICES_PATH:-/app/data/voices}"

# Function to download default CosyVoice 2 model
download_default_model() {
    local model_dir="${MODELS_PATH:-/app/data/models}/CosyVoice2-0.5B"
    
    if [ ! -d "$model_dir" ] || [ -z "$(ls -A "$model_dir" 2>/dev/null)" ]; then
        echo "Downloading CosyVoice 2 base model (0.5B)..."
        mkdir -p "$model_dir"
        
        # Use Python to download via modelscope or huggingface
        python3 -c "
import os
import sys
sys.path.insert(0, '/app')

model_dir = os.environ.get('MODEL_DIR', '/app/data/models/CosyVoice2-0.5B')
print(f'Downloading to {model_dir}...')

# Try ModelScope first (faster in China)
try:
    from modelscope import snapshot_download
    snapshot_download('iic/CosyVoice2-0.5B', local_dir=model_dir)
    print('Model downloaded successfully from ModelScope!')
    sys.exit(0)
except Exception as e:
    print(f'ModelScope download failed: {e}')

# Fallback to HuggingFace
try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
    hf_snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir=model_dir)
    print('Model downloaded successfully from HuggingFace!')
    sys.exit(0)
except Exception as e:
    print(f'HuggingFace download failed: {e}')

print('ERROR: All download methods failed.')
sys.exit(1)
" || {
            echo "ERROR: Failed to download CosyVoice 2 model."
            echo "Please manually download the model to ${MODELS_PATH:-/app/data/models}/CosyVoice2-0.5B"
            echo "You can download from:"
            echo "  - ModelScope: https://modelscope.cn/models/iic/CosyVoice2-0.5B"
            echo "  - HuggingFace: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B"
            exit 1
        }
    else
        echo "CosyVoice 2 model already exists at $model_dir"
    fi
}

# Function to verify model loads correctly
verify_model() {
    echo "Verifying model can be loaded..."
    python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    model_dir = '/app/data/models/CosyVoice2-0.5B'
    model = CosyVoice2(model_dir, load_jit=False, load_onnx=False, fp16=False)
    spks = model.list_available_spks()
    print(f'Available base speakers: {spks}')
    print('Model verification successful!')
except Exception as e:
    print(f'Model verification failed: {e}')
    sys.exit(1)
" || {
        echo "WARNING: Model verification failed. The model may be corrupted."
        echo "Consider re-downloading by removing the model directory and restarting."
    }
}

# Run setup
download_default_model
verify_model

echo "=== lightTTS Entrypoint Complete ==="
echo "Starting application..."

# Execute the main command
exec "$@"

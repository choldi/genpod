# ==============================================================================
# Etapa de construcción
# ==============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Añadimos python3.10-dev y libsox-dev/sox para que pyworld y los efectos de audio compilen
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    gcc \
    git \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user "transformers<4.38"

# Clonar CosyVoice e instalar sus dependencias específicas
RUN git clone --recursive https://github.com/QwenAudio/CosyVoice.git /opt/cosyvoice
WORKDIR /opt/cosyvoice

# FIX DEFINITIVO (de tu versión original):
# 1. Instalar setuptools<71 (que aún incluye pkg_resources) y wheel
RUN pip install --no-cache-dir --user "setuptools<71.0.0" wheel

# 2. Instalar openai-whisper primero con --no-build-isolation para que use el setuptools<71
RUN pip install --no-cache-dir --user --no-build-isolation openai-whisper==20231117

# 3. Instalar el resto de requisitos de CosyVoice
RUN pip install --no-cache-dir --user -r requirements.txt

WORKDIR /app

# ==============================================================================
# Etapa final
# ==============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copiar las dependencias de Python instaladas en el builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiar el código fuente de CosyVoice a la imagen final
COPY --from=builder /opt/cosyvoice /opt/cosyvoice

COPY --chown=appuser:appuser . .

RUN mkdir -p /app/data/models /app/data/voices /app/tmp/matplotlib /app/tmp/huggingface && \
    chown -R appuser:appuser /app/data /app/tmp

RUN chmod +x /app/docker/entrypoint.sh

ENV PATH=/home/appuser/.local/bin:$PATH
# Añadir CosyVoice y /app al PYTHONPATH para que los imports funcionen
ENV PYTHONPATH=/opt/cosyvoice:/opt/cosyvoice/third_party/Matcha-TTS:/app:$PYTHONPATH
ENV NUMBA_DISABLE_JIT=1
ENV NUMBA_CACHE_DIR=/tmp
ENV MPLCONFIGDIR=/app/tmp/matplotlib
ENV TRANSFORMERS_CACHE=/app/tmp/huggingface
   
USER appuser

# Esta combinación es la que mantiene el contenedor vivo:
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

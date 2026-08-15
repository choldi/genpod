# ==============================================================================
# Etapa 1: Construcción (Aquí sí necesitamos las herramientas de desarrollo)
# ==============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    gcc \
    cmake \
    git \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user "transformers<4.38"

# Clonar con profundidad 1 para no descargar el historial de Git innecesario
RUN git clone --depth 1 --recursive https://github.com/QwenAudio/CosyVoice.git /opt/cosyvoice
WORKDIR /opt/cosyvoice

# FIX para compilación de dependencias antiguas
RUN pip install --no-cache-dir --user "setuptools<71.0.0" wheel
RUN pip install --no-cache-dir --user --no-build-isolation openai-whisper==20231117
RUN pip install --no-cache-dir --user -r requirements.txt

WORKDIR /app

# ==============================================================================
# Etapa 2: Ejecución (Imagen final ligera)
# ==============================================================================
# ¡CAMBIO CLAVE! Usamos 'runtime' en lugar de 'devel'. Ahorra ~8-10 GB.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Solo instalamos lo estrictamente necesario para ejecutar (no para compilar)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox2 \
    sox \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copiar solo las dependencias de Python compiladas, no las herramientas de build
COPY --from=builder /root/.local /home/appuser/.local

# Copiar el código de CosyVoice, pero eliminamos la carpeta .git para ahorrar espacio
COPY --from=builder /opt/cosyvoice /opt/cosyvoice
RUN rm -rf /opt/cosyvoice/.git /opt/cosyvoice/.gitmodules

COPY --chown=appuser:appuser . .

# Crear directorios de trabajo y asignar permisos
RUN mkdir -p /app/data/models /app/data/voices /app/tmp/matplotlib /app/tmp/huggingface && \
    chown -R appuser:appuser /app/data /app/tmp

RUN chmod +x /app/docker/entrypoint.sh

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/opt/cosyvoice:/opt/cosyvoice/third_party/Matcha-TTS:/app:$PYTHONPATH
ENV NUMBA_DISABLE_JIT=1
ENV NUMBA_CACHE_DIR=/tmp
ENV MPLCONFIGDIR=/app/tmp/matplotlib
ENV TRANSFORMERS_CACHE=/app/tmp/huggingface
   
USER appuser

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


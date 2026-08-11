# Etapa de construcción
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Clonar CosyVoice e instalar sus dependencias específicas
RUN git clone --recursive https://github.com/QwenAudio/CosyVoice.git /opt/cosyvoice
WORKDIR /opt/cosyvoice

# FIX: Instalar setuptools para proporcionar pkg_resources (necesario para compilar openai-whisper)
RUN pip install --no-cache-dir --user --upgrade setuptools
RUN pip install --no-cache-dir --user -r requirements.txt

WORKDIR /app

# Etapa final
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copiar las dependencias de Python instaladas en el builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiar el código fuente de CosyVoice a la imagen final
COPY --from=builder /opt/cosyvoice /opt/cosyvoice

COPY --chown=appuser:appuser . .

RUN mkdir -p /app/data/models /app/data/voices && chown -R appuser:appuser /app/data

RUN chmod +x /app/docker/entrypoint.sh

ENV PATH=/home/appuser/.local/bin:$PATH
# Añadir CosyVoice y /app al PYTHONPATH para que los imports funcionen
ENV PYTHONPATH=/opt/cosyvoice:/app:$PYTHONPATH

USER appuser

ENTRYPOINT ["/app/docker/entrypoint.sh"]

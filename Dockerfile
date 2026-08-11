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

# Clonar e instalar CosyVoice en la etapa builder
RUN git clone https://github.com/QwenAudio/CosyVoice.git /opt/cosyvoice
WORKDIR /opt/cosyvoice
RUN git submodule update --init --recursive
RUN pip install --no-cache-dir --user .

WORKDIR /app

# Etapa final
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

RUN mkdir -p /app/data/models /app/data/voices && chown -R appuser:appuser /app/data

RUN chmod +x /app/docker/entrypoint.sh

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

USER appuser

ENTRYPOINT ["/app/docker/entrypoint.sh"]


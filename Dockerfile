# Usamos una imagen base con CUDA 12.1 y Ubuntu 22.04 (estándar para CosyVoice)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Configuración de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Instalar dependencias del sistema necesarias para CosyVoice y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    git \
    build-essential \
    cmake \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# 1. Copiar e instalar los requisitos BASE de TU proyecto primero
# (Esto permite cachear esta capa si solo cambias el código de CosyVoice)
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 2. Clonar el repositorio oficial de CosyVoice (rama main para soporte v3)
# Es crucial usar --recursive para descargar los submódulos (ej. Matcha-TTS)
RUN git clone --recursive https://github.com/QwenAudio/CosyVoice.git /app/CosyVoice

# 3. Configurar PYTHONPATH para incluir CosyVoice y sus submódulos
# Esto es vital porque CosyVoice ya no tiene setup.py, así que lo tratamos como un módulo fuente
ENV PYTHONPATH="${PYTHONPATH}:/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS"

# 4. Instalar las dependencias ESPECÍFICAS de CosyVoice desde su requirements.txt
RUN pip install --user -r /app/CosyVoice/requirements.txt

# 5. Copiar el resto de tu aplicación
COPY . .

# Asegurar permisos de ejecución en scripts propios
RUN chmod +x docker/entrypoint.sh

# Puerto por defecto
EXPOSE 8000

# Comando de entrada
CMD ["docker/entrypoint.sh"]

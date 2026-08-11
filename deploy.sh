#!/bin/bash
set -e # Salir inmediatamente si hay un error

# Colores para la salida
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Despliegue de GenPod (lightTTS / CosyVoice 2)  ${NC}"
echo -e "${GREEN}==================================================${NC}"

# 1. Verificar prerequisitos
echo -e "\n${YELLOW}[1/5] Verificando prerequisitos...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker no está instalado. Instálalo primero.${NC}"
    exit 1
fi
if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose no está instalado.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker y Docker Compose detectados.${NC}"

# Verificar NVIDIA (opcional)
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ GPU NVIDIA detectada.${NC}"
    HAS_GPU=true
else
    echo -e "${YELLOW}⚠️  No se detectó NVIDIA GPU. Se usará modo CPU.${NC}"
fi

# 2. Configuración del entorno
echo -e "\n${YELLOW}[2/5] Configurando entorno...${NC}"
if [ ! -f .env ]; then
    echo "Creando .env desde .env.example..."
    cp .env.example .env
fi

# Preguntar al usuario si hay GPU
if [ "$HAS_GPU" = true ]; then
    read -p "¿Deseas usar la GPU para la inferencia? (s/N): " USE_GPU
    if [[ "$USE_GPU" =~ ^[Ss]$ ]]; then
        sed -i 's/DEVICE=cpu/DEVICE=cuda/' .env
        echo -e "${GREEN}✅ Configurado para usar GPU (DEVICE=cuda).${NC}"
        
        # Verificar NVIDIA Container Toolkit
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo -e "${RED}⚠️  Advertencia: NVIDIA Container Toolkit no parece estar configurado en Docker.${NC}"
            echo -e "${YELLOW}   El despliegue podría fallar. Asegúrate de haber instalado nvidia-container-toolkit.${NC}"
        fi
    else
        echo -e "${YELLOW}✅ Configurado para usar CPU (DEVICE=cpu).${NC}"
    fi
else
    echo -e "${YELLOW}✅ Configurado para usar CPU (DEVICE=cpu).${NC}"
fi

# 3. Preparar directorios de datos
echo -e "\n${YELLOW}[3/5] Preparando directorios de datos...${NC}"
mkdir -p data/models data/voices
echo -e "${GREEN}✅ Directorios data/models y data/voices listos.${NC}"

# 4. Construir y arrancar
echo -e "\n${YELLOW}[4/5] Construyendo imagen Docker e iniciando servicios...${NC}"
echo -e "${YELLOW}⚠️  NOTA: La primera vez descargará el modelo CosyVoice 2 (~2-4GB). Esto puede tardar unos minutos.${NC}"

if [ "$HAS_GPU" = true ] && [[ "$USE_GPU" =~ ^[Ss]$ ]]; then
    echo "Arrancando con perfil GPU..."
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
else
    echo "Arrancando con perfil CPU..."
    docker compose up -d --build
fi

# 5. Verificación
echo -e "\n${YELLOW}[5/5] Verificando despliegue...${NC}"
sleep 5 # Dar un momento a que el contenedor inicie

if docker compose ps | grep -q "Up"; then
    echo -e "${GREEN}✅ ¡Contenedor arrancado correctamente!${NC}"
    echo -e "${GREEN}🌐 API disponible en: http://localhost:8000${NC}"
    echo -e "${GREEN}📚 Documentación Swagger en: http://localhost:8000/docs${NC}"
    
    echo -e "\n${YELLOW}Comprobando endpoint de salud...${NC}"
    curl -s http://localhost:8000/health | python3 -m json.tool || echo "⚠️  La API aún está cargando el modelo. Revisa los logs con: docker compose logs -f"
else
    echo -e "${RED}❌ El contenedor no se inició correctamente.${NC}"
    echo -e "${YELLOW}Revisa los logs con: docker compose logs --tail 50${NC}"
    exit 1
fi

echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}  ¡Despliegue completado con éxito!              ${NC}"
echo -e "${GREEN}==================================================${NC}"


#!/bin/bash
# scripts/aider_session.sh
# Portable: detecta automáticamente las rutas del proyecto.
# Uso: ./scripts/aider_session.sh [argumentos para aider]

set -euo pipefail

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# Detectar directorio del script y del proyecto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Función para logging consistente
log_info() { echo -e "${YELLOW}[INFO] $*${NC}"; }
log_success() { echo -e "${GREEN}[OK] $*${NC}"; }
log_warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }
log_error() { echo -e "${RED}[ERROR] $*${NC}"; }

# ── 1. Cargar .env si existe ──
if [ -f .env ]; then
  log_info "Cargando variables de entorno desde .env..."
  set -a
  source .env
  set +a
else
  log_warn "No se encontró .env. Las API keys deben estar en el entorno."
fi

# ── 2. Ejecutar pre_prompt.sh si existe ──
if [ -f scripts/pre_prompt.sh ]; then
  log_info "Generando contexto fresco..."
  bash scripts/pre_prompt.sh
else
  log_warn "scripts/pre_prompt.sh no encontrado. Continuando sin contexto fresco."
fi

# ── 3. Seleccionar modelo ──
MODEL_FILE=".aider_model_selection.yml"
if [ ! -f "$MODEL_FILE" ]; then
  log_warn "$MODEL_FILE no encontrado. Usando modelo por defecto."
  SELECTED_MODEL="openai/local-qwen"
else
  log_info "Modelos disponibles:"
  
  # Extraer modelos con python (más fiable que yq)
  python3 -c "
import yaml, sys
with open('$MODEL_FILE') as f:
    data = yaml.safe_load(f)
for i, m in enumerate(data.get('models', []), 1):
    print(f'  {i}. {m[\"name\"]} ({m[\"type\"]})')
"
  
  echo ""
  read -p "Selecciona un modelo (número, Enter para el primero): " CHOICE
  CHOICE=${CHOICE:-1}
  
  SELECTED_MODEL=$(python3 -c "
import yaml
with open('$MODEL_FILE') as f:
    data = yaml.safe_load(f)
models = data.get('models', [])
idx = $CHOICE - 1
if 0 <= idx < len(models):
    print(models[idx]['id'])
else:
    print('')
")
  
  if [ -z "$SELECTED_MODEL" ]; then
    log_warn "Selección inválida. Usando modelo por defecto."
    SELECTED_MODEL="openai/local-qwen"
  fi
  
  log_success "Modelo seleccionado: ${SELECTED_MODEL}"
  
  # Configurar variables de entorno específicas del modelo
  eval $(python3 -c "
import yaml, os
with open('$MODEL_FILE') as f:
    data = yaml.safe_load(f)
for m in data.get('models', []):
    if m['id'] == '$SELECTED_MODEL':
        for k, v in m.get('env', {}).items():
            expanded = os.path.expandvars(v)
            print(f'export {k}=\"{expanded}\"')
        break
")
fi

# ── 4. Construir argumentos de Aider ──
AIDER_ARGS=("--model" "$SELECTED_MODEL")

# Si hay un modelo weak/editor en el archivo, usarlos
WEAK_MODEL=$(python3 -c "
import yaml
with open('$MODEL_FILE') as f:
    data = yaml.safe_load(f)
for m in data.get('models', []):
    if m.get('type') == 'weak':
        print(m['id'])
        break
")
if [ -n "$WEAK_MODEL" ] && [ "$WEAK_MODEL" != "$SELECTED_MODEL" ]; then
  AIDER_ARGS+=("--weak-model" "$WEAK_MODEL")
fi

EDITOR_MODEL=$(python3 -c "
import yaml
with open('$MODEL_FILE') as f:
    data = yaml.safe_load(f)
for m in data.get('models', []):
    if m.get('type') == 'editor':
        print(m['id'])
        break
")
if [ -n "$EDITOR_MODEL" ]; then
  AIDER_ARGS+=("--editor-model" "$EDITOR_MODEL")
fi

# Añadir contexto fresco si se generó
if [ -f .aider_fresh_context.md ]; then
  AIDER_ARGS+=("--read" ".aider_fresh_context.md")
fi

# Añadir argumentos pasados al script
AIDER_ARGS+=("$@")

# Añadir argumentos extra desde variable de entorno
if [ -n "${AIDER_EXTRA_ARGS:-}" ]; then
  for arg in $AIDER_EXTRA_ARGS; do
    AIDER_ARGS+=("$arg")
  done
fi

# ── 5. Ejecutar Aider ──
log_info "Iniciando sesión con modelo ${SELECTED_MODEL}..."
log_info "Argumentos: ${AIDER_ARGS[*]}"
aider "${AIDER_ARGS[@]}"
exit_code=$?

# ── 6. Actualizar memoria ──
TEMPLATE="docs/AIDER_MEMORY_TEMPLATE.md"
OUTPUT="docs/AIDER_MEMORY.md"
PHASE_FILE=".aider_phase"

if [ -f "$TEMPLATE" ]; then
  log_info "Actualizando memoria en $OUTPUT..."
  
  DATE=$(date '+%d de %B de %Y %H:%M')
  BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
  RECENT_COMMITS=$(git log --oneline -5 2>/dev/null || echo "No git history")
  MODIFIED_FILES=$(git diff --name-only HEAD~5 2>/dev/null | sort -u | sed 's/^/  - /' || echo "  - (no git diff)")
  TODOS=$(grep -rn "TODO\|FIXME\|HACK" src/ --include="*.py" --include="*.js" --include="*.ts" --include="*.java" --include="*.go" --include="*.rs" 2>/dev/null | head -15 | sed 's/^/  - /' || echo "  - (none)")
  
  PHASE=$(cat "$PHASE_FILE" 2>/dev/null || echo "1 (Inicial)")
  STATE_FILE="docs/AIDER_STATE.md"
  STATE=$(cat "$STATE_FILE" 2>/dev/null || echo "Desarrollo activo de la Fase ${PHASE}.")
  
  cp "$TEMPLATE" "$OUTPUT"

  export AIDER_DATE="$DATE"
  export AIDER_PHASE="$PHASE"
  export AIDER_BRANCH="$BRANCH"
  export AIDER_STATE="$STATE"
  export AIDER_RECENT_COMMITS="$RECENT_COMMITS"
  export AIDER_MODIFIED_FILES="$MODIFIED_FILES"
  export AIDER_TODOS="$TODOS"

  # Generar OUTPUT usando Python (evita "argument list too long")
  python3 << 'PYEOF'
import os
from pathlib import Path

template = Path("docs/AIDER_MEMORY_TEMPLATE.md").read_text(encoding="utf-8")
output_file = Path("docs/AIDER_MEMORY.md")

replaces = {
    "{{DATE}}": os.environ.get("AIDER_DATE", ""),
    "{{PHASE}}": os.environ.get("AIDER_PHASE", ""),
    "{{BRANCH}}": os.environ.get("AIDER_BRANCH", ""),
    "{{STATE}}": os.environ.get("AIDER_STATE", ""),
    "{{RECENT_COMMITS}}": os.environ.get("AIDER_RECENT_COMMITS", ""),
    "{{MODIFIED_FILES}}": os.environ.get("AIDER_MODIFIED_FILES", ""),
    "{{TODOS}}": os.environ.get("AIDER_TODOS", ""),
}

for old, new in replaces.items():
    template = template.replace(old, new)

output_file.write_text(template, encoding="utf-8")
print("Memoria actualizada correctamente.")
PYEOF
  
  log_success "Memoria actualizada en $OUTPUT"
else
  log_warn "Plantilla $TEMPLATE no encontrada. Saltando actualización de memoria."
fi

log_success "Sesión finalizada (código $exit_code)."
exit $exit_code

# Terminus Media Server (TMS) - Arquitectura Base v3.0

**Fecha:** 25 de Julio de 2026
**Estado:** Definición Arquitectónica Aprobada
**Contexto:** Proyecto personal, despliegue Docker, backend SeaweedFS existente, desarrollo asistido por IA (Aider + Litellm + OpenRouter).

---

## 1. Visión General y Principios

**Terminus Media Server (TMS)** es un servidor de medios doméstico autocontenido en Docker. Su propósito es ofrecer navegación, búsqueda, adquisición y reproducción de contenido a través de una **Interfaz de Usuario de Terminal (TUI) auténtica**, tanto en navegador web como en Termux.

### Principios Fundamentales

1. **No Invasivo:** Los archivos multimedia originales nunca se modifican. Los metadatos se gestionan en rutas externas (`/data/metadata/`).
2. **TUI como Núcleo:** La interfaz de terminal es una capa arquitectónica reutilizable denominada "Terminus Core".
3. **Compatibilidad con Ecosistema:** Se reutilizará la lógica de plugins de Kodi mediante un adaptador en el worker.
4. **Autocontenido:** Despliegue con `docker compose up`. Configuración íntegra por variables de entorno.

---

## 2. Filosofía de la Interfaz de Usuario (TUI)

### 2.1. Terminus Core (Framework TUI Base)

Se desarrollará un framework interno agnóstico a la lógica de negocio para construir aplicaciones TUI.

- **Motor de Renderizado (Frontend Web):** Basado en `xterm.js` + `@xterm/addon-fit`. Una librería de widgets TUI gestionará ventanas, listas, inputs y diálogos renderizados como caracteres de terminal.
- **Cliente Nativo (Termux/Shell):** Aplicación en Go independiente. Utilizará `tview` para renderizar la misma estructura de widgets. Es un cliente "tonto" que pinta lo que el backend le indica.
- **Protocolo de Comunicación:** WebSocket bidireccional. El backend envía instrucciones de renderizado abstractas (JSON) y el cliente las traduce a su toolkit TUI nativo. El cliente envía eventos de teclado/ratón al backend.

[Usuario] <--> [Terminal (xterm.js / App Nativa)] <--WebSocket--> [Backend API (Lógica de UI)]
text


### 2.2. Experiencia de Reproducción de Vídeo

- **Regla:** El vídeo NUNCA se incrusta dentro de la terminal basada en texto.
- **Cliente Web:** El backend notifica al frontend con una URL de stream. Se abre una ventana superpuesta (`<div>` flotante) con un reproductor HTML5 (Video.js). La terminal subyacente queda visible.
- **Cliente Nativo (Termux):** El cliente recibe la URL del stream y lanza un reproductor externo configurable (mpv, VLC) como proceso hijo. La terminal queda libre.

### 2.3. Vistas Principales y Navegación

La interfaz debe ser operable 100% por teclado. El ratón funciona como entrada secundaria en el terminal emulado del navegador.

- Menú Principal
- Explorador de Colecciones (vista de rejilla/lista)
- Búsqueda (barra de filtrado superior, resultados dinámicos)
- Detalle del Medio (sinopsis, reparto, carátula ASCII opcional)
- Reproductor (solo barra de progreso y metadatos; el vídeo en ventana aparte)
- Configuración y Administración

---

## 3. Decisiones Arquitectónicas (ADRs)

### ADR-001: Estrategia de Streaming y Transcodificación (Opción C)

- **Estado:** Aceptado
- **Contexto:** Las pruebas de transcodificación solo con CPU no son satisfactorias. El sistema debe ser honesto con el usuario sobre las capacidades del hardware.
- **Decisión:** Flujo de decisión en cascada para la reproducción:
    1.  **Direct Play:** Si el códec del archivo es compatible con el cliente, `tms-api` genera una Presigned URL de SeaweedFS. El cliente descarga directo.
    2.  **Pre-transcodificación (CPU asíncrona):** Si no es compatible y no hay GPU, se rechaza la reproducción inmediata. Se ofrece "Pre-transcodificar". El worker transcodifica completo a SeaweedFS y, al acabar, `tms-api` entrega una Presigned URL.
    3.  **Transcodificación al vuelo (GPU):** Si no es compatible y `ENABLE_GPU=true`, `tms-worker` ejecuta FFmpeg emitiendo a stdout. `tms-api` actúa de proxy, transmitiendo al cliente vía HTTP Chunked Transfer o HLS. **Prohibido usar Presigned URLs para streaming en vivo.**

### ADR-002: Integración con SeaweedFS y Ciclo de Vida de Caché

- **Estado:** Aceptado
- **Contexto:** Usar Redis como buffer de streaming es un anti-patrón para flujos binarios de alto throughput.
- **Decisión:** Redis queda liberado exclusivamente para sesiones, colas de tareas (Celery) y estado UI.
- **Caché:** `tms-worker` escribe la caché de transcodificación en SeaweedFS. Debe ejecutar un cron interno que elimine archivos de caché con `last_accessed > 24h`.

### ADR-003: Motor de Torrents y Privacidad

- **Estado:** Aceptado
- **Contexto:** WebTorrent en backend obligaría a introducir Node.js. Las versiones recientes de `libtorrent` soportan el protocolo WebTorrent nativamente.
- **Decisión:** Mantener `libtorrent` (libtorrent-rasterbar) en `tms-worker`. Se habilitará `sequential_download=True` para permitir streaming durante la descarga.
- **Privacidad y Seeding:** TMS no añade telemetría. Por defecto, `seed_mode=False` y se detendrá la descarga al alcanzar el 100% (ratio 0) para no consumir ancho de banda del usuario en segundo plano.

### ADR-004: Convenciones de Desarrollo

- **Estado:** Aceptado
- **Decisión:** Arquitectura Hexagonal estricta. Tests como especificación. Límite de 300 LOC por archivo. Tipado estricto (`mypy --strict`). Uso obligatorio de `ruff` para formateo antes de commits.

### ADR-006: Estrategia de logging
- **Estado:** Aceptado
- **Contexto:** Se necesita un sistema de trazabilidad homogéneo para todos los componentes (backend, worker, clientes).
- **Decisión:** Logging estándar de Python (`logging`) y `console` en JavaScript, controlado por variables de entorno `TMS_LOG_LEVEL` y `TMS_LOG_FORCE`.


---

## 4. Diagramas C4 (Modelo Mermaid)

### 4.1. Nivel 1: Diagrama de Contexto

```mermaid
C4Context
  title C4 Nivel 1: Contexto del Sistema TMS

  Person(user_web, "Usuario (Web)", "Navega y reproduce desde navegador moderno.")
  Person(user_termux, "Usuario (Termux)", "Navega y reproduce desde Android/Termux.")
  
  System(tms, "Terminus Media Server (TMS)", "Servidor de medios autocontenido con TUI.")
  
  System_Ext(seaweedfs, "SeaweedFS", "Almacenamiento distribuido (S3).")
  System_Ext(tmdb, "TheMovieDB / TVDB", "API externa para metadatos.")
  System_Ext(opensub, "OpenSubtitles", "API externa para subtítulos.")

  Rel(user_web, tms, "Interactúa vía", "WebSocket / HTTPS")
  Rel(user_termux, tms, "Interactúa vía", "WebSocket")
  Rel(tms, seaweedfs, "Lee/Escribe medios y caché", "S3 API / HTTP Range")
  Rel(tms, tmdb, "Consulta metadatos", "HTTPS REST")
  Rel(tms, opensub, "Busca subtítulos", "HTTPS REST")

4.2. Nivel 2: Diagrama de Contenedores

C4Container
  title C4 Nivel 2: Contenedores Docker de TMS

  Person(user, "Usuario", "Admin o Estándar")

  System_Boundary(tms_boundary, "TMS (Docker Compose)") {
    Container(proxy, "tms-proxy", "Nginx/Traefik", "Proxy inverso, SSL, enrutamiento.")
    Container(api, "tms-api", "Python 3.12 + FastAPI", "API, Auth, URLs firmadas, Proxy de stream en vivo.")
    Container(worker, "tms-worker", "Python + FFmpeg + libtorrent", "Transcodificación, scraping, torrents.")
    ContainerDb(db, "tms-db", "PostgreSQL 16", "Persistencia.")
    ContainerQueue(redis, "tms-redis", "Redis 7", "Colas y sesiones.")
  }

  System_Ext(seaweedfs, "SeaweedFS", "Almacenamiento S3 externo")

  Rel(user, proxy, "HTTPS / WSS")
  Rel(proxy, api, "HTTP / WebSocket")
  Rel(api, db, "SQL", "Lectura/Escritura")
  Rel(api, redis, "Pub/Sub & Cache")
  Rel(api, worker, "Encola tareas", "Redis Queue")
  
  Rel(api, seaweedfs, "Genera URLs firmadas (Direct Play)", "S3 API")
  Rel(worker, seaweedfs, "Lee/Escribe caché transcodificación", "S3 API")
  Rel(worker, api, "Stream binario (Transcodificación GPU al vuelo)", "HTTP Chunked / HLS")

5. Esquema de Base de Datos (PostgreSQL)
sql

-- Habilitar extensión para búsquedas sin acentos
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Usuarios
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'standard' CHECK (role IN ('admin', 'standard')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Repositorios (colecciones)
CREATE TABLE repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    type VARCHAR(20) NOT NULL CHECK (type IN ('local', 's3', 'nas')),
    path VARCHAR(500) NOT NULL,
    config JSONB DEFAULT '{}',
    is_readonly BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Elementos multimedia
CREATE TABLE media_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    original_title VARCHAR(255),
    year INT,
    media_type VARCHAR(20) NOT NULL CHECK (media_type IN ('movie', 'show', 'episode')),
    show_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    season_number INT,
    episode_number INT,
    tmdb_id VARCHAR(50),
    file_path VARCHAR(500) NOT NULL,
    metadata_path VARCHAR(500),
    video_codec VARCHAR(20),
    audio_codec VARCHAR(20),
    file_hash VARCHAR(16),
    file_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_media_repository ON media_items(repository_id);
CREATE INDEX idx_media_tmdb ON media_items(tmdb_id);
CREATE INDEX idx_media_search ON media_items USING GIN (
    to_tsvector('spanish', title || ' ' || COALESCE(original_title, ''))
);

-- Progreso de usuarios
CREATE TABLE user_progress (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    media_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'no_visto' CHECK (status IN ('no_visto', 'viendo', 'completado')),
    timestamp_seconds INT DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, media_id)
);

CREATE INDEX idx_progress_user ON user_progress(user_id);

-- Caché de subtítulos
CREATE TABLE subtitle_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    media_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    language VARCHAR(10) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    downloaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(media_id, language, provider)
);

-- Trigger para actualizar automáticamente updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_media_items_updated_at 
    BEFORE UPDATE ON media_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_progress_updated_at 
    BEFORE UPDATE ON user_progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

6. Estructura de Directorios del Proyecto
text

tms/
├── .aider.conf.yml
├── .gitignore
├── CONVENTIONS.md
├── docker-compose.yml
├── docs/
│   └── TMS_Arquitectura_Base.md
├── src/
│   ├── tms-api/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py
│   │       ├── domain/
│   │       ├── application/
│   │       ├── infrastructure/
│   │       └── interfaces/
│   ├── tms-worker/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py
│   │       ├── transcoder/
│   │       ├── torrent/
│   │       ├── scraper/
│   │       └── subtitles/
│   └── tms-client-go/
│       ├── go.mod
│       ├── main.go
│       └── tui/
└── tests/
    ├── unit/
    └── integration/

7. Stack Tecnológico
Componente	Tecnología	Versión
API Backend	Python + FastAPI	3.12 / 0.115+
Worker	Python + Celery	3.12
Cliente Nativo	Go + tview	1.23+
Frontend Web	xterm.js + Video.js	5.x / 8.x
Base de Datos	PostgreSQL	16
Caché/Colas	Redis	7
Almacenamiento	SeaweedFS (S3 API)	3.x
Transcodificación	FFmpeg	7.x
Torrents	libtorrent-rasterbar	2.x
Proxy Inverso	Nginx o Traefik	latest

Próximos Pasos:

    Revisar esta especificación y resolver dudas en reunión de kick-off.

    Elaborar diagramas de arquitectura (C4 Model: Contexto, Contenedores, Componentes).

    Desglosar la Fase 1 en issues del backlog (GitHub Projects / Jira).

    Definir el esquema de base de datos inicial (entidades: Usuario, Repositorio, Media, Progreso).

    Prototipar el protocolo de renderizado TUI (JSON schema para widgets).


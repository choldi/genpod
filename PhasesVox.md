# Plan de Implementación: Soporte para VoxCPM en LightTTS

## Resumen Ejecutivo

Este documento describe el plan por fases para añadir soporte completo para el modelo **VoxCPM** en LightTTS, manteniendo compatibilidad total con los modelos CosyVoice 2/3 existentes.

## Estado Actual

- [x] Parámetro `model` presente en API (`TTSRequest`) y `engine.synthesize()`
- [x] Parámetro `model` implementado en la lógica de síntesis
- [x] Soporte para múltiples modelos cargados simultáneamente
- [x] Sintetizadores desacoplados con despacho por `model_type`
- [x] `VoiceRegistry` asocia voces a modelos con filtrado por modelo

---

## Fase 1: Refactorización Multi-Modelo 🎯

**Objetivo**: Arquitectura base para soportar múltiples modelos simultáneamente.

### Entregables Completados

| Componente | Cambios | Estado |
|------------|---------|--------|
| `ModelLoader` | Carga múltiple, caché, descubrimiento automático | [x] |
| `VoiceRegistry` | Asociación voz-modelo, filtrado por modelo | [x] |
| `BaseSynthesizer` | Parámetro `model_type`, despacho por modelo | [x] |
| `ClonedSynthesizer` | Parámetro `model_type`, despacho por modelo | [x] |
| `LightTTSEngine` | Carga perezosa, selección por voz/parámetro, caché sintetizadores | [x] |

### Características Implementadas

- [x] **Carga perezosa (Lazy Loading)**: Los modelos se cargan solo cuando se necesitan
- [x] **Caché de modelos**: Múltiples modelos en memoria simultáneamente
- [x] **Selección automática de modelo**: 
- [ ] Parámetro explícito `model` en `synthesize()`
- [ ] Inferencia desde `voice_id` (formato `model_type:speaker_id`)
- [ ] Fallback a modelo por defecto
- [x] **API extendida**:
- [ ] `get_available_models()` - Lista modelos detectados
- [ ] `list_voices(model_type)` - Filtrado por modelo
- [ ] `unload_model(model_type)` - Gestión de memoria
- [ ] `clone_voice(..., model=...)` - Asociación modelo-voz al clonar

---

## Fase 2: Soporte Completo para VoxCPM 🚀

**Objetivo**: Implementar soporte completo para VoxCPM con compatibilidad total con CosyVoice 2/3.

### Entregables Pendientes

| Componente | Cambios | Estado |
|------------|---------|--------|
| `TTSRequest` | Añadir parámetro `model` explícito | [ ] |
| `LightTTSEngine` | Lógica de selección de modelo por `voice_id` | [ ] |
| `VoiceRegistry` | Fallback a modelo por defecto | [ ] |
| `BaseSynthesizer` | Soporte para VoxCPM | [ ] |
| `ClonedSynthesizer` | Soporte para VoxCPM | [ ] |
| `API` | Métodos `get_available_models()` y `list_voices(model_type)` | [ ] |

### Características Implementadas en esta Fase

- [x] **Parámetro explícito `model` en `synthesize()`**: 
- [x] **Inferencia desde `voice_id`** (formato `model_type:speaker_id`)
- [x] **Fallback a modelo por defecto** cuando no se especifica `model` ni se puede inferir
- [x] **Soporte para VoxCPM** en todos los componentes
- [x] **API extendida**:
  - `get_available_models()` - Devuelve lista de modelos detectados
  - `list_voices(model_type)` - Filtra voces por tipo de modelo
  - `unload_model(model_type)` - Libera memoria de modelos no usados
  - `clone_voice(..., model=...)` - Asocia modelo específico al clonar voz

### Consideraciones Adicionales

- [x] **Compatibilidad cruzada**: Todos los modelos (CosyVoice 2/3, VoxCPM) deben ser tratados de forma uniforme
- [x] **Gestión de errores**: Validación de modelos y manejo de excepciones específicas
- [x] **Documentación**: Actualización de la documentación para reflejar los nuevos parámetros y funcionalidades

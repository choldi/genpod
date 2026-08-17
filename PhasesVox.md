# Plan de Implementación: Soporte para VoxCPM en LightTTS

## Resumen Ejecutivo

Este documento describe el plan por fases para añadir soporte completo para el modelo **VoxCPM** en LightTTS, manteniendo compatibilidad total con los modelos CosyVoice 2/3 existentes.

## Estado Actual

- ✅ Parámetro `model` presente en API (`TTSRequest`) y `engine.synthesize()`
- ❌ Parámetro `model` ignorado en implementación actual
- ❌ Solo un modelo cargado a la vez (`self._model`)
- ❌ Sintetizadores acoplados a CosyVoice
- ❌ `VoiceRegistry` no asocia voces a modelos

---

## Fase 1: Refactorización Multi-Modelo (EN PROGRESO) 🎯

**Objetivo**: Arquitectura base para soportar múltiples modelos simultáneamente.

### Entregables Completados

| Componente | Cambios | Estado |
|------------|---------|--------|
| `ModelLoader` | Carga múltiple, caché, descubrimiento automático | ✅ |
| `VoiceRegistry` | Asociación voz-modelo, filtrado por modelo | ✅ |
| `BaseSynthesizer` | Parámetro `model_type`, despacho por modelo | ✅ |
| `ClonedSynthesizer` | Parámetro `model_type`, despacho por modelo | ✅ |
| `LightTTSEngine` | Carga perezosa, selección por voz/parámetro, caché sintetizadores | ✅ |

### Características Implementadas

1. **Carga perezosa (Lazy Loading)**: Los modelos se cargan solo cuando se necesitan
2. **Caché de modelos**: Múltiples modelos en memoria simultáneamente
3. **Selección automática de modelo**: 
   - Parámetro explícito `model` en `synthesize()`
   - Inferencia desde `voice_id` (formato `model_type:speaker_id`)
   - Fallback a modelo por defecto
4. **API extendida**:
   - `get_available_models()` - Lista modelos detectados
   - `list_voices(model_type)` - Filtrado por modelo
   - `unload_model(model_type)` - Gestión de memoria
   - `clone_voice(..., model=...)` - Asociación modelo-voz al clonar

### Formato de Voice ID


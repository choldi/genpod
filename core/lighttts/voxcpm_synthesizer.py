from typing import Any
from pydantic import BaseSettings
import os

class VoxCPMSynthesizer(BaseSynthesizer):
    """Synthesizer for VoxCPM models with streaming and emotion control support."""
    
    def __init__(
        self,
        model: str,
        voices_path: str,
        voice_manager: VoiceManager,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(device, **kwargs)
        self.model = model
        self.voices_path = voices_path
        self.voice_manager = voice_manager
        self._check_model_exists()
    
    def _check_model_exists(self):
        """Verificar si el modelo existe en el directorio de modelos."""
        model_path = os.path.join("./data/models", f"{self.model}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo '{self.model}' no encontrado")

    def synthesize(self, text: str) -> bytes:
        """Synthesize the given text using the selected model."""
        # Verificar si el modelo existe
        self._check_model_exists()
        
        # Realizar la síntesis de voz
        # ...

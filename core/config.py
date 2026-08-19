from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    # Rutas por defecto para contenedor Docker
    MODELS_PATH: str = "/models"
    VOICES_PATH: str = "/voices"
    
    # Configuración de modelo legacy (compatibilidad)
    model_path: str = "./data/models/voxcpm_v1.0.pth"
    model_version: str = "v1.0"
    device: str = "cuda"

    model_config = ConfigDict(
        protected_namespaces=(),  # Evita warning con campos que empiezan por "model_"
        extra="ignore"            # Ignora variables de entorno no definidas
    )

    @property
    def DEVICE(self) -> str:
        """Alias en mayúsculas para compatibilidad con código que use settings.DEVICE"""
        return self.device


# Crear la instancia que se importa en otros módulos
settings = Settings()

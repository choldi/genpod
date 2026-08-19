from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "./data/models/voxcpm_v1.0.pth"
    model_version: str = "v1.0"
    device: str = "cuda"
    # ... otras configuraciones ...

    @property
    def DEVICE(self) -> str:
        """Alias en mayúsculas para compatibilidad con código que use settings.DEVICE"""
        return self.device

# Crear la instancia que se importa en otros módulos
settings = Settings()

"""Model loading and detection for LightTTS."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import torch

from core.config import settings
from core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles CosyVoice model loading and version detection."""

    def __init__(self, models_path: str, device: str) -> None:
        self.models_path = Path(models_path)
        self.device = device
        self._model: Optional[object] = None
        self._model_version: str = "unknown"
        self._load_wav = None

    def load(self) -> Tuple[object, str, callable]:
        """Load the CosyVoice model (supports v1, v2, and v3)."""
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3
            from cosyvoice.utils.file_utils import load_wav

            self._load_wav = load_wav

            model_dir = self._detect_model_dir()
            logger.info(f"Loading model from {model_dir} on {self.device}...")

            if (model_dir / "cosyvoice3.yaml").exists():
                logger.info("Detected CosyVoice 3 model")
                self._model = CosyVoice3(str(model_dir), fp16=(self.device == "cuda"))
                self._model_version = "v3"
            elif (model_dir / "cosyvoice2.yaml").exists():
                logger.info("Detected CosyVoice 2 model")
                self._model = CosyVoice2(str(model_dir), load_jit=False, fp16=(self.device == "cuda"))
                self._model_version = "v2"
            else:
                logger.info("Detected CosyVoice v1 (SFT) model")
                self._model = CosyVoice(str(model_dir), load_jit=False, fp16=(self.device == "cuda"))
                self._model_version = "v1"

            logger.info(f"Model loaded successfully (version: {self._model_version})")
            return self._model, self._model_version, self._load_wav

        except ImportError as e:
            raise ModelLoadError(
                "CosyVoice library not installed or outdated. "
                "Please ensure the latest version is cloned and PYTHONPATH is set correctly."
            ) from e
        except Exception as e:
            raise ModelLoadError(f"Failed to load CosyVoice model: {e}") from e

    def _detect_model_dir(self) -> Path:
        """Detect which CosyVoice model version is available."""
        if (self.models_path / "CosyVoice3-0.5B" / "cosyvoice3.yaml").exists():
            return self.models_path / "CosyVoice3-0.5B"
        if (self.models_path / "CosyVoice2-0.5B" / "cosyvoice2.yaml").exists():
            return self.models_path / "CosyVoice2-0.5B"
        if (self.models_path / "CosyVoice-300M-SFT" / "cosyvoice.yaml").exists():
            return self.models_path / "CosyVoice-300M-SFT"

        model_dirs = list(self.models_path.glob("CosyVoice*"))
        if model_dirs:
            return model_dirs[0]

        raise ModelLoadError(
            f"No CosyVoice model found in {self.models_path}. "
            "Please download the model weights first."
        )

    @property
    def model(self) -> Optional[object]:
        return self._model

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def load_wav(self) -> Optional[callable]:
        return self._load_wav

"""Model Loader for CosyVoice and VoxCPM models."""

import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Callable
import torch

from core.config import settings
from core.exceptions import ModelLoadError
from core.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Handles CosyVoice and VoxCPM model loading and version detection."""

    # Model configurations
    MODEL_CONFIGS = {
        "cosyvoice2": {
            "model_dir_name": "CosyVoice2-0.5B",
            "model_file_patterns": ["*.pt", "*.bin", "*.safetensors"],
            "load_func": "load_cosyvoice2",
        },
        "cosyvoice3": {
            "model_dir_name": "CosyVoice3-0.5B",
            "model_file_patterns": ["*.pt", "*.bin", "*.safetensors"],
            "load_func": "load_cosyvoice3",
        },
        "voxcpm": {
            "model_dir_name": "VoxCPM",
            "model_file_patterns": ["*.pt", "*.bin", "*.safetensors", "*.json", "*.yaml", "*.yml"],
            "load_func": "load_voxcpm",
        },
    }

    def __init__(self, models_path: str, device: str = "cpu") -> None:
        """Initialize the model loader.
        
        Args:
            models_path: Path to models directory
            device: Device to load models on (cpu, cuda, mps)
        """
        self.models_path = Path(models_path)
        self.device = device
        self._loaded_models: Dict[str, Tuple[Any, str, Callable]] = {}
        self._model_configs: Dict[str, Dict] = {}
        
        # Discover available models
        self._discover_models()

    def _discover_models(self) -> None:
        """Discover available models in the models directory."""
        logger.info(f"Discovering models in {self.models_path}")
        
        for model_type, config in self.MODEL_CONFIGS.items():
            model_dir = self.models_path / config["model_dir_name"]
            if model_dir.exists():
                model_files = []
                for pattern in config["model_file_patterns"]:
                    model_files.extend(list(model_dir.glob(pattern)))
                
                if model_files:
                    self._model_configs[model_type] = {
                        "path": model_dir,
                        "files": model_files,
                        "config": config,
                    }
                    logger.info(f"Found {model_type} model at {model_dir}")
                else:
                    logger.warning(f"Model directory exists but no model files found: {model_dir}")
            else:
                logger.debug(f"Model directory not found: {model_dir}")

    def get_available_models(self) -> Dict[str, Dict]:
        """Get information about available models."""
        return {
            model_type: {
                "path": str(info["path"]),
                "files": [str(f) for f in info["files"]],
                "type": model_type,
            }
            for model_type, info in self._model_configs.items()
        }

    def load(self, model_type: str = "cosyvoice2") -> Tuple[Any, str, Callable]:
        """Load a specific model type.
        
        Args:
            model_type: Type of model to load (cosyvoice2, cosyvoice3, voxcpm)
            
        Returns:
            Tuple of (model, version, load_wav_function)
            
        Raises:
            ModelLoadError: If model type not available or loading fails
        """
        if model_type in self._loaded_models:
            logger.info(f"Returning cached {model_type} model")
            return self._loaded_models[model_type]

        if model_type not in self._model_configs:
            available = list(self._model_configs.keys())
            raise ModelLoadError(
                f"Model type '{model_type}' not available. Available models: {available}"
            )

        logger.info(f"Loading {model_type} model...")
        model_info = self._model_configs[model_type]
        model_path = model_info["path"]
        config = model_info["config"]

        try:
            # Dynamic import to avoid loading all model dependencies at once
            if model_type.startswith("cosyvoice"):
                model, version, load_wav = self._load_cosyvoice(model_path, model_type)
            elif model_type == "voxcpm":
                model, version, load_wav = self._load_voxcpm(model_path)
            else:
                raise ModelLoadError(f"Unknown model type: {model_type}")

            self._loaded_models[model_type] = (model, version, load_wav)
            logger.info(f"{model_type} model loaded successfully (version: {version})")
            return model, version, load_wav

        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}", exc_info=True)
            raise ModelLoadError(f"{model_type} model loading failed: {e}") from e

    def _load_cosyvoice(self, model_path: Path, model_type: str) -> Tuple[Any, str, Callable]:
        """Load CosyVoice 2 or 3 model."""
        # Import here to avoid dependency issues if not installed
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice3
            from cosyvoice.utils.file_utils import load_wav
        except ImportError as e:
            raise ModelLoadError(f"CosyVoice not installed: {e}") from e

        if model_type == "cosyvoice2":
            model = CosyVoice2(str(model_path), load_jit=False, load_trt=False, fp16=False)
            version = "cosyvoice2"
        elif model_type == "cosyvoice3":
            model = CosyVoice3(str(model_path), load_jit=False, load_trt=False, fp16=False)
            version = "cosyvoice3"
        else:
            raise ModelLoadError(f"Unknown CosyVoice type: {model_type}")

        return model, version, load_wav

    def _load_voxcpm(self, model_path: Path) -> Tuple[Any, str, Callable]:
        """Load VoxCPM model."""
        try:
            # Try to import VoxCPM - this assumes a typical structure
            # Adjust imports based on actual VoxCPM package structure
            import sys
            sys.path.insert(0, str(model_path))
            
            # Try different possible import patterns for VoxCPM
            try:
                from voxcpm import VoxCPM
            except ImportError:
                try:
                    from models.voxcpm import VoxCPM
                except ImportError:
                    try:
                        from voxcpm.model import VoxCPM
                    except ImportError as e:
                        raise ModelLoadError(f"VoxCPM module not found. Tried multiple import patterns: {e}") from e
            
            # Load model - VoxCPM might use different loading patterns
            # Check for config file first
            config_files = list(model_path.glob("*.yaml")) + list(model_path.glob("*.yml")) + list(model_path.glob("*.json"))
            config_path = config_files[0] if config_files else None
            
            # Check for model weights
            model_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
            if not model_files:
                raise ModelLoadError(f"No model weights found in {model_path}")
            
            model_weights = model_files[0]
            
            # Initialize VoxCPM model
            # This assumes VoxCPM has a from_pretrained or similar method
            if hasattr(VoxCPM, 'from_pretrained'):
                model = VoxCPM.from_pretrained(str(model_path), device=self.device)
            elif hasattr(VoxCPM, 'load_from_checkpoint'):
                model = VoxCPM.load_from_checkpoint(str(model_weights), map_location=self.device)
            else:
                # Fallback: direct instantiation
                model = VoxCPM(str(model_path), device=self.device)
            
            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()
            
            # Get version info
            version = "voxcpm-1.0"
            if config_path:
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    version = config.get('version', version)
                except Exception:
                    pass
            
            # Define load_wav function for VoxCPM (uses torchaudio typically)
            def load_wav_voxcpm(wav_path: str) -> torch.Tensor:
                import torchaudio
                waveform, sample_rate = torchaudio.load(wav_path)
                # Resample to 24kHz if needed (VoxCPM default)
                if sample_rate != 24000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                    waveform = resampler(waveform)
                return waveform
            
            logger.info(f"VoxCPM model loaded from {model_path}")
            return model, version, load_wav_voxcpm
            
        except ImportError as e:
            raise ModelLoadError(f"VoxCPM not installed or import failed: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load VoxCPM model: {e}", exc_info=True)
            raise ModelLoadError(f"VoxCPM model loading failed: {e}") from e

    def unload_model(self, model_type: str) -> bool:
        """Unload a specific model from memory."""
        if model_type in self._loaded_models:
            del self._loaded_models[model_type]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded {model_type} model")
            return True
        return False

    def unload_all(self) -> None:
        """Unload all models."""
        self._loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded all models")

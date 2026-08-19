"""Model Loader for CosyVoice and VoxCPM models with robust loading and health checks."""

import logging
import os
import time
import subprocess
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Callable, List
import torch

from core.config import settings
from core.exceptions import ModelLoadError, GPUOutOfMemoryError
from core.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Handles CosyVoice and VoxCPM model loading with package management and health checks."""

    # Model configurations with package requirements
    MODEL_CONFIGS = {
        "cosyvoice2": {
            "model_dir_name": "CosyVoice2-0.5B",
            "model_file_patterns": ["*.pt", "*.bin", "*.safetensors"],
            "package_name": "cosyvoice",
            "min_version": "0.1.0",
            "load_func": "load_cosyvoice2",
        },
        "cosyvoice3": {
            "model_dir_name": "CosyVoice3-0.5B",
            "model_file_patterns": ["*.pt", "*.bin", "*.safetensors"],
            "package_name": "cosyvoice",
            "min_version": "0.1.0",
            "load_func": "load_cosyvoice3",
        },
        "voxcpm": {
            "model_dir_name": "VoxCPM2",
            "model_file_patterns": ["*.pth", "*.bin", "*.safetensors", "*.json", "*.yaml", "*.yml"],
            "package_name": "voxcpm",
            "min_version": "1.0.0",
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
        self._package_cache: Dict[str, bool] = {}
        
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
                    logger.info(f"Found {model_type} model at {model_dir} ({len(model_files)} files)")
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
                "package": info["config"]["package_name"],
            }
            for model_type, info in self._model_configs.items()
        }

    def ensure_package_installed(self, model_type: str) -> bool:
        """Ensure required package is installed for model type."""
        if model_type in self._package_cache:
            return self._package_cache[model_type]
        
        config = self.MODEL_CONFIGS.get(model_type)
        if not config:
            logger.error(f"Unknown model type: {model_type}")
            return False
        
        package_name = config["package_name"]
        min_version = config.get("min_version", "0.0.0")
        
        try:
            # Check if package is already installed
            spec = importlib.util.find_spec(package_name)
            if spec is not None:
                # Check version
                module = importlib.import_module(package_name)
                version = getattr(module, "__version__", "0.0.0")
                logger.info(f"Package {package_name} v{version} already installed")
                self._package_cache[model_type] = True
                return True
        except ImportError:
            pass
        
        # Try to install package
        logger.info(f"Installing package {package_name} (>= {min_version})...")
        try:
            # Use pip to install
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                f"{package_name}>={min_version}",
                "--no-cache-dir"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name}")
                self._package_cache[model_type] = True
                return True
            else:
                logger.error(f"Failed to install {package_name}: {result.stderr}")
                self._package_cache[model_type] = False
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout installing {package_name}")
            self._package_cache[model_type] = False
            return False
        except Exception as e:
            logger.error(f"Error installing {package_name}: {e}")
            self._package_cache[model_type] = False
            return False

    def load(self, model_type: str = "cosyvoice2") -> Tuple[Any, str, Callable]:
        """Load a specific model type with health checks.
        
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
                f"Model type '{model_type}' not available. Available models: {available}",
                model_type=model_type
            )

        # Ensure package is installed
        if not self.ensure_package_installed(model_type):
            raise ModelLoadError(
                f"Required package for {model_type} not available",
                model_type=model_type
            )

        logger.info(f"Loading {model_type} model...")
        model_info = self._model_configs[model_type]
        model_path = model_info["path"]
        config = model_info["config"]

        # Retry logic for transient errors
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Dynamic import to avoid loading all model dependencies at once
                if model_type.startswith("cosyvoice"):
                    model, version, load_wav = self._load_cosyvoice(model_path, model_type)
                elif model_type == "voxcpm":
                    model, version, load_wav = self._load_voxcpm(model_path)
                else:
                    raise ModelLoadError(f"Unknown model type: {model_type}", model_type=model_type)

                # Health check
                if not self._health_check(model, model_type):
                    raise ModelLoadError(f"Health check failed for {model_type}", model_type=model_type)

                self._loaded_models[model_type] = (model, version, load_wav)
                logger.info(f"{model_type} model loaded successfully (version: {version})")
                return model, version, load_wav

            except GPUOutOfMemoryError:
                # Don't retry OOM errors
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {model_type}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                # Clear CUDA cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.error(f"Failed to load {model_type} model after {max_retries} attempts: {last_error}")
        raise ModelLoadError(f"{model_type} model loading failed: {last_error}", model_type=model_type) from last_error

    def _health_check(self, model: Any, model_type: str) -> bool:
        """Perform basic health check on loaded model."""
        try:
            # Check model has required attributes
            if model_type.startswith("cosyvoice"):
                # CosyVoice models should have inference methods
                required_attrs = ["inference_sft", "inference_zero_shot"]
                for attr in required_attrs:
                    if not hasattr(model, attr):
                        logger.warning(f"Model missing attribute: {attr}")
                        return False
            elif model_type == "voxcpm":
                # VoxCPM should have generate method
                if not hasattr(model, "generate") and not hasattr(model, "inference"):
                    logger.warning("VoxCPM model missing generate/inference method")
                    return False
            
            # Check model is on correct device
            if hasattr(model, "device"):
                if str(model.device) != self.device and self.device != "cpu":
                    logger.warning(f"Model device mismatch: {model.device} vs {self.device}")
            
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def _load_cosyvoice(self, model_path: Path, model_type: str) -> Tuple[Any, str, Callable]:
        """Load CosyVoice 2 or 3 model with error handling."""
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice3
            from cosyvoice.utils.file_utils import load_wav
        except ImportError as e:
            raise ModelLoadError(f"CosyVoice not installed: {e}", model_type=model_type) from e

        try:
            if model_type == "cosyvoice2":
                model = CosyVoice2(str(model_path), load_jit=False, load_trt=False, fp16=False)
                version = "cosyvoice2"
            elif model_type == "cosyvoice3":
                model = CosyVoice3(str(model_path), load_jit=False, load_trt=False, fp16=False)
                version = "cosyvoice3"
            else:
                raise ModelLoadError(f"Unknown CosyVoice type: {model_type}", model_type=model_type)

            # Move to device
            if hasattr(model, 'model') and hasattr(model.model, 'to'):
                model.model.to(self.device)
            elif hasattr(model, 'to'):
                model.to(self.device)

            return model, version, load_wav

        except torch.cuda.OutOfMemoryError as e:
            raise GPUOutOfMemoryError(model_type) from e
        except Exception as e:
            logger.error(f"Failed to load {model_type}: {e}", exc_info=True)
            raise ModelLoadError(f"{model_type} loading failed: {e}", model_type=model_type) from e

    def _load_voxcpm(self, model_path: Path) -> Tuple[Any, str, Callable]:
        """Load VoxCPM model with robust package handling."""
        try:
            # Ensure VoxCPM package is available
            if not self.ensure_package_installed("voxcpm"):
                raise ModelLoadError("VoxCPM package not available", model_type="voxcpm")

            # Import VoxCPM - try multiple import patterns
            VoxCPM = None
            import_patterns = [
                "voxcpm.VoxCPM",
                "voxcpm.model.VoxCPM",
                "models.voxcpm.VoxCPM",
                "voxcpm.VoxCPMModel",
            ]
            
            for pattern in import_patterns:
                try:
                    module_path, class_name = pattern.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    VoxCPM = getattr(module, class_name)
                    logger.info(f"Successfully imported VoxCPM from {pattern}")
                    break
                except (ImportError, AttributeError):
                    continue
            
            if VoxCPM is None:
                raise ModelLoadError("Could not import VoxCPM class from any known pattern")

            # Find config and model files
            config_files = list(model_path.glob("*.yaml")) + list(model_path.glob("*.yml")) + list(model_path.glob("*.json"))
            config_path = config_files[0] if config_files else None
            
            model_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
            if not model_files:
                raise ModelLoadError(f"No model weights found in {model_path}")
            
            model_weights = model_files[0]
            logger.info(f"Loading VoxCPM from {model_weights} with config {config_path}")

            # Initialize VoxCPM model
            try:
                if hasattr(VoxCPM, 'from_pretrained'):
                    model = VoxCPM.from_pretrained(str(model_path), device=self.device)
                elif hasattr(VoxCPM, 'load_from_checkpoint'):
                    model = VoxCPM.load_from_checkpoint(str(model_weights), map_location=self.device)
                else:
                    # Direct instantiation with config
                    model = VoxCPM(str(model_path), device=self.device)
            except Exception as e:
                logger.error(f"VoxCPM initialization failed: {e}", exc_info=True)
                raise ModelLoadError(f"VoxCPM initialization failed: {e}", model_type="voxcpm") from e

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
            
            # Define load_wav function for VoxCPM
            def load_wav_voxcpm(wav_path: str) -> torch.Tensor:
                import torchaudio
                waveform, sample_rate = torchaudio.load(wav_path)
                # Resample to 24kHz if needed (VoxCPM default)
                if sample_rate != 24000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                    waveform = resampler(waveform)
                return waveform
            
            logger.info(f"VoxCPM model loaded successfully from {model_path}")
            return model, version, load_wav_voxcpm
            
        except torch.cuda.OutOfMemoryError as e:
            raise GPUOutOfMemoryError("voxcpm") from e
        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Failed to load VoxCPM model: {e}", exc_info=True)
            raise ModelLoadError(f"VoxCPM model loading failed: {e}", model_type="voxcpm") from e

    def unload_model(self, model_type: str) -> bool:
        """Unload a specific model from memory."""
        if model_type in self._loaded_models:
            del self._loaded_models[model_type]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_type}")
            return True
        return False

    def unload_all(self) -> None:
        """Unload all models."""
        self._loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded all models")

    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        if model_type not in self._loaded_models:
            raise ValueError(f"Model not loaded: {model_type}")
        model, version, _ = self._loaded_models[model_type]
        return {
            "model_type": model_type,
            "version": version,
            "device": self.device,
            "loaded": True,
        }

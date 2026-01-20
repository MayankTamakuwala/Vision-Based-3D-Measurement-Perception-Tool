"""MiDaS model loading and management."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Callable
import yaml

from ..utils.logger import get_logger
from ..utils.exceptions import ModelLoadError
from ..utils.validators import validate_model_type

logger = get_logger(__name__)


class ModelLoader:
    """
    Handles loading and caching of MiDaS depth estimation models.

    Supports multiple MiDaS variants:
    - DPT_Large: Highest quality, ~3GB memory
    - DPT_Hybrid: Balanced quality/speed, ~2GB memory
    - MiDaS_small: Fast inference, ~1GB memory
    """

    def __init__(self, cache_dir: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory to cache downloaded models
            config_path: Path to model configuration YAML file
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config(config_path)
        self._model_cache = {}  # In-memory cache for loaded models

        # Set torch hub cache directory
        torch.hub.set_dir(str(self.cache_dir))

        logger.info(f"Model cache directory: {self.cache_dir}")

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load model configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "model_config.yaml"

        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default model configuration."""
        return {
            'models': {
                'DPT_Large': {
                    'hub_name': 'intel-isl/MiDaS',
                    'model_type': 'DPT_Large',
                    'input_size': 384,
                    'description': 'Highest quality, slower inference',
                    'memory_requirement': '~3GB',
                },
                'DPT_Hybrid': {
                    'hub_name': 'intel-isl/MiDaS',
                    'model_type': 'DPT_Hybrid',
                    'input_size': 384,
                    'description': 'Balanced quality and speed',
                    'memory_requirement': '~2GB',
                },
                'MiDaS_small': {
                    'hub_name': 'intel-isl/MiDaS',
                    'model_type': 'MiDaS_small',
                    'input_size': 256,
                    'description': 'Fast inference, good for real-time',
                    'memory_requirement': '~1GB',
                },
            }
        }

    def load_model(
        self,
        model_type: str = "DPT_Large",
        device: Optional[torch.device] = None
    ) -> Tuple[nn.Module, Callable]:
        """
        Load a MiDaS model and its transform.

        Args:
            model_type: Model variant (DPT_Large, DPT_Hybrid, MiDaS_small)
            device: Device to load model on

        Returns:
            Tuple of (model, transform_function)

        Raises:
            ModelLoadError: If model loading fails
        """
        # Validate model type
        model_type = validate_model_type(model_type)

        # Check in-memory cache
        cache_key = f"{model_type}_{device}"
        if cache_key in self._model_cache:
            logger.info(f"Using cached model: {model_type}")
            return self._model_cache[cache_key]

        # Load model and transform
        try:
            logger.info(f"Loading model: {model_type}")
            model_config = self.config['models'].get(model_type, {})
            hub_name = model_config.get('hub_name', 'intel-isl/MiDaS')

            # Load model from torch hub
            model = torch.hub.load(
                hub_name,
                model_type,
                pretrained=True,
                trust_repo=True  # Trust Intel ISL repository
            )

            # Load corresponding transform
            midas_transforms = torch.hub.load(hub_name, "transforms")

            # Select appropriate transform based on model type
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform

            # Move model to device and set to eval mode
            if device:
                model = model.to(device)

            model.eval()

            # Cache the loaded model
            self._model_cache[cache_key] = (model, transform)

            logger.info(f"Model loaded successfully: {model_type}")
            logger.info(f"Input size: {model_config.get('input_size', 'unknown')}")

            return model, transform

        except Exception as e:
            raise ModelLoadError(f"Failed to load model {model_type}: {str(e)}")

    def get_model_info(self, model_type: str) -> dict:
        """
        Get information about a model variant.

        Args:
            model_type: Model variant name

        Returns:
            Dictionary with model information
        """
        model_type = validate_model_type(model_type)
        return self.config['models'].get(model_type, {})

    def list_available_models(self) -> list:
        """
        List all available model variants.

        Returns:
            List of model names
        """
        return list(self.config['models'].keys())

    def get_recommended_model(self, use_case: str = "balanced") -> str:
        """
        Get recommended model for a specific use case.

        Args:
            use_case: One of 'high_quality', 'balanced', 'real_time', 'low_memory'

        Returns:
            Recommended model name
        """
        recommendations = self.config.get('recommendations', {})
        return recommendations.get(use_case, "DPT_Hybrid")

    def clear_cache(self):
        """Clear in-memory model cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared")

    def preload_model(self, model_type: str, device: torch.device):
        """
        Preload a model into memory cache.

        Args:
            model_type: Model variant to preload
            device: Device to load model on
        """
        logger.info(f"Preloading model: {model_type}")
        self.load_model(model_type, device)

    def get_input_size(self, model_type: str) -> int:
        """
        Get the expected input size for a model.

        Args:
            model_type: Model variant name

        Returns:
            Input size (height/width in pixels)
        """
        model_info = self.get_model_info(model_type)
        return model_info.get('input_size', 384)

    def __repr__(self) -> str:
        """String representation of model loader."""
        cached_models = len(self._model_cache)
        return f"ModelLoader(cache_dir='{self.cache_dir}', cached_models={cached_models})"

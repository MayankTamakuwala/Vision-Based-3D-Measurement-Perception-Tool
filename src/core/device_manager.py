"""Device management and GPU optimization for depth estimation."""

import torch
from typing import Optional, Dict
import yaml
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.exceptions import DeviceError

logger = get_logger(__name__)


class DeviceManager:
    """
    Manages device selection and GPU optimization settings.

    Supports CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU fallback.
    Includes H100-specific optimizations for maximum performance.
    """

    def __init__(self, config_path: Optional[str] = None, preferred_device: Optional[str] = None):
        """
        Initialize device manager.

        Args:
            config_path: Path to device configuration YAML file
            preferred_device: Override device selection (cuda, mps, cpu)
        """
        self.config = self._load_config(config_path)
        self.device = self._detect_device(preferred_device)
        self.device_name = self._get_device_name()
        self._configure_optimizations()

        logger.info(f"Device initialized: {self.device} ({self.device_name})")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load device configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "device_config.yaml"

        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default device configuration."""
        return {
            'h100': {
                'enabled': True,
                'use_tf32': True,
                'mixed_precision': 'fp16',
                'cudnn_benchmark': True,
                'cudnn_deterministic': False,
            },
            'memory': {
                'max_batch_size': 16,
                'empty_cache_interval': 10,
                'pin_memory': True,
                'allow_growth': True,
            },
            'device_priority': ['cuda', 'mps', 'cpu'],
            'performance': {
                'num_threads': 4,
                'prefetch_factor': 2,
                'persistent_workers': True,
            }
        }

    def _detect_device(self, preferred_device: Optional[str]) -> torch.device:
        """
        Detect and select the best available device.

        Args:
            preferred_device: User-specified device preference

        Returns:
            Selected torch device
        """
        if preferred_device:
            return self._validate_device(preferred_device)

        # Check device priority order from config
        priority = self.config.get('device_priority', ['cuda', 'mps', 'cpu'])

        for device_type in priority:
            if device_type == 'cuda' and torch.cuda.is_available():
                return torch.device('cuda')
            elif device_type == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')

        # Fallback to CPU
        logger.warning("No GPU detected. Using CPU (this will be slow).")
        return torch.device('cpu')

    def _validate_device(self, device_str: str) -> torch.device:
        """
        Validate that the requested device is available.

        Args:
            device_str: Device string (cuda, mps, cpu)

        Returns:
            Validated torch device

        Raises:
            DeviceError: If requested device is not available
        """
        device_str = device_str.lower()

        if device_str == 'cuda':
            if not torch.cuda.is_available():
                raise DeviceError("CUDA requested but not available")
            return torch.device('cuda')
        elif device_str == 'mps':
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                raise DeviceError("MPS requested but not available")
            return torch.device('mps')
        elif device_str == 'cpu':
            return torch.device('cpu')
        else:
            raise DeviceError(f"Invalid device: {device_str}. Use 'cuda', 'mps', or 'cpu'")

    def _get_device_name(self) -> str:
        """Get human-readable device name."""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name(0)
        elif self.device.type == 'mps':
            return "Apple Silicon (MPS)"
        else:
            return "CPU"

    def _configure_optimizations(self):
        """Configure device-specific optimizations."""
        if self.device.type == 'cuda':
            self._configure_cuda_optimizations()
        elif self.device.type == 'cpu':
            self._configure_cpu_optimizations()

    def _configure_cuda_optimizations(self):
        """Configure CUDA-specific optimizations."""
        h100_config = self.config.get('h100', {})

        # Enable TF32 for faster matrix operations (H100/A100)
        if h100_config.get('use_tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for matrix operations")

        # Enable cuDNN benchmarking for optimal convolution algorithms
        if h100_config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")

        # Deterministic mode (disable for performance)
        torch.backends.cudnn.deterministic = h100_config.get('cudnn_deterministic', False)

        # Log GPU info
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {gpu_props.name}")
            logger.info(f"Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            logger.info(f"Total Memory: {gpu_props.total_memory / 1024**3:.2f} GB")

    def _configure_cpu_optimizations(self):
        """Configure CPU-specific optimizations."""
        perf_config = self.config.get('performance', {})
        num_threads = perf_config.get('num_threads', 4)

        torch.set_num_threads(num_threads)
        logger.info(f"CPU threads set to: {num_threads}")

    def is_h100(self) -> bool:
        """Check if running on H100 GPU."""
        if self.device.type == 'cuda' and torch.cuda.is_available():
            return "H100" in torch.cuda.get_device_name(0)
        return False

    def get_optimal_batch_size(self, default: int = 8) -> int:
        """
        Get optimal batch size based on device capabilities.

        Args:
            default: Default batch size

        Returns:
            Recommended batch size
        """
        if self.device.type == 'cuda':
            memory_config = self.config.get('memory', {})
            return memory_config.get('max_batch_size', 16)
        elif self.device.type == 'mps':
            return 4  # Apple Silicon has limited memory bandwidth
        else:
            return 1  # CPU is slow, use small batches

    def get_mixed_precision_dtype(self) -> torch.dtype:
        """
        Get the appropriate dtype for mixed precision training.

        Returns:
            torch.dtype for mixed precision (float16, bfloat16, or float32)
        """
        if self.device.type == 'cuda':
            h100_config = self.config.get('h100', {})
            precision = h100_config.get('mixed_precision', 'fp16')

            if precision == 'fp16':
                return torch.float16
            elif precision == 'bf16':
                return torch.bfloat16
            else:
                return torch.float32
        else:
            # MPS and CPU use float32
            return torch.float32

    def empty_cache(self):
        """Clear GPU cache to free up memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")

    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory info in MB
        """
        if self.device.type == 'cuda':
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
            }
        else:
            return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0}

    def __repr__(self) -> str:
        """String representation of device manager."""
        return f"DeviceManager(device={self.device}, name='{self.device_name}')"

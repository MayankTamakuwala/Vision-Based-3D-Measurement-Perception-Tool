"""Performance metrics and timing utilities."""

import time
from contextlib import contextmanager
from typing import Dict, Optional
import torch
import psutil


class PerformanceMetrics:
    """Track and compute performance metrics."""

    def __init__(self):
        """Initialize performance metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.total_time = 0.0
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
        self.num_images = 0

    def add_inference_time(self, time_ms: float):
        """Add an inference time measurement."""
        self.inference_times.append(time_ms)
        self.total_time += time_ms

    def add_preprocessing_time(self, time_ms: float):
        """Add a preprocessing time measurement."""
        self.preprocessing_times.append(time_ms)

    def add_postprocessing_time(self, time_ms: float):
        """Add a postprocessing time measurement."""
        self.postprocessing_times.append(time_ms)

    def increment_images(self, count: int = 1):
        """Increment the number of processed images."""
        self.num_images += count

    def get_summary(self) -> Dict[str, float]:
        """
        Get a summary of performance metrics.

        Returns:
            Dictionary containing performance statistics
        """
        summary = {
            'total_images': self.num_images,
            'total_time_ms': self.total_time,
        }

        if self.inference_times:
            summary.update({
                'avg_inference_ms': sum(self.inference_times) / len(self.inference_times),
                'min_inference_ms': min(self.inference_times),
                'max_inference_ms': max(self.inference_times),
            })

        if self.preprocessing_times:
            summary['avg_preprocessing_ms'] = sum(self.preprocessing_times) / len(self.preprocessing_times)

        if self.postprocessing_times:
            summary['avg_postprocessing_ms'] = sum(self.postprocessing_times) / len(self.postprocessing_times)

        # Calculate FPS
        if self.total_time > 0 and self.num_images > 0:
            summary['fps'] = (self.num_images * 1000) / self.total_time  # Convert ms to seconds

        return summary


@contextmanager
def timer(name: str = "Operation", logger=None):
    """
    Context manager for timing operations.

    Args:
        name: Name of the operation being timed
        logger: Optional logger to log timing information

    Yields:
        Dictionary that will contain the elapsed time in milliseconds

    Example:
        >>> with timer("Model inference") as t:
        ...     result = model(input)
        >>> print(f"Took {t['elapsed_ms']:.2f} ms")
    """
    timing = {}
    start_time = time.perf_counter()

    try:
        yield timing
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timing['elapsed_ms'] = elapsed_ms

        if logger:
            logger.info(f"{name} took {elapsed_ms:.2f} ms")


def get_gpu_memory_info() -> Optional[Dict[str, float]]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with GPU memory info (in MB) or None if no GPU available
    """
    if not torch.cuda.is_available():
        return None

    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def get_cpu_memory_info() -> Dict[str, float]:
    """
    Get current CPU memory usage.

    Returns:
        Dictionary with CPU memory info (in MB)
    """
    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        'rss_mb': mem_info.rss / 1024**2,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024**2,  # Virtual Memory Size
    }


def get_system_info() -> Dict[str, any]:
    """
    Get system information including GPU and CPU details.

    Returns:
        Dictionary with system information
    """
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_percent': psutil.virtual_memory().percent,
    }

    # GPU information
    if torch.cuda.is_available():
        info.update({
            'gpu_available': True,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
        })
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info.update({
            'gpu_available': True,
            'gpu_name': 'Apple Silicon (MPS)',
            'gpu_count': 1,
        })
    else:
        info['gpu_available'] = False

    return info


def format_time(ms: float) -> str:
    """
    Format time in milliseconds to human-readable string.

    Args:
        ms: Time in milliseconds

    Returns:
        Formatted time string
    """
    if ms < 1000:
        return f"{ms:.2f} ms"
    elif ms < 60000:
        return f"{ms/1000:.2f} s"
    else:
        return f"{ms/60000:.2f} min"

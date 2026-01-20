"""Main depth estimation engine."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Union
from pathlib import Path
import time

from .device_manager import DeviceManager
from .model_loader import ModelLoader
from ..preprocessing.image_loader import ImageLoader
from ..preprocessing.transforms import prepare_image_for_inference, prepare_batch_for_inference
from ..postprocessing.depth_processor import DepthProcessor
from ..postprocessing.normalizer import DepthNormalizer
from ..utils.logger import get_logger
from ..utils.exceptions import ModelLoadError, PreprocessingError
from ..utils.metrics import timer, PerformanceMetrics

logger = get_logger(__name__)


class DepthEstimator:
    """
    Main depth estimation engine using MiDaS models.

    Orchestrates the complete pipeline:
    1. Device management and GPU optimization
    2. Model loading and caching
    3. Image preprocessing
    4. Inference
    5. Depth postprocessing
    """

    def __init__(
        self,
        model_type: str = "DPT_Large",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        optimize: bool = True
    ):
        """
        Initialize depth estimator.

        Args:
            model_type: MiDaS model variant (DPT_Large, DPT_Hybrid, MiDaS_small)
            device: Device to use (cuda, mps, cpu). If None, auto-detects.
            cache_dir: Directory to cache models
            optimize: Whether to apply performance optimizations

        Raises:
            ModelLoadError: If model loading fails
        """
        logger.info(f"Initializing DepthEstimator with model: {model_type}")

        # Initialize device manager
        self.device_manager = DeviceManager(preferred_device=device)
        self.device = self.device_manager.device

        # Initialize model loader
        self.model_loader = ModelLoader(cache_dir=cache_dir)

        # Load model and transform
        try:
            self.model, self.transform = self.model_loader.load_model(
                model_type=model_type,
                device=self.device
            )
            self.model_type = model_type
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize model: {str(e)}")

        # Initialize processors
        self.image_loader = ImageLoader(use_pil=False)
        self.depth_processor = DepthProcessor()
        self.normalizer = DepthNormalizer()

        # Performance metrics
        self.metrics = PerformanceMetrics()

        # Optimization settings
        self.optimize = optimize
        if optimize:
            self._apply_optimizations()

        logger.info(f"DepthEstimator initialized on {self.device}")

    def _apply_optimizations(self):
        """Apply performance optimizations."""
        # Set model to inference mode
        self.model.eval()

        # Disable gradient computation
        torch.set_grad_enabled(False)

        # Enable JIT compilation if possible
        if self.device.type == 'cuda':
            try:
                # Warm up GPU
                dummy_input = torch.randn(1, 3, 384, 384).to(self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                logger.info("GPU warm-up completed")
            except Exception as e:
                logger.warning(f"GPU warm-up failed: {str(e)}")

    def estimate_depth(
        self,
        image: np.ndarray,
        normalize: bool = True,
        remove_outliers: bool = True
    ) -> np.ndarray:
        """
        Estimate depth from a single image.

        Args:
            image: Input RGB image (H, W, 3)
            normalize: Whether to normalize output depth to [0, 1]
            remove_outliers: Whether to remove outlier depth values

        Returns:
            Depth map array (H, W)

        Raises:
            PreprocessingError: If image preprocessing fails
        """
        try:
            original_shape = image.shape[:2]

            # Preprocessing
            with timer("Preprocessing", logger) as t:
                input_tensor = prepare_image_for_inference(image, self.transform, self.device)
            self.metrics.add_preprocessing_time(t['elapsed_ms'])

            # Inference
            with timer("Inference", logger) as t:
                with torch.no_grad():
                    if self.device.type == 'cuda' and self.optimize:
                        # Use automatic mixed precision for faster inference
                        with torch.cuda.amp.autocast(dtype=self.device_manager.get_mixed_precision_dtype()):
                            depth_tensor = self.model(input_tensor)
                    else:
                        depth_tensor = self.model(input_tensor)
            self.metrics.add_inference_time(t['elapsed_ms'])

            # Convert to numpy
            depth = depth_tensor.squeeze().cpu().numpy()

            # Postprocessing
            with timer("Postprocessing", logger) as t:
                depth = self.depth_processor.process_pipeline(
                    depth,
                    original_shape=original_shape,
                    smooth=False,
                    remove_outliers=remove_outliers
                )

                if normalize:
                    depth = self.normalizer.normalize_minmax(depth, target_range=(0.0, 1.0))

            self.metrics.add_postprocessing_time(t['elapsed_ms'])
            self.metrics.increment_images()

            logger.debug(f"Depth estimation complete: shape={depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]")
            return depth

        except Exception as e:
            logger.error(f"Depth estimation failed: {str(e)}")
            raise

    def estimate_depth_from_file(
        self,
        image_path: Union[str, Path],
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth from an image file.

        Args:
            image_path: Path to image file
            normalize: Whether to normalize depth

        Returns:
            Tuple of (original_image, depth_map)
        """
        image = self.image_loader.load_single(image_path)
        depth = self.estimate_depth(image, normalize=normalize)
        return image, depth

    def estimate_batch(
        self,
        images: List[np.ndarray],
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> List[np.ndarray]:
        """
        Estimate depth for a batch of images.

        Args:
            images: List of RGB images
            batch_size: Batch size for processing (if None, uses optimal size)
            normalize: Whether to normalize depths

        Returns:
            List of depth maps
        """
        if batch_size is None:
            batch_size = self.device_manager.get_optimal_batch_size()

        logger.info(f"Processing {len(images)} images in batches of {batch_size}")

        all_depths = []
        num_batches = (len(images) + batch_size - 1) // batch_size

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"Processing batch {batch_num}/{num_batches}")

            # Process each image in batch individually for now
            # (Could be optimized for true batch processing)
            batch_depths = [self.estimate_depth(img, normalize=normalize) for img in batch]
            all_depths.extend(batch_depths)

            # Clear cache periodically
            if batch_num % 10 == 0:
                self.device_manager.empty_cache()

        logger.info(f"Batch processing complete: {len(all_depths)} depths generated")
        return all_depths

    def estimate_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.jpg",
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> List[Tuple[Path, np.ndarray, np.ndarray]]:
        """
        Estimate depth for all images in a directory.

        Args:
            directory: Directory containing images
            pattern: Glob pattern for image files
            batch_size: Batch size for processing
            normalize: Whether to normalize depths

        Returns:
            List of (path, image, depth) tuples
        """
        # Load all images
        loaded_images = self.image_loader.load_batch(directory, pattern=pattern)

        if not loaded_images:
            logger.warning(f"No images found in {directory} matching pattern {pattern}")
            return []

        logger.info(f"Found {len(loaded_images)} images in {directory}")

        # Separate paths and images
        paths, images = zip(*loaded_images)

        # Estimate depths
        depths = self.estimate_batch(list(images), batch_size=batch_size, normalize=normalize)

        # Combine results
        results = list(zip(paths, images, depths))
        return results

    def get_depth_stats(self, depth: np.ndarray) -> dict:
        """
        Get statistics for a depth map.

        Args:
            depth: Depth map

        Returns:
            Dictionary with depth statistics
        """
        return self.normalizer.get_depth_stats(depth)

    def get_performance_summary(self) -> dict:
        """
        Get performance metrics summary.

        Returns:
            Dictionary with performance statistics
        """
        return self.metrics.get_summary()

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics.reset()
        logger.info("Performance metrics reset")

    def clear_cache(self):
        """Clear model cache and GPU memory."""
        self.model_loader.clear_cache()
        self.device_manager.empty_cache()
        logger.info("Cache cleared")

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'device_name': self.device_manager.device_name,
            'optimizations_enabled': self.optimize,
            **self.model_loader.get_model_info(self.model_type)
        }

    def __repr__(self) -> str:
        """String representation of depth estimator."""
        return (
            f"DepthEstimator("
            f"model={self.model_type}, "
            f"device={self.device}, "
            f"optimize={self.optimize})"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.clear_cache()

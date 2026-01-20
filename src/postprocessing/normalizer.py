"""Depth map normalization utilities."""

import numpy as np
from typing import Tuple, Dict

from ..utils.logger import get_logger
from ..utils.exceptions import PostprocessingError
from ..utils.validators import validate_depth_map

logger = get_logger(__name__)


class DepthNormalizer:
    """
    Handles normalization of depth maps to standard ranges.

    Supports various normalization strategies:
    - Min-max normalization
    - Percentile-based normalization
    - Z-score normalization
    """

    def __init__(self):
        """Initialize depth normalizer."""
        pass

    def normalize_minmax(
        self,
        depth: np.ndarray,
        target_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        Normalize depth map using min-max normalization.

        Args:
            depth: Input depth map
            target_range: Target (min, max) range for normalized values

        Returns:
            Normalized depth map

        Raises:
            PostprocessingError: If normalization fails
        """
        try:
            validate_depth_map(depth)

            min_val = depth.min()
            max_val = depth.max()

            if max_val == min_val:
                logger.warning("Depth map has constant values, returning zeros")
                return np.zeros_like(depth)

            # Normalize to [0, 1]
            normalized = (depth - min_val) / (max_val - min_val)

            # Scale to target range
            target_min, target_max = target_range
            normalized = normalized * (target_max - target_min) + target_min

            logger.debug(f"Min-max normalized to range {target_range}")
            return normalized

        except Exception as e:
            raise PostprocessingError(f"Failed to normalize depth map: {str(e)}")

    def normalize_percentile(
        self,
        depth: np.ndarray,
        pmin: float = 2.0,
        pmax: float = 98.0,
        target_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        Normalize depth map using percentile-based normalization.

        This is more robust to outliers than min-max normalization.

        Args:
            depth: Input depth map
            pmin: Lower percentile (0-100)
            pmax: Upper percentile (0-100)
            target_range: Target (min, max) range for normalized values

        Returns:
            Normalized depth map
        """
        try:
            validate_depth_map(depth)

            vmin = np.percentile(depth, pmin)
            vmax = np.percentile(depth, pmax)

            if vmax == vmin:
                logger.warning("Percentile range is zero, using min-max")
                return self.normalize_minmax(depth, target_range)

            # Clip and normalize
            clipped = np.clip(depth, vmin, vmax)
            normalized = (clipped - vmin) / (vmax - vmin)

            # Scale to target range
            target_min, target_max = target_range
            normalized = normalized * (target_max - target_min) + target_min

            logger.debug(f"Percentile normalized with p=[{pmin}, {pmax}]")
            return normalized

        except Exception as e:
            raise PostprocessingError(f"Failed to apply percentile normalization: {str(e)}")

    def normalize_zscore(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize depth map using z-score normalization (standardization).

        Args:
            depth: Input depth map

        Returns:
            Normalized depth map (mean=0, std=1)
        """
        try:
            validate_depth_map(depth)

            mean = depth.mean()
            std = depth.std()

            if std == 0:
                logger.warning("Standard deviation is zero, returning centered values")
                return depth - mean

            normalized = (depth - mean) / std

            logger.debug("Applied z-score normalization")
            return normalized

        except Exception as e:
            raise PostprocessingError(f"Failed to apply z-score normalization: {str(e)}")

    def normalize_to_uint8(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to uint8 range [0, 255].

        Args:
            depth: Input depth map

        Returns:
            Depth map as uint8 array
        """
        normalized = self.normalize_minmax(depth, target_range=(0, 255))
        return normalized.astype(np.uint8)

    def normalize_to_uint16(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to uint16 range [0, 65535].

        Args:
            depth: Input depth map

        Returns:
            Depth map as uint16 array
        """
        normalized = self.normalize_minmax(depth, target_range=(0, 65535))
        return normalized.astype(np.uint16)

    def get_depth_stats(self, depth: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics for a depth map.

        Args:
            depth: Input depth map

        Returns:
            Dictionary with depth statistics
        """
        try:
            validate_depth_map(depth)

            stats = {
                'min': float(depth.min()),
                'max': float(depth.max()),
                'mean': float(depth.mean()),
                'median': float(np.median(depth)),
                'std': float(depth.std()),
                'percentile_1': float(np.percentile(depth, 1)),
                'percentile_5': float(np.percentile(depth, 5)),
                'percentile_25': float(np.percentile(depth, 25)),
                'percentile_75': float(np.percentile(depth, 75)),
                'percentile_95': float(np.percentile(depth, 95)),
                'percentile_99': float(np.percentile(depth, 99)),
            }

            logger.debug(f"Computed depth stats: min={stats['min']:.2f}, max={stats['max']:.2f}")
            return stats

        except Exception as e:
            raise PostprocessingError(f"Failed to compute depth stats: {str(e)}")

    def adaptive_normalize(
        self,
        depth: np.ndarray,
        method: str = "percentile"
    ) -> np.ndarray:
        """
        Apply adaptive normalization based on depth statistics.

        Args:
            depth: Input depth map
            method: Normalization method ('minmax', 'percentile', 'zscore')

        Returns:
            Normalized depth map
        """
        method = method.lower()

        if method == "minmax":
            return self.normalize_minmax(depth)
        elif method == "percentile":
            return self.normalize_percentile(depth)
        elif method == "zscore":
            return self.normalize_zscore(depth)
        else:
            logger.warning(f"Unknown method '{method}', using minmax")
            return self.normalize_minmax(depth)

    def denormalize(
        self,
        normalized_depth: np.ndarray,
        original_min: float,
        original_max: float
    ) -> np.ndarray:
        """
        Denormalize a depth map back to its original range.

        Args:
            normalized_depth: Normalized depth map (assumed to be in [0, 1])
            original_min: Original minimum value
            original_max: Original maximum value

        Returns:
            Denormalized depth map
        """
        try:
            denormalized = normalized_depth * (original_max - original_min) + original_min
            logger.debug(f"Denormalized to range [{original_min}, {original_max}]")
            return denormalized

        except Exception as e:
            raise PostprocessingError(f"Failed to denormalize depth map: {str(e)}")

    def create_depth_histogram(
        self,
        depth: np.ndarray,
        bins: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create histogram of depth values.

        Args:
            depth: Input depth map
            bins: Number of histogram bins

        Returns:
            Tuple of (histogram, bin_edges)
        """
        try:
            validate_depth_map(depth)

            hist, bin_edges = np.histogram(depth.flatten(), bins=bins)

            logger.debug(f"Created depth histogram with {bins} bins")
            return hist, bin_edges

        except Exception as e:
            raise PostprocessingError(f"Failed to create depth histogram: {str(e)}")

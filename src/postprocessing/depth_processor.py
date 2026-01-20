"""Depth map processing and refinement utilities."""

import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import get_logger
from ..utils.exceptions import PostprocessingError
from ..utils.validators import validate_depth_map

logger = get_logger(__name__)


class DepthProcessor:
    """
    Handles post-processing operations on depth maps.

    Operations include:
    - Resizing to original image dimensions
    - Smoothing/filtering
    - Inversion (MiDaS outputs inverse depth)
    """

    def __init__(self):
        """Initialize depth processor."""
        pass

    def resize_to_original(
        self,
        depth: np.ndarray,
        original_shape: Tuple[int, int],
        interpolation: int = cv2.INTER_CUBIC
    ) -> np.ndarray:
        """
        Resize depth map to match original image dimensions.

        Args:
            depth: Depth map array
            original_shape: (height, width) of original image
            interpolation: Interpolation method

        Returns:
            Resized depth map

        Raises:
            PostprocessingError: If resizing fails
        """
        try:
            validate_depth_map(depth)

            height, width = original_shape
            resized = cv2.resize(depth, (width, height), interpolation=interpolation)

            logger.debug(f"Resized depth from {depth.shape} to {resized.shape}")
            return resized

        except Exception as e:
            raise PostprocessingError(f"Failed to resize depth map: {str(e)}")

    def apply_bilateral_filter(
        self,
        depth: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """
        Apply bilateral filter to smooth depth map while preserving edges.

        Args:
            depth: Input depth map
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space

        Returns:
            Filtered depth map
        """
        try:
            validate_depth_map(depth)

            # Normalize to 0-255 for filtering
            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)

            # Apply bilateral filter
            filtered = cv2.bilateralFilter(depth_normalized, d, sigma_color, sigma_space)

            # Denormalize back to original range
            filtered = filtered.astype(np.float32) / 255.0
            filtered = filtered * (depth.max() - depth.min()) + depth.min()

            logger.debug("Applied bilateral filter to depth map")
            return filtered

        except Exception as e:
            raise PostprocessingError(f"Failed to apply bilateral filter: {str(e)}")

    def apply_gaussian_blur(
        self,
        depth: np.ndarray,
        kernel_size: int = 5,
        sigma: float = 0
    ) -> np.ndarray:
        """
        Apply Gaussian blur to smooth depth map.

        Args:
            depth: Input depth map
            kernel_size: Size of Gaussian kernel (must be odd)
            sigma: Gaussian kernel standard deviation

        Returns:
            Blurred depth map
        """
        try:
            validate_depth_map(depth)

            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

            blurred = cv2.GaussianBlur(depth, (kernel_size, kernel_size), sigma)

            logger.debug(f"Applied Gaussian blur with kernel size {kernel_size}")
            return blurred

        except Exception as e:
            raise PostprocessingError(f"Failed to apply Gaussian blur: {str(e)}")

    def apply_median_filter(
        self,
        depth: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Apply median filter to remove noise from depth map.

        Args:
            depth: Input depth map
            kernel_size: Size of median filter kernel (must be odd)

        Returns:
            Filtered depth map
        """
        try:
            validate_depth_map(depth)

            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Convert to appropriate format for median blur
            if depth.dtype != np.uint8:
                # Normalize to 0-255
                depth_uint8 = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
                filtered = cv2.medianBlur(depth_uint8, kernel_size)
                # Convert back to float
                filtered = filtered.astype(np.float32) / 255.0 * (depth.max() - depth.min()) + depth.min()
            else:
                filtered = cv2.medianBlur(depth, kernel_size)

            logger.debug(f"Applied median filter with kernel size {kernel_size}")
            return filtered

        except Exception as e:
            raise PostprocessingError(f"Failed to apply median filter: {str(e)}")

    def invert_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Invert depth values (MiDaS outputs inverse depth by default).

        Args:
            depth: Input depth map

        Returns:
            Inverted depth map
        """
        try:
            validate_depth_map(depth)

            # Avoid division by zero
            epsilon = 1e-6
            inverted = 1.0 / (depth + epsilon)

            logger.debug("Inverted depth values")
            return inverted

        except Exception as e:
            raise PostprocessingError(f"Failed to invert depth: {str(e)}")

    def clip_depth(
        self,
        depth: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> np.ndarray:
        """
        Clip depth values to a specified range.

        Args:
            depth: Input depth map
            min_val: Minimum value (None for no min clipping)
            max_val: Maximum value (None for no max clipping)

        Returns:
            Clipped depth map
        """
        try:
            validate_depth_map(depth)

            clipped = depth.copy()

            if min_val is not None:
                clipped = np.maximum(clipped, min_val)

            if max_val is not None:
                clipped = np.minimum(clipped, max_val)

            logger.debug(f"Clipped depth to range [{min_val}, {max_val}]")
            return clipped

        except Exception as e:
            raise PostprocessingError(f"Failed to clip depth: {str(e)}")

    def remove_outliers(
        self,
        depth: np.ndarray,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0
    ) -> np.ndarray:
        """
        Remove outlier depth values using percentile-based clipping.

        Args:
            depth: Input depth map
            percentile_low: Lower percentile for clipping
            percentile_high: Upper percentile for clipping

        Returns:
            Depth map with outliers removed
        """
        try:
            validate_depth_map(depth)

            low_val = np.percentile(depth, percentile_low)
            high_val = np.percentile(depth, percentile_high)

            clipped = np.clip(depth, low_val, high_val)

            logger.debug(f"Removed outliers using percentiles [{percentile_low}, {percentile_high}]")
            return clipped

        except Exception as e:
            raise PostprocessingError(f"Failed to remove outliers: {str(e)}")

    def fill_holes(self, depth: np.ndarray, hole_value: float = 0.0) -> np.ndarray:
        """
        Fill holes (invalid regions) in depth map using inpainting.

        Args:
            depth: Input depth map
            hole_value: Value that represents holes/invalid regions

        Returns:
            Depth map with holes filled
        """
        try:
            validate_depth_map(depth)

            # Create mask of holes
            mask = (np.abs(depth - hole_value) < 1e-6).astype(np.uint8)

            if mask.sum() == 0:
                logger.debug("No holes found in depth map")
                return depth

            # Convert depth to uint8 for inpainting
            depth_uint8 = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)

            # Inpaint
            filled = cv2.inpaint(depth_uint8, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Convert back to float
            filled = filled.astype(np.float32) / 255.0 * (depth.max() - depth.min()) + depth.min()

            logger.debug(f"Filled {mask.sum()} hole pixels")
            return filled

        except Exception as e:
            raise PostprocessingError(f"Failed to fill holes: {str(e)}")

    def process_pipeline(
        self,
        depth: np.ndarray,
        original_shape: Optional[Tuple[int, int]] = None,
        smooth: bool = False,
        remove_outliers: bool = True
    ) -> np.ndarray:
        """
        Apply a complete processing pipeline to depth map.

        Args:
            depth: Input depth map
            original_shape: Original image dimensions for resizing
            smooth: Whether to apply smoothing filter
            remove_outliers: Whether to remove outlier values

        Returns:
            Processed depth map
        """
        processed = depth.copy()

        # Remove outliers
        if remove_outliers:
            processed = self.remove_outliers(processed)

        # Apply smoothing
        if smooth:
            processed = self.apply_bilateral_filter(processed)

        # Resize to original dimensions
        if original_shape:
            processed = self.resize_to_original(processed, original_shape)

        return processed

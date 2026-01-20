"""Image transformation utilities for MiDaS preprocessing."""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional

from ..utils.logger import get_logger
from ..utils.exceptions import PreprocessingError
from ..utils.validators import validate_image_array

logger = get_logger(__name__)


def prepare_image_for_inference(
    image: np.ndarray,
    transform: callable,
    device: torch.device
) -> torch.Tensor:
    """
    Prepare an image for MiDaS inference.

    Args:
        image: Input image in RGB format (H, W, C)
        transform: MiDaS transform function
        device: Target device for tensor

    Returns:
        Preprocessed image tensor ready for inference

    Raises:
        PreprocessingError: If preprocessing fails
    """
    try:
        validate_image_array(image)

        # Apply MiDaS transform
        input_batch = transform(image).to(device)

        logger.debug(f"Transformed image shape: {input_batch.shape}")
        return input_batch

    except Exception as e:
        raise PreprocessingError(f"Failed to prepare image for inference: {str(e)}")


def prepare_batch_for_inference(
    images: list,
    transform: callable,
    device: torch.device
) -> torch.Tensor:
    """
    Prepare a batch of images for MiDaS inference.

    Args:
        images: List of images in RGB format (H, W, C)
        transform: MiDaS transform function
        device: Target device for tensor

    Returns:
        Batched tensor ready for inference

    Raises:
        PreprocessingError: If preprocessing fails
    """
    try:
        # Transform each image
        transformed = [transform(img) for img in images]

        # Stack into a batch
        batch = torch.stack(transformed).to(device)

        logger.debug(f"Batch shape: {batch.shape}")
        return batch

    except Exception as e:
        raise PreprocessingError(f"Failed to prepare batch for inference: {str(e)}")


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: int,
    interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio.

    Args:
        image: Input image
        target_size: Target size for the shorter side
        interpolation: Interpolation method

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    # Calculate new dimensions
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    logger.debug(f"Resized from {(h, w)} to {(new_h, new_w)}")

    return resized


def pad_to_size(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 0
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image to target size.

    Args:
        image: Input image
        target_size: (height, width) tuple
        pad_value: Value to use for padding

    Returns:
        Tuple of (padded_image, (top, bottom, left, right) padding)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate padding
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad image
    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[pad_value] * image.shape[2]
        )
    else:
        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=pad_value
        )

    return padded, (top, bottom, left, right)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize image using ImageNet mean and std.

    Args:
        image: Input image (0-255 range or 0-1 range)
        mean: Mean values for each channel
        std: Std values for each channel

    Returns:
        Normalized image
    """
    # Convert to float and scale to [0, 1] if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Normalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    normalized = (image - mean) / std

    return normalized


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize an image back to [0, 255] range.

    Args:
        image: Normalized image
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Denormalized image in uint8 format
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    denormalized = (image * std) + mean
    denormalized = np.clip(denormalized * 255, 0, 255).astype(np.uint8)

    return denormalized


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.

    Args:
        image: RGB image

    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in RGB format.

    Args:
        image: Input image (grayscale or RGB)

    Returns:
        RGB image
    """
    if len(image.shape) == 2:
        # Grayscale to RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA to RGB
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        return image


class ImagePreprocessor:
    """
    Handles image preprocessing pipeline for depth estimation.
    """

    def __init__(self, target_size: Optional[int] = None):
        """
        Initialize preprocessor.

        Args:
            target_size: Optional target size for resizing
        """
        self.target_size = target_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to an image.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Ensure RGB format
        image = ensure_rgb(image)

        # Resize if target size is specified
        if self.target_size:
            image = resize_with_aspect_ratio(image, self.target_size)

        return image

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Allow using preprocessor as a callable."""
        return self.preprocess(image)

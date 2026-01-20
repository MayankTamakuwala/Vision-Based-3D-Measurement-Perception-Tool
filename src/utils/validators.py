"""Input validation utilities for the Vision-Based 3D Measurement & Perception Tool."""

from pathlib import Path
from typing import List, Tuple, Union
import numpy as np

from .exceptions import InvalidInputError


def validate_image_path(path: Union[str, Path]) -> Path:
    """
    Validate that an image file exists and is readable.

    Args:
        path: Path to image file

    Returns:
        Validated Path object

    Raises:
        InvalidInputError: If path is invalid or file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise InvalidInputError(f"Image file not found: {path}")

    if not path.is_file():
        raise InvalidInputError(f"Path is not a file: {path}")

    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        raise InvalidInputError(
            f"Invalid image format: {path.suffix}. "
            f"Supported formats: {', '.join(valid_extensions)}"
        )

    return path


def validate_directory(path: Union[str, Path], create_if_missing: bool = False) -> Path:
    """
    Validate that a directory exists.

    Args:
        path: Path to directory
        create_if_missing: If True, create directory if it doesn't exist

    Returns:
        Validated Path object

    Raises:
        InvalidInputError: If path is invalid or directory doesn't exist
    """
    path = Path(path)

    if not path.exists():
        if create_if_missing:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise InvalidInputError(f"Directory not found: {path}")

    if not path.is_dir():
        raise InvalidInputError(f"Path is not a directory: {path}")

    return path


def validate_image_array(image: np.ndarray) -> np.ndarray:
    """
    Validate that an image array has correct shape and dtype.

    Args:
        image: Image array to validate

    Returns:
        Validated image array

    Raises:
        InvalidInputError: If image array is invalid
    """
    if not isinstance(image, np.ndarray):
        raise InvalidInputError(f"Image must be numpy array, got {type(image)}")

    if image.ndim not in [2, 3]:
        raise InvalidInputError(
            f"Image must be 2D (grayscale) or 3D (color), got shape {image.shape}"
        )

    if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
        raise InvalidInputError(
            f"Image must have 1, 3, or 4 channels, got {image.shape[2]}"
        )

    if image.size == 0:
        raise InvalidInputError("Image array is empty")

    return image


def validate_depth_map(depth: np.ndarray) -> np.ndarray:
    """
    Validate that a depth map has correct shape and values.

    Args:
        depth: Depth map to validate

    Returns:
        Validated depth map

    Raises:
        InvalidInputError: If depth map is invalid
    """
    if not isinstance(depth, np.ndarray):
        raise InvalidInputError(f"Depth map must be numpy array, got {type(depth)}")

    if depth.ndim != 2:
        raise InvalidInputError(
            f"Depth map must be 2D, got shape {depth.shape}"
        )

    if depth.size == 0:
        raise InvalidInputError("Depth map is empty")

    if not np.isfinite(depth).all():
        raise InvalidInputError("Depth map contains non-finite values (NaN or Inf)")

    return depth


def validate_point(point: Tuple[int, int], image_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Validate that a point is within image boundaries.

    Args:
        point: (x, y) coordinates
        image_shape: (height, width) of image

    Returns:
        Validated point

    Raises:
        InvalidInputError: If point is invalid or out of bounds
    """
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise InvalidInputError(f"Point must be a tuple/list of 2 values, got {point}")

    x, y = point
    height, width = image_shape

    if not (0 <= x < width and 0 <= y < height):
        raise InvalidInputError(
            f"Point ({x}, {y}) is out of bounds for image shape {image_shape}"
        )

    return (int(x), int(y))


def validate_bbox(bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Validate that a bounding box is within image boundaries.

    Args:
        bbox: (x1, y1, x2, y2) coordinates
        image_shape: (height, width) of image

    Returns:
        Validated bounding box

    Raises:
        InvalidInputError: If bounding box is invalid
    """
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        raise InvalidInputError(f"Bounding box must be a tuple/list of 4 values, got {bbox}")

    x1, y1, x2, y2 = bbox
    height, width = image_shape

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise InvalidInputError(
            f"Invalid bounding box ({x1}, {y1}, {x2}, {y2}) for image shape {image_shape}"
        )

    return (int(x1), int(y1), int(x2), int(y2))


def validate_model_type(model_type: str) -> str:
    """
    Validate that model type is supported.

    Args:
        model_type: Model type string

    Returns:
        Validated model type

    Raises:
        InvalidInputError: If model type is not supported
    """
    valid_models = {'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'}

    if model_type not in valid_models:
        raise InvalidInputError(
            f"Invalid model type: {model_type}. "
            f"Supported models: {', '.join(valid_models)}"
        )

    return model_type


def validate_device(device: str) -> str:
    """
    Validate that device string is valid.

    Args:
        device: Device string (cuda, mps, cpu)

    Returns:
        Validated device string

    Raises:
        InvalidInputError: If device is not valid
    """
    valid_devices = {'cuda', 'mps', 'cpu'}

    if device not in valid_devices:
        raise InvalidInputError(
            f"Invalid device: {device}. "
            f"Valid devices: {', '.join(valid_devices)}"
        )

    return device


def validate_batch_size(batch_size: int, max_batch_size: int = 32) -> int:
    """
    Validate batch size parameter.

    Args:
        batch_size: Batch size to validate
        max_batch_size: Maximum allowed batch size

    Returns:
        Validated batch size

    Raises:
        InvalidInputError: If batch size is invalid
    """
    if not isinstance(batch_size, int):
        raise InvalidInputError(f"Batch size must be an integer, got {type(batch_size)}")

    if batch_size < 1:
        raise InvalidInputError(f"Batch size must be >= 1, got {batch_size}")

    if batch_size > max_batch_size:
        raise InvalidInputError(
            f"Batch size {batch_size} exceeds maximum {max_batch_size}"
        )

    return batch_size

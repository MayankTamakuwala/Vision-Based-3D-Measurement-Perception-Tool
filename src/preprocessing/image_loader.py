"""Image loading utilities for single images, batches, and directories."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image

from ..utils.logger import get_logger
from ..utils.exceptions import PreprocessingError, InvalidInputError
from ..utils.validators import validate_image_path, validate_directory

logger = get_logger(__name__)


class ImageLoader:
    """
    Handles loading images from various sources.

    Supports:
    - Single images from file paths
    - Batch loading from directories
    - Various image formats (JPEG, PNG, etc.)
    """

    def __init__(self, use_pil: bool = False):
        """
        Initialize image loader.

        Args:
            use_pil: If True, use PIL for loading. Otherwise use OpenCV.
        """
        self.use_pil = use_pil

    def load_single(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load a single image from file.

        Args:
            path: Path to image file

        Returns:
            Image as numpy array in RGB format (H, W, C)

        Raises:
            PreprocessingError: If image loading fails
        """
        path = validate_image_path(path)

        try:
            if self.use_pil:
                image = self._load_with_pil(path)
            else:
                image = self._load_with_opencv(path)

            logger.debug(f"Loaded image: {path.name}, shape: {image.shape}")
            return image

        except Exception as e:
            raise PreprocessingError(f"Failed to load image {path}: {str(e)}")

    def _load_with_opencv(self, path: Path) -> np.ndarray:
        """Load image using OpenCV and convert BGR to RGB."""
        image = cv2.imread(str(path))

        if image is None:
            raise PreprocessingError(f"OpenCV failed to load image: {path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_with_pil(self, path: Path) -> np.ndarray:
        """Load image using PIL."""
        image = Image.open(path)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return np.array(image)

    def load_batch(
        self,
        directory: Union[str, Path],
        pattern: str = "*.jpg",
        max_images: Optional[int] = None,
        recursive: bool = False
    ) -> List[Tuple[Path, np.ndarray]]:
        """
        Load multiple images from a directory.

        Args:
            directory: Directory containing images
            pattern: Glob pattern for image files (e.g., "*.jpg", "*.png")
            max_images: Maximum number of images to load (None for all)
            recursive: If True, search subdirectories recursively

        Returns:
            List of (path, image) tuples

        Raises:
            PreprocessingError: If directory access fails
        """
        directory = validate_directory(directory)

        try:
            # Find all matching image files
            if recursive:
                image_paths = list(directory.rglob(pattern))
            else:
                image_paths = list(directory.glob(pattern))

            # Sort paths for consistent ordering
            image_paths.sort()

            # Limit number of images if specified
            if max_images:
                image_paths = image_paths[:max_images]

            logger.info(f"Found {len(image_paths)} images in {directory}")

            # Load all images
            loaded_images = []
            for path in image_paths:
                try:
                    image = self.load_single(path)
                    loaded_images.append((path, image))
                except Exception as e:
                    logger.warning(f"Skipping {path.name}: {str(e)}")
                    continue

            logger.info(f"Successfully loaded {len(loaded_images)} images")
            return loaded_images

        except Exception as e:
            raise PreprocessingError(f"Failed to load batch from {directory}: {str(e)}")

    def load_multiple(self, paths: List[Union[str, Path]]) -> List[Tuple[Path, np.ndarray]]:
        """
        Load multiple images from a list of paths.

        Args:
            paths: List of image file paths

        Returns:
            List of (path, image) tuples
        """
        loaded_images = []

        for path in paths:
            try:
                path = Path(path)
                image = self.load_single(path)
                loaded_images.append((path, image))
            except Exception as e:
                logger.warning(f"Skipping {path}: {str(e)}")
                continue

        return loaded_images

    def get_image_info(self, path: Union[str, Path]) -> dict:
        """
        Get information about an image without loading the full data.

        Args:
            path: Path to image file

        Returns:
            Dictionary with image metadata
        """
        path = validate_image_path(path)

        try:
            if self.use_pil:
                with Image.open(path) as img:
                    return {
                        'path': str(path),
                        'format': img.format,
                        'mode': img.mode,
                        'size': img.size,  # (width, height)
                        'width': img.width,
                        'height': img.height,
                    }
            else:
                # Use OpenCV to get image info
                image = cv2.imread(str(path))
                if image is None:
                    raise PreprocessingError(f"Failed to read image: {path}")

                return {
                    'path': str(path),
                    'shape': image.shape,  # (height, width, channels)
                    'height': image.shape[0],
                    'width': image.shape[1],
                    'channels': image.shape[2] if len(image.shape) > 2 else 1,
                    'dtype': str(image.dtype),
                }

        except Exception as e:
            raise PreprocessingError(f"Failed to get image info for {path}: {str(e)}")

    def validate_batch(self, images: List[np.ndarray]) -> bool:
        """
        Validate that a batch of images can be processed together.

        Args:
            images: List of image arrays

        Returns:
            True if batch is valid

        Raises:
            InvalidInputError: If batch validation fails
        """
        if not images:
            raise InvalidInputError("Empty image batch")

        # Check that all images have the same number of channels
        channels = [img.shape[2] if len(img.shape) > 2 else 1 for img in images]
        if len(set(channels)) > 1:
            raise InvalidInputError(
                f"Images in batch have different channel counts: {set(channels)}"
            )

        return True

    def resize_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Resize an image to target size.

        Args:
            image: Input image array
            target_size: (width, height) tuple
            interpolation: OpenCV interpolation method

        Returns:
            Resized image
        """
        return cv2.resize(image, target_size, interpolation=interpolation)

    def __repr__(self) -> str:
        """String representation of image loader."""
        backend = "PIL" if self.use_pil else "OpenCV"
        return f"ImageLoader(backend='{backend}')"

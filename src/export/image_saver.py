"""Image saving utilities for depth maps and visualizations."""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.exceptions import ExportError
from ..utils.validators import validate_directory

logger = get_logger(__name__)


class ImageSaver:
    """
    Handles saving depth maps and visualizations to disk.

    Supports multiple formats:
    - PNG (lossless, recommended for depth maps)
    - JPEG (lossy, smaller file size)
    - TIFF (16-bit support for high-precision depth)
    """

    def __init__(self, output_dir: Union[str, Path] = "./data/output"):
        """
        Initialize image saver.

        Args:
            output_dir: Base directory for saving outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Image saver initialized: {self.output_dir}")

    def save_depth_map(
        self,
        depth: np.ndarray,
        filename: str,
        format: str = "png",
        normalize: bool = True,
        subdirectory: Optional[str] = None
    ) -> Path:
        """
        Save depth map to file.

        Args:
            depth: Depth map array
            filename: Output filename (without extension)
            format: Image format (png, jpg, tiff)
            normalize: Whether to normalize to uint8/uint16
            subdirectory: Optional subdirectory within output_dir

        Returns:
            Path to saved file

        Raises:
            ExportError: If saving fails
        """
        try:
            # Determine output path
            if subdirectory:
                output_path = self.output_dir / subdirectory
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = self.output_dir

            # Add extension
            filepath = output_path / f"{filename}.{format.lower()}"

            # Normalize depth based on format
            if format.lower() in ['png', 'jpg', 'jpeg']:
                if normalize:
                    depth_save = (depth * 255).astype(np.uint8)
                else:
                    depth_save = depth.astype(np.uint8)
            elif format.lower() in ['tiff', 'tif']:
                if normalize:
                    depth_save = (depth * 65535).astype(np.uint16)
                else:
                    depth_save = depth.astype(np.uint16)
            else:
                raise ExportError(f"Unsupported format: {format}")

            # Save image
            success = cv2.imwrite(str(filepath), depth_save)

            if not success:
                raise ExportError(f"Failed to write image: {filepath}")

            logger.debug(f"Saved depth map: {filepath.name}")
            return filepath

        except Exception as e:
            raise ExportError(f"Failed to save depth map: {str(e)}")

    def save_colored_depth(
        self,
        colored_depth: np.ndarray,
        filename: str,
        format: str = "png",
        subdirectory: Optional[str] = None
    ) -> Path:
        """
        Save colorized depth map.

        Args:
            colored_depth: RGB colored depth map
            filename: Output filename
            format: Image format
            subdirectory: Optional subdirectory

        Returns:
            Path to saved file
        """
        try:
            # Determine output path
            if subdirectory:
                output_path = self.output_dir / subdirectory
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = self.output_dir

            filepath = output_path / f"{filename}.{format.lower()}"

            # Convert RGB to BGR for OpenCV
            colored_bgr = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)

            # Save
            success = cv2.imwrite(str(filepath), colored_bgr)

            if not success:
                raise ExportError(f"Failed to write image: {filepath}")

            logger.debug(f"Saved colored depth: {filepath.name}")
            return filepath

        except Exception as e:
            raise ExportError(f"Failed to save colored depth: {str(e)}")

    def save_overlay(
        self,
        overlay: np.ndarray,
        filename: str,
        format: str = "png",
        subdirectory: Optional[str] = None
    ) -> Path:
        """
        Save overlay image.

        Args:
            overlay: RGB overlay image
            filename: Output filename
            format: Image format
            subdirectory: Optional subdirectory

        Returns:
            Path to saved file
        """
        return self.save_colored_depth(overlay, filename, format, subdirectory)

    def save_comparison(
        self,
        images: List[np.ndarray],
        filename: str,
        labels: Optional[List[str]] = None,
        format: str = "png"
    ) -> Path:
        """
        Save side-by-side comparison of multiple images.

        Args:
            images: List of images to compare
            filename: Output filename
            labels: Optional labels for each image
            format: Image format

        Returns:
            Path to saved file
        """
        try:
            # Ensure all images have the same height
            heights = [img.shape[0] for img in images]
            target_height = max(heights)

            # Resize images to same height
            resized_images = []
            for img in images:
                if img.shape[0] != target_height:
                    aspect_ratio = img.shape[1] / img.shape[0]
                    new_width = int(target_height * aspect_ratio)
                    resized = cv2.resize(img, (new_width, target_height))
                else:
                    resized = img
                resized_images.append(resized)

            # Concatenate horizontally
            comparison = np.hstack(resized_images)

            # Add labels if provided
            if labels:
                comparison = self._add_labels(comparison, labels, len(images))

            filepath = self.output_dir / f"{filename}.{format.lower()}"

            # Convert RGB to BGR if needed
            if len(comparison.shape) == 3 and comparison.shape[2] == 3:
                comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

            success = cv2.imwrite(str(filepath), comparison)

            if not success:
                raise ExportError(f"Failed to write comparison: {filepath}")

            logger.debug(f"Saved comparison: {filepath.name}")
            return filepath

        except Exception as e:
            raise ExportError(f"Failed to save comparison: {str(e)}")

    def _add_labels(
        self,
        image: np.ndarray,
        labels: List[str],
        num_images: int
    ) -> np.ndarray:
        """Add text labels to comparison image."""
        labeled = image.copy()
        width_per_image = image.shape[1] // num_images

        for i, label in enumerate(labels):
            x = i * width_per_image + 10
            y = 30
            cv2.putText(
                labeled, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA
            )

        return labeled

    def save_batch(
        self,
        results: List[Tuple[str, np.ndarray, np.ndarray]],
        save_depth: bool = True,
        save_colored: bool = True,
        save_overlay: bool = True,
        colormap: str = "viridis"
    ) -> List[Path]:
        """
        Save batch processing results.

        Args:
            results: List of (filename, image, depth) tuples
            save_depth: Whether to save grayscale depth
            save_colored: Whether to save colored depth
            save_overlay: Whether to save overlay
            colormap: Colormap for colored depth

        Returns:
            List of saved file paths
        """
        from ..postprocessing.colormap_generator import ColormapGenerator

        colormap_gen = ColormapGenerator(default_colormap=colormap)
        saved_paths = []

        logger.info(f"Saving batch results: {len(results)} images")

        for filename, image, depth in results:
            try:
                # Save grayscale depth
                if save_depth:
                    path = self.save_depth_map(
                        depth, f"{filename}_depth",
                        subdirectory="depth_maps"
                    )
                    saved_paths.append(path)

                # Save colored depth
                if save_colored:
                    colored = colormap_gen.apply_colormap(depth, colormap=colormap)
                    path = self.save_colored_depth(
                        colored, f"{filename}_depth_color",
                        subdirectory="depth_maps"
                    )
                    saved_paths.append(path)

                # Save overlay
                if save_overlay:
                    colored = colormap_gen.apply_colormap(depth, colormap=colormap)
                    overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)
                    path = self.save_overlay(
                        overlay, f"{filename}_overlay",
                        subdirectory="overlays"
                    )
                    saved_paths.append(path)

            except Exception as e:
                logger.warning(f"Failed to save results for {filename}: {str(e)}")

        logger.info(f"Saved {len(saved_paths)} files")
        return saved_paths

    def create_timestamped_filename(self, base_name: str, extension: str = "png") -> str:
        """
        Create a filename with timestamp.

        Args:
            base_name: Base name for file
            extension: File extension

        Returns:
            Filename with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"

    def get_output_path(self, filename: str, subdirectory: Optional[str] = None) -> Path:
        """
        Get full output path for a filename.

        Args:
            filename: Filename
            subdirectory: Optional subdirectory

        Returns:
            Full path
        """
        if subdirectory:
            return self.output_dir / subdirectory / filename
        return self.output_dir / filename

    def __repr__(self) -> str:
        """String representation of image saver."""
        return f"ImageSaver(output_dir='{self.output_dir}')"

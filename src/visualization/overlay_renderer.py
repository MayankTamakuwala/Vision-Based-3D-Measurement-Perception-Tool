"""Depth overlay rendering utilities."""

import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import get_logger
from ..utils.exceptions import VisualizationError
from ..utils.validators import validate_image_array

logger = get_logger(__name__)


class OverlayRenderer:
    """
    Renders depth maps overlaid on original images.

    Supports:
    - Alpha blending
    - Multiple blend modes
    - Annotations (lines, points, text)
    """

    def __init__(self):
        """Initialize overlay renderer."""
        pass

    def blend_depth_with_image(
        self,
        image: np.ndarray,
        depth_colored: np.ndarray,
        alpha: float = 0.5,
        beta: Optional[float] = None
    ) -> np.ndarray:
        """
        Blend colored depth map with original image.

        Args:
            image: Original RGB image
            depth_colored: Colored depth map (RGB)
            alpha: Weight for original image (0.0 to 1.0)
            beta: Weight for depth (if None, uses 1-alpha)

        Returns:
            Blended RGB image

        Raises:
            VisualizationError: If blending fails
        """
        try:
            validate_image_array(image)
            validate_image_array(depth_colored)

            if beta is None:
                beta = 1.0 - alpha

            # Ensure same dimensions
            if image.shape != depth_colored.shape:
                depth_colored = cv2.resize(
                    depth_colored,
                    (image.shape[1], image.shape[0])
                )

            # Blend
            blended = cv2.addWeighted(image, alpha, depth_colored, beta, 0)

            logger.debug(f"Blended image with depth (alpha={alpha}, beta={beta})")
            return blended

        except Exception as e:
            raise VisualizationError(f"Failed to blend depth with image: {str(e)}")

    def create_side_by_side(
        self,
        image: np.ndarray,
        depth_colored: np.ndarray,
        add_separator: bool = True,
        separator_width: int = 2,
        separator_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Create side-by-side comparison of image and depth.

        Args:
            image: Original image
            depth_colored: Colored depth map
            add_separator: Whether to add a vertical separator line
            separator_width: Width of separator in pixels
            separator_color: RGB color of separator

        Returns:
            Side-by-side comparison image
        """
        try:
            # Ensure same height
            if image.shape[0] != depth_colored.shape[0]:
                depth_colored = cv2.resize(
                    depth_colored,
                    (int(depth_colored.shape[1] * image.shape[0] / depth_colored.shape[0]),
                    image.shape[0])
                )

            if add_separator:
                # Create separator
                separator = np.full(
                    (image.shape[0], separator_width, 3),
                    separator_color,
                    dtype=np.uint8
                )
                combined = np.hstack([image, separator, depth_colored])
            else:
                combined = np.hstack([image, depth_colored])

            logger.debug("Created side-by-side comparison")
            return combined

        except Exception as e:
            raise VisualizationError(f"Failed to create side-by-side: {str(e)}")

    def draw_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 0, 0),
        radius: int = 5,
        thickness: int = -1
    ) -> np.ndarray:
        """
        Draw a point on image.

        Args:
            image: Input image
            point: (x, y) coordinates
            color: RGB color
            radius: Circle radius
            thickness: Circle thickness (-1 for filled)

        Returns:
            Image with point drawn
        """
        result = image.copy()
        cv2.circle(result, point, radius, color, thickness)
        return result

    def draw_line(
        self,
        image: np.ndarray,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw a line between two points.

        Args:
            image: Input image
            point1: First point (x, y)
            point2: Second point (x, y)
            color: RGB color
            thickness: Line thickness

        Returns:
            Image with line drawn
        """
        result = image.copy()
        cv2.line(result, point1, point2, color, thickness)
        return result

    def draw_text(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.6,
        thickness: int = 2,
        background: bool = True,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        bg_alpha: float = 0.7
    ) -> np.ndarray:
        """
        Draw text on image with optional background.

        Args:
            image: Input image
            text: Text to draw
            position: (x, y) position
            color: Text RGB color
            font_scale: Font scale
            thickness: Text thickness
            background: Whether to draw background
            bg_color: Background RGB color
            bg_alpha: Background transparency

        Returns:
            Image with text drawn
        """
        result = image.copy()

        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = position

        # Draw background
        if background:
            # Create overlay
            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (x - 5, y - text_height - 5),
                (x + text_width + 5, y + baseline + 5),
                bg_color,
                -1
            )
            # Blend overlay
            result = cv2.addWeighted(overlay, bg_alpha, result, 1 - bg_alpha, 0)

        # Draw text
        cv2.putText(
            result, text, (x, y),
            font, font_scale, color, thickness, cv2.LINE_AA
        )

        return result

    def annotate_measurement(
        self,
        image: np.ndarray,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        distance: float,
        units: str = "units"
    ) -> np.ndarray:
        """
        Draw a measurement annotation between two points.

        Args:
            image: Input image
            point1: First point
            point2: Second point
            distance: Measured distance
            units: Distance units

        Returns:
            Annotated image
        """
        result = image.copy()

        # Draw line
        result = self.draw_line(result, point1, point2, color=(0, 255, 0), thickness=2)

        # Draw points
        result = self.draw_point(result, point1, color=(255, 0, 0), radius=6)
        result = self.draw_point(result, point2, color=(255, 0, 0), radius=6)

        # Draw distance label at midpoint
        mid_x = (point1[0] + point2[0]) // 2
        mid_y = (point1[1] + point2[1]) // 2
        label = f"{distance:.2f} {units}"

        result = self.draw_text(result, label, (mid_x, mid_y))

        return result

    def create_depth_overlay(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        colormap: str = "viridis",
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create complete depth overlay with colormap.

        Args:
            image: Original image
            depth: Depth map
            colormap: Colormap name
            alpha: Blend alpha

        Returns:
            Depth overlay image
        """
        from ..postprocessing.colormap_generator import ColormapGenerator

        # Generate colored depth
        colormap_gen = ColormapGenerator(default_colormap=colormap)
        depth_colored = colormap_gen.apply_colormap(depth, colormap=colormap)

        # Blend with original
        overlay = self.blend_depth_with_image(image, depth_colored, alpha=alpha)

        return overlay

    def __repr__(self) -> str:
        """String representation of overlay renderer."""
        return "OverlayRenderer()"

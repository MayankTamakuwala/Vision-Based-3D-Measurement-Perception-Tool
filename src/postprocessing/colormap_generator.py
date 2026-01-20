"""Colormap generation for depth map visualization."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Union

from ..utils.logger import get_logger
from ..utils.exceptions import VisualizationError
from ..utils.validators import validate_depth_map

logger = get_logger(__name__)


class ColormapGenerator:
    """
    Generates colorized visualizations of depth maps.

    Supports multiple color schemes:
    - viridis, plasma, inferno, magma (perceptually uniform)
    - jet, turbo (traditional)
    - custom colormaps
    """

    # Available colormap names
    AVAILABLE_COLORMAPS = [
        'viridis', 'plasma', 'inferno', 'magma',
        'jet', 'turbo', 'rainbow', 'hot', 'cool',
        'gray', 'bone', 'copper', 'spring', 'summer',
        'autumn', 'winter'
    ]

    def __init__(self, default_colormap: str = 'viridis'):
        """
        Initialize colormap generator.

        Args:
            default_colormap: Default colormap to use
        """
        self.default_colormap = default_colormap
        self._validate_colormap(default_colormap)

    def _validate_colormap(self, colormap: str) -> str:
        """Validate colormap name."""
        if colormap not in self.AVAILABLE_COLORMAPS:
            logger.warning(
                f"Colormap '{colormap}' not in recommended list. "
                f"Available: {', '.join(self.AVAILABLE_COLORMAPS[:5])}..."
            )
        return colormap

    def apply_colormap(
        self,
        depth: np.ndarray,
        colormap: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Apply colormap to depth map.

        Args:
            depth: Input depth map (2D array)
            colormap: Colormap name (if None, uses default)
            normalize: Whether to normalize depth to [0, 1] before coloring

        Returns:
            RGB image with colormap applied (H, W, 3)

        Raises:
            VisualizationError: If colormap application fails
        """
        try:
            validate_depth_map(depth)

            colormap = colormap or self.default_colormap

            # Normalize depth to [0, 1] if requested
            if normalize:
                depth_normalized = self._normalize_depth(depth)
            else:
                depth_normalized = depth

            # Get matplotlib colormap
            cmap = cm.get_cmap(colormap)

            # Apply colormap (returns RGBA)
            colored = cmap(depth_normalized)

            # Convert to RGB uint8
            rgb = (colored[:, :, :3] * 255).astype(np.uint8)

            logger.debug(f"Applied colormap '{colormap}' to depth map")
            return rgb

        except Exception as e:
            raise VisualizationError(f"Failed to apply colormap: {str(e)}")

    def apply_colormap_opencv(
        self,
        depth: np.ndarray,
        colormap_type: int = cv2.COLORMAP_VIRIDIS,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Apply colormap using OpenCV (faster but fewer options).

        Args:
            depth: Input depth map
            colormap_type: OpenCV colormap constant
            normalize: Whether to normalize depth first

        Returns:
            RGB image with colormap applied
        """
        try:
            validate_depth_map(depth)

            # Normalize to uint8
            if normalize:
                depth_uint8 = self._to_uint8(depth)
            else:
                depth_uint8 = depth.astype(np.uint8)

            # Apply OpenCV colormap (returns BGR)
            colored_bgr = cv2.applyColorMap(depth_uint8, colormap_type)

            # Convert BGR to RGB
            colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

            logger.debug(f"Applied OpenCV colormap (type={colormap_type})")
            return colored_rgb

        except Exception as e:
            raise VisualizationError(f"Failed to apply OpenCV colormap: {str(e)}")

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth to [0, 1] range."""
        min_val = depth.min()
        max_val = depth.max()

        if max_val == min_val:
            return np.zeros_like(depth)

        return (depth - min_val) / (max_val - min_val)

    def _to_uint8(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth to uint8 [0, 255] range."""
        normalized = self._normalize_depth(depth)
        return (normalized * 255).astype(np.uint8)

    def create_depth_heatmap(
        self,
        depth: np.ndarray,
        colormap: Optional[str] = None
    ) -> np.ndarray:
        """
        Create a heatmap visualization of depth.

        Args:
            depth: Input depth map
            colormap: Colormap name

        Returns:
            RGB heatmap image
        """
        colormap = colormap or self.default_colormap
        return self.apply_colormap(depth, colormap=colormap, normalize=True)

    def create_grayscale_depth(
        self,
        depth: np.ndarray,
        invert: bool = False
    ) -> np.ndarray:
        """
        Create grayscale visualization of depth.

        Args:
            depth: Input depth map
            invert: Whether to invert colors (far=white, near=black)

        Returns:
            Grayscale image (uint8)
        """
        try:
            validate_depth_map(depth)

            grayscale = self._to_uint8(depth)

            if invert:
                grayscale = 255 - grayscale

            logger.debug("Created grayscale depth visualization")
            return grayscale

        except Exception as e:
            raise VisualizationError(f"Failed to create grayscale depth: {str(e)}")

    def create_custom_colormap(
        self,
        depth: np.ndarray,
        colors: list,
        positions: Optional[list] = None
    ) -> np.ndarray:
        """
        Create a custom colormap from a list of colors.

        Args:
            depth: Input depth map
            colors: List of RGB tuples, e.g., [(0,0,255), (0,255,0), (255,0,0)]
            positions: Optional list of positions [0,1] for each color

        Returns:
            RGB image with custom colormap
        """
        try:
            from matplotlib.colors import LinearSegmentedColormap

            validate_depth_map(depth)

            # Create custom colormap
            if positions is None:
                positions = np.linspace(0, 1, len(colors))

            # Normalize colors to [0, 1]
            colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors]

            # Create colormap
            cmap = LinearSegmentedColormap.from_list('custom', list(zip(positions, colors_normalized)))

            # Normalize depth
            depth_normalized = self._normalize_depth(depth)

            # Apply colormap
            colored = cmap(depth_normalized)
            rgb = (colored[:, :, :3] * 255).astype(np.uint8)

            logger.debug("Applied custom colormap")
            return rgb

        except Exception as e:
            raise VisualizationError(f"Failed to create custom colormap: {str(e)}")

    def create_depth_legend(
        self,
        depth_range: tuple,
        colormap: Optional[str] = None,
        height: int = 256,
        width: int = 30
    ) -> np.ndarray:
        """
        Create a colorbar/legend for depth values.

        Args:
            depth_range: (min, max) depth values
            colormap: Colormap name
            height: Height of legend in pixels
            width: Width of legend in pixels

        Returns:
            RGB legend image
        """
        try:
            colormap = colormap or self.default_colormap

            # Create gradient from 0 to 1
            gradient = np.linspace(0, 1, height)[:, np.newaxis]
            gradient = np.repeat(gradient, width, axis=1)

            # Flip vertically so max is at top
            gradient = np.flipud(gradient)

            # Apply colormap
            legend = self.apply_colormap(gradient, colormap=colormap, normalize=False)

            logger.debug(f"Created depth legend for range {depth_range}")
            return legend

        except Exception as e:
            raise VisualizationError(f"Failed to create depth legend: {str(e)}")

    def get_colormap_names(self) -> list:
        """Get list of available colormap names."""
        return self.AVAILABLE_COLORMAPS.copy()

    def preview_colormaps(
        self,
        depth: np.ndarray,
        colormaps: Optional[list] = None
    ) -> dict:
        """
        Preview multiple colormaps on the same depth map.

        Args:
            depth: Input depth map
            colormaps: List of colormap names (if None, uses default set)

        Returns:
            Dictionary mapping colormap names to colored images
        """
        if colormaps is None:
            colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'turbo']

        previews = {}
        for cmap in colormaps:
            try:
                previews[cmap] = self.apply_colormap(depth, colormap=cmap)
            except Exception as e:
                logger.warning(f"Failed to preview colormap '{cmap}': {str(e)}")

        return previews

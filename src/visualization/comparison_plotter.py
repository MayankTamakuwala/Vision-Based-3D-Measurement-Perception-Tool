"""Matplotlib-based comparison plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.exceptions import VisualizationError

logger = get_logger(__name__)


class ComparisonPlotter:
    """
    Creates matplotlib-based comparison visualizations.

    Supports:
    - Side-by-side comparisons
    - Grid layouts
    - Depth histograms
    - Depth profiles
    """

    def __init__(self, figsize: Tuple[int, int] = (15, 5), dpi: int = 100):
        """
        Initialize comparison plotter.

        Args:
            figsize: Default figure size (width, height) in inches
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_side_by_side(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        overlay: Optional[np.ndarray] = None,
        titles: Optional[List[str]] = None,
        cmap: str = 'viridis'
    ) -> plt.Figure:
        """
        Create side-by-side comparison plot.

        Args:
            image: Original RGB image
            depth: Depth map
            overlay: Optional overlay image
            titles: Optional titles for each subplot
            cmap: Colormap for depth

        Returns:
            Matplotlib figure
        """
        try:
            num_images = 3 if overlay is not None else 2
            fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5), dpi=self.dpi)

            if num_images == 2:
                axes = [axes[0], axes[1]]
            else:
                axes = [axes[0], axes[1], axes[2]]

            # Default titles
            if titles is None:
                titles = ['Original', 'Depth', 'Overlay'][:num_images]

            # Plot original image
            axes[0].imshow(image)
            axes[0].set_title(titles[0], fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # Plot depth map
            im = axes[1].imshow(depth, cmap=cmap)
            axes[1].set_title(titles[1], fontsize=12, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            # Plot overlay if provided
            if overlay is not None:
                axes[2].imshow(overlay)
                axes[2].set_title(titles[2], fontsize=12, fontweight='bold')
                axes[2].axis('off')

            plt.tight_layout()
            logger.debug("Created side-by-side comparison plot")

            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to create side-by-side plot: {str(e)}")

    def plot_depth_histogram(
        self,
        depth: np.ndarray,
        bins: int = 256,
        title: str = "Depth Distribution"
    ) -> plt.Figure:
        """
        Plot depth value histogram.

        Args:
            depth: Depth map
            bins: Number of histogram bins
            title: Plot title

        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)

            # Flatten depth and create histogram
            depth_flat = depth.flatten()
            ax.hist(depth_flat, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)

            ax.set_xlabel('Depth Value', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add statistics
            stats_text = f"Mean: {depth.mean():.3f}\nMedian: {np.median(depth):.3f}\nStd: {depth.std():.3f}"
            ax.text(
                0.98, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9
            )

            plt.tight_layout()
            logger.debug("Created depth histogram")

            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to create histogram: {str(e)}")

    def plot_depth_profile(
        self,
        depth: np.ndarray,
        line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        title: str = "Depth Profile"
    ) -> plt.Figure:
        """
        Plot depth profile along a line.

        Args:
            depth: Depth map
            line_coords: ((x1, y1), (x2, y2)) line coordinates (if None, uses center horizontal)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=self.dpi)

            # Default line: horizontal center
            if line_coords is None:
                y = depth.shape[0] // 2
                profile = depth[y, :]
                x_coords = np.arange(depth.shape[1])
                line_start = (0, y)
                line_end = (depth.shape[1] - 1, y)
            else:
                (x1, y1), (x2, y2) = line_coords
                # Sample along line using Bresenham's algorithm (approximate)
                num_samples = max(abs(x2 - x1), abs(y2 - y1))
                x_coords = np.linspace(x1, x2, num_samples).astype(int)
                y_coords = np.linspace(y1, y2, num_samples).astype(int)
                profile = depth[y_coords, x_coords]
                x_coords = np.arange(num_samples)
                line_start = (x1, y1)
                line_end = (x2, y2)

            # Show depth map with line
            im = ax1.imshow(depth, cmap='viridis')
            ax1.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r-', linewidth=2)
            ax1.scatter(*line_start, c='red', s=100, marker='o', zorder=5)
            ax1.scatter(*line_end, c='red', s=100, marker='o', zorder=5)
            ax1.set_title('Depth Map with Profile Line', fontsize=11, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

            # Plot profile
            ax2.plot(x_coords, profile, 'b-', linewidth=2)
            ax2.fill_between(x_coords, profile, alpha=0.3)
            ax2.set_xlabel('Position along line', fontsize=10)
            ax2.set_ylabel('Depth', fontsize=10)
            ax2.set_title('Depth Profile', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            logger.debug("Created depth profile")

            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to create depth profile: {str(e)}")

    def plot_grid(
        self,
        images: List[np.ndarray],
        titles: Optional[List[str]] = None,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        cmap: Optional[str] = None
    ) -> plt.Figure:
        """
        Create grid layout of images.

        Args:
            images: List of images to display
            titles: Optional titles for each image
            rows: Number of rows (auto-calculated if None)
            cols: Number of columns (auto-calculated if None)
            cmap: Colormap (if applicable)

        Returns:
            Matplotlib figure
        """
        try:
            num_images = len(images)

            # Calculate grid dimensions
            if rows is None and cols is None:
                cols = int(np.ceil(np.sqrt(num_images)))
                rows = int(np.ceil(num_images / cols))
            elif rows is None:
                rows = int(np.ceil(num_images / cols))
            elif cols is None:
                cols = int(np.ceil(num_images / rows))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=self.dpi)

            # Flatten axes array
            if rows * cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, (ax, img) in enumerate(zip(axes, images)):
                if cmap and len(img.shape) == 2:
                    im = ax.imshow(img, cmap=cmap)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.imshow(img)

                if titles and idx < len(titles):
                    ax.set_title(titles[idx], fontsize=10, fontweight='bold')

                ax.axis('off')

            # Hide unused subplots
            for idx in range(num_images, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            logger.debug(f"Created grid plot: {rows}x{cols}")

            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to create grid plot: {str(e)}")

    def save_figure(self, fig: plt.Figure, filepath: Path, dpi: Optional[int] = None):
        """
        Save matplotlib figure to file.

        Args:
            fig: Matplotlib figure
            filepath: Output file path
            dpi: DPI for saved image (if None, uses figure DPI)
        """
        try:
            dpi = dpi or self.dpi
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            logger.debug(f"Saved figure: {filepath}")
            plt.close(fig)

        except Exception as e:
            raise VisualizationError(f"Failed to save figure: {str(e)}")

    def __repr__(self) -> str:
        """String representation of comparison plotter."""
        return f"ComparisonPlotter(figsize={self.figsize}, dpi={self.dpi})"

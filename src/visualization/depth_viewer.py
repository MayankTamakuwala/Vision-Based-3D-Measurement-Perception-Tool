"""3D visualization of depth maps as point clouds."""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple

from ..utils.logger import get_logger
from ..utils.exceptions import VisualizationError
from ..utils.validators import validate_image_array

logger = get_logger(__name__)


class DepthViewer:
    """
    Creates 3D visualizations of depth maps.

    Supports:
    - Point cloud generation from depth maps
    - Interactive 3D plots with Plotly
    - Color mapping from original images
    - Configurable sampling and filtering
    """

    def __init__(
        self,
        point_skip: int = 1,
        depth_scale: float = 100.0,
        invert_depth: bool = False
    ):
        """
        Initialize depth viewer.

        Args:
            point_skip: Sample every Nth point (higher = faster, lower quality)
            depth_scale: Scale factor for depth values
            invert_depth: Invert depth values (near/far flip)
        """
        self.point_skip = point_skip
        self.depth_scale = depth_scale
        self.invert_depth = invert_depth

        logger.info(f"Initialized DepthViewer (skip={point_skip}, scale={depth_scale})")

    def depth_to_point_cloud(
        self,
        depth: np.ndarray,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert depth map to 3D point cloud.

        Args:
            depth: Depth map (H, W)
            image: Optional RGB image for point colors (H, W, 3)
            mask: Optional binary mask (H, W) - only include masked points

        Returns:
            Tuple of (points, colors) where:
                points: (N, 3) array of [x, y, z] coordinates
                colors: (N, 3) array of RGB colors or None

        Raises:
            VisualizationError: If depth map is invalid
        """
        try:
            validate_image_array(depth)

            if depth.ndim != 2:
                raise VisualizationError("Depth map must be 2D array")

            # Get dimensions
            height, width = depth.shape

            # Create coordinate grid
            y_coords, x_coords = np.meshgrid(
                np.arange(0, height, self.point_skip),
                np.arange(0, width, self.point_skip),
                indexing='ij'
            )

            # Sample depth values
            z_coords = depth[::self.point_skip, ::self.point_skip]

            # Apply mask if provided
            if mask is not None:
                mask_sampled = mask[::self.point_skip, ::self.point_skip]
                valid = mask_sampled.astype(bool)
            else:
                valid = np.ones_like(z_coords, dtype=bool)

            # Invert depth if requested (for visualization)
            if self.invert_depth:
                z_coords = -z_coords

            # Scale depth
            z_coords = z_coords * self.depth_scale

            # Flatten and filter
            x_flat = x_coords[valid].flatten()
            y_flat = y_coords[valid].flatten()
            z_flat = z_coords[valid].flatten()

            # Stack into point cloud
            points = np.stack([x_flat, y_flat, z_flat], axis=1)

            # Extract colors if image provided
            colors = None
            if image is not None:
                if image.shape[:2] != depth.shape:
                    logger.warning("Image and depth shapes don't match, skipping colors")
                else:
                    # Sample colors
                    colors_sampled = image[::self.point_skip, ::self.point_skip]
                    colors = colors_sampled[valid].reshape(-1, 3)

            logger.info(f"Generated point cloud with {len(points)} points")
            return points, colors

        except Exception as e:
            raise VisualizationError(f"Failed to generate point cloud: {str(e)}")

    def create_interactive_plot(
        self,
        depth: np.ndarray,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        title: str = "3D Depth Visualization",
        colorscale: str = "Viridis",
        point_size: int = 2
    ) -> go.Figure:
        """
        Create interactive 3D plot with Plotly.

        Args:
            depth: Depth map
            image: Optional RGB image for coloring
            mask: Optional binary mask
            title: Plot title
            colorscale: Plotly colorscale name
            point_size: Size of points in plot

        Returns:
            Plotly Figure object

        Raises:
            VisualizationError: If plotting fails
        """
        try:
            # Generate point cloud
            points, colors = self.depth_to_point_cloud(depth, image, mask)

            if len(points) == 0:
                raise VisualizationError("Point cloud is empty")

            # Extract coordinates
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            # Determine coloring
            if colors is not None:
                # Use RGB colors from image
                rgb_strings = [
                    f'rgb({int(c[0])},{int(c[1])},{int(c[2])})'
                    for c in colors
                ]
                marker_dict = {
                    'size': point_size,
                    'color': rgb_strings,
                    'opacity': 0.8
                }
            else:
                # Color by depth
                marker_dict = {
                    'size': point_size,
                    'color': z,
                    'colorscale': colorscale,
                    'colorbar': {'title': 'Depth'},
                    'opacity': 0.8
                }

            # Create scatter plot
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=marker_dict,
                hovertemplate=(
                    'X: %{x:.0f}<br>'
                    'Y: %{y:.0f}<br>'
                    'Depth: %{z:.2f}<br>'
                    '<extra></extra>'
                )
            )

            # Create figure
            fig = go.Figure(data=[scatter])

            # Update layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X (pixels)',
                    yaxis_title='Y (pixels)',
                    zaxis_title='Depth',
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700
            )

            logger.info("Created interactive 3D plot")
            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to create 3D plot: {str(e)}")

    def create_rotating_animation(
        self,
        depth: np.ndarray,
        image: Optional[np.ndarray] = None,
        num_frames: int = 36,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create rotating animation of point cloud.

        Args:
            depth: Depth map
            image: Optional RGB image
            num_frames: Number of rotation frames
            output_path: Optional path to save animation

        Returns:
            Plotly Figure with animation
        """
        try:
            # Generate point cloud
            points, colors = self.depth_to_point_cloud(depth, image)

            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            # Determine coloring
            if colors is not None:
                rgb_strings = [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in colors]
                color_value = rgb_strings
            else:
                color_value = z

            # Create frames for rotation
            frames = []
            angles = np.linspace(0, 360, num_frames, endpoint=False)

            for angle in angles:
                # Calculate camera position
                rad = np.radians(angle)
                eye_x = 1.5 * np.cos(rad)
                eye_y = 1.5 * np.sin(rad)

                frame = go.Frame(
                    layout=dict(
                        scene=dict(
                            camera=dict(
                                eye=dict(x=eye_x, y=eye_y, z=1.5)
                            )
                        )
                    ),
                    name=f'frame_{int(angle)}'
                )
                frames.append(frame)

            # Create base scatter
            scatter = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=color_value,
                    colorscale='Viridis' if colors is None else None,
                    opacity=0.8
                )
            )

            # Create figure
            fig = go.Figure(
                data=[scatter],
                frames=frames
            )

            # Add play button
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 100},
                                'fromcurrent': True,
                                'mode': 'immediate'
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0},
                                'mode': 'immediate'
                            }]
                        }
                    ]
                }],
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Depth',
                    aspectmode='data'
                ),
                height=700
            )

            if output_path:
                fig.write_html(output_path)
                logger.info(f"Saved rotating animation to {output_path}")

            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to create animation: {str(e)}")

    def create_surface_plot(
        self,
        depth: np.ndarray,
        colorscale: str = "Viridis",
        title: str = "Depth Surface"
    ) -> go.Figure:
        """
        Create 3D surface plot of depth map.

        Args:
            depth: Depth map
            colorscale: Plotly colorscale
            title: Plot title

        Returns:
            Plotly Figure
        """
        try:
            validate_image_array(depth)

            if self.invert_depth:
                depth = -depth

            # Downsample if too large
            if depth.shape[0] > 100 or depth.shape[1] > 100:
                step = max(depth.shape[0] // 100, depth.shape[1] // 100)
                depth_sampled = depth[::step, ::step]
            else:
                depth_sampled = depth

            # Create surface
            surface = go.Surface(
                z=depth_sampled * self.depth_scale,
                colorscale=colorscale,
                colorbar=dict(title='Depth')
            )

            fig = go.Figure(data=[surface])

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Depth',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700
            )

            logger.info("Created 3D surface plot")
            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to create surface plot: {str(e)}")

    def save_point_cloud(
        self,
        depth: np.ndarray,
        output_path: str,
        image: Optional[np.ndarray] = None,
        format: str = "ply"
    ):
        """
        Save point cloud to file.

        Args:
            depth: Depth map
            output_path: Output file path
            image: Optional RGB image
            format: File format ('ply', 'xyz', 'npy')

        Raises:
            VisualizationError: If saving fails
        """
        try:
            points, colors = self.depth_to_point_cloud(depth, image)

            if format == "npy":
                # Save as numpy arrays
                if colors is not None:
                    data = np.concatenate([points, colors], axis=1)
                else:
                    data = points
                np.save(output_path, data)

            elif format == "xyz":
                # Simple XYZ format
                with open(output_path, 'w') as f:
                    for i, point in enumerate(points):
                        if colors is not None:
                            color = colors[i]
                            f.write(f"{point[0]} {point[1]} {point[2]} "
                                   f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
                        else:
                            f.write(f"{point[0]} {point[1]} {point[2]}\n")

            elif format == "ply":
                # PLY format with colors
                with open(output_path, 'w') as f:
                    # Header
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(points)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    if colors is not None:
                        f.write("property uchar red\n")
                        f.write("property uchar green\n")
                        f.write("property uchar blue\n")
                    f.write("end_header\n")

                    # Data
                    for i, point in enumerate(points):
                        if colors is not None:
                            color = colors[i]
                            f.write(f"{point[0]} {point[1]} {point[2]} "
                                   f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
                        else:
                            f.write(f"{point[0]} {point[1]} {point[2]}\n")

            else:
                raise VisualizationError(f"Unsupported format: {format}")

            logger.info(f"Saved point cloud to {output_path} ({format} format)")

        except Exception as e:
            raise VisualizationError(f"Failed to save point cloud: {str(e)}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DepthViewer("
            f"skip={self.point_skip}, "
            f"scale={self.depth_scale}, "
            f"invert={self.invert_depth})"
        )

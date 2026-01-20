#!/usr/bin/env python3
"""Gradio web interface for quick depth estimation demos."""

import gradio as gr
import numpy as np
import sys
from pathlib import Path
import cv2
from typing import Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.postprocessing.colormap_generator import ColormapGenerator
from src.visualization.overlay_renderer import OverlayRenderer
from src.visualization.depth_viewer import DepthViewer
from src.measurement.distance_calculator import DistanceCalculator
from src.utils.logger import setup_logger

logger = setup_logger("gradio_app", level="INFO")

# Global state
depth_estimator = None
colormap_gen = ColormapGenerator()
overlay_renderer = OverlayRenderer()
depth_viewer = DepthViewer(point_skip=3)
distance_calc = DistanceCalculator()

# Store current depth and image
current_depth = None
current_image = None
selected_points = []


def initialize_model(model_type: str) -> str:
    """Initialize depth estimation model."""
    global depth_estimator
    try:
        logger.info(f"Loading model: {model_type}")
        depth_estimator = DepthEstimator(model_type=model_type, optimize=True)
        return f"✅ Model {model_type} loaded successfully on {depth_estimator.device_manager.device_name}"
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return f"❌ Error loading model: {str(e)}"


def process_image(
    image: np.ndarray,
    model_type: str,
    colormap: str,
    show_overlay: bool
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Process uploaded image and generate depth map.

    Args:
        image: Input RGB image
        model_type: Model variant to use
        colormap: Colormap for visualization
        show_overlay: Whether to show overlay

    Returns:
        Tuple of (depth_colored, output_image, info_text)
    """
    global depth_estimator, current_depth, current_image, selected_points

    try:
        # Reset points
        selected_points = []

        # Initialize model if needed
        if depth_estimator is None or depth_estimator.model_type != model_type:
            initialize_model(model_type)

        # Store current image
        current_image = image

        # Estimate depth
        logger.info("Estimating depth...")
        current_depth = depth_estimator.estimate_depth(image, normalize=True)

        # Create colored depth map
        depth_colored = colormap_gen.apply_colormap(current_depth, colormap=colormap)

        # Create output image
        if show_overlay:
            output_image = overlay_renderer.blend_depth_with_image(
                image, depth_colored, alpha=0.5
            )
        else:
            output_image = depth_colored

        # Get performance stats
        perf = depth_estimator.get_performance_summary()

        info_text = f"""
### Processing Complete ✅

**Image Size:** {image.shape[1]}x{image.shape[0]}
**Model:** {model_type}
**Device:** {depth_estimator.device_manager.device_name}
**Inference Time:** {perf.get('avg_inference_ms', 0):.2f} ms
**Depth Range:** [{current_depth.min():.3f}, {current_depth.max():.3f}]

Click on the depth map to measure distances between points!
"""

        return depth_colored, output_image, info_text

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        error_img = np.zeros_like(image)
        return error_img, error_img, f"❌ Error: {str(e)}"


def handle_point_click(image: np.ndarray, evt: gr.SelectData) -> Tuple[np.ndarray, str]:
    """
    Handle clicks on depth map for distance measurement.

    Args:
        image: Current depth colored image
        evt: Gradio SelectData event

    Returns:
        Tuple of (annotated_image, measurement_text)
    """
    global current_depth, current_image, selected_points

    if current_depth is None:
        return image, "⚠️ Please process an image first"

    try:
        # Get click coordinates
        x, y = int(evt.index[0]), int(evt.index[1])
        selected_points.append((x, y))

        # Annotate image with point
        annotated = image.copy()
        cv2.circle(annotated, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(
            annotated, f"P{len(selected_points)}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

        measurement_text = f"**Selected Points:** {len(selected_points)}\n\n"

        # If two points selected, calculate distance
        if len(selected_points) == 2:
            p1, p2 = selected_points[0], selected_points[1]

            # Draw line
            cv2.line(annotated, p1, p2, (0, 255, 0), 2)

            # Calculate distance
            distance = distance_calc.point_to_point_distance(
                current_depth, p1, p2, method="euclidean"
            )

            # Get depth values
            depth1 = current_depth[p1[1], p1[0]]
            depth2 = current_depth[p2[1], p2[0]]

            measurement_text += f"""
**Point 1:** ({p1[0]}, {p1[1]}) - Depth: {depth1:.3f}
**Point 2:** ({p2[0]}, {p2[1]}) - Depth: {depth2:.3f}
**Distance:** {distance:.2f} pixels
**Depth Difference:** {abs(depth2 - depth1):.3f}

Click "Clear Points" to start new measurement.
"""

            # Reset for next measurement
            selected_points = []
        else:
            measurement_text += f"Point {len(selected_points)}: ({x}, {y})\n"
            measurement_text += "Click another point to measure distance."

        return annotated, measurement_text

    except Exception as e:
        logger.error(f"Error handling click: {e}")
        return image, f"❌ Error: {str(e)}"


def clear_points() -> Tuple[Optional[np.ndarray], str]:
    """Clear selected measurement points."""
    global selected_points, current_depth
    selected_points = []

    if current_depth is not None:
        # Regenerate clean depth colored image
        depth_colored = colormap_gen.apply_colormap(current_depth)
        return depth_colored, "Points cleared. Click to select new points."

    return None, "No active depth map."


def generate_3d_view(point_skip: int, colorscale: str):
    """Generate 3D point cloud visualization."""
    global current_depth, current_image

    if current_depth is None:
        return None, "⚠️ Please process an image first"

    try:
        # Update point skip
        depth_viewer.point_skip = point_skip

        # Generate 3D plot
        logger.info("Generating 3D visualization...")
        fig = depth_viewer.create_interactive_plot(
            current_depth,
            image=current_image,
            title="Interactive 3D Depth Visualization",
            colorscale=colorscale,
            point_size=2
        )

        return fig, f"✅ 3D visualization generated ({point_skip}x sampling)"

    except Exception as e:
        logger.error(f"Error generating 3D view: {e}")
        return None, f"❌ Error: {str(e)}"


# Build Gradio interface
def build_interface():
    """Build the Gradio interface."""

    with gr.Blocks(title="Vision3D - Depth Estimation", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎯 Vision-Based 3D Measurement & Perception Tool

        Upload an image to estimate depth, measure distances, and explore in 3D!
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📥 Input")
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=300
                )

                model_dropdown = gr.Dropdown(
                    choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                    value="MiDaS_small",
                    label="Model (small=faster, large=better)"
                )

                colormap_dropdown = gr.Dropdown(
                    choices=["viridis", "plasma", "inferno", "magma", "jet", "turbo"],
                    value="viridis",
                    label="Colormap"
                )

                overlay_checkbox = gr.Checkbox(
                    label="Show Overlay",
                    value=True
                )

                process_btn = gr.Button("🚀 Process Image", variant="primary")

                info_box = gr.Markdown("Upload an image and click Process!")

            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 🎨 Depth Map")
                depth_output = gr.Image(
                    label="Depth Visualization",
                    type="numpy",
                    height=300,
                    interactive=False
                )

                gr.Markdown("### 📊 Result")
                result_output = gr.Image(
                    label="Final Output",
                    type="numpy",
                    height=300,
                    interactive=False
                )

        # Measurement section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📏 Measurement Tool")
                gr.Markdown("Click two points on the depth map to measure distance")

                measurement_output = gr.Markdown("Click on the depth map above...")
                clear_btn = gr.Button("🗑️ Clear Points")

        # 3D Visualization section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🌐 3D Visualization")

                with gr.Row():
                    point_skip_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Point Sampling (higher = faster)"
                    )

                    colorscale_3d = gr.Dropdown(
                        choices=["Viridis", "Plasma", "Blues", "Earth", "Rainbow"],
                        value="Viridis",
                        label="3D Colorscale"
                    )

                view_3d_btn = gr.Button("🎲 Generate 3D View", variant="secondary")

                plot_3d = gr.Plot(label="Interactive 3D Point Cloud")
                view_3d_status = gr.Markdown("")

        # Event handlers
        process_btn.click(
            fn=process_image,
            inputs=[input_image, model_dropdown, colormap_dropdown, overlay_checkbox],
            outputs=[depth_output, result_output, info_box]
        )

        depth_output.select(
            fn=handle_point_click,
            inputs=[depth_output],
            outputs=[depth_output, measurement_output]
        )

        clear_btn.click(
            fn=clear_points,
            outputs=[depth_output, measurement_output]
        )

        view_3d_btn.click(
            fn=generate_3d_view,
            inputs=[point_skip_slider, colorscale_3d],
            outputs=[plot_3d, view_3d_status]
        )

        # Examples
        gr.Markdown("### 📚 Examples")
        gr.Markdown("Upload your own images or try the processing on sample data!")

    return app


def main():
    """Launch the Gradio app."""
    logger.info("=" * 70)
    logger.info("Vision-Based 3D Measurement & Perception Tool")
    logger.info("Gradio Web Interface")
    logger.info("=" * 70)

    app = build_interface()

    logger.info("Launching Gradio interface...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        pwa=True
    )


if __name__ == "__main__":
    main()

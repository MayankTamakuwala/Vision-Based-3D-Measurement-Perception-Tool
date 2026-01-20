#!/usr/bin/env python3
"""Streamlit dashboard for Vision-Based 3D Measurement & Perception Tool."""

import streamlit as st
import numpy as np
import sys
from pathlib import Path
import cv2
import time
from PIL import Image
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.preprocessing.image_loader import ImageLoader
from src.postprocessing.colormap_generator import ColormapGenerator
from src.postprocessing.depth_processor import DepthProcessor
from src.visualization.overlay_renderer import OverlayRenderer
from src.visualization.depth_viewer import DepthViewer
from src.visualization.comparison_plotter import ComparisonPlotter
from src.measurement.distance_calculator import DistanceCalculator
from src.measurement.scale_estimator import ScaleEstimator
from src.measurement.object_analyzer import ObjectAnalyzer
from src.export.json_exporter import JSONExporter
from src.export.image_saver import ImageSaver
from src.utils.logger import setup_logger

logger = setup_logger("streamlit_app", level="INFO")

# Page configuration
st.set_page_config(
    page_title="Vision3D - Depth Estimation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'depth_estimator' not in st.session_state:
        st.session_state.depth_estimator = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_depth' not in st.session_state:
        st.session_state.current_depth = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0
    if 'selected_points' not in st.session_state:
        st.session_state.selected_points = []
    if 'calibration' not in st.session_state:
        st.session_state.calibration = None


@st.cache_resource
def load_model(model_type: str, optimize: bool = True):
    """Load depth estimation model (cached)."""
    logger.info(f"Loading model: {model_type}")
    return DepthEstimator(model_type=model_type, optimize=optimize)


def sidebar_config():
    """Render sidebar configuration."""
    st.sidebar.title("⚙️ Configuration")

    # Model selection
    st.sidebar.subheader("Model Settings")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
        help="Larger models = better quality, slower inference"
    )

    optimize = st.sidebar.checkbox("GPU Optimization", value=True)

    # Visualization settings
    st.sidebar.subheader("Visualization")
    colormap = st.sidebar.selectbox(
        "Colormap",
        ["viridis", "plasma", "inferno", "magma", "jet", "turbo", "coolwarm"],
        index=0
    )

    overlay_alpha = st.sidebar.slider(
        "Overlay Transparency",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    # Processing settings
    st.sidebar.subheader("Processing")
    normalize = st.sidebar.checkbox("Normalize Depth", value=True)
    remove_outliers = st.sidebar.checkbox("Remove Outliers", value=True)
    smooth_depth = st.sidebar.checkbox("Smooth Depth", value=False)

    return {
        'model_type': model_type,
        'optimize': optimize,
        'colormap': colormap,
        'overlay_alpha': overlay_alpha,
        'normalize': normalize,
        'remove_outliers': remove_outliers,
        'smooth_depth': smooth_depth
    }


def page_single_image():
    """Single image processing page."""
    st.markdown('<p class="main-header">📷 Single Image Processing</p>', unsafe_allow_html=True)

    config = sidebar_config()

    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Supported formats: PNG, JPG, JPEG, BMP"
    )

    if uploaded_file:
        # Load image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Convert to RGB if needed
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        st.session_state.current_image = image_np

        # Display original
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image_np, use_container_width=True)
            st.caption(f"Size: {image_np.shape[1]}x{image_np.shape[0]}")

        # Process button
        if st.button("🚀 Process Image", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Load model
                    estimator = load_model(config['model_type'], config['optimize'])
                    st.session_state.depth_estimator = estimator

                    # Estimate depth
                    start_time = time.time()
                    depth = estimator.estimate_depth(
                        image_np,
                        normalize=config['normalize'],
                        remove_outliers=config['remove_outliers']
                    )
                    processing_time = time.time() - start_time
                    st.session_state.processing_time = processing_time

                    # Apply smoothing if requested
                    if config['smooth_depth']:
                        depth_processor = DepthProcessor()
                        depth = depth_processor.apply_bilateral_filter(depth)

                    st.session_state.current_depth = depth

                    st.success(f"✅ Processing complete in {processing_time:.3f}s")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    return

        # Display results
        if st.session_state.current_depth is not None:
            depth = st.session_state.current_depth

            with col2:
                st.subheader("Depth Map")

                # Generate colored depth map
                colormap_gen = ColormapGenerator()
                depth_colored = colormap_gen.apply_colormap(depth, colormap=config['colormap'])

                st.image(depth_colored, use_container_width=True)
                st.caption(f"Range: [{depth.min():.3f}, {depth.max():.3f}]")

            # Overlay
            st.subheader("Depth Overlay")
            overlay_renderer = OverlayRenderer()
            overlay = overlay_renderer.blend_depth_with_image(
                image_np,
                depth_colored,
                alpha=config['overlay_alpha']
            )
            st.image(overlay, use_container_width=True)

            # Statistics
            st.subheader("📊 Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Processing Time", f"{st.session_state.processing_time:.3f}s")
            with col2:
                st.metric("Mean Depth", f"{depth.mean():.3f}")
            with col3:
                st.metric("Std Dev", f"{depth.std():.3f}")
            with col4:
                perf = st.session_state.depth_estimator.get_performance_summary()
                st.metric("FPS", f"{perf.get('fps', 0):.2f}")

            # Export options
            st.subheader("💾 Export")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Download Depth Map"):
                    # Convert to PIL Image
                    depth_img = Image.fromarray((depth * 255).astype(np.uint8))
                    st.download_button(
                        "Download PNG",
                        data=depth_img.tobytes(),
                        file_name="depth_map.png",
                        mime="image/png"
                    )

            with col2:
                if st.button("Download Overlay"):
                    overlay_img = Image.fromarray(overlay)
                    st.download_button(
                        "Download PNG",
                        data=overlay_img.tobytes(),
                        file_name="depth_overlay.png",
                        mime="image/png"
                    )

            with col3:
                if st.button("Download Metadata"):
                    json_exporter = JSONExporter()
                    metadata = json_exporter.create_metadata(
                        depth=depth,
                        image_path="uploaded_image",
                        model_type=config['model_type']
                    )
                    metadata_str = json.dumps(metadata, indent=2)
                    st.download_button(
                        "Download JSON",
                        data=metadata_str,
                        file_name="depth_metadata.json",
                        mime="application/json"
                    )


def page_measurements():
    """Measurement and analysis page."""
    st.markdown('<p class="main-header">📏 Measurements & Analysis</p>', unsafe_allow_html=True)

    if st.session_state.current_depth is None:
        st.warning("⚠️ Please process an image first in the Single Image page")
        return

    depth = st.session_state.current_depth
    image = st.session_state.current_image

    # Measurement tools
    st.subheader("Measurement Tools")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Distance", "🎯 Calibration", "🔍 Object Detection", "📈 Depth Profile"
    ])

    with tab1:
        st.write("**Point-to-Point Distance Measurement**")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Point 1")
            p1_x = st.number_input("X1", 0, depth.shape[1]-1, 100)
            p1_y = st.number_input("Y1", 0, depth.shape[0]-1, 100)

        with col2:
            st.write("Point 2")
            p2_x = st.number_input("X2", 0, depth.shape[1]-1, 200)
            p2_y = st.number_input("Y2", 0, depth.shape[0]-1, 200)

        if st.button("Calculate Distance"):
            calc = DistanceCalculator()
            distance = calc.point_to_point_distance(
                depth, (p1_x, p1_y), (p2_x, p2_y)
            )

            depth1 = depth[p1_y, p1_x]
            depth2 = depth[p2_y, p2_x]

            st.success(f"**Distance:** {distance:.2f} pixels")
            st.info(f"**Depth Difference:** {abs(depth2 - depth1):.3f}")

            # Visualize
            overlay_renderer = OverlayRenderer()
            annotated = overlay_renderer.annotate_distance(
                image, depth, (p1_x, p1_y), (p2_x, p2_y)
            )
            st.image(annotated, use_container_width=True)

    with tab2:
        st.write("**Scale Calibration**")
        st.write("Use a known reference distance to calibrate depth to real-world units")

        known_distance = st.number_input(
            "Known Distance (meters)",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1
        )

        col1, col2 = st.columns(2)
        with col1:
            ref_x1 = st.number_input("Reference X1", 0, depth.shape[1]-1, 100, key="ref_x1")
            ref_y1 = st.number_input("Reference Y1", 0, depth.shape[0]-1, 100, key="ref_y1")

        with col2:
            ref_x2 = st.number_input("Reference X2", 0, depth.shape[1]-1, 200, key="ref_x2")
            ref_y2 = st.number_input("Reference Y2", 0, depth.shape[0]-1, 200, key="ref_y2")

        if st.button("Calibrate"):
            scale_est = ScaleEstimator()
            calibration = scale_est.calibrate(
                depth,
                known_distance=known_distance,
                reference_points=[(ref_x1, ref_y1), (ref_x2, ref_y2)],
                unit="meters"
            )

            st.session_state.calibration = calibration

            st.success(f"✅ Calibration complete!")
            st.info(f"**Scale Factor:** {calibration.scale_factor:.6f}")

    with tab3:
        st.write("**Depth-Based Object Detection**")

        threshold = st.slider("Depth Threshold", 0.0, 1.0, 0.5, 0.01)
        min_area = st.slider("Minimum Object Area (pixels)", 100, 10000, 1000, 100)

        if st.button("Detect Objects"):
            analyzer = ObjectAnalyzer()

            objects = analyzer.detect_objects(
                depth,
                depth_threshold=threshold,
                min_area=min_area
            )

            st.success(f"Found {len(objects)} objects")

            # Visualize
            overlay_renderer = OverlayRenderer()
            annotated = overlay_renderer.annotate_objects(image, objects)
            st.image(annotated, use_container_width=True)

            # Show object details
            if objects:
                st.write("**Detected Objects:**")
                for i, obj in enumerate(objects[:5]):  # Show first 5
                    with st.expander(f"Object {i+1}"):
                        st.write(f"- Area: {obj['area']} pixels")
                        st.write(f"- Centroid: ({obj['centroid'][0]:.0f}, {obj['centroid'][1]:.0f})")
                        st.write(f"- Mean Depth: {obj['mean_depth']:.3f}")
                        st.write(f"- Bounding Box: {obj['bbox']}")

    with tab4:
        st.write("**Depth Profile Analysis**")

        profile_type = st.radio(
            "Profile Type",
            ["Horizontal", "Vertical", "Custom Line"],
            horizontal=True
        )

        if profile_type in ["Horizontal", "Vertical"]:
            position = st.slider(
                f"{'Y' if profile_type == 'Horizontal' else 'X'} Position",
                0,
                depth.shape[0 if profile_type == 'Horizontal' else 1] - 1,
                depth.shape[0 if profile_type == 'Horizontal' else 1] // 2
            )

            if st.button("Generate Profile"):
                import matplotlib.pyplot as plt

                if profile_type == "Horizontal":
                    profile = depth[position, :]
                    x_label = "X Position"
                else:
                    profile = depth[:, position]
                    x_label = "Y Position"

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(profile)
                ax.set_xlabel(x_label)
                ax.set_ylabel("Depth")
                ax.set_title(f"{profile_type} Depth Profile")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)


def page_3d_visualization():
    """3D visualization page."""
    st.markdown('<p class="main-header">🌐 3D Visualization</p>', unsafe_allow_html=True)

    if st.session_state.current_depth is None:
        st.warning("⚠️ Please process an image first in the Single Image page")
        return

    depth = st.session_state.current_depth
    image = st.session_state.current_image

    # 3D settings
    st.subheader("3D Visualization Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        point_skip = st.slider("Point Sampling", 1, 10, 3, help="Higher = fewer points, faster")

    with col2:
        depth_scale = st.slider("Depth Scale", 0.1, 5.0, 1.0, 0.1)

    with col3:
        colorscale = st.selectbox(
            "Colorscale",
            ["Viridis", "Plasma", "Blues", "Earth", "Rainbow", "Jet"]
        )

    use_image_colors = st.checkbox("Use Image Colors", value=True)

    if st.button("🎲 Generate 3D Visualization", type="primary"):
        with st.spinner("Generating 3D point cloud..."):
            try:
                viewer = DepthViewer(
                    point_skip=point_skip,
                    depth_scale=depth_scale,
                    invert_depth=True
                )

                fig = viewer.create_interactive_plot(
                    depth,
                    image=image if use_image_colors else None,
                    title="Interactive 3D Depth Point Cloud",
                    colorscale=colorscale,
                    point_size=2
                )

                st.plotly_chart(fig, use_container_width=True)

                st.success("✅ 3D visualization complete!")

                # Surface plot option
                if st.checkbox("Show Surface Plot"):
                    surface_fig = viewer.create_surface_plot(
                        depth,
                        colorscale=colorscale,
                        title="3D Depth Surface"
                    )
                    st.plotly_chart(surface_fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def page_batch_processing():
    """Batch processing page."""
    st.markdown('<p class="main-header">📁 Batch Processing</p>', unsafe_allow_html=True)

    st.info("💡 For batch processing, please use the CLI tool: `python scripts/run_batch.py`")

    st.code("""
# Example batch processing command:
python scripts/run_batch.py input_dir/ -o output/ -m DPT_Hybrid -c viridis

# Options:
#   -o, --output: Output directory
#   -m, --model: Model type (DPT_Large, DPT_Hybrid, MiDaS_small)
#   -c, --colormap: Colormap for visualization
#   -p, --pattern: File pattern (*.jpg, *.png, etc.)
    """, language="bash")

    st.subheader("Batch Processing Features")
    st.write("""
    - Process entire directories of images
    - Consistent settings across all images
    - Progress tracking and performance metrics
    - Automatic output organization
    - JSON metadata for each image
    """)


def main():
    """Main app function."""
    init_session_state()

    # Header
    st.sidebar.image("https://via.placeholder.com/300x100.png?text=Vision3D", use_container_width=True)

    # Navigation
    st.sidebar.title("📑 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "📷 Single Image",
            "📏 Measurements",
            "🌐 3D Visualization",
            "📁 Batch Processing",
            "ℹ️ About"
        ]
    )

    # Route to pages
    if page == "📷 Single Image":
        page_single_image()
    elif page == "📏 Measurements":
        page_measurements()
    elif page == "🌐 3D Visualization":
        page_3d_visualization()
    elif page == "📁 Batch Processing":
        page_batch_processing()
    elif page == "ℹ️ About":
        st.markdown('<p class="main-header">ℹ️ About</p>', unsafe_allow_html=True)
        st.markdown("""
        ## Vision-Based 3D Measurement & Perception Tool

        An ML-driven perception system that converts 2D images into 3D spatial understanding
        using monocular depth estimation with MiDaS models.

        ### Features
        - 🎯 Monocular depth estimation (DPT_Large, DPT_Hybrid, MiDaS_small)
        - 🚀 GPU-optimized inference (H100/A100/MPS)
        - 📏 Distance measurements and calibration
        - 🔍 Object detection and analysis
        - 🌐 Interactive 3D visualization
        - 💾 Multiple export formats

        ### Tech Stack
        - **Core**: PyTorch, OpenCV, NumPy
        - **Models**: MiDaS (Intel ISL)
        - **UI**: Streamlit, Gradio
        - **Visualization**: Matplotlib, Plotly

        ### Resources
        - [GitHub Repository](https://github.com/yourusername/Vision-Based-3D-Measurement-Perception-Tool)
        - [Documentation](docs/)
        - [MiDaS Paper](https://arxiv.org/abs/1907.01341)

        ---
        Built with ❤️ for AR/ML research and computer vision applications
        """)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Vision3D** v1.0.0")
    st.sidebar.markdown("© 2025 - MIT License")


if __name__ == "__main__":
    main()

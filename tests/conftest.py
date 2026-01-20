"""Pytest fixtures for Vision3D tests."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import cv2

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.core.device_manager import DeviceManager
from src.preprocessing.image_loader import ImageLoader
from src.postprocessing.colormap_generator import ColormapGenerator
from src.measurement.distance_calculator import DistanceCalculator
from src.visualization.overlay_renderer import OverlayRenderer


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp(prefix="vision3d_test_")
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image_small():
    """Create small test image (256x256)."""
    # Create RGB image with gradient
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img[i, :, 0] = i  # Red gradient
        img[:, i, 1] = i  # Green gradient
    img[:, :, 2] = 128  # Blue constant
    return img


@pytest.fixture
def sample_image_large():
    """Create larger test image (512x512)."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Create checkerboard pattern
    for i in range(0, 512, 64):
        for j in range(0, 512, 64):
            if (i // 64 + j // 64) % 2 == 0:
                img[i:i+64, j:j+64] = [255, 255, 255]
    return img


@pytest.fixture
def sample_depth_map():
    """Create sample depth map (256x256)."""
    # Create depth map with distance from center
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    depth = np.sqrt((x - cx)**2 + (y - cy)**2)
    # Normalize to [0, 1]
    depth = depth / depth.max()
    return depth.astype(np.float32)


@pytest.fixture
def sample_depth_map_with_objects():
    """Create depth map with distinct objects at different depths."""
    depth = np.ones((256, 256), dtype=np.float32) * 0.8

    # Object 1: Close circle
    cv2.circle(depth, (80, 80), 30, 0.3, -1)

    # Object 2: Far rectangle
    depth[150:200, 150:220] = 0.9

    # Object 3: Medium distance
    cv2.ellipse(depth, (180, 80), (40, 25), 0, 0, 360, 0.6, -1)

    return depth


@pytest.fixture
def sample_video_frames():
    """Generate list of sample video frames."""
    frames = []
    for i in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Moving square
        x_pos = 50 + i * 20
        cv2.rectangle(frame, (x_pos, 100), (x_pos + 50, 150), (0, 255, 0), -1)
        frames.append(frame)
    return frames


@pytest.fixture
def device_manager():
    """Create device manager instance."""
    return DeviceManager()


@pytest.fixture
def depth_estimator_light():
    """Create depth estimator with light model (MiDaS_small)."""
    return DepthEstimator(model_type="MiDaS_small", optimize=False)


@pytest.fixture(scope="session")
def depth_estimator_cached():
    """Create cached depth estimator for multiple tests (session scope)."""
    return DepthEstimator(model_type="MiDaS_small", optimize=False)


@pytest.fixture
def image_loader():
    """Create image loader instance."""
    return ImageLoader()


@pytest.fixture
def colormap_generator():
    """Create colormap generator instance."""
    return ColormapGenerator()


@pytest.fixture
def distance_calculator():
    """Create distance calculator instance."""
    return DistanceCalculator()


@pytest.fixture
def overlay_renderer():
    """Create overlay renderer instance."""
    return OverlayRenderer()


@pytest.fixture
def sample_calibration_data():
    """Sample calibration data for scale estimation."""
    return {
        'known_distance': 1.0,
        'reference_points': [(100, 100), (200, 200)],
        'unit': 'meters'
    }


@pytest.fixture
def mock_model_weights():
    """Create mock model weights for testing without downloading."""
    # This would be used to mock torch.hub.load in unit tests
    pass


@pytest.fixture(autouse=True)
def reset_numpy_random():
    """Reset numpy random seed before each test for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def sample_point_cloud():
    """Generate sample 3D point cloud data."""
    n_points = 1000
    points = np.random.rand(n_points, 3) * 100
    colors = np.random.randint(0, 256, (n_points, 3), dtype=np.uint8)
    return points, colors


@pytest.fixture
def performance_metrics_data():
    """Sample performance metrics for testing."""
    return {
        'inference_times': [0.05, 0.045, 0.048, 0.052, 0.046],
        'total_images': 5,
        'device': 'cpu',
        'model_type': 'MiDaS_small'
    }


# Markers for conditional test execution
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring model download"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to integration tests
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.slow)

        # Add GPU marker to tests with 'gpu' in name
        if "gpu" in item.name.lower():
            item.add_marker(pytest.mark.requires_gpu)

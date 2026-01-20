"""Tests for DepthEstimator."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.utils.exceptions import ModelLoadError, InvalidInputError


@pytest.mark.unit
@pytest.mark.requires_model
class TestDepthEstimator:
    """Test cases for DepthEstimator."""

    def test_initialization(self, depth_estimator_light):
        """Test DepthEstimator initialization."""
        estimator = depth_estimator_light

        assert estimator.model is not None
        assert estimator.transform is not None
        assert estimator.model_type == "MiDaS_small"

    def test_estimate_depth_small_image(self, depth_estimator_light, sample_image_small):
        """Test depth estimation on small image."""
        estimator = depth_estimator_light
        depth = estimator.estimate_depth(sample_image_small)

        assert depth.shape == sample_image_small.shape[:2]
        assert depth.dtype == np.float32
        assert depth.min() >= 0
        assert depth.max() <= 1

    def test_estimate_depth_normalization(self, depth_estimator_light, sample_image_small):
        """Test depth normalization."""
        estimator = depth_estimator_light

        # With normalization
        depth_norm = estimator.estimate_depth(sample_image_small, normalize=True)
        assert depth_norm.min() >= 0
        assert depth_norm.max() <= 1

        # Without normalization
        depth_raw = estimator.estimate_depth(sample_image_small, normalize=False)
        assert not (depth_raw.min() == 0 and depth_raw.max() == 1)

    def test_batch_processing(self, depth_estimator_light, sample_image_small):
        """Test batch depth estimation."""
        estimator = depth_estimator_light

        # Create batch of 3 images
        batch = [sample_image_small, sample_image_small, sample_image_small]
        depths = estimator.estimate_depth_batch(batch)

        assert len(depths) == 3
        for depth in depths:
            assert depth.shape == sample_image_small.shape[:2]

    def test_invalid_input_shape(self, depth_estimator_light):
        """Test error handling for invalid input."""
        estimator = depth_estimator_light

        # 1D array should fail
        with pytest.raises((InvalidInputError, ValueError)):
            estimator.estimate_depth(np.zeros(100))

        # 4D array should fail
        with pytest.raises((InvalidInputError, ValueError)):
            estimator.estimate_depth(np.zeros((10, 256, 256, 3)))

    def test_grayscale_image(self, depth_estimator_light):
        """Test depth estimation on grayscale image."""
        estimator = depth_estimator_light

        # Create grayscale image
        gray_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        depth = estimator.estimate_depth(gray_img)

        assert depth.shape == (256, 256)
        assert depth.dtype == np.float32

    def test_performance_tracking(self, depth_estimator_light, sample_image_small):
        """Test performance metrics tracking."""
        estimator = depth_estimator_light

        # Process image
        estimator.estimate_depth(sample_image_small)

        # Get performance summary
        perf = estimator.get_performance_summary()

        assert 'total_inferences' in perf
        assert 'avg_inference_ms' in perf
        assert perf['total_inferences'] > 0

    def test_model_type_validation(self):
        """Test model type validation."""
        # Valid model type
        estimator = DepthEstimator(model_type="MiDaS_small", optimize=False)
        assert estimator.model_type == "MiDaS_small"

        # Invalid model type should raise error
        with pytest.raises(ModelLoadError):
            DepthEstimator(model_type="InvalidModel")

    def test_depth_range_consistency(self, depth_estimator_light, sample_image_small):
        """Test that depth values are consistent across runs."""
        estimator = depth_estimator_light

        depth1 = estimator.estimate_depth(sample_image_small)
        depth2 = estimator.estimate_depth(sample_image_small)

        # Should be very similar (small numerical differences allowed)
        np.testing.assert_allclose(depth1, depth2, rtol=1e-5)

    def test_reset_performance_metrics(self, depth_estimator_light, sample_image_small):
        """Test performance metrics reset."""
        estimator = depth_estimator_light

        # Process image
        estimator.estimate_depth(sample_image_small)

        # Reset metrics
        estimator.performance_tracker.reset()

        perf = estimator.get_performance_summary()
        assert perf['total_inferences'] == 0


@pytest.mark.slow
@pytest.mark.requires_model
class TestDepthEstimatorModels:
    """Test different model variants."""

    def test_dpt_large(self, sample_image_small):
        """Test DPT_Large model."""
        estimator = DepthEstimator(model_type="DPT_Large", optimize=False)
        depth = estimator.estimate_depth(sample_image_small)

        assert depth.shape == sample_image_small.shape[:2]

    def test_dpt_hybrid(self, sample_image_small):
        """Test DPT_Hybrid model."""
        estimator = DepthEstimator(model_type="DPT_Hybrid", optimize=False)
        depth = estimator.estimate_depth(sample_image_small)

        assert depth.shape == sample_image_small.shape[:2]

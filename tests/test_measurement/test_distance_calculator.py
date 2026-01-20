"""Tests for DistanceCalculator."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.measurement.distance_calculator import DistanceCalculator
from src.utils.exceptions import MeasurementError


@pytest.mark.unit
class TestDistanceCalculator:
    """Test cases for DistanceCalculator."""

    def test_initialization(self):
        """Test DistanceCalculator initialization."""
        calc = DistanceCalculator()
        assert calc is not None

    def test_point_to_point_distance_euclidean(self, sample_depth_map):
        """Test Euclidean distance calculation."""
        calc = DistanceCalculator()

        p1 = (100, 100)
        p2 = (150, 150)

        distance = calc.point_to_point_distance(
            sample_depth_map, p1, p2, method="euclidean"
        )

        assert isinstance(distance, float)
        assert distance > 0

        # Distance should be approximately sqrt(50^2 + 50^2 + depth_diff^2)
        # which is >= sqrt(50^2 + 50^2) ~ 70.7
        assert distance >= 70.0

    def test_point_to_point_distance_manhattan(self, sample_depth_map):
        """Test Manhattan distance calculation."""
        calc = DistanceCalculator()

        p1 = (100, 100)
        p2 = (150, 150)

        distance = calc.point_to_point_distance(
            sample_depth_map, p1, p2, method="manhattan"
        )

        assert isinstance(distance, float)
        assert distance > 0

        # Manhattan distance for 50x50 displacement is at least 100
        assert distance >= 100

    def test_horizontal_distance(self, sample_depth_map):
        """Test horizontal distance (same y)."""
        calc = DistanceCalculator()

        p1 = (50, 100)
        p2 = (150, 100)

        distance = calc.point_to_point_distance(
            sample_depth_map, p1, p2, method="euclidean"
        )

        # Mainly horizontal movement
        assert distance > 0

    def test_vertical_distance(self, sample_depth_map):
        """Test vertical distance (same x)."""
        calc = DistanceCalculator()

        p1 = (100, 50)
        p2 = (100, 150)

        distance = calc.point_to_point_distance(
            sample_depth_map, p1, p2, method="euclidean"
        )

        # Mainly vertical movement
        assert distance > 0

    def test_region_depth_analysis(self, sample_depth_map):
        """Test region depth analysis."""
        calc = DistanceCalculator()

        region = (50, 50, 150, 150)  # (x1, y1, x2, y2)

        analysis = calc.analyze_region_depth(sample_depth_map, region)

        assert 'mean_depth' in analysis
        assert 'min_depth' in analysis
        assert 'max_depth' in analysis
        assert 'std_depth' in analysis
        assert 'area_pixels' in analysis

        assert analysis['area_pixels'] == 100 * 100

    def test_depth_difference_map(self, sample_depth_map):
        """Test depth difference map generation."""
        calc = DistanceCalculator()

        depth2 = sample_depth_map * 1.2  # Scaled version

        diff_map = calc.depth_difference_map(sample_depth_map, depth2)

        assert diff_map.shape == sample_depth_map.shape
        assert np.all(diff_map >= 0)  # Absolute differences

    def test_invalid_points(self, sample_depth_map):
        """Test error handling for invalid points."""
        calc = DistanceCalculator()

        # Point outside depth map
        p1 = (100, 100)
        p2 = (300, 300)  # Out of bounds for 256x256

        with pytest.raises((MeasurementError, IndexError)):
            calc.point_to_point_distance(sample_depth_map, p1, p2)

    def test_same_point_distance(self, sample_depth_map):
        """Test distance between same point (should be 0)."""
        calc = DistanceCalculator()

        p1 = (100, 100)
        p2 = (100, 100)

        distance = calc.point_to_point_distance(sample_depth_map, p1, p2)

        assert distance == 0.0

    def test_depth_profile_horizontal(self, sample_depth_map):
        """Test horizontal depth profile."""
        calc = DistanceCalculator()

        y = 128
        profile = calc.get_depth_profile(sample_depth_map, y=y, direction='horizontal')

        assert len(profile) == sample_depth_map.shape[1]
        assert isinstance(profile, np.ndarray)

    def test_depth_profile_vertical(self, sample_depth_map):
        """Test vertical depth profile."""
        calc = DistanceCalculator()

        x = 128
        profile = calc.get_depth_profile(sample_depth_map, x=x, direction='vertical')

        assert len(profile) == sample_depth_map.shape[0]
        assert isinstance(profile, np.ndarray)

    def test_average_depth_along_line(self, sample_depth_map):
        """Test average depth calculation along a line."""
        calc = DistanceCalculator()

        p1 = (50, 50)
        p2 = (200, 200)

        avg_depth = calc.average_depth_along_line(sample_depth_map, p1, p2)

        assert isinstance(avg_depth, float)
        assert 0 <= avg_depth <= 1

    def test_multiple_measurements(self, sample_depth_map):
        """Test multiple consecutive measurements."""
        calc = DistanceCalculator()

        points = [(50, 50), (100, 100), (150, 150), (200, 200)]

        distances = []
        for i in range(len(points) - 1):
            dist = calc.point_to_point_distance(
                sample_depth_map, points[i], points[i+1]
            )
            distances.append(dist)

        assert len(distances) == 3
        assert all(d > 0 for d in distances)


@pytest.mark.unit
class TestDistanceCalculatorEdgeCases:
    """Test edge cases for DistanceCalculator."""

    def test_zero_depth_map(self):
        """Test with all-zero depth map."""
        calc = DistanceCalculator()
        depth = np.zeros((100, 100), dtype=np.float32)

        p1 = (25, 25)
        p2 = (75, 75)

        distance = calc.point_to_point_distance(depth, p1, p2)

        # Only x,y distance, no depth component
        expected = np.sqrt(50**2 + 50**2)
        assert np.isclose(distance, expected, rtol=0.01)

    def test_uniform_depth_map(self):
        """Test with uniform depth values."""
        calc = DistanceCalculator()
        depth = np.ones((100, 100), dtype=np.float32) * 0.5

        p1 = (25, 25)
        p2 = (75, 75)

        distance = calc.point_to_point_distance(depth, p1, p2)

        # Only x,y distance since depths are equal
        expected = np.sqrt(50**2 + 50**2)
        assert np.isclose(distance, expected, rtol=0.01)

    def test_single_pixel_depth_map(self):
        """Test with 1x1 depth map."""
        calc = DistanceCalculator()
        depth = np.array([[0.5]], dtype=np.float32)

        p1 = (0, 0)
        p2 = (0, 0)

        distance = calc.point_to_point_distance(depth, p1, p2)
        assert distance == 0.0

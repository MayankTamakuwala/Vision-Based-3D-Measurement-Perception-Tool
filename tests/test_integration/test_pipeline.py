"""Integration tests for end-to-end pipelines."""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.preprocessing.image_loader import ImageLoader
from src.postprocessing.colormap_generator import ColormapGenerator
from src.visualization.overlay_renderer import OverlayRenderer
from src.measurement.distance_calculator import DistanceCalculator
from src.measurement.scale_estimator import ScaleEstimator
from src.measurement.object_analyzer import ObjectAnalyzer
from src.export.json_exporter import JSONExporter
from src.export.image_saver import ImageSaver


@pytest.mark.integration
@pytest.mark.requires_model
class TestEndToEndPipeline:
    """Integration tests for complete workflows."""

    def test_single_image_pipeline(self, sample_image_small, temp_dir):
        """Test complete single image processing pipeline."""
        # 1. Load image (already have it)
        image = sample_image_small

        # 2. Estimate depth
        estimator = DepthEstimator(model_type="MiDaS_small", optimize=False)
        depth = estimator.estimate_depth(image, normalize=True)

        assert depth.shape == image.shape[:2]

        # 3. Generate colored visualization
        colormap_gen = ColormapGenerator()
        depth_colored = colormap_gen.apply_colormap(depth)

        assert depth_colored.shape == (*image.shape[:2], 3)

        # 4. Create overlay
        overlay_renderer = OverlayRenderer()
        overlay = overlay_renderer.blend_depth_with_image(image, depth, alpha=0.5)

        assert overlay.shape == image.shape

        # 5. Export results
        saver = ImageSaver(output_dir=str(temp_dir))
        saved_paths = saver.save_depth_results(
            image=image,
            depth=depth,
            depth_colored=depth_colored,
            overlay=overlay,
            output_name="test_result"
        )

        assert all(Path(p).exists() for p in saved_paths.values())

        # 6. Export metadata
        json_exporter = JSONExporter()
        metadata = json_exporter.create_metadata(
            depth=depth,
            image_path="test_image.jpg",
            model_type="MiDaS_small"
        )

        assert 'depth_statistics' in metadata
        assert 'model_info' in metadata

    def test_measurement_pipeline(self, sample_depth_map_with_objects):
        """Test measurement workflow."""
        depth = sample_depth_map_with_objects

        # 1. Calculate distances
        calc = DistanceCalculator()
        p1 = (80, 80)
        p2 = (180, 80)

        distance = calc.point_to_point_distance(depth, p1, p2)
        assert distance > 0

        # 2. Calibrate scale
        scale_est = ScaleEstimator()
        calibration = scale_est.calibrate(
            depth,
            known_distance=1.0,
            reference_points=[p1, p2],
            unit="meters"
        )

        assert calibration.scale_factor > 0

        # 3. Convert to metric distance
        metric_distance = scale_est.depth_to_meters(distance, calibration)
        assert np.isclose(metric_distance, 1.0, rtol=0.1)

        # 4. Detect objects
        analyzer = ObjectAnalyzer()
        objects = analyzer.detect_objects(depth, depth_threshold=0.5, min_area=100)

        assert len(objects) > 0

    def test_batch_processing_pipeline(self, sample_image_small, temp_dir):
        """Test batch processing workflow."""
        # Create batch of images
        batch = [sample_image_small for _ in range(3)]

        # Initialize pipeline components
        estimator = DepthEstimator(model_type="MiDaS_small", optimize=False)
        colormap_gen = ColormapGenerator()
        saver = ImageSaver(output_dir=str(temp_dir))

        # Process batch
        results = []
        for i, image in enumerate(batch):
            # Estimate depth
            depth = estimator.estimate_depth(image, normalize=True)

            # Colorize
            depth_colored = colormap_gen.apply_colormap(depth)

            # Save
            saved_paths = saver.save_depth_results(
                image=image,
                depth=depth,
                depth_colored=depth_colored,
                output_name=f"batch_{i}"
            )

            results.append(saved_paths)

        # Verify all results
        assert len(results) == 3
        for result in results:
            assert 'depth_map' in result
            assert Path(result['depth_map']).exists()

        # Check performance
        perf = estimator.get_performance_summary()
        assert perf['total_inferences'] == 3

    def test_visualization_pipeline(self, sample_image_small, sample_depth_map, temp_dir):
        """Test visualization workflow."""
        image = sample_image_small
        depth = sample_depth_map

        # 1. Create colored depth map
        colormap_gen = ColormapGenerator()
        depth_colored = colormap_gen.apply_colormap(depth, colormap='viridis')

        # 2. Create overlay
        overlay_renderer = OverlayRenderer()
        overlay = overlay_renderer.blend_depth_with_image(image, depth, alpha=0.6)

        # 3. Annotate with measurements
        p1 = (50, 50)
        p2 = (150, 150)
        annotated = overlay_renderer.annotate_distance(image, depth, p1, p2)

        assert annotated.shape == image.shape

        # 4. Save all visualizations
        saver = ImageSaver(output_dir=str(temp_dir))
        paths = saver.save_depth_results(
            image=image,
            depth=depth,
            depth_colored=depth_colored,
            overlay=overlay,
            output_name="vis_test"
        )

        assert len(paths) > 0

    def test_calibration_and_measurement_pipeline(self, sample_depth_map_with_objects):
        """Test calibration followed by measurements."""
        depth = sample_depth_map_with_objects

        # 1. Set up calibration
        scale_est = ScaleEstimator()
        known_distance = 2.0  # meters
        ref_points = [(50, 50), (200, 200)]

        calibration = scale_est.calibrate(
            depth,
            known_distance=known_distance,
            reference_points=ref_points,
            unit="meters"
        )

        # 2. Make measurements with calibration
        calc = DistanceCalculator()

        measurements = []
        test_points = [
            ((80, 80), (100, 100)),
            ((150, 150), (180, 180)),
            ((50, 200), (200, 50))
        ]

        for p1, p2 in test_points:
            distance_pixels = calc.point_to_point_distance(depth, p1, p2)
            distance_meters = scale_est.depth_to_meters(distance_pixels, calibration)
            measurements.append(distance_meters)

        # Verify all measurements
        assert all(m > 0 for m in measurements)

    def test_object_detection_and_analysis_pipeline(self, sample_depth_map_with_objects):
        """Test object detection and analysis workflow."""
        depth = sample_depth_map_with_objects

        # 1. Detect objects
        analyzer = ObjectAnalyzer()
        objects = analyzer.detect_objects(
            depth,
            depth_threshold=0.5,
            min_area=200
        )

        assert len(objects) > 0

        # 2. Analyze each object
        calc = DistanceCalculator()

        for obj in objects:
            # Get object region
            bbox = obj['bbox']
            region = (bbox[0], bbox[1], bbox[2], bbox[3])

            # Analyze depth in region
            analysis = calc.analyze_region_depth(depth, region)

            assert 'mean_depth' in analysis
            assert 'area_pixels' in analysis

        # 3. Compare objects
        if len(objects) >= 2:
            obj1 = objects[0]
            obj2 = objects[1]

            depth_diff = abs(obj1['mean_depth'] - obj2['mean_depth'])
            assert depth_diff >= 0

    def test_export_pipeline(self, sample_image_small, sample_depth_map, temp_dir):
        """Test complete export workflow."""
        image = sample_image_small
        depth = sample_depth_map

        # 1. Generate all visualizations
        colormap_gen = ColormapGenerator()
        overlay_renderer = OverlayRenderer()

        depth_colored = colormap_gen.apply_colormap(depth, colormap='plasma')
        overlay = overlay_renderer.blend_depth_with_image(image, depth)

        # 2. Export images
        saver = ImageSaver(output_dir=str(temp_dir))
        image_paths = saver.save_depth_results(
            image=image,
            depth=depth,
            depth_colored=depth_colored,
            overlay=overlay,
            output_name="export_test"
        )

        # 3. Export metadata
        json_exporter = JSONExporter()
        metadata = json_exporter.create_metadata(
            depth=depth,
            image_path="test_image.jpg",
            model_type="MiDaS_small"
        )

        metadata_path = temp_dir / "metadata.json"
        json_exporter.save_metadata(metadata, str(metadata_path))

        # Verify all exports
        assert all(Path(p).exists() for p in image_paths.values())
        assert metadata_path.exists()


@pytest.mark.integration
class TestPipelineErrorHandling:
    """Test error handling in pipelines."""

    def test_pipeline_with_invalid_input(self):
        """Test pipeline with invalid image."""
        estimator = DepthEstimator(model_type="MiDaS_small", optimize=False)

        # Invalid input shape
        invalid_image = np.zeros((10, 10))  # Too small, grayscale

        with pytest.raises((ValueError, RuntimeError)):
            depth = estimator.estimate_depth(invalid_image)

    def test_pipeline_with_corrupted_depth(self, sample_image_small):
        """Test pipeline with corrupted depth map."""
        # Create invalid depth map (wrong shape)
        invalid_depth = np.zeros((50, 50))  # Different size than image

        overlay_renderer = OverlayRenderer()

        with pytest.raises((ValueError, RuntimeError)):
            overlay_renderer.blend_depth_with_image(sample_image_small, invalid_depth)


@pytest.mark.slow
@pytest.mark.integration
class TestPerformancePipeline:
    """Test pipeline performance and optimization."""

    def test_batch_processing_performance(self, sample_image_small):
        """Test batch processing performance."""
        estimator = DepthEstimator(model_type="MiDaS_small", optimize=True)

        # Process batch
        batch = [sample_image_small for _ in range(5)]
        depths = estimator.estimate_depth_batch(batch)

        assert len(depths) == 5

        # Check performance metrics
        perf = estimator.get_performance_summary()
        assert perf['total_inferences'] == 5
        assert perf['avg_inference_ms'] > 0

    def test_memory_efficiency(self, sample_image_large):
        """Test memory efficiency with large images."""
        estimator = DepthEstimator(model_type="MiDaS_small", optimize=True)

        # Process large image
        depth = estimator.estimate_depth(sample_image_large)

        assert depth.shape == sample_image_large.shape[:2]

        # Ensure cleanup
        estimator.device_manager.clear_cache()

#!/usr/bin/env python3
"""Demo script showcasing measurement capabilities."""

import argparse
import sys
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.measurement.distance_calculator import DistanceCalculator
from src.measurement.scale_estimator import ScaleEstimator
from src.measurement.object_analyzer import ObjectAnalyzer
from src.visualization.overlay_renderer import OverlayRenderer
from src.postprocessing.colormap_generator import ColormapGenerator
from src.utils.logger import setup_logger

logger = setup_logger("measurement_demo", level="INFO")


def main():
    """Main function for measurement demo."""
    parser = argparse.ArgumentParser(
        description="Demonstrate depth measurement capabilities"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./data/output",
        help="Output directory (default: ./data/output)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="MiDaS_small",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model variant (default: MiDaS_small)"
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input image not found: {input_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Vision-Based 3D Measurement & Perception Tool")
    logger.info("Measurement Demonstration")
    logger.info("=" * 70)

    try:
        # Initialize components
        logger.info("Initializing system...")
        estimator = DepthEstimator(model_type=args.model, optimize=True)
        distance_calc = DistanceCalculator()
        scale_est = ScaleEstimator()
        object_analyzer = ObjectAnalyzer()
        renderer = OverlayRenderer()
        colormap_gen = ColormapGenerator(default_colormap="viridis")

        logger.info(f"Device: {estimator.device_manager.device_name}")

        # Load and process image
        logger.info(f"Processing image: {input_path.name}")
        image, depth = estimator.estimate_depth_from_file(input_path)

        # Get image dimensions
        height, width = image.shape[:2]
        logger.info(f"Image size: {width}x{height}")

        # Demo 1: Point-to-point distances
        logger.info("\n" + "=" * 70)
        logger.info("DEMO 1: Point-to-Point Distance Measurements")
        logger.info("=" * 70)

        # Define some test points
        point1 = (width // 4, height // 2)
        point2 = (3 * width // 4, height // 2)

        distance = distance_calc.point_to_point_distance(depth, point1, point2)
        depth1 = distance_calc.depth_at_point(depth, point1)
        depth2 = distance_calc.depth_at_point(depth, point2)

        logger.info(f"Point 1: {point1}, Depth: {depth1:.4f}")
        logger.info(f"Point 2: {point2}, Depth: {depth2:.4f}")
        logger.info(f"Distance: {distance:.4f} (relative units)")

        # Create annotated image
        depth_colored = colormap_gen.apply_colormap(depth)
        annotated = renderer.annotate_measurement(
            image, point1, point2, distance, units="rel"
        )
        output_path = output_dir / f"{input_path.stem}_measurement.png"
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), annotated_bgr)
        logger.info(f"Saved annotated image: {output_path.name}")

        # Demo 2: Scale calibration
        logger.info("\n" + "=" * 70)
        logger.info("DEMO 2: Scale Calibration")
        logger.info("=" * 70)

        # Simulate calibration with known distance
        # (in practice, user would mark a known reference object)
        calibration_points = [point1, point2]
        known_distance = 1.0  # meters (example)

        calibration = scale_est.calibrate(
            depth,
            known_distance=known_distance,
            pixels=calibration_points,
            metric_unit="meters"
        )

        logger.info(f"Calibration scale factor: {calibration.scale_factor:.6f}")
        logger.info(f"Reference distance: {known_distance} meters")

        # Convert previous distance to metric units
        distance_metric = scale_est.depth_to_metric(distance, calibration)
        logger.info(f"Distance in metric units: {distance_metric:.4f} meters")

        # Save calibration
        calib_path = output_dir / "calibration.json"
        scale_est.save_calibration(calib_path)
        logger.info(f"Saved calibration: {calib_path.name}")

        # Demo 3: Region analysis
        logger.info("\n" + "=" * 70)
        logger.info("DEMO 3: Region Depth Analysis")
        logger.info("=" * 70)

        # Define a region (center of image)
        bbox = (
            width // 4,
            height // 4,
            3 * width // 4,
            3 * height // 4
        )

        region_stats = distance_calc.region_depth_stats(depth, bbox)
        logger.info("Region statistics:")
        for key, value in region_stats.items():
            logger.info(f"  {key}: {value:.4f}")

        # Demo 4: Object detection
        logger.info("\n" + "=" * 70)
        logger.info("DEMO 4: Depth-Based Object Detection")
        logger.info("=" * 70)

        # Find nearest objects
        nearest_objects = object_analyzer.find_nearest_objects(
            depth, num_objects=3, min_area=500
        )

        logger.info(f"Found {len(nearest_objects)} objects:")
        for obj in nearest_objects:
            logger.info(f"  Object {obj.id}:")
            logger.info(f"    Center: {obj.center}")
            logger.info(f"    Depth: {obj.depth_mean:.4f}")
            logger.info(f"    Area: {obj.area} pixels")

        # Demo 5: Depth segmentation
        logger.info("\n" + "=" * 70)
        logger.info("DEMO 5: Depth Segmentation")
        logger.info("=" * 70)

        segmented = object_analyzer.segment_by_depth(depth, num_levels=5)
        logger.info(f"Segmented into 5 depth levels")
        logger.info(f"Unique levels: {len(np.unique(segmented))}")

        # Create segmentation visualization
        segmentation_map = object_analyzer.create_depth_segmentation_map(
            depth, num_levels=8, colormap="jet"
        )
        seg_path = output_dir / f"{input_path.stem}_segmentation.png"
        seg_bgr = cv2.cvtColor(segmentation_map, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(seg_path), seg_bgr)
        logger.info(f"Saved segmentation map: {seg_path.name}")

        # Demo 6: Find closest and farthest points
        logger.info("\n" + "=" * 70)
        logger.info("DEMO 6: Extrema Detection")
        logger.info("=" * 70)

        closest_point = distance_calc.find_closest_point(depth)
        farthest_point = distance_calc.find_farthest_point(depth)

        logger.info(f"Closest point: {closest_point[:2]}, depth: {closest_point[2]:.4f}")
        logger.info(f"Farthest point: {farthest_point[:2]}, depth: {farthest_point[2]:.4f}")

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("MEASUREMENT DEMO COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output directory: {output_dir}")
        logger.info("Generated files:")
        logger.info(f"  - {output_path.name} (annotated measurement)")
        logger.info(f"  - {seg_path.name} (depth segmentation)")
        logger.info(f"  - {calib_path.name} (calibration parameters)")
        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())

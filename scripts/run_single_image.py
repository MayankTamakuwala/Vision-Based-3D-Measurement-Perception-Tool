#!/usr/bin/env python3
"""CLI script for depth estimation on a single image."""

import argparse
import sys
from pathlib import Path
import cv2
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.postprocessing.colormap_generator import ColormapGenerator
from src.utils.logger import setup_logger

logger = setup_logger("vision3d", level="INFO", log_file="depth_estimation.log")


def main():
    """Main function for single image depth estimation."""
    parser = argparse.ArgumentParser(
        description="Estimate depth from a single image using MiDaS"
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
        default="DPT_Large",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model variant (default: DPT_Large)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "-c", "--colormap",
        type=str,
        default="viridis",
        help="Colormap for visualization (default: viridis)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize depth output"
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw depth as numpy array (.npy)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Model cache directory (default: ./models)"
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

    # Output paths
    output_name = input_path.stem
    depth_gray_path = output_dir / f"{output_name}_depth_gray.png"
    depth_color_path = output_dir / f"{output_name}_depth_color.png"
    overlay_path = output_dir / f"{output_name}_overlay.png"
    raw_depth_path = output_dir / f"{output_name}_depth.npy"

    logger.info("=" * 60)
    logger.info("Vision-Based 3D Measurement & Perception Tool")
    logger.info("Single Image Depth Estimation")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {output_dir}")

    try:
        # Initialize depth estimator
        logger.info("Initializing depth estimator...")
        start_time = time.time()

        estimator = DepthEstimator(
            model_type=args.model,
            device=args.device,
            cache_dir=args.cache_dir,
            optimize=True
        )

        init_time = time.time() - start_time
        logger.info(f"Initialization complete ({init_time:.2f}s)")
        logger.info(f"Device: {estimator.device_manager.device_name}")

        # Load and process image
        logger.info("Loading image...")
        image, depth = estimator.estimate_depth_from_file(
            input_path,
            normalize=not args.no_normalize
        )

        # Get depth statistics
        stats = estimator.get_depth_stats(depth)
        logger.info("Depth Statistics:")
        logger.info(f"  Min: {stats['min']:.4f}")
        logger.info(f"  Max: {stats['max']:.4f}")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Median: {stats['median']:.4f}")

        # Get performance metrics
        perf = estimator.get_performance_summary()
        logger.info("Performance:")
        logger.info(f"  Inference: {perf.get('avg_inference_ms', 0):.2f} ms")
        if 'fps' in perf:
            logger.info(f"  FPS: {perf['fps']:.2f}")

        # Save outputs
        logger.info("Saving outputs...")

        # 1. Grayscale depth map
        depth_gray = (depth * 255).astype('uint8')
        cv2.imwrite(str(depth_gray_path), depth_gray)
        logger.info(f"  Saved grayscale depth: {depth_gray_path.name}")

        # 2. Colorized depth map
        colormap_gen = ColormapGenerator(default_colormap=args.colormap)
        depth_colored = colormap_gen.apply_colormap(depth, colormap=args.colormap)
        depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(depth_color_path), depth_colored_bgr)
        logger.info(f"  Saved colored depth: {depth_color_path.name}")

        # 3. Overlay depth on original image
        depth_colored_resized = cv2.resize(depth_colored, (image.shape[1], image.shape[0]))
        overlay = cv2.addWeighted(image, 0.6, depth_colored_resized, 0.4, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        logger.info(f"  Saved overlay: {overlay_path.name}")

        # 4. Raw depth array (optional)
        if args.save_raw:
            import numpy as np
            np.save(raw_depth_path, depth)
            logger.info(f"  Saved raw depth: {raw_depth_path.name}")

        logger.info("=" * 60)
        logger.info("SUCCESS! Depth estimation complete.")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

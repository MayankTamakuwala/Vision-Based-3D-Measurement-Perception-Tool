#!/usr/bin/env python3
"""CLI script for batch depth estimation on multiple images."""

import argparse
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.export.image_saver import ImageSaver
from src.export.json_exporter import JSONExporter
from src.utils.logger import setup_logger

logger = setup_logger("vision3d_batch", level="INFO", log_file="batch_processing.log")


def main():
    """Main function for batch depth estimation."""
    parser = argparse.ArgumentParser(
        description="Batch depth estimation on multiple images"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing input images"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./data/output",
        help="Output directory (default: ./data/output)"
    )
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="*.jpg",
        help="Image file pattern (default: *.jpg)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="DPT_Hybrid",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model variant (default: DPT_Hybrid)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        help="Batch size for processing (default: auto)"
    )
    parser.add_argument(
        "-c", "--colormap",
        type=str,
        default="viridis",
        help="Colormap for visualization (default: viridis)"
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        default=True,
        help="Save grayscale depth maps"
    )
    parser.add_argument(
        "--save-colored",
        action="store_true",
        default=True,
        help="Save colored depth maps"
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=True,
        help="Save overlay images"
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        default=True,
        help="Save JSON metadata"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to process"
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Vision-Based 3D Measurement & Perception Tool")
    logger.info("Batch Depth Estimation")
    logger.info("=" * 70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Pattern: {args.pattern}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {output_dir}")

    try:
        # Initialize depth estimator
        logger.info("Initializing depth estimator...")
        start_time = time.time()

        estimator = DepthEstimator(
            model_type=args.model,
            device=args.device,
            optimize=True
        )

        init_time = time.time() - start_time
        logger.info(f"Initialization complete ({init_time:.2f}s)")
        logger.info(f"Device: {estimator.device_manager.device_name}")

        # Process directory
        logger.info("Searching for images...")
        image_files = list(input_dir.glob(args.pattern))

        if args.max_images:
            image_files = image_files[:args.max_images]

        if not image_files:
            logger.warning(f"No images found matching pattern: {args.pattern}")
            return 0

        logger.info(f"Found {len(image_files)} images")

        # Process images
        logger.info("Processing images...")
        processing_start = time.time()

        results = estimator.estimate_from_directory(
            directory=input_dir,
            pattern=args.pattern,
            batch_size=args.batch_size,
            normalize=True
        )

        if args.max_images:
            results = results[:args.max_images]

        processing_time = time.time() - processing_start

        # Get performance metrics
        perf = estimator.get_performance_summary()
        logger.info("Processing complete!")
        logger.info(f"Total time: {processing_time:.2f}s")
        logger.info(f"Average inference: {perf.get('avg_inference_ms', 0):.2f} ms")
        logger.info(f"Throughput: {perf.get('fps', 0):.2f} FPS")

        # Save outputs
        logger.info("Saving outputs...")

        # Initialize savers
        image_saver = ImageSaver(output_dir=output_dir)
        json_exporter = JSONExporter(output_dir=output_dir / "metadata")

        # Prepare results for saving
        save_results = [
            (path.stem, image, depth)
            for path, image, depth in results
        ]

        # Save images
        if args.save_depth or args.save_colored or args.save_overlay:
            saved_paths = image_saver.save_batch(
                save_results,
                save_depth=args.save_depth,
                save_colored=args.save_colored,
                save_overlay=args.save_overlay,
                colormap=args.colormap
            )
            logger.info(f"Saved {len(saved_paths)} image files")

        # Save metadata
        if args.save_metadata:
            batch_metadata = []
            for path, image, depth in results:
                stats = estimator.get_depth_stats(depth)
                metadata = {
                    'filename': path.name,
                    'depth_stats': stats,
                    'model': args.model,
                    'colormap': args.colormap
                }
                batch_metadata.append(metadata)

            json_exporter.export_batch_metadata(batch_metadata, filename="batch_results")
            logger.info("Saved batch metadata")

            # Save performance report
            json_exporter.export_performance_report(perf, filename="performance_report")
            logger.info("Saved performance report")

        # Print summary
        logger.info("=" * 70)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Processed: {len(results)} images")
        logger.info(f"Total time: {processing_time:.2f}s")
        logger.info(f"Average per image: {processing_time / len(results):.2f}s")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

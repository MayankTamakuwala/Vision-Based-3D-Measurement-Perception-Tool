#!/usr/bin/env python3
"""CLI script for depth estimation on video files."""

import argparse
import sys
from pathlib import Path
import cv2
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.preprocessing.video_processor import VideoProcessor
from src.postprocessing.colormap_generator import ColormapGenerator
from src.export.image_saver import ImageSaver
from src.utils.logger import setup_logger

logger = setup_logger("vision3d_video", level="INFO", log_file="video_processing.log")


def main():
    """Main function for video depth estimation."""
    parser = argparse.ArgumentParser(
        description="Depth estimation on video files"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./data/output/video",
        help="Output directory (default: ./data/output/video)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="MiDaS_small",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model variant (default: MiDaS_small for speed)"
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "-c", "--colormap",
        type=str,
        default="viridis",
        help="Colormap for visualization (default: viridis)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save output as video file"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames"
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        help="Output video FPS (default: same as input)"
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Vision-Based 3D Measurement & Perception Tool")
    logger.info("Video Depth Estimation")
    logger.info("=" * 70)

    try:
        # Initialize video processor
        logger.info("Opening video...")
        video = VideoProcessor(str(input_path))
        video_info = video.get_video_info()

        logger.info(f"Video: {video_info['filename']}")
        logger.info(f"Resolution: {video_info['width']}x{video_info['height']}")
        logger.info(f"FPS: {video_info['fps']:.2f}")
        logger.info(f"Total frames: {video_info['total_frames']}")
        logger.info(f"Duration: {video_info['duration_formatted']}")

        # Initialize depth estimator
        logger.info("Initializing depth estimator...")
        estimator = DepthEstimator(
            model_type=args.model,
            optimize=True
        )
        logger.info(f"Device: {estimator.device_manager.device_name}")

        # Initialize visualization
        colormap_gen = ColormapGenerator(default_colormap=args.colormap)

        # Prepare video writer if saving video
        video_writer = None
        if args.save_video:
            output_fps = args.output_fps or video_info['fps']
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = output_dir / f"{input_path.stem}_depth.mp4"

            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                output_fps,
                (video_info['width'], video_info['height'])
            )
            logger.info(f"Output video: {output_video_path.name}")

        # Process frames
        logger.info("Processing video frames...")
        start_time = time.time()

        frames_processed = 0
        frame_times = []

        # Create progress bar
        total_frames_to_process = min(
            args.max_frames or video_info['total_frames'],
            video_info['total_frames'] // args.sample_rate
        )

        with tqdm(total=total_frames_to_process, desc="Processing", unit="frame") as pbar:
            for frame_num, frame in video.extract_frames(
                sample_rate=args.sample_rate,
                max_frames=args.max_frames
            ):
                frame_start = time.time()

                # Estimate depth
                depth = estimator.estimate_depth(frame, normalize=True)

                # Create colored visualization
                depth_colored = colormap_gen.apply_colormap(depth, colormap=args.colormap)

                # Create overlay
                overlay = cv2.addWeighted(frame, 0.5, depth_colored, 0.5, 0)

                # Save to video
                if video_writer:
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    video_writer.write(overlay_bgr)

                # Save individual frames if requested
                if args.save_frames:
                    frame_dir = output_dir / "frames"
                    frame_dir.mkdir(exist_ok=True)
                    frame_path = frame_dir / f"frame_{frame_num:06d}.png"
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(frame_path), overlay_bgr)

                frames_processed += 1
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'FPS': f'{1.0 / frame_time:.2f}',
                    'Avg': f'{frames_processed / sum(frame_times):.2f}'
                })

        processing_time = time.time() - start_time

        # Clean up
        video.close()
        if video_writer:
            video_writer.release()

        # Get performance metrics
        perf = estimator.get_performance_summary()

        # Print summary
        logger.info("=" * 70)
        logger.info("VIDEO PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Processed: {frames_processed} frames")
        logger.info(f"Total time: {processing_time:.2f}s")
        logger.info(f"Average FPS: {frames_processed / processing_time:.2f}")
        logger.info(f"Average inference: {perf.get('avg_inference_ms', 0):.2f} ms")
        logger.info(f"Output directory: {output_dir}")

        if args.save_video:
            logger.info(f"Output video: {output_video_path.name}")
        if args.save_frames:
            logger.info(f"Individual frames: {output_dir / 'frames'}")

        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        if video_writer:
            video_writer.release()
        return 1

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        if video_writer:
            video_writer.release()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CLI script for real-time webcam depth estimation."""

import argparse
import sys
from pathlib import Path
import cv2
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.depth_estimator import DepthEstimator
from src.preprocessing.webcam_handler import WebcamHandler
from src.postprocessing.colormap_generator import ColormapGenerator
from src.utils.logger import setup_logger

logger = setup_logger("vision3d_webcam", level="INFO")


def main():
    """Main function for webcam depth estimation."""
    parser = argparse.ArgumentParser(
        description="Real-time webcam depth estimation"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="MiDaS_small",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model variant (default: MiDaS_small for speed)"
    )
    parser.add_argument(
        "-c", "--colormap",
        type=str,
        default="viridis",
        help="Colormap for visualization (default: viridis)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height (default: 480)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window (for headless mode)"
    )
    parser.add_argument(
        "--save-recording",
        type=str,
        help="Save recording to video file"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Vision-Based 3D Measurement & Perception Tool")
    logger.info("Real-Time Webcam Depth Estimation")
    logger.info("=" * 70)

    try:
        # Initialize webcam
        logger.info(f"Opening camera {args.camera}...")
        webcam = WebcamHandler(
            camera_id=args.camera,
            width=args.width,
            height=args.height
        )
        webcam.start_capture()

        camera_info = webcam.get_camera_info()
        logger.info(f"Camera: {camera_info['width']}x{camera_info['height']}")

        # Initialize depth estimator
        logger.info("Initializing depth estimator...")
        estimator = DepthEstimator(
            model_type=args.model,
            optimize=True
        )
        logger.info(f"Device: {estimator.device_manager.device_name}")

        # Initialize visualization
        colormap_gen = ColormapGenerator(default_colormap=args.colormap)

        # Video writer for recording
        video_writer = None
        if args.save_recording:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                args.save_recording,
                fourcc,
                30.0,  # FPS
                (camera_info['width'], camera_info['height'])
            )
            logger.info(f"Recording to: {args.save_recording}")

        logger.info("\n" + "=" * 70)
        logger.info("STARTING REAL-TIME DEPTH ESTIMATION")
        logger.info("Press 'q' to quit, 's' to save screenshot")
        logger.info("=" * 70 + "\n")

        frame_count = 0
        fps_update_interval = 30
        fps_start_time = time.time()

        while True:
            # Get frame from webcam
            frame = webcam.get_latest_frame()

            if frame is None:
                continue

            frame_count += 1

            # Estimate depth
            depth = estimator.estimate_depth(frame, normalize=True)

            # Create colored visualization
            depth_colored = colormap_gen.apply_colormap(depth, colormap=args.colormap)

            # Create side-by-side display
            display_frame = np.hstack([frame, depth_colored])

            # Add FPS overlay
            if frame_count % fps_update_interval == 0:
                elapsed = time.time() - fps_start_time
                current_fps = fps_update_interval / elapsed
                fps_start_time = time.time()
            else:
                current_fps = estimator.get_performance_summary().get('fps', 0)

            cv2.putText(
                display_frame,
                f"FPS: {current_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                "Original",
                (10, camera_info['height'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                "Depth",
                (camera_info['width'] + 10, camera_info['height'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Save recording if enabled
            if video_writer:
                # Save just the depth overlay for recording
                overlay = cv2.addWeighted(frame, 0.5, depth_colored, 0.5, 0)
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                video_writer.write(overlay_bgr)

            # Display
            if not args.no_display:
                display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Real-Time Depth Estimation", display_bgr)

                key = cv2.waitKey(1) & 0xFF

                # Handle key presses
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(screenshot_path, display_bgr)
                    logger.info(f"Screenshot saved: {screenshot_path}")

        # Clean up
        webcam.stop_capture()
        if video_writer:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        # Print statistics
        webcam_stats = webcam.get_fps_stats()
        perf = estimator.get_performance_summary()

        logger.info("\n" + "=" * 70)
        logger.info("SESSION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total frames: {webcam_stats['frames']}")
        logger.info(f"Session duration: {webcam_stats['elapsed']:.2f}s")
        logger.info(f"Average FPS: {webcam_stats['fps']:.2f}")
        logger.info(f"Average inference: {perf.get('avg_inference_ms', 0):.2f} ms")

        if args.save_recording:
            logger.info(f"Recording saved: {args.save_recording}")

        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        if 'webcam' in locals():
            webcam.stop_capture()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        return 1

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        if 'webcam' in locals():
            webcam.stop_capture()
        if 'video_writer' in locals() and video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        return 1


if __name__ == "__main__":
    sys.exit(main())

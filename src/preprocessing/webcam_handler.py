"""Real-time webcam capture for depth estimation."""

import cv2
import numpy as np
from typing import Optional, Tuple
import time
from queue import Queue
import threading

from ..utils.logger import get_logger
from ..utils.exceptions import PreprocessingError

logger = get_logger(__name__)


class WebcamHandler:
    """
    Handles real-time webcam capture for depth estimation.

    Supports:
    - Multi-camera support
    - Frame buffering
    - Threaded capture for smoother performance
    - Resolution and FPS configuration
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        buffer_size: int = 2
    ):
        """
        Initialize webcam handler.

        Args:
            camera_id: Camera device ID (0 for default camera)
            width: Desired frame width (None for camera default)
            height: Desired frame height (None for camera default)
            fps: Desired FPS (None for camera default)
            buffer_size: Frame buffer size for threaded capture

        Raises:
            PreprocessingError: If camera cannot be opened
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_capturing = False
        self.thread = None
        self.frame_queue = Queue(maxsize=buffer_size)
        self.latest_frame = None
        self.frame_count = 0
        self.start_time = None

        # Desired settings
        self.desired_width = width
        self.desired_height = height
        self.desired_fps = fps

        logger.info(f"Initializing webcam handler for camera {camera_id}")

    def start_capture(self) -> bool:
        """
        Start capturing from webcam.

        Returns:
            True if successful, False otherwise

        Raises:
            PreprocessingError: If camera cannot be opened
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                raise PreprocessingError(f"Failed to open camera {self.camera_id}")

            # Set camera properties if specified
            if self.desired_width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
            if self.desired_height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
            if self.desired_fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

            # Get actual camera properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"Camera opened: {self.width}x{self.height} @ {self.fps:.2f} fps"
            )

            # Start capture thread
            self.is_capturing = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

            logger.info("Webcam capture started")
            return True

        except Exception as e:
            raise PreprocessingError(f"Failed to start capture: {str(e)}")

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        while self.is_capturing:
            ret, frame = self.cap.read()

            if not ret:
                logger.warning("Failed to read frame from camera")
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update latest frame
            self.latest_frame = frame_rgb
            self.frame_count += 1

            # Add to queue (non-blocking, drop if full)
            if not self.frame_queue.full():
                self.frame_queue.put(frame_rgb)

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next frame from the buffer.

        Args:
            timeout: Timeout in seconds

        Returns:
            Frame array or None if timeout

        Raises:
            PreprocessingError: If not capturing
        """
        if not self.is_capturing:
            raise PreprocessingError("Camera not capturing. Call start_capture() first.")

        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except:
            # Queue empty or timeout
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame (may skip frames).

        Returns:
            Latest frame array or None if no frames captured
        """
        return self.latest_frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame (OpenCV-compatible interface).

        Returns:
            Tuple of (success, frame)
        """
        frame = self.get_latest_frame()
        return (frame is not None, frame)

    def get_fps_stats(self) -> dict:
        """
        Get FPS statistics.

        Returns:
            Dictionary with FPS info
        """
        if self.start_time is None:
            return {'elapsed': 0, 'frames': 0, 'fps': 0}

        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        return {
            'elapsed': elapsed,
            'frames': self.frame_count,
            'fps': fps
        }

    def stop_capture(self):
        """Stop capturing and release resources."""
        self.is_capturing = False

        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info("Webcam capture stopped")

    def is_opened(self) -> bool:
        """Check if camera is open and capturing."""
        return self.cap is not None and self.cap.isOpened() and self.is_capturing

    def get_camera_info(self) -> dict:
        """
        Get camera information.

        Returns:
            Dictionary with camera properties
        """
        if not self.cap:
            return {}

        return {
            'camera_id': self.camera_id,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'backend': self.cap.getBackendName()
        }

    def set_resolution(self, width: int, height: int):
        """
        Change camera resolution.

        Args:
            width: New width
            height: New height
        """
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Update actual values
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Resolution set to {self.width}x{self.height}")

    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.stop_capture()

    def __repr__(self) -> str:
        """String representation of webcam handler."""
        status = "capturing" if self.is_capturing else "stopped"
        return f"WebcamHandler(camera_id={self.camera_id}, status={status})"

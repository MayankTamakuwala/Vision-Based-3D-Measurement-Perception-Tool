"""Video processing utilities for frame extraction and analysis."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, List, Tuple, Optional, Dict
from datetime import timedelta

from ..utils.logger import get_logger
from ..utils.exceptions import PreprocessingError

logger = get_logger(__name__)


class VideoProcessor:
    """
    Handles video file processing and frame extraction.

    Supports:
    - Frame-by-frame extraction
    - Sampling strategies (uniform, key-frames)
    - Video metadata extraction
    - Various video codecs via OpenCV
    """

    def __init__(self, video_path: str):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file

        Raises:
            PreprocessingError: If video cannot be opened
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise PreprocessingError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise PreprocessingError(f"Failed to open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        logger.info(
            f"Opened video: {self.video_path.name} "
            f"({self.width}x{self.height}, {self.total_frames} frames, {self.fps:.2f} fps)"
        )

    def extract_frames(
        self,
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Extract frames from video.

        Args:
            sample_rate: Extract every Nth frame (1 = all frames)
            max_frames: Maximum number of frames to extract
            start_frame: Starting frame index
            end_frame: Ending frame index (None for end of video)

        Yields:
            Tuple of (frame_number, frame_array)

        Raises:
            PreprocessingError: If frame extraction fails
        """
        try:
            # Reset to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if end_frame is None:
                end_frame = self.total_frames

            frames_extracted = 0
            current_frame = start_frame

            while current_frame < end_frame:
                if max_frames and frames_extracted >= max_frames:
                    break

                # Set to desired frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {current_frame}")
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield current_frame, frame_rgb

                frames_extracted += 1
                current_frame += sample_rate

            logger.info(f"Extracted {frames_extracted} frames from video")

        except Exception as e:
            raise PreprocessingError(f"Failed to extract frames: {str(e)}")

    def extract_frames_at_timestamps(
        self,
        timestamps: List[float]
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames at specific timestamps.

        Args:
            timestamps: List of timestamps in seconds

        Returns:
            List of (timestamp, frame) tuples
        """
        try:
            frames = []

            for timestamp in timestamps:
                # Calculate frame number
                frame_num = int(timestamp * self.fps)

                if frame_num >= self.total_frames:
                    logger.warning(f"Timestamp {timestamp}s exceeds video duration")
                    continue

                # Set to desired frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

                ret, frame = self.cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((timestamp, frame_rgb))

            logger.info(f"Extracted {len(frames)} frames at specified timestamps")
            return frames

        except Exception as e:
            raise PreprocessingError(f"Failed to extract frames at timestamps: {str(e)}")

    def extract_key_frames(
        self,
        threshold: float = 30.0,
        max_frames: Optional[int] = None
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract key frames based on scene change detection.

        Args:
            threshold: Scene change threshold (higher = fewer key frames)
            max_frames: Maximum number of key frames

        Returns:
            List of (frame_number, frame) tuples
        """
        try:
            key_frames = []
            prev_frame = None

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            for frame_num in range(self.total_frames):
                if max_frames and len(key_frames) >= max_frames:
                    break

                ret, frame = self.cap.read()
                if not ret:
                    break

                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is None:
                    # First frame is always a key frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    key_frames.append((frame_num, frame_rgb))
                else:
                    # Calculate difference from previous frame
                    diff = cv2.absdiff(gray, prev_frame)
                    mean_diff = np.mean(diff)

                    if mean_diff > threshold:
                        # Scene change detected
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        key_frames.append((frame_num, frame_rgb))

                prev_frame = gray

            logger.info(f"Extracted {len(key_frames)} key frames")
            return key_frames

        except Exception as e:
            raise PreprocessingError(f"Failed to extract key frames: {str(e)}")

    def get_video_info(self) -> Dict:
        """
        Get comprehensive video information.

        Returns:
            Dictionary with video metadata
        """
        info = {
            'path': str(self.video_path),
            'filename': self.video_path.name,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration_seconds': self.duration,
            'duration_formatted': str(timedelta(seconds=int(self.duration))),
            'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC))
        }

        return info

    def sample_uniform(
        self,
        num_samples: int
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract uniformly sampled frames from video.

        Args:
            num_samples: Number of frames to sample

        Returns:
            List of (frame_number, frame) tuples
        """
        try:
            if num_samples >= self.total_frames:
                # Sample all frames
                sample_rate = 1
            else:
                # Calculate sample rate
                sample_rate = self.total_frames // num_samples

            frames = []
            for frame_num, frame in self.extract_frames(sample_rate=sample_rate, max_frames=num_samples):
                frames.append((frame_num, frame))

            return frames

        except Exception as e:
            raise PreprocessingError(f"Failed to sample uniformly: {str(e)}")

    def get_frame_at_index(self, frame_num: int) -> np.ndarray:
        """
        Get a specific frame by index.

        Args:
            frame_num: Frame index

        Returns:
            Frame array

        Raises:
            PreprocessingError: If frame cannot be retrieved
        """
        try:
            if frame_num < 0 or frame_num >= self.total_frames:
                raise PreprocessingError(f"Frame index {frame_num} out of range")

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()

            if not ret:
                raise PreprocessingError(f"Failed to read frame {frame_num}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb

        except Exception as e:
            raise PreprocessingError(f"Failed to get frame at index: {str(e)}")

    def reset(self):
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.debug("Video reset to beginning")

    def close(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            logger.debug(f"Closed video: {self.video_path.name}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()

    def __repr__(self) -> str:
        """String representation of video processor."""
        return (
            f"VideoProcessor("
            f"file='{self.video_path.name}', "
            f"frames={self.total_frames}, "
            f"fps={self.fps:.2f})"
        )

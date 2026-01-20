"""Scale estimation and calibration for depth measurements."""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass, asdict

from ..utils.logger import get_logger
from ..utils.exceptions import CalibrationError
from ..utils.validators import validate_depth_map

logger = get_logger(__name__)


@dataclass
class CalibrationParams:
    """Calibration parameters for depth-to-metric conversion."""
    scale_factor: float
    reference_distance: float
    reference_pixels: List[Tuple[int, int]]
    depth_unit: str = "relative"
    metric_unit: str = "meters"
    timestamp: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ScaleEstimator:
    """
    Handles scale estimation and calibration for depth measurements.

    Converts relative depth values to real-world metric units using
    reference objects with known dimensions.
    """

    def __init__(self):
        """Initialize scale estimator."""
        self.calibration_params: Optional[CalibrationParams] = None

    def calibrate(
        self,
        depth: np.ndarray,
        known_distance: float,
        pixels: List[Tuple[int, int]],
        metric_unit: str = "meters"
    ) -> CalibrationParams:
        """
        Calibrate scale using a known distance between points.

        Args:
            depth: Depth map array
            known_distance: Known real-world distance
            pixels: List of pixel coordinates marking the known distance
            metric_unit: Unit of the known distance

        Returns:
            Calibration parameters

        Raises:
            CalibrationError: If calibration fails
        """
        try:
            validate_depth_map(depth)

            if len(pixels) < 2:
                raise CalibrationError("Need at least 2 points for calibration")

            # Calculate depth distance in relative units
            from .distance_calculator import DistanceCalculator
            calc = DistanceCalculator()

            depth_distance = calc.point_to_point_distance(
                depth, pixels[0], pixels[1], method="euclidean"
            )

            if depth_distance == 0:
                raise CalibrationError("Depth distance is zero, cannot calibrate")

            # Calculate scale factor
            scale_factor = known_distance / depth_distance

            # Create calibration parameters
            from datetime import datetime
            params = CalibrationParams(
                scale_factor=scale_factor,
                reference_distance=known_distance,
                reference_pixels=pixels,
                metric_unit=metric_unit,
                timestamp=datetime.now().isoformat()
            )

            self.calibration_params = params

            logger.info(
                f"Calibration complete: scale_factor={scale_factor:.6f}, "
                f"reference={known_distance} {metric_unit}"
            )

            return params

        except Exception as e:
            raise CalibrationError(f"Failed to calibrate: {str(e)}")

    def calibrate_from_bbox(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int],
        known_width: Optional[float] = None,
        known_height: Optional[float] = None,
        metric_unit: str = "meters"
    ) -> CalibrationParams:
        """
        Calibrate using a bounding box with known dimensions.

        Args:
            depth: Depth map array
            bbox: Bounding box (x1, y1, x2, y2)
            known_width: Known real-world width (if None, uses height)
            known_height: Known real-world height (if None, uses width)
            metric_unit: Unit of measurements

        Returns:
            Calibration parameters
        """
        try:
            if known_width is None and known_height is None:
                raise CalibrationError("Must provide either known width or height")

            x1, y1, x2, y2 = bbox

            if known_width is not None:
                # Use horizontal distance
                pixels = [(x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2)]
                known_distance = known_width
            else:
                # Use vertical distance
                pixels = [((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2)]
                known_distance = known_height

            return self.calibrate(depth, known_distance, pixels, metric_unit)

        except Exception as e:
            raise CalibrationError(f"Failed to calibrate from bbox: {str(e)}")

    def depth_to_metric(
        self,
        depth_value: Union[float, np.ndarray],
        calibration: Optional[CalibrationParams] = None
    ) -> Union[float, np.ndarray]:
        """
        Convert depth value(s) from relative units to metric units.

        Args:
            depth_value: Depth value or array
            calibration: Calibration parameters (if None, uses stored params)

        Returns:
            Converted depth value(s) in metric units

        Raises:
            CalibrationError: If no calibration parameters available
        """
        try:
            params = calibration or self.calibration_params

            if params is None:
                raise CalibrationError("No calibration parameters available")

            metric_value = depth_value * params.scale_factor

            return metric_value

        except Exception as e:
            raise CalibrationError(f"Failed to convert to metric: {str(e)}")

    def metric_to_depth(
        self,
        metric_value: Union[float, np.ndarray],
        calibration: Optional[CalibrationParams] = None
    ) -> Union[float, np.ndarray]:
        """
        Convert metric value(s) to relative depth units.

        Args:
            metric_value: Metric value or array
            calibration: Calibration parameters (if None, uses stored params)

        Returns:
            Converted value(s) in relative depth units
        """
        try:
            params = calibration or self.calibration_params

            if params is None:
                raise CalibrationError("No calibration parameters available")

            if params.scale_factor == 0:
                raise CalibrationError("Scale factor is zero")

            depth_value = metric_value / params.scale_factor

            return depth_value

        except Exception as e:
            raise CalibrationError(f"Failed to convert to depth units: {str(e)}")

    def save_calibration(
        self,
        filepath: Union[str, Path],
        calibration: Optional[CalibrationParams] = None
    ):
        """
        Save calibration parameters to JSON file.

        Args:
            filepath: Output file path
            calibration: Calibration parameters (if None, uses stored params)

        Raises:
            CalibrationError: If saving fails
        """
        try:
            params = calibration or self.calibration_params

            if params is None:
                raise CalibrationError("No calibration parameters to save")

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(params.to_dict(), f, indent=2)

            logger.info(f"Saved calibration parameters: {filepath}")

        except Exception as e:
            raise CalibrationError(f"Failed to save calibration: {str(e)}")

    def load_calibration(self, filepath: Union[str, Path]) -> CalibrationParams:
        """
        Load calibration parameters from JSON file.

        Args:
            filepath: Input file path

        Returns:
            Calibration parameters

        Raises:
            CalibrationError: If loading fails
        """
        try:
            filepath = Path(filepath)

            if not filepath.exists():
                raise CalibrationError(f"Calibration file not found: {filepath}")

            with open(filepath, 'r') as f:
                data = json.load(f)

            params = CalibrationParams(**data)
            self.calibration_params = params

            logger.info(f"Loaded calibration parameters: {filepath}")
            return params

        except Exception as e:
            raise CalibrationError(f"Failed to load calibration: {str(e)}")

    def estimate_scale_from_multiple(
        self,
        depth: np.ndarray,
        measurements: List[Dict]
    ) -> CalibrationParams:
        """
        Estimate scale using multiple known measurements (averaging).

        Args:
            depth: Depth map array
            measurements: List of dicts with 'pixels' and 'distance' keys

        Returns:
            Averaged calibration parameters
        """
        try:
            if not measurements:
                raise CalibrationError("No measurements provided")

            scale_factors = []

            for measurement in measurements:
                pixels = measurement['pixels']
                known_distance = measurement['distance']

                params = self.calibrate(depth, known_distance, pixels)
                scale_factors.append(params.scale_factor)

            # Average scale factors
            avg_scale_factor = np.mean(scale_factors)
            std_scale_factor = np.std(scale_factors)

            logger.info(
                f"Estimated scale from {len(measurements)} measurements: "
                f"avg={avg_scale_factor:.6f}, std={std_scale_factor:.6f}"
            )

            # Use first measurement as reference, but with averaged scale
            first_measurement = measurements[0]
            params = CalibrationParams(
                scale_factor=avg_scale_factor,
                reference_distance=first_measurement['distance'],
                reference_pixels=first_measurement['pixels'],
                notes=f"Averaged from {len(measurements)} measurements (std={std_scale_factor:.6f})"
            )

            self.calibration_params = params
            return params

        except Exception as e:
            raise CalibrationError(f"Failed to estimate scale from multiple: {str(e)}")

    def get_calibration_quality(
        self,
        depth: np.ndarray,
        calibration: Optional[CalibrationParams] = None
    ) -> Dict[str, float]:
        """
        Assess calibration quality by re-measuring reference distance.

        Args:
            depth: Depth map array
            calibration: Calibration parameters

        Returns:
            Dictionary with quality metrics
        """
        try:
            params = calibration or self.calibration_params

            if params is None:
                raise CalibrationError("No calibration parameters available")

            # Re-measure reference distance
            from .distance_calculator import DistanceCalculator
            calc = DistanceCalculator()

            measured_depth = calc.point_to_point_distance(
                depth,
                params.reference_pixels[0],
                params.reference_pixels[1]
            )

            # Convert to metric
            measured_metric = self.depth_to_metric(measured_depth, params)

            # Calculate error
            error = abs(measured_metric - params.reference_distance)
            error_percent = (error / params.reference_distance) * 100

            quality = {
                'reference_distance': params.reference_distance,
                'measured_distance': measured_metric,
                'error': error,
                'error_percent': error_percent,
                'scale_factor': params.scale_factor
            }

            logger.debug(f"Calibration quality: error={error_percent:.2f}%")
            return quality

        except Exception as e:
            raise CalibrationError(f"Failed to assess calibration quality: {str(e)}")

    def is_calibrated(self) -> bool:
        """Check if calibration parameters are available."""
        return self.calibration_params is not None

    def clear_calibration(self):
        """Clear stored calibration parameters."""
        self.calibration_params = None
        logger.info("Calibration parameters cleared")

    def __repr__(self) -> str:
        """String representation of scale estimator."""
        if self.is_calibrated():
            return (
                f"ScaleEstimator(calibrated=True, "
                f"scale_factor={self.calibration_params.scale_factor:.6f})"
            )
        return "ScaleEstimator(calibrated=False)"

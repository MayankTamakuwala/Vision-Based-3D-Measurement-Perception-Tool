"""Distance calculation utilities for depth measurements."""

import numpy as np
from typing import Tuple, List, Optional, Dict

from ..utils.logger import get_logger
from ..utils.exceptions import MeasurementError
from ..utils.validators import validate_depth_map, validate_point, validate_bbox

logger = get_logger(__name__)


class DistanceCalculator:
    """
    Calculates distances and measurements from depth maps.

    Supports:
    - Point-to-point distances
    - Region average depth
    - Depth difference maps
    - Relative distance calculations
    """

    def __init__(self):
        """Initialize distance calculator."""
        pass

    def point_to_point_distance(
        self,
        depth: np.ndarray,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        method: str = "euclidean"
    ) -> float:
        """
        Calculate distance between two points in depth space.

        Args:
            depth: Depth map array
            point1: First point (x, y)
            point2: Second point (x, y)
            method: Distance calculation method ('euclidean', 'manhattan', 'depth_only')

        Returns:
            Distance value (relative units)

        Raises:
            MeasurementError: If calculation fails
        """
        try:
            validate_depth_map(depth)
            point1 = validate_point(point1, depth.shape)
            point2 = validate_point(point2, depth.shape)

            x1, y1 = point1
            x2, y2 = point2

            # Get depth values
            depth1 = depth[y1, x1]
            depth2 = depth[y2, x2]

            if method == "depth_only":
                # Only depth difference
                distance = abs(depth1 - depth2)

            elif method == "manhattan":
                # Manhattan distance in 3D space
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                dz = abs(depth2 - depth1)
                distance = dx + dy + dz

            else:  # euclidean (default)
                # Euclidean distance in 3D space (pixel coords + depth)
                dx = x2 - x1
                dy = y2 - y1
                dz = depth2 - depth1
                distance = np.sqrt(dx**2 + dy**2 + dz**2)

            logger.debug(f"Distance between {point1} and {point2}: {distance:.4f}")
            return float(distance)

        except Exception as e:
            raise MeasurementError(f"Failed to calculate point-to-point distance: {str(e)}")

    def region_average_depth(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate average depth within a bounding box region.

        Args:
            depth: Depth map array
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Average depth value

        Raises:
            MeasurementError: If calculation fails
        """
        try:
            validate_depth_map(depth)
            bbox = validate_bbox(bbox, depth.shape)

            x1, y1, x2, y2 = bbox
            region = depth[y1:y2, x1:x2]

            avg_depth = float(np.mean(region))

            logger.debug(f"Average depth in region {bbox}: {avg_depth:.4f}")
            return avg_depth

        except Exception as e:
            raise MeasurementError(f"Failed to calculate region average depth: {str(e)}")

    def region_depth_stats(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict[str, float]:
        """
        Calculate depth statistics within a region.

        Args:
            depth: Depth map array
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Dictionary with depth statistics
        """
        try:
            validate_depth_map(depth)
            bbox = validate_bbox(bbox, depth.shape)

            x1, y1, x2, y2 = bbox
            region = depth[y1:y2, x1:x2]

            stats = {
                'mean': float(np.mean(region)),
                'median': float(np.median(region)),
                'min': float(np.min(region)),
                'max': float(np.max(region)),
                'std': float(np.std(region)),
                'range': float(np.max(region) - np.min(region))
            }

            return stats

        except Exception as e:
            raise MeasurementError(f"Failed to calculate region stats: {str(e)}")

    def depth_at_point(
        self,
        depth: np.ndarray,
        point: Tuple[int, int],
        neighborhood_size: int = 1
    ) -> float:
        """
        Get depth value at a point with optional neighborhood averaging.

        Args:
            depth: Depth map array
            point: Point (x, y)
            neighborhood_size: Size of neighborhood for averaging (1 = no averaging)

        Returns:
            Depth value
        """
        try:
            validate_depth_map(depth)
            point = validate_point(point, depth.shape)

            x, y = point

            if neighborhood_size <= 1:
                return float(depth[y, x])

            # Average over neighborhood
            half_size = neighborhood_size // 2
            y1 = max(0, y - half_size)
            y2 = min(depth.shape[0], y + half_size + 1)
            x1 = max(0, x - half_size)
            x2 = min(depth.shape[1], x + half_size + 1)

            neighborhood = depth[y1:y2, x1:x2]
            return float(np.mean(neighborhood))

        except Exception as e:
            raise MeasurementError(f"Failed to get depth at point: {str(e)}")

    def depth_difference_map(
        self,
        depth: np.ndarray,
        reference_point: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create a map of depth differences relative to a reference point.

        Args:
            depth: Depth map array
            reference_point: Reference point (x, y)

        Returns:
            Depth difference map
        """
        try:
            validate_depth_map(depth)
            reference_point = validate_point(reference_point, depth.shape)

            x, y = reference_point
            reference_depth = depth[y, x]

            diff_map = depth - reference_depth

            logger.debug(f"Created depth difference map relative to {reference_point}")
            return diff_map

        except Exception as e:
            raise MeasurementError(f"Failed to create depth difference map: {str(e)}")

    def find_closest_point(
        self,
        depth: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[int, int, float]:
        """
        Find the closest point (minimum depth) in the depth map.

        Args:
            depth: Depth map array
            threshold: Optional depth threshold (ignore values below this)

        Returns:
            Tuple of (x, y, depth_value)
        """
        try:
            validate_depth_map(depth)

            # Apply threshold if specified
            if threshold is not None:
                valid_mask = depth >= threshold
                if not valid_mask.any():
                    raise MeasurementError("No valid points above threshold")
                masked_depth = np.where(valid_mask, depth, np.inf)
            else:
                masked_depth = depth

            # Find minimum
            min_idx = np.argmin(masked_depth)
            y, x = np.unravel_index(min_idx, depth.shape)
            min_depth = depth[y, x]

            logger.debug(f"Closest point: ({x}, {y}) with depth {min_depth:.4f}")
            return int(x), int(y), float(min_depth)

        except Exception as e:
            raise MeasurementError(f"Failed to find closest point: {str(e)}")

    def find_farthest_point(
        self,
        depth: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[int, int, float]:
        """
        Find the farthest point (maximum depth) in the depth map.

        Args:
            depth: Depth map array
            threshold: Optional depth threshold (ignore values above this)

        Returns:
            Tuple of (x, y, depth_value)
        """
        try:
            validate_depth_map(depth)

            # Apply threshold if specified
            if threshold is not None:
                valid_mask = depth <= threshold
                if not valid_mask.any():
                    raise MeasurementError("No valid points below threshold")
                masked_depth = np.where(valid_mask, depth, -np.inf)
            else:
                masked_depth = depth

            # Find maximum
            max_idx = np.argmax(masked_depth)
            y, x = np.unravel_index(max_idx, depth.shape)
            max_depth = depth[y, x]

            logger.debug(f"Farthest point: ({x}, {y}) with depth {max_depth:.4f}")
            return int(x), int(y), float(max_depth)

        except Exception as e:
            raise MeasurementError(f"Failed to find farthest point: {str(e)}")

    def path_distance(
        self,
        depth: np.ndarray,
        points: List[Tuple[int, int]]
    ) -> float:
        """
        Calculate total distance along a path of points.

        Args:
            depth: Depth map array
            points: List of points (x, y) defining the path

        Returns:
            Total path distance
        """
        try:
            if len(points) < 2:
                raise MeasurementError("Path must have at least 2 points")

            total_distance = 0.0
            for i in range(len(points) - 1):
                dist = self.point_to_point_distance(depth, points[i], points[i + 1])
                total_distance += dist

            logger.debug(f"Path distance over {len(points)} points: {total_distance:.4f}")
            return total_distance

        except Exception as e:
            raise MeasurementError(f"Failed to calculate path distance: {str(e)}")

    def create_measurement_annotation(
        self,
        depth: np.ndarray,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        units: str = "units"
    ) -> Dict:
        """
        Create a complete measurement annotation with all relevant data.

        Args:
            depth: Depth map array
            point1: First point
            point2: Second point
            units: Distance units label

        Returns:
            Dictionary with measurement data
        """
        try:
            distance = self.point_to_point_distance(depth, point1, point2)
            depth1 = self.depth_at_point(depth, point1)
            depth2 = self.depth_at_point(depth, point2)

            annotation = {
                'type': 'point_to_point',
                'point1': {'x': point1[0], 'y': point1[1], 'depth': depth1},
                'point2': {'x': point2[0], 'y': point2[1], 'depth': depth2},
                'distance': distance,
                'depth_difference': abs(depth2 - depth1),
                'units': units
            }

            return annotation

        except Exception as e:
            raise MeasurementError(f"Failed to create measurement annotation: {str(e)}")

    def __repr__(self) -> str:
        """String representation of distance calculator."""
        return "DistanceCalculator()"

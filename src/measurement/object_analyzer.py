"""Object-level depth analysis and segmentation."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..utils.exceptions import MeasurementError
from ..utils.validators import validate_depth_map

logger = get_logger(__name__)


@dataclass
class ObjectInfo:
    """Information about a detected object."""
    id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # (x, y)
    area: int
    depth_mean: float
    depth_median: float
    depth_min: float
    depth_max: float
    depth_std: float


class ObjectAnalyzer:
    """
    Analyzes objects and regions based on depth information.

    Supports:
    - Depth-based segmentation
    - Object detection
    - Dimensional estimation
    - Spatial relationships
    """

    def __init__(self):
        """Initialize object analyzer."""
        pass

    def segment_by_depth(
        self,
        depth: np.ndarray,
        num_levels: int = 5,
        method: str = "uniform"
    ) -> np.ndarray:
        """
        Segment depth map into discrete levels.

        Args:
            depth: Depth map array
            num_levels: Number of depth levels
            method: Segmentation method ('uniform', 'quantile')

        Returns:
            Segmented depth map with integer labels

        Raises:
            MeasurementError: If segmentation fails
        """
        try:
            validate_depth_map(depth)

            if method == "quantile":
                # Use quantile-based thresholds
                thresholds = [np.percentile(depth, i * 100 / num_levels)
                             for i in range(1, num_levels)]
            else:  # uniform
                # Use uniform thresholds
                min_depth = depth.min()
                max_depth = depth.max()
                thresholds = np.linspace(min_depth, max_depth, num_levels + 1)[1:-1]

            # Create segmented map
            segmented = np.zeros_like(depth, dtype=np.int32)
            for i, threshold in enumerate(thresholds):
                segmented[depth > threshold] = i + 1

            logger.debug(f"Segmented depth into {num_levels} levels using {method} method")
            return segmented

        except Exception as e:
            raise MeasurementError(f"Failed to segment by depth: {str(e)}")

    def find_objects_by_depth(
        self,
        depth: np.ndarray,
        depth_range: Tuple[float, float],
        min_area: int = 100
    ) -> List[ObjectInfo]:
        """
        Find objects within a specific depth range.

        Args:
            depth: Depth map array
            depth_range: (min_depth, max_depth) range
            min_area: Minimum object area in pixels

        Returns:
            List of detected objects

        Raises:
            MeasurementError: If object detection fails
        """
        try:
            validate_depth_map(depth)

            min_depth, max_depth = depth_range

            # Create binary mask for depth range
            mask = ((depth >= min_depth) & (depth <= max_depth)).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            objects = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                if area < min_area:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, x + w, y + h)

                # Calculate center
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = x + w // 2, y + h // 2

                # Extract depth statistics for this object
                x1, y1, x2, y2 = bbox
                object_depth = depth[y1:y2, x1:x2]
                object_mask = mask[y1:y2, x1:x2]

                # Get depth values only within the object
                depth_values = object_depth[object_mask > 0]

                if len(depth_values) == 0:
                    continue

                obj_info = ObjectInfo(
                    id=i,
                    bbox=bbox,
                    center=(cx, cy),
                    area=int(area),
                    depth_mean=float(np.mean(depth_values)),
                    depth_median=float(np.median(depth_values)),
                    depth_min=float(np.min(depth_values)),
                    depth_max=float(np.max(depth_values)),
                    depth_std=float(np.std(depth_values))
                )

                objects.append(obj_info)

            logger.info(f"Found {len(objects)} objects in depth range [{min_depth:.2f}, {max_depth:.2f}]")
            return objects

        except Exception as e:
            raise MeasurementError(f"Failed to find objects by depth: {str(e)}")

    def find_nearest_objects(
        self,
        depth: np.ndarray,
        num_objects: int = 5,
        min_area: int = 100
    ) -> List[ObjectInfo]:
        """
        Find the N nearest objects in the scene.

        Args:
            depth: Depth map array
            num_objects: Number of objects to find
            min_area: Minimum object area

        Returns:
            List of nearest objects sorted by depth
        """
        try:
            # Segment depth map
            segmented = self.segment_by_depth(depth, num_levels=10, method="quantile")

            all_objects = []

            # Find objects in each depth level
            for level in range(1, 11):
                level_mask = (segmented == level).astype(np.uint8)
                min_depth = depth[level_mask > 0].min() if level_mask.any() else 0
                max_depth = depth[level_mask > 0].max() if level_mask.any() else 1

                objects = self.find_objects_by_depth(
                    depth,
                    (min_depth, max_depth),
                    min_area=min_area
                )
                all_objects.extend(objects)

            # Sort by depth and take N nearest
            all_objects.sort(key=lambda obj: obj.depth_mean)
            nearest = all_objects[:num_objects]

            logger.info(f"Found {len(nearest)} nearest objects")
            return nearest

        except Exception as e:
            raise MeasurementError(f"Failed to find nearest objects: {str(e)}")

    def estimate_object_dimensions(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int],
        scale_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Estimate object dimensions from bounding box and depth.

        Args:
            depth: Depth map array
            bbox: Bounding box (x1, y1, x2, y2)
            scale_factor: Calibration scale factor for metric units

        Returns:
            Dictionary with dimension estimates
        """
        try:
            x1, y1, x2, y2 = bbox

            # Pixel dimensions
            width_px = x2 - x1
            height_px = y2 - y1

            # Average depth in region
            region_depth = depth[y1:y2, x1:x2]
            avg_depth = np.mean(region_depth)

            # Estimated dimensions (rough approximation)
            # This is a simplified model - real conversion would need camera calibration
            width_relative = width_px * avg_depth * scale_factor
            height_relative = height_px * avg_depth * scale_factor

            dimensions = {
                'width_pixels': width_px,
                'height_pixels': height_px,
                'width_relative': width_relative,
                'height_relative': height_relative,
                'average_depth': avg_depth,
                'area_pixels': width_px * height_px
            }

            return dimensions

        except Exception as e:
            raise MeasurementError(f"Failed to estimate object dimensions: {str(e)}")

    def compute_spatial_relationships(
        self,
        objects: List[ObjectInfo]
    ) -> List[Dict]:
        """
        Compute spatial relationships between objects.

        Args:
            objects: List of objects to analyze

        Returns:
            List of relationship dictionaries
        """
        try:
            relationships = []

            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects):
                    if i >= j:
                        continue

                    # Distance between centers
                    dx = obj2.center[0] - obj1.center[0]
                    dy = obj2.center[1] - obj1.center[1]
                    distance_2d = np.sqrt(dx**2 + dy**2)

                    # Depth difference
                    depth_diff = obj2.depth_mean - obj1.depth_mean

                    # Relative position
                    if abs(dx) > abs(dy):
                        position = "right" if dx > 0 else "left"
                    else:
                        position = "below" if dy > 0 else "above"

                    # Depth relationship
                    if abs(depth_diff) < 0.1:
                        depth_rel = "same_depth"
                    elif depth_diff > 0:
                        depth_rel = "behind"
                    else:
                        depth_rel = "in_front"

                    relationship = {
                        'object1_id': obj1.id,
                        'object2_id': obj2.id,
                        'distance_2d': float(distance_2d),
                        'depth_difference': float(depth_diff),
                        'relative_position': position,
                        'depth_relationship': depth_rel
                    }

                    relationships.append(relationship)

            logger.debug(f"Computed {len(relationships)} spatial relationships")
            return relationships

        except Exception as e:
            raise MeasurementError(f"Failed to compute spatial relationships: {str(e)}")

    def create_depth_segmentation_map(
        self,
        depth: np.ndarray,
        num_levels: int = 8,
        colormap: str = "jet"
    ) -> np.ndarray:
        """
        Create a color-coded segmentation map.

        Args:
            depth: Depth map array
            num_levels: Number of depth levels
            colormap: Colormap name

        Returns:
            RGB segmentation map
        """
        try:
            segmented = self.segment_by_depth(depth, num_levels=num_levels)

            # Normalize to 0-255
            segmented_norm = (segmented * 255 / num_levels).astype(np.uint8)

            # Apply colormap
            cmap = cv2.COLORMAP_JET if colormap == "jet" else cv2.COLORMAP_VIRIDIS
            colored = cv2.applyColorMap(segmented_norm, cmap)

            # Convert BGR to RGB
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

            return colored_rgb

        except Exception as e:
            raise MeasurementError(f"Failed to create segmentation map: {str(e)}")

    def __repr__(self) -> str:
        """String representation of object analyzer."""
        return "ObjectAnalyzer()"

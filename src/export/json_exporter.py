"""JSON metadata export utilities."""

import json
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.exceptions import ExportError

logger = get_logger(__name__)


class JSONExporter:
    """
    Handles exporting depth estimation metadata to JSON format.

    Exports:
    - Depth statistics
    - Processing parameters
    - Performance metrics
    - Measurement data
    """

    def __init__(self, output_dir: Union[str, Path] = "./data/output/metadata"):
        """
        Initialize JSON exporter.

        Args:
            output_dir: Directory for saving JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"JSON exporter initialized: {self.output_dir}")

    def create_metadata_dict(
        self,
        depth: np.ndarray,
        model_info: Optional[Dict] = None,
        processing_params: Optional[Dict] = None,
        measurements: Optional[List[Dict]] = None,
        performance: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive metadata dictionary.

        Args:
            depth: Depth map array
            model_info: Model information
            processing_params: Processing parameters
            measurements: List of measurement dictionaries
            performance: Performance metrics

        Returns:
            Metadata dictionary
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'version': '0.1.0',
        }

        # Depth statistics
        metadata['depth_stats'] = {
            'shape': list(depth.shape),
            'min': float(depth.min()),
            'max': float(depth.max()),
            'mean': float(depth.mean()),
            'median': float(np.median(depth)),
            'std': float(depth.std()),
            'dtype': str(depth.dtype),
        }

        # Model information
        if model_info:
            metadata['model'] = model_info

        # Processing parameters
        if processing_params:
            metadata['processing'] = processing_params

        # Measurements
        if measurements:
            metadata['measurements'] = measurements

        # Performance metrics
        if performance:
            metadata['performance'] = performance

        return metadata

    def export_depth_metadata(
        self,
        depth: np.ndarray,
        filename: str,
        **kwargs
    ) -> Path:
        """
        Export depth map metadata to JSON file.

        Args:
            depth: Depth map array
            filename: Output filename (without .json extension)
            **kwargs: Additional metadata fields

        Returns:
            Path to saved JSON file

        Raises:
            ExportError: If export fails
        """
        try:
            filepath = self.output_dir / f"{filename}.json"

            # Create metadata
            metadata = self.create_metadata_dict(depth, **kwargs)

            # Write JSON
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Exported metadata: {filepath.name}")
            return filepath

        except Exception as e:
            raise ExportError(f"Failed to export metadata: {str(e)}")

    def export_batch_metadata(
        self,
        batch_results: List[Dict[str, Any]],
        filename: str = "batch_results"
    ) -> Path:
        """
        Export batch processing results to JSON.

        Args:
            batch_results: List of result dictionaries
            filename: Output filename

        Returns:
            Path to saved JSON file
        """
        try:
            filepath = self.output_dir / f"{filename}.json"

            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(batch_results),
                'results': batch_results
            }

            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Exported batch metadata: {filepath.name}")
            return filepath

        except Exception as e:
            raise ExportError(f"Failed to export batch metadata: {str(e)}")

    def export_performance_report(
        self,
        performance_data: Dict[str, Any],
        filename: str = "performance_report"
    ) -> Path:
        """
        Export performance metrics report.

        Args:
            performance_data: Performance metrics dictionary
            filename: Output filename

        Returns:
            Path to saved JSON file
        """
        try:
            filepath = self.output_dir / f"{filename}.json"

            report = {
                'timestamp': datetime.now().isoformat(),
                'performance': performance_data
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Exported performance report: {filepath.name}")
            return filepath

        except Exception as e:
            raise ExportError(f"Failed to export performance report: {str(e)}")

    def load_metadata(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load metadata from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Metadata dictionary

        Raises:
            ExportError: If loading fails
        """
        try:
            filepath = Path(filepath)

            with open(filepath, 'r') as f:
                metadata = json.load(f)

            logger.debug(f"Loaded metadata: {filepath.name}")
            return metadata

        except Exception as e:
            raise ExportError(f"Failed to load metadata: {str(e)}")

    def __repr__(self) -> str:
        """String representation of JSON exporter."""
        return f"JSONExporter(output_dir='{self.output_dir}')"

"""Custom exceptions for the Vision-Based 3D Measurement & Perception Tool."""


class Vision3DError(Exception):
    """Base exception for all vision 3D perception errors."""
    pass


class ModelLoadError(Vision3DError):
    """Raised when model loading fails."""
    pass


class InvalidInputError(Vision3DError):
    """Raised when input validation fails."""
    pass


class DeviceError(Vision3DError):
    """Raised when device configuration or detection fails."""
    pass


class MeasurementError(Vision3DError):
    """Raised when measurement computation fails."""
    pass


class CalibrationError(Vision3DError):
    """Raised when calibration process fails."""
    pass


class PreprocessingError(Vision3DError):
    """Raised when preprocessing operations fail."""
    pass


class PostprocessingError(Vision3DError):
    """Raised when postprocessing operations fail."""
    pass


class VisualizationError(Vision3DError):
    """Raised when visualization operations fail."""
    pass


class ExportError(Vision3DError):
    """Raised when export operations fail."""
    pass

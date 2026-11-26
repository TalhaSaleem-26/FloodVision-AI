class InferenceError(Exception):
    """Base exception for inference-related errors."""


class ArtifactNotFoundError(InferenceError):
    """Required production artifact is missing."""


class ArtifactIntegrityError(InferenceError):
    """Artifact integrity verification failed."""


class ModelCompatibilityError(InferenceError):
    """Unsupported model version or configuration."""


class ModelLoadError(InferenceError):
    """Model reconstruction or loading failed."""


class DeviceUnavailableError(InferenceError):
    """Requested inference device is unavailable."""


class InputValidationError(InferenceError):
    """Input imagery validation failed."""


class PreprocessingError(InferenceError):
    """Input preprocessing failed."""


class PredictionError(InferenceError):
    """Model prediction failed."""


class PostprocessingError(InferenceError):
    """Prediction postprocessing failed."""


class GeospatialError(InferenceError):
    """Geospatial processing failed."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from affine import Affine
from rasterio.crs import CRS


# =====================================================
# 1. RASTER METADATA
# =====================================================

@dataclass(frozen=True)
class RasterMetadata:
    """Geospatial metadata associated with an input raster."""

    path: Path
    width: int
    height: int
    count: int

    crs: CRS | None
    transform: Affine

    dtypes: tuple[str, ...]
    nodata_values: tuple[float | None, ...]
    band_descriptions: tuple[str | None, ...]

    band_order_verified: bool


# =====================================================
# 2. RAW SAR INPUT
# =====================================================

@dataclass
class SARInput:
    """Raw SAR imagery and its validity mask."""

    image: np.ndarray
    valid_mask: np.ndarray
    metadata: RasterMetadata


# =====================================================
# 3. PREPROCESSED INPUT
# =====================================================

@dataclass
class PreprocessedInput:
    """Model-compatible input and associated metadata."""

    tensor: torch.Tensor
    valid_mask: np.ndarray
    metadata: RasterMetadata

    @property
    def valid_pixels(self) -> int:

        return int(self.valid_mask.sum())

    @property
    def invalid_pixels(self) -> int:

        return int(
            self.valid_mask.size - self.valid_pixels
        )

    @property
    def total_pixels(self) -> int:

        return int(self.valid_mask.size)

    @property
    def valid_percentage(self) -> float:

        if self.total_pixels == 0:
            return 0.0

        return (
            self.valid_pixels / self.total_pixels
        ) * 100


# =====================================================
# 4. RAW MODEL PREDICTION
# =====================================================

@dataclass
class RawPrediction:
    """Raw model output before postprocessing."""

    logits: torch.Tensor

    valid_mask: np.ndarray
    metadata: RasterMetadata

    model_version: str
    device: str

    inference_time_ms: float


# =====================================================
# 5. PREDICTION STATISTICS
# =====================================================

@dataclass(frozen=True)
class PredictionStatistics:
    """Pixel-level segmentation statistics."""

    total_pixels: int
    valid_pixels: int
    invalid_pixels: int

    water_pixels: int
    non_water_pixels: int

    water_coverage: float
    non_water_coverage: float
    valid_percentage: float

    mean_water_probability: float


# =====================================================
# 6. FINAL PREDICTION RESULT
# =====================================================

@dataclass
class PredictionResult:
    """Postprocessed water segmentation result."""

    mask: np.ndarray
    probabilities: np.ndarray

    valid_mask: np.ndarray
    metadata: RasterMetadata

    statistics: PredictionStatistics

    model_version: str
    threshold: float
    device: str

    inference_time_ms: float
    postprocessing_time_ms: float

    @property
    def water_coverage(self) -> float:

        return self.statistics.water_coverage

    @property
    def processing_time_ms(self) -> float:

        return (
            self.inference_time_ms
            + self.postprocessing_time_ms
        )

    def to_summary(self) -> dict:

        return {
            "source_file": self.metadata.path.name,

            "model_version": self.model_version,
            "device": self.device,
            "threshold": self.threshold,

            "width": self.metadata.width,
            "height": self.metadata.height,

            "crs": (
                str(self.metadata.crs)
                if self.metadata.crs is not None
                else None
            ),

            "band_order_verified": (
                self.metadata.band_order_verified
            ),

            "statistics": {
                "total_pixels": (
                    self.statistics.total_pixels
                ),
                "valid_pixels": (
                    self.statistics.valid_pixels
                ),
                "invalid_pixels": (
                    self.statistics.invalid_pixels
                ),
                "water_pixels": (
                    self.statistics.water_pixels
                ),
                "non_water_pixels": (
                    self.statistics.non_water_pixels
                ),
                "water_coverage": (
                    self.statistics.water_coverage
                ),
                "non_water_coverage": (
                    self.statistics.non_water_coverage
                ),
                "valid_percentage": (
                    self.statistics.valid_percentage
                ),
                "mean_water_probability": (
                    self.statistics.mean_water_probability
                ),
            },

            "timing": {
                "inference_ms": self.inference_time_ms,
                "postprocessing_ms": (
                    self.postprocessing_time_ms
                ),
                "processing_ms": self.processing_time_ms,
            },
        }

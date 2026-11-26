
from dataclasses import dataclass, asdict
from pathlib import Path
import math

from .exceptions import InputValidationError
from .tiling import TilingConfig


@dataclass(frozen=True)
class PipelineConfig:

    # Model configuration
    device: str = "auto"

    # Input validation
    require_crs: bool = True
    allow_unverified_band_order: bool = False

    # Tiling configuration
    tile_size: int = 512
    overlap: int = 128
    min_blend_weight: float = 0.001

    # Storage configuration
    storage: str = "auto"
    memory_budget_mib: int = 128

    # Resource limits
    max_pixels: int = 100_000_000
    max_tiles: int = 1000

    # Vectorization configuration
    connectivity: int = 4
    max_vector_pixels: int = 4_000_000
    max_regions: int = 100_000

    # Geometry processing
    simplification_tolerance_m: float = 5.0
    max_area_change_percentage: float = 1.0

    # Geometry fidelity
    min_geometry_iou: float = 1.0
    max_invalid_encroachment: int = 0

    # Raster export
    raster_block_size: int = 256

    # Temporary storage
    work_dir: Path | None = None

    def __post_init__(self):

        integer_fields = (
            "tile_size",
            "overlap",
            "memory_budget_mib",
            "max_pixels",
            "max_tiles",
            "connectivity",
            "max_vector_pixels",
            "max_regions",
            "max_invalid_encroachment",
            "raster_block_size"
        )

        for name in integer_fields:

            if type(getattr(self, name)) is not int:
                raise InputValidationError(
                    f"{name} must be an integer."
                )

        if not isinstance(self.device, str):
            raise InputValidationError(
                "Invalid device configuration."
            )

        if self.tile_size != 512:
            raise InputValidationError(
                "Production model requires 512 x 512 tiles."
            )

        if not 0 <= self.overlap < self.tile_size:
            raise InputValidationError(
                "Invalid tile overlap."
            )

        if (
            not math.isfinite(self.min_blend_weight)
            or not 0 < self.min_blend_weight <= 1
        ):
            raise InputValidationError(
                "Invalid minimum blending weight."
            )

        if self.storage not in (
            "auto", "memory", "disk"
        ):
            raise InputValidationError(
                "Unsupported storage mode."
            )

        positive_fields = (
            "memory_budget_mib",
            "max_pixels",
            "max_tiles",
            "max_vector_pixels",
            "max_regions"
        )

        for name in positive_fields:

            if getattr(self, name) <= 0:
                raise InputValidationError(
                    f"{name} must be positive."
                )

        if self.max_vector_pixels > self.max_pixels:
            raise InputValidationError(
                "Vector pixel limit exceeds raster limit."
            )

        if self.connectivity not in (4, 8):
            raise InputValidationError(
                "Connectivity must be 4 or 8."
            )

        if (
            not math.isfinite(
                self.simplification_tolerance_m
            )
            or self.simplification_tolerance_m < 0
        ):
            raise InputValidationError(
                "Invalid simplification tolerance."
            )

        if (
            not math.isfinite(
                self.max_area_change_percentage
            )
            or not 0 <= self.max_area_change_percentage <= 100
        ):
            raise InputValidationError(
                "Invalid maximum area change."
            )

        if (
            not math.isfinite(self.min_geometry_iou)
            or not 0 < self.min_geometry_iou <= 1
        ):
            raise InputValidationError(
                "Invalid minimum geometry IoU."
            )

        if self.max_invalid_encroachment != 0:
            raise InputValidationError(
                "Invalid pixel encroachment is not allowed."
            )

        if (
            self.raster_block_size < 16
            or self.raster_block_size % 16 != 0
        ):
            raise InputValidationError(
                "Raster block size must be a positive multiple of 16."
            )

        if self.work_dir is not None:

            object.__setattr__(
                self,
                "work_dir",
                Path(self.work_dir).expanduser().resolve()
            )

    def get_tiling_config(self):

        return TilingConfig(
            tile_size=self.tile_size,
            overlap=self.overlap,
            min_blend_weight=self.min_blend_weight
        )

    def to_dict(self):

        result = asdict(self)

        if self.work_dir is not None:
            result["work_dir"] = str(self.work_dir)

        return result

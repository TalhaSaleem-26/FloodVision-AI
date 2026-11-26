
from pathlib import Path

import numpy as np
import rasterio

from rasterio.errors import RasterioError

from .exceptions import InputValidationError
from .schemas import RasterMetadata, SARInput


class InputValidator:
    """Validate Sentinel-1 SAR imagery for inference."""

    SUPPORTED_EXTENSIONS = {".tif", ".tiff"}
    EXPECTED_BANDS = ("VV", "VH")
    EXPECTED_CHANNELS = 2

    TILE_HEIGHT = 512
    TILE_WIDTH = 512

    def __init__(
        self,
        require_crs=True,
        allow_unverified_band_order=False
    ):

        self.require_crs = require_crs

        self.allow_unverified_band_order = (
            allow_unverified_band_order
        )

    # =====================================================
    # 1. FILE VALIDATION
    # =====================================================

    def validate_path(self, path):

        path = Path(path).expanduser().resolve()

        if not path.is_file():
            raise InputValidationError(
                f"Input file not found: {path}"
            )

        if (
            path.suffix.lower()
            not in self.SUPPORTED_EXTENSIONS
        ):
            raise InputValidationError(
                "Only .tif and .tiff files are supported."
            )

        if path.stat().st_size == 0:
            raise InputValidationError(
                "Input file is empty."
            )

        return path

    # =====================================================
    # 2. BAND ORDER VALIDATION
    # =====================================================

    def validate_band_order(self, descriptions):

        normalized = tuple(
            str(description).strip().upper()
            if description else None
            for description in descriptions
        )

        if normalized == self.EXPECTED_BANDS:
            return True

        # Reject known contradictory band descriptions.
        for expected, actual in zip(
            self.EXPECTED_BANDS,
            normalized
        ):

            if (
                actual in self.EXPECTED_BANDS
                and actual != expected
            ):
                raise InputValidationError(
                    f"Incorrect band order: {normalized}. "
                    "Expected VV followed by VH."
                )

        if self.allow_unverified_band_order:
            return False

        raise InputValidationError(
            "Unable to verify VV/VH band order. "
            "Confirm the input channel order explicitly."
        )

    # =====================================================
    # 3. RASTER METADATA VALIDATION
    # =====================================================

    def validate_raster(
        self,
        path,
        require_tile_size=True
    ):

        path = self.validate_path(path)

        try:

            with rasterio.open(path) as src:

                if src.driver != "GTiff":
                    raise InputValidationError(
                        f"Unsupported raster driver: {src.driver}"
                    )

                if src.count != self.EXPECTED_CHANNELS:
                    raise InputValidationError(
                        f"Expected 2 channels, got {src.count}."
                    )

                if src.height <= 0 or src.width <= 0:
                    raise InputValidationError(
                        "Invalid raster dimensions."
                    )

                if require_tile_size:

                    if (
                        src.height != self.TILE_HEIGHT
                        or src.width != self.TILE_WIDTH
                    ):
                        raise InputValidationError(
                            "Direct inference requires "
                            "512 x 512 imagery. "
                            f"Received {src.height} x {src.width}."
                        )

                for dtype in src.dtypes:

                    if np.dtype(dtype).kind not in (
                        "i", "u", "f"
                    ):
                        raise InputValidationError(
                            f"Unsupported raster dtype: {dtype}"
                        )

                if self.require_crs and src.crs is None:
                    raise InputValidationError(
                        "Raster CRS is missing."
                    )

                transform_values = np.asarray(
                    tuple(src.transform),
                    dtype=np.float64
                )

                if not np.isfinite(transform_values).all():
                    raise InputValidationError(
                        "Invalid raster transform."
                    )

                if src.transform.determinant == 0:
                    raise InputValidationError(
                        "Raster transform is not invertible."
                    )

                band_order_verified = (
                    self.validate_band_order(
                        src.descriptions
                    )
                )

                metadata = RasterMetadata(
                    path=path,
                    width=src.width,
                    height=src.height,
                    count=src.count,
                    crs=src.crs,
                    transform=src.transform,
                    dtypes=tuple(src.dtypes),
                    nodata_values=tuple(src.nodatavals),
                    band_descriptions=tuple(
                        src.descriptions
                    ),
                    band_order_verified=band_order_verified
                )

        except (RasterioError, OSError) as exc:
            raise InputValidationError(
                f"Unable to open raster: {path}"
            ) from exc

        return metadata

    # =====================================================
    # 4. RAW ARRAY VALIDATION
    # =====================================================

    def validate_array(
        self,
        image,
        valid_mask,
        require_tile_size=True
    ):

        if not isinstance(image, np.ndarray):
            raise InputValidationError(
                "Image must be a NumPy array."
            )

        if image.ndim != 3:
            raise InputValidationError(
                "Expected image shape (C, H, W)."
            )

        if image.shape[0] != self.EXPECTED_CHANNELS:
            raise InputValidationError(
                "Expected exactly 2 SAR channels."
            )

        if image.dtype.kind not in ("i", "u", "f"):
            raise InputValidationError(
                "Image must contain real numeric values."
            )

        if image.shape[1] <= 0 or image.shape[2] <= 0:
            raise InputValidationError(
                "Invalid image dimensions."
            )

        if require_tile_size:

            if image.shape[1:] != (
                self.TILE_HEIGHT,
                self.TILE_WIDTH
            ):
                raise InputValidationError(
                    f"Invalid tile shape: {image.shape}"
                )

        if not isinstance(valid_mask, np.ndarray):
            raise InputValidationError(
                "Valid mask must be a NumPy array."
            )

        if valid_mask.shape != image.shape[1:]:
            raise InputValidationError(
                "Valid mask dimensions do not match image."
            )

        if valid_mask.dtype != np.bool_:
            raise InputValidationError(
                "Valid mask must have boolean dtype."
            )

        effective_mask = (
            valid_mask
            & np.isfinite(image).all(axis=0)
        )

        if not effective_mask.any():
            raise InputValidationError(
                "Image contains no valid input pixels."
            )

        return effective_mask

    # =====================================================
    # 5. STRUCTURED INPUT VALIDATION
    # =====================================================

    def validate_input(
        self,
        sar_input: SARInput,
        require_tile_size=True
    ):

        metadata = sar_input.metadata

        expected_shape = (
            metadata.count,
            metadata.height,
            metadata.width
        )

        if sar_input.image.shape != expected_shape:
            raise InputValidationError(
                "Image shape does not match raster metadata."
            )

        return self.validate_array(
            image=sar_input.image,
            valid_mask=sar_input.valid_mask,
            require_tile_size=require_tile_size
        )


from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio

from rasterio.errors import RasterioError

from .exceptions import InputValidationError

from .schemas import (
    RasterMetadata,
    SARInput,
)

from .validator import InputValidator

from .tiling import (
    TilingConfig,
    TileWindow,
    SlidingWindowGenerator,
)


# =====================================================
# 1. STRUCTURED TILE RESULT
# =====================================================

@dataclass
class RasterTile:
    """A padded SAR tile and its original raster window."""

    window: TileWindow
    sar_input: SARInput

    @property
    def valid_pixels(self) -> int:

        return int(
            self.sar_input.valid_mask.sum()
        )

    @property
    def is_empty(self) -> bool:

        return self.valid_pixels == 0

    @property
    def padding_pixels(self) -> int:

        return (
            self.window.tile_size ** 2
            - self.window.height * self.window.width
        )

    @property
    def original_shape(self) -> tuple[int, int]:

        return (
            self.window.height,
            self.window.width
        )


# =====================================================
# 2. MEMORY-EFFICIENT RASTER READER
# =====================================================

class RasterTileReader:
    """Read georeferenced SAR imagery one tile at a time."""

    def __init__(
        self,
        config: TilingConfig | None = None,
        require_crs: bool = True,
        allow_unverified_band_order: bool = False
    ):

        self.config = config or TilingConfig()

        self.generator = SlidingWindowGenerator(
            self.config
        )

        self.validator = InputValidator(
            require_crs=require_crs,
            allow_unverified_band_order=(
                allow_unverified_band_order
            )
        )

    # =================================================
    # 3. VALIDATE ORIGINAL RASTER
    # =================================================

    def get_metadata(
        self,
        path: str | Path
    ) -> RasterMetadata:

        return self.validator.validate_raster(
            path,
            require_tile_size=False
        )

    # =================================================
    # 4. READ INDIVIDUAL WINDOW
    # =================================================

    def read_window(
        self,
        src,
        metadata: RasterMetadata,
        window: TileWindow
    ) -> RasterTile:

        if window.tile_size != self.config.tile_size:

            raise InputValidationError(
                "Window tile size mismatch."
            )

        if (
            window.row_off < 0
            or window.col_off < 0
            or window.height <= 0
            or window.width <= 0
            or window.height > window.tile_size
            or window.width > window.tile_size
            or window.row_end > metadata.height
            or window.col_end > metadata.width
        ):

            raise InputValidationError(
                "Tile window exceeds raster boundaries."
            )

        raster_window = window.rasterio_window

        try:

            # Read only the requested spatial window.
            image = src.read(
                window=raster_window,
                out_dtype="float32"
            )

            band_masks = src.read_masks(
                window=raster_window
            )

        except (RasterioError, OSError) as exc:

            raise InputValidationError(
                f"Failed to read tile {window.index}."
            ) from exc

        expected_shape = (
            2,
            window.height,
            window.width
        )

        if image.shape != expected_shape:

            raise InputValidationError(
                f"Unexpected tile shape: {image.shape}"
            )

        if band_masks.shape != expected_shape:

            raise InputValidationError(
                "Unexpected raster mask dimensions."
            )

        # Pixel is valid only if both SAR bands are valid.
        valid_mask = (
            (band_masks > 0).all(axis=0)
            & np.isfinite(image).all(axis=0)
        )

        # Remove invalid raw values.
        image[:, ~valid_mask] = 0.0

        # =============================================
        # 5. PAD SMALL WINDOWS
        # =============================================

        tile_size = self.config.tile_size

        padded_image = np.zeros(
            (2, tile_size, tile_size),
            dtype=np.float32
        )

        padded_mask = np.zeros(
            (tile_size, tile_size),
            dtype=bool
        )

        padded_image[
            :,
            :window.height,
            :window.width
        ] = image

        padded_mask[
            :window.height,
            :window.width
        ] = valid_mask

        # =============================================
        # 6. PRESERVE TILE GEOREFERENCING
        # =============================================

        tile_transform = src.window_transform(
            raster_window
        )

        tile_metadata = replace(
            metadata,
            width=tile_size,
            height=tile_size,
            transform=tile_transform,
            dtypes=("float32", "float32")
        )

        sar_input = SARInput(
            image=padded_image,
            valid_mask=padded_mask,
            metadata=tile_metadata
        )

        return RasterTile(
            window=window,
            sar_input=sar_input
        )

    # =================================================
    # 7. STREAM ALL RASTER TILES
    # =================================================

    def iter_tiles(
        self,
        path: str | Path
    ) -> Iterator[RasterTile]:

        metadata = self.get_metadata(path)

        try:

            with rasterio.open(metadata.path) as src:

                if (
                    src.height != metadata.height
                    or src.width != metadata.width
                    or src.count != metadata.count
                    or src.crs != metadata.crs
                    or src.transform != metadata.transform
                ):

                    raise InputValidationError(
                        "Raster metadata changed before reading."
                    )

                for window in self.generator.generate(
                    height=metadata.height,
                    width=metadata.width
                ):

                    yield self.read_window(
                        src=src,
                        metadata=metadata,
                        window=window
                    )

        except (RasterioError, OSError) as exc:

            raise InputValidationError(
                f"Unable to read raster: {metadata.path}"
            ) from exc

    # =================================================
    # 8. TOTAL WINDOW COUNT
    # =================================================

    def count_tiles(
        self,
        path: str | Path
    ) -> int:

        metadata = self.get_metadata(path)

        return self.generator.count(
            height=metadata.height,
            width=metadata.width
        )

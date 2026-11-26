
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from rasterio.windows import Window


# =====================================================
# 1. TILING CONFIGURATION
# =====================================================

@dataclass(frozen=True)
class TilingConfig:

    tile_size: int = 512
    overlap: int = 128
    min_blend_weight: float = 0.001

    def __post_init__(self):

        if type(self.tile_size) is not int:
            raise ValueError(
                "Tile size must be an integer."
            )

        if self.tile_size != 512:
            raise ValueError(
                "Current model requires tile size 512."
            )

        if type(self.overlap) is not int:
            raise ValueError(
                "Overlap must be an integer."
            )

        if not 0 <= self.overlap < self.tile_size:
            raise ValueError(
                "Overlap must be between 0 and tile size."
            )

        if not 0 < self.min_blend_weight <= 1:
            raise ValueError(
                "Invalid minimum blending weight."
            )

    @property
    def stride(self) -> int:

        return self.tile_size - self.overlap


# =====================================================
# 2. TILE WINDOW
# =====================================================

@dataclass(frozen=True)
class TileWindow:

    index: int

    row_off: int
    col_off: int

    height: int
    width: int

    tile_size: int

    @property
    def row_end(self) -> int:

        return self.row_off + self.height

    @property
    def col_end(self) -> int:

        return self.col_off + self.width

    @property
    def pad_bottom(self) -> int:

        return self.tile_size - self.height

    @property
    def pad_right(self) -> int:

        return self.tile_size - self.width

    @property
    def requires_padding(self) -> bool:

        return (
            self.pad_bottom > 0
            or self.pad_right > 0
        )

    @property
    def rasterio_window(self) -> Window:

        return Window(
            col_off=self.col_off,
            row_off=self.row_off,
            width=self.width,
            height=self.height
        )

    @property
    def raster_slices(self):

        return (
            slice(self.row_off, self.row_end),
            slice(self.col_off, self.col_end)
        )


# =====================================================
# 3. SLIDING WINDOW GENERATOR
# =====================================================

class SlidingWindowGenerator:

    def __init__(
        self,
        config: TilingConfig | None = None
    ):

        self.config = config or TilingConfig()

    def generate_positions(
        self,
        length: int
    ) -> list[int]:

        if type(length) is not int or length <= 0:
            raise ValueError(
                "Raster dimension must be a positive integer."
            )

        tile_size = self.config.tile_size
        stride = self.config.stride

        if length <= tile_size:
            return [0]

        last_position = length - tile_size

        positions = list(
            range(
                0,
                last_position + 1,
                stride
            )
        )

        # Anchor final tile to raster boundary.
        if positions[-1] != last_position:

            positions.append(last_position)

        return positions

    def generate(
        self,
        height: int,
        width: int
    ) -> Iterator[TileWindow]:

        rows = self.generate_positions(height)
        cols = self.generate_positions(width)

        tile_size = self.config.tile_size

        index = 0

        for row in rows:

            for col in cols:

                actual_height = min(
                    tile_size,
                    height - row
                )

                actual_width = min(
                    tile_size,
                    width - col
                )

                yield TileWindow(
                    index=index,
                    row_off=row,
                    col_off=col,
                    height=actual_height,
                    width=actual_width,
                    tile_size=tile_size
                )

                index += 1

    def count(
        self,
        height: int,
        width: int
    ) -> int:

        return (
            len(self.generate_positions(height))
            * len(self.generate_positions(width))
        )


# =====================================================
# 4. BLENDING WEIGHTS
# =====================================================

class TileWeightGenerator:

    def __init__(
        self,
        config: TilingConfig | None = None
    ):

        self.config = config or TilingConfig()

        self.weights = self._create_weights()

    def _create_weights(self) -> np.ndarray:

        tile_size = self.config.tile_size

        if self.config.overlap == 0:

            return np.ones(
                (tile_size, tile_size),
                dtype=np.float32
            )

        # Hann window gives higher importance
        # to predictions near the tile center.

        weights_1d = np.hanning(
            tile_size
        ).astype(np.float32)

        weights_2d = np.outer(
            weights_1d,
            weights_1d
        )

        weights_2d /= weights_2d.max()

        # Prevent zero-weight raster boundaries.
        weights_2d = np.maximum(
            weights_2d,
            self.config.min_blend_weight
        )

        return weights_2d.astype(np.float32)

    def get_weights(
        self,
        window: TileWindow | None = None
    ) -> np.ndarray:

        if window is None:

            return self.weights.copy()

        if window.tile_size != self.config.tile_size:
            raise ValueError(
                "Window tile size mismatch."
            )

        if not (
            0 < window.height <= window.tile_size
            and 0 < window.width <= window.tile_size
        ):
            raise ValueError(
                "Invalid window dimensions."
            )

        return self.weights[
            :window.height,
            :window.width
        ].copy()

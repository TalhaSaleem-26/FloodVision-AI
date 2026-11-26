
from pathlib import Path
from tempfile import TemporaryDirectory
import math

import numpy as np

from .exceptions import PostprocessingError

from .tiling import TileWindow


class WeightedProbabilityAccumulator:
    """
    Accumulate overlapping tile probabilities using
    weighted averaging.

    Supports memory and disk-backed storage.
    """

    def __init__(
        self,
        height: int,
        width: int,
        storage: str = "memory",
        work_dir: str | Path | None = None
    ):

        if (
            type(height) is not int
            or type(width) is not int
            or height <= 0
            or width <= 0
        ):
            raise ValueError(
                "Raster dimensions must be positive integers."
            )

        if storage not in {"memory", "disk"}:
            raise ValueError(
                "Storage must be memory or disk."
            )

        self.height = height
        self.width = width
        self.shape = (height, width)

        self.storage = storage

        self.tiles_added = 0

        self._closed = False
        self._finalized = False

        self._temp_dir = None

        self.probability_sum = None
        self.weight_sum = None

        self.probabilities = None
        self.mask = None

        if self.storage == "disk":

            if work_dir is not None:

                work_dir = Path(work_dir)

                work_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

            self._temp_dir = TemporaryDirectory(
                prefix="floodvision_accumulator_",
                dir=work_dir
            )

        try:

            self.probability_sum = self._allocate(
                "probability_sum",
                np.float32,
                0.0
            )

            self.weight_sum = self._allocate(
                "weight_sum",
                np.float32,
                0.0
            )

        except Exception:

            self.close()

            raise

    # =================================================
    # 1. ARRAY ALLOCATION
    # =================================================

    def _allocate(
        self,
        name,
        dtype,
        fill_value
    ):

        if self.storage == "memory":

            return np.full(
                self.shape,
                fill_value,
                dtype=dtype
            )

        path = (
            Path(self._temp_dir.name)
            / f"{name}.dat"
        )

        array = np.memmap(
            path,
            dtype=dtype,
            mode="w+",
            shape=self.shape
        )

        array[:] = fill_value

        return array

    # =================================================
    # 2. STATE VALIDATION
    # =================================================

    def _ensure_open(self):

        if self._closed:

            raise RuntimeError(
                "Accumulator is already closed."
            )

    # =================================================
    # 3. ADD INDIVIDUAL TILE
    # =================================================

    def add(
        self,
        window: TileWindow,
        probabilities: np.ndarray,
        valid_mask: np.ndarray,
        weights: np.ndarray
    ):

        self._ensure_open()

        if self._finalized:

            raise RuntimeError(
                "Cannot add tiles after finalization."
            )

        if not isinstance(window, TileWindow):

            raise PostprocessingError(
                "Expected TileWindow object."
            )

        if (
            window.row_off < 0
            or window.col_off < 0
            or window.height <= 0
            or window.width <= 0
            or window.height > window.tile_size
            or window.width > window.tile_size
            or window.row_end > self.height
            or window.col_end > self.width
        ):

            raise PostprocessingError(
                "Tile exceeds accumulator boundaries."
            )

        expected_shape = (
            window.height,
            window.width
        )

        for name, array in (
            ("Probabilities", probabilities),
            ("Validity mask", valid_mask),
            ("Weights", weights)
        ):

            if not isinstance(array, np.ndarray):

                raise PostprocessingError(
                    f"{name} must be a NumPy array."
                )

            if array.shape != expected_shape:

                raise PostprocessingError(
                    f"Incorrect {name} shape: "
                    f"{array.shape}"
                )

        if valid_mask.dtype != np.bool_:

            raise PostprocessingError(
                "Validity mask must have boolean dtype."
            )

        if not np.issubdtype(
            probabilities.dtype,
            np.floating
        ):

            raise PostprocessingError(
                "Probabilities must be floating point."
            )

        if not np.issubdtype(
            weights.dtype,
            np.floating
        ):

            raise PostprocessingError(
                "Weights must be floating point."
            )

        if not np.isfinite(weights).all():

            raise PostprocessingError(
                "Blending weights contain NaN or Inf."
            )

        if np.any(weights <= 0) or np.any(weights > 1):

            raise PostprocessingError(
                "Blending weights must be in (0, 1]."
            )

        if not valid_mask.any():

            return False

        valid_probabilities = probabilities[valid_mask]

        if not np.isfinite(
            valid_probabilities
        ).all():

            raise PostprocessingError(
                "Valid probabilities contain NaN or Inf."
            )

        if (
            np.any(valid_probabilities < 0)
            or np.any(valid_probabilities > 1)
        ):

            raise PostprocessingError(
                "Probabilities must be between 0 and 1."
            )

        # Invalid pixels must contribute neither
        # probabilities nor blending weights.

        effective_weights = np.where(
            valid_mask,
            weights,
            0.0
        ).astype(np.float32)

        safe_probabilities = np.where(
            valid_mask,
            probabilities,
            0.0
        ).astype(np.float32)

        weighted_probabilities = (
            safe_probabilities
            * effective_weights
        )

        rows, cols = window.raster_slices

        self.probability_sum[
            rows, cols
        ] += weighted_probabilities

        self.weight_sum[
            rows, cols
        ] += effective_weights

        self.tiles_added += 1

        return True

    # =================================================
    # 4. FINALIZE SEGMENTATION
    # =================================================

    def finalize(
        self,
        threshold: float = 0.5,
        chunk_rows: int = 256
    ):

        self._ensure_open()

        if self._finalized:

            raise RuntimeError(
                "Accumulator already finalized."
            )

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 < threshold < 1
        ):

            raise ValueError(
                "Invalid segmentation threshold."
            )

        if type(chunk_rows) is not int or chunk_rows <= 0:

            raise ValueError(
                "chunk_rows must be a positive integer."
            )

        self.probabilities = self._allocate(
            "final_probabilities",
            np.float32,
            np.nan
        )

        self.mask = self._allocate(
            "final_mask",
            np.int8,
            -1
        )

        valid_pixels = 0
        water_pixels = 0
        non_water_pixels = 0

        probability_total = 0.0

        for start in range(
            0,
            self.height,
            chunk_rows
        ):

            end = min(
                start + chunk_rows,
                self.height
            )

            sums = self.probability_sum[start:end]

            weights = self.weight_sum[start:end]

            valid = weights > 0

            if not np.isfinite(sums).all():
                raise PostprocessingError(
                    "Accumulated probabilities are not finite."
                )

            if not np.isfinite(weights).all():
                raise PostprocessingError(
                    "Accumulated weights are not finite."
                )

            probability_chunk = np.full(
                sums.shape,
                np.nan,
                dtype=np.float32
            )

            np.divide(
                sums,
                weights,
                out=probability_chunk,
                where=valid
            )

            # Protect against minor floating-point drift.
            np.clip(
                probability_chunk,
                0.0,
                1.0,
                out=probability_chunk
            )

            mask_chunk = np.full(
                sums.shape,
                -1,
                dtype=np.int8
            )

            mask_chunk[valid] = (
                probability_chunk[valid] >= threshold
            ).astype(np.int8)

            self.probabilities[
                start:end
            ] = probability_chunk

            self.mask[
                start:end
            ] = mask_chunk

            valid_pixels += int(valid.sum())

            water_pixels += int(
                (mask_chunk == 1).sum()
            )

            non_water_pixels += int(
                (mask_chunk == 0).sum()
            )

            probability_total += float(
                probability_chunk[valid].sum(
                    dtype=np.float64
                )
            )

        total_pixels = self.height * self.width

        invalid_pixels = total_pixels - valid_pixels

        statistics = {
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "invalid_pixels": invalid_pixels,

            "water_pixels": water_pixels,
            "non_water_pixels": non_water_pixels,

            "water_coverage": (
                water_pixels / valid_pixels * 100
                if valid_pixels else None
            ),

            "non_water_coverage": (
                non_water_pixels / valid_pixels * 100
                if valid_pixels else None
            ),

            "valid_percentage": (
                valid_pixels / total_pixels * 100
            ),

            "mean_water_probability": (
                probability_total / valid_pixels
                if valid_pixels else None
            ),

            "tiles_added": self.tiles_added
        }

        self._finalized = True

        self.flush()

        return (
            self.mask,
            self.probabilities,
            statistics
        )

    # =================================================
    # 5. FLUSH DISK-BACKED ARRAYS
    # =================================================

    def flush(self):

        self._ensure_open()

        for array in (
            self.probability_sum,
            self.weight_sum,
            self.probabilities,
            self.mask
        ):

            if isinstance(array, np.memmap):

                array.flush()

    # =================================================
    # 6. RESOURCE CLEANUP
    # =================================================

    def close(self):

        if self._closed:
            return

        for name in (
            "probability_sum",
            "weight_sum",
            "probabilities",
            "mask"
        ):

            array = getattr(self, name, None)

            if isinstance(array, np.memmap):

                array.flush()

                # Explicitly release mapping for Windows.
                mapping = getattr(array, "_mmap", None)

                if mapping is not None:
                    mapping.close()

            setattr(self, name, None)

        if self._temp_dir is not None:

            self._temp_dir.cleanup()

            self._temp_dir = None

        self._closed = True

    # =================================================
    # 7. CONTEXT MANAGER
    # =================================================

    def __enter__(self):

        self._ensure_open()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):

        self.close()

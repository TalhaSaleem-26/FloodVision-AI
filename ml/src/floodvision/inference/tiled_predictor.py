
from contextlib import contextmanager, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import time

import numpy as np

from .schemas import RasterMetadata

from .exceptions import PredictionError

from .model_loader import ModelLoader
from .preprocessor import SARPreprocessor
from .predictor import FloodPredictor
from .postprocessor import SegmentationPostprocessor

from .tiling import (
    TilingConfig,
    TileWeightGenerator,
)

from .tile_reader import RasterTileReader

from .accumulator import (
    WeightedProbabilityAccumulator,
)


# =====================================================
# 1. TILED PREDICTION RESULT
# =====================================================

@dataclass
class TiledPredictionResult:

    mask: np.ndarray
    probabilities: np.ndarray

    metadata: RasterMetadata
    statistics: dict

    model_version: str
    device: str
    threshold: float

    total_tiles: int
    processed_tiles: int
    skipped_tiles: int

    tile_size: int
    overlap: int
    storage: str

    inference_time_ms: float
    processing_time_ms: float

    @property
    def water_coverage(self) -> float | None:

        return self.statistics["water_coverage"]

    @property
    def average_tile_inference_ms(self) -> float:

        if self.processed_tiles == 0:
            return 0.0

        return (
            self.inference_time_ms
            / self.processed_tiles
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

            "tiling": {
                "tile_size": self.tile_size,
                "overlap": self.overlap,
                "stride": (
                    self.tile_size - self.overlap
                ),
                "storage": self.storage,
                "total_tiles": self.total_tiles,
                "processed_tiles": self.processed_tiles,
                "skipped_tiles": self.skipped_tiles,
            },

            "statistics": self.statistics.copy(),

            "timing": {
                "inference_ms": self.inference_time_ms,

                "average_tile_inference_ms": (
                    self.average_tile_inference_ms
                ),

                "processing_ms": (
                    self.processing_time_ms
                ),
            },
        }


# =====================================================
# 2. PRODUCTION TILED INFERENCE ENGINE
# =====================================================

class TiledInferenceEngine:

    def __init__(
        self,
        loader: ModelLoader,
        config: TilingConfig | None = None,
        device: str = "auto",
        require_crs: bool = True,
        allow_unverified_band_order: bool = False,
        memory_budget_mib: int = 128,
        max_pixels: int = 100_000_000,
        max_tiles: int = 10_000,
    ):

        if (
            type(memory_budget_mib) is not int
            or memory_budget_mib <= 0
        ):
            raise ValueError(
                "Memory budget must be a positive integer."
            )

        if type(max_pixels) is not int or max_pixels <= 0:
            raise ValueError("Invalid maximum pixel count.")

        if type(max_tiles) is not int or max_tiles <= 0:
            raise ValueError("Invalid maximum tile count.")

        self.loader = loader
        self.config = config or TilingConfig()

        self.memory_budget_bytes = (
            memory_budget_mib * 1024 * 1024
        )

        self.max_pixels = max_pixels
        self.max_tiles = max_tiles

        self.predictor = FloodPredictor(
            loader=loader,
            device=device
        )

        self.threshold = float(
            loader.inference_config["threshold"]
        )

        self.preprocessor = SARPreprocessor(
            config_path=(
                loader.bundle_dir
                / "preprocessing.yaml"
            ),
            require_crs=require_crs,
            allow_unverified_band_order=(
                allow_unverified_band_order
            )
        )

        self.postprocessor = (
            SegmentationPostprocessor(
                threshold=self.threshold
            )
        )

        self.reader = RasterTileReader(
            config=self.config,
            require_crs=require_crs,
            allow_unverified_band_order=(
                allow_unverified_band_order
            )
        )

        self.weight_generator = TileWeightGenerator(
            self.config
        )

    # =================================================
    # 3. AUTOMATIC STORAGE SELECTION
    # =================================================

    def resolve_storage(
        self,
        height: int,
        width: int,
        storage: str = "auto"
    ) -> str:

        if storage not in {
            "auto", "memory", "disk"
        }:

            raise ValueError(
                "Storage must be auto, memory or disk."
            )

        # Two float32 accumulation arrays,
        # one float32 probability array,
        # and one int8 segmentation mask.
        required_bytes = height * width * 13

        if storage == "memory":

            if required_bytes > self.memory_budget_bytes:

                raise PredictionError(
                    "Raster exceeds accumulator memory budget. "
                    "Use disk-backed storage."
                )

            return "memory"

        if storage == "disk":
            return "disk"

        if required_bytes <= self.memory_budget_bytes:
            return "memory"

        return "disk"

    # =================================================
    # 4. PROCESS COMPLETE RASTER
    # =================================================

    @contextmanager
    def predict(
        self,
        path: str | Path,
        storage: str = "auto",
        work_dir: str | Path | None = None,
        chunk_rows: int = 256,
    ) -> Iterator[TiledPredictionResult]:

        start_time = time.perf_counter()

        metadata = self.reader.get_metadata(path)

        total_pixels = metadata.height * metadata.width

        if total_pixels > self.max_pixels:

            raise PredictionError(
                "Raster exceeds maximum supported pixel count."
            )

        expected_tiles = self.reader.generator.count(
            metadata.height,
            metadata.width
        )

        if expected_tiles > self.max_tiles:

            raise PredictionError(
                "Raster exceeds maximum supported tile count."
            )

        selected_storage = self.resolve_storage(
            height=metadata.height,
            width=metadata.width,
            storage=storage
        )

        total_tiles = 0
        processed_tiles = 0
        skipped_tiles = 0

        total_inference_ms = 0.0

        with WeightedProbabilityAccumulator(
            height=metadata.height,
            width=metadata.width,
            storage=selected_storage,
            work_dir=work_dir
        ) as accumulator:

            with closing(
                self.reader.iter_tiles(metadata.path)
            ) as stream:

                for raster_tile in stream:

                    total_tiles += 1

                    window = raster_tile.window

                    # Completely invalid windows
                    # must not enter the model.
                    if raster_tile.is_empty:

                        skipped_tiles += 1

                        continue

                    processed = (
                        self.preprocessor.preprocess(
                            raster_tile.sar_input
                        )
                    )

                    raw = self.predictor(processed)

                    tile_result = self.postprocessor(raw)

                    h = window.height
                    w = window.width

                    probabilities = (
                        tile_result.probabilities[:h, :w]
                    )

                    valid_mask = (
                        tile_result.valid_mask[:h, :w]
                    )

                    weights = (
                        self.weight_generator.get_weights(
                            window
                        )
                    )

                    added = accumulator.add(
                        window=window,
                        probabilities=probabilities,
                        valid_mask=valid_mask,
                        weights=weights
                    )

                    if not added:

                        raise PredictionError(
                            "Valid tile was not accumulated."
                        )

                    processed_tiles += 1

                    total_inference_ms += (
                        raw.inference_time_ms
                    )

            # Validate tile accounting.
            if total_tiles != expected_tiles:

                raise PredictionError(
                    "Unexpected number of raster tiles."
                )

            if (
                processed_tiles + skipped_tiles
                != total_tiles
            ):

                raise PredictionError(
                    "Tile accounting mismatch."
                )

            mask, probabilities, statistics = (
                accumulator.finalize(
                    threshold=self.threshold,
                    chunk_rows=chunk_rows
                )
            )

            if (
                statistics["tiles_added"]
                != processed_tiles
            ):

                raise PredictionError(
                    "Accumulated tile count mismatch."
                )

            processing_time_ms = (
                time.perf_counter() - start_time
            ) * 1000

            result = TiledPredictionResult(
                mask=mask,
                probabilities=probabilities,

                metadata=metadata,
                statistics=statistics,

                model_version=(
                    self.predictor.model_version
                ),

                device=str(self.predictor.device),

                threshold=self.threshold,

                total_tiles=total_tiles,
                processed_tiles=processed_tiles,
                skipped_tiles=skipped_tiles,

                tile_size=self.config.tile_size,
                overlap=self.config.overlap,
                storage=selected_storage,

                inference_time_ms=total_inference_ms,
                processing_time_ms=processing_time_ms
            )

            # Result remains usable while the
            # accumulator context is open.
            yield result

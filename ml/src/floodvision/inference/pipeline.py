
import time
import uuid

from pathlib import Path

from .exceptions import (
    InputValidationError,
    GeospatialError
)

from .pipeline_config import PipelineConfig
from .pipeline_result import PipelineResult

from .model_loader import ModelLoader
from .tiled_predictor import TiledInferenceEngine

from .geospatial import GeospatialAnalyzer
from .area_analyzer import AreaAnalyzer

from .vectorizer import WaterVectorizer
from .geometry_processor import GeometryProcessor
from .geometry_fidelity import GeometryFidelityValidator

from .raster_exporter import RasterExporter
from .vector_exporter import VectorExporter

from .artifact_manager import ArtifactManager


class FloodInferencePipeline:

    def __init__(
        self,
        bundle_dir,
        device=None,
        config=None
    ):

        if config is None:

            config = PipelineConfig(
                device=device or "auto"
            )

        elif device is not None and device != config.device:

            raise InputValidationError(
                "Conflicting device configuration."
            )

        if not isinstance(config, PipelineConfig):
            raise InputValidationError(
                "Expected PipelineConfig."
            )

        self.config = config

        self.loader = ModelLoader(bundle_dir)

        self.model = self.loader.load(
            device=config.device
        )

        self.model_info = self.loader.get_model_info()

        self.engine = TiledInferenceEngine(
            loader=self.loader,
            config=config.get_tiling_config(),
            device=config.device,
            require_crs=config.require_crs,
            allow_unverified_band_order=(
                config.allow_unverified_band_order
            ),
            memory_budget_mib=config.memory_budget_mib,
            max_pixels=config.max_pixels,
            max_tiles=config.max_tiles
        )

        self.raster_exporter = RasterExporter(
            block_size=config.raster_block_size
        )

        self.vectorizer = WaterVectorizer(
            connectivity=config.connectivity,
            max_pixels=config.max_vector_pixels,
            max_regions=config.max_regions
        )

        self.geometry_processor = GeometryProcessor(
            tolerance_m=(
                config.simplification_tolerance_m
            ),
            max_area_change_percentage=(
                config.max_area_change_percentage
            )
        )

        self.fidelity_validator = (
            GeometryFidelityValidator(
                min_iou=config.min_geometry_iou,
                max_invalid_encroachment=(
                    config.max_invalid_encroachment
                )
            )
        )

        self.vector_exporter = VectorExporter(
            model_version=self.model_info["model_version"],
            threshold=self.model_info["threshold"]
        )

    def get_model_info(self):

        return self.loader.get_model_info()

    def _export_rasters(
        self,
        prediction,
        manager
    ):

        self.raster_exporter.export_mask(
            mask=prediction.mask,
            metadata=prediction.metadata,
            output_path=manager.get_path(
                "water_mask.tif"
            ),
            model_version=prediction.model_version,
            threshold=prediction.threshold
        )

        manager.register(
            "water_mask",
            "water_mask.tif"
        )

        self.raster_exporter.export_probabilities(
            probabilities=prediction.probabilities,
            metadata=prediction.metadata,
            output_path=manager.get_path(
                "water_probability.tif"
            ),
            model_version=prediction.model_version,
            threshold=prediction.threshold
        )

        manager.register(
            "water_probability",
            "water_probability.tif"
        )

    def _process_vectors(
        self,
        prediction,
        manager,
        result
    ):

        vector_result = self.vectorizer.vectorize(
            prediction.mask,
            prediction.metadata
        )

        original_fidelity = (
            self.fidelity_validator.validate(
                vector_result,
                prediction.mask,
                prediction.metadata
            )
        )

        if not original_fidelity["passed"]:

            raise GeospatialError(
                "Original geometry fidelity validation failed."
            )

        result.vectorization = (
            vector_result.to_summary()
        )

        self.vector_exporter.export(
            result=vector_result,
            output_path=manager.get_path(
                "water_regions_original.geojson"
            ),
            source_file=result.source_file
        )

        manager.register(
            "water_regions_original",
            "water_regions_original.geojson"
        )

        processed = self.geometry_processor.process(
            vector_result
        )

        result.geometry_processing = (
            processed.to_summary()
        )

        try:

            simplified_fidelity = (
                self.fidelity_validator.validate(
                    processed,
                    prediction.mask,
                    prediction.metadata
                )
            )

        except GeospatialError as error:

            simplified_fidelity = {
                "passed": False,
                "reason": str(error)
            }

        use_simplified = (
            simplified_fidelity["passed"]
        )

        result.geometry_fidelity = {
            "original": original_fidelity,
            "simplified": simplified_fidelity,
            "fallback_used": not use_simplified,
            "selected_variant": (
                "processed"
                if use_simplified
                else "original"
            )
        }

        if use_simplified:

            self.vector_exporter.export(
                result=processed,
                output_path=manager.get_path(
                    "water_regions_simplified.geojson"
                ),
                source_file=result.source_file
            )

            manager.register(
                "water_regions_simplified",
                "water_regions_simplified.geojson"
            )

        else:

            result.warnings.append(
                "Simplified geometry failed fidelity "
                "validation. Original geometry retained."
            )

    def predict(
        self,
        input_path,
        output_dir,
        export_rasters=True,
        export_vectors=True,
        calculate_area=True
    ):

        start = time.perf_counter()

        input_path = Path(
            input_path
        ).expanduser().resolve()

        output_dir = Path(
            output_dir
        ).expanduser().resolve()

        if not input_path.is_file():
            raise InputValidationError(
                f"Input raster not found: {input_path}"
            )

        for flag in (
            export_rasters,
            export_vectors,
            calculate_area
        ):

            if type(flag) is not bool:
                raise InputValidationError(
                    "Export options must be boolean."
                )

        analysis_id = uuid.uuid4().hex

        with ArtifactManager(output_dir) as manager:

            with self.engine.predict(
                path=input_path,
                storage=self.config.storage,
                work_dir=self.config.work_dir
            ) as prediction:

                summary = prediction.to_summary()

                metadata = prediction.metadata

                geo_info = GeospatialAnalyzer(
                    metadata
                ).analyze()

                result = PipelineResult(
                    analysis_id=analysis_id,
                    source_file=input_path.name,
                    model_version=prediction.model_version,
                    device=prediction.device,
                    threshold=prediction.threshold,
                    output_dir=output_dir,
                    geospatial_metadata=geo_info.to_dict(),
                    statistics=dict(prediction.statistics),
                    tiling=dict(summary["tiling"]),
                    timing=dict(summary["timing"])
                )

                if not metadata.band_order_verified:

                    result.warnings.append(
                        "SAR band order could not be verified."
                    )

                if result.statistics["valid_pixels"] == 0:

                    result.warnings.append(
                        "Input raster contains no valid pixels."
                    )

                if calculate_area:

                    area_start = time.perf_counter()

                    area_stats = AreaAnalyzer(
                        metadata
                    ).analyze(
                        prediction.mask
                    )

                    result.area_analytics = (
                        area_stats.to_dict()
                    )

                    result.timing["area_analysis_ms"] = (
                        time.perf_counter() - area_start
                    ) * 1000

                if export_rasters:

                    raster_start = time.perf_counter()

                    self._export_rasters(
                        prediction,
                        manager
                    )

                    result.timing["raster_export_ms"] = (
                        time.perf_counter() - raster_start
                    ) * 1000

                if export_vectors:

                    vector_start = time.perf_counter()

                    self._process_vectors(
                        prediction,
                        manager,
                        result
                    )

                    result.timing["vector_processing_ms"] = (
                        time.perf_counter() - vector_start
                    ) * 1000

            # Temporary prediction arrays are closed here.

            result.artifacts = dict(
                manager.artifacts
            )

            result.timing["pipeline_processing_ms"] = (
                time.perf_counter() - start
            ) * 1000

            manager.write_json(
                "analysis_report.json",
                result.to_dict()
            )

            manager.register(
                "analysis_report",
                "analysis_report.json"
            )

            manager.write_manifest(
                analysis_id
            )

            manager.register(
                "manifest",
                "manifest.json"
            )

            manager.commit()

            result.artifacts = dict(
                manager.artifacts
            )

        return result

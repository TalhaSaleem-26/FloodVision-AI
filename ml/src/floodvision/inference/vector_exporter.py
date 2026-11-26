
import json
import math
import os
import tempfile

from pathlib import Path

from shapely.geometry import mapping
from shapely.geometry.polygon import orient
from shapely.geometry import MultiPolygon

from .exceptions import GeospatialError
from .vectorizer import VectorizationResult
from .geometry_processor import GeometryProcessingResult


class VectorExporter:

    def __init__(
        self,
        model_version: str = "1.0.0",
        threshold: float = 0.5
    ):

        if not isinstance(model_version, str) or not model_version.strip():
            raise GeospatialError(
                "Invalid model version."
            )

        if (
            not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise GeospatialError(
                "Invalid prediction threshold."
            )

        self.model_version = model_version
        self.threshold = float(threshold)

    # =============================================
    # 1. GEOMETRY VALIDATION
    # =============================================

    def _validate_geometry(self, geometry):

        if geometry.is_empty:
            raise GeospatialError(
                "Cannot export empty geometry."
            )

        if not geometry.is_valid:
            raise GeospatialError(
                "Cannot export invalid geometry."
            )

        if geometry.geom_type not in (
            "Polygon",
            "MultiPolygon"
        ):
            raise GeospatialError(
                "Unsupported geometry type."
            )

        bounds = geometry.bounds

        if not all(
            math.isfinite(value)
            for value in bounds
        ):
            raise GeospatialError(
                "Non-finite geometry bounds."
            )

        min_x, min_y, max_x, max_y = bounds

        if not (
            -180 <= min_x <= max_x <= 180
            and -90 <= min_y <= max_y <= 90
        ):
            raise GeospatialError(
                "Invalid WGS84 coordinates."
            )

    # =============================================
    # 2. RING ORIENTATION
    # =============================================

    def _orient_geometry(self, geometry):

        if geometry.geom_type == "Polygon":

            return orient(
                geometry,
                sign=1.0
            )

        return MultiPolygon([
            orient(part, sign=1.0)
            for part in geometry.geoms
        ])

    # =============================================
    # 3. CREATE GEOJSON FEATURE
    # =============================================

    def _create_feature(
        self,
        region_id,
        geometry,
        properties
    ):

        self._validate_geometry(geometry)

        geometry = self._orient_geometry(
            geometry
        )

        return {
            "type": "Feature",
            "id": int(region_id),
            "geometry": mapping(geometry),
            "properties": {
                **properties,
                "region_id": int(region_id),
                "classification": "predicted_water",
                "model_version": self.model_version,
                "threshold": self.threshold
            }
        }

    # =============================================
    # 4. BUILD FEATURE COLLECTION
    # =============================================

    def build_feature_collection(
        self,
        result,
        source_file: str
    ):

        if not isinstance(
            result,
            (
                VectorizationResult,
                GeometryProcessingResult
            )
        ):
            raise GeospatialError(
                "Unsupported vector result."
            )

        if result.crs != "EPSG:4326":
            raise GeospatialError(
                "GeoJSON requires WGS84 coordinates."
            )

        features = []
        bounds = []
        seen_ids = set()

        is_processed = isinstance(
            result,
            GeometryProcessingResult
        )

        for region in result.regions:

            if region.region_id in seen_ids:
                raise GeospatialError(
                    "Duplicate region identifier."
                )

            seen_ids.add(region.region_id)

            if is_processed:

                properties = {
                    **region.to_dict(),

                    # Canonical research area
                    "area_m2":
                        region.original_area_m2,

                    "area_km2":
                        region.original_area_m2
                        / 1_000_000,

                    "area_hectares":
                        region.original_area_m2
                        / 10_000,

                    # Display geometry area
                    "display_area_m2":
                        region.area_m2,

                    "geometry_variant":
                        "processed"
                }

            else:

                properties = {
                    **region.to_dict(),
                    "geometry_variant":
                        "original"
                }

            area = properties["area_m2"]

            if not math.isfinite(area) or area <= 0:
                raise GeospatialError(
                    "Invalid region area."
                )

            feature = self._create_feature(
                region_id=region.region_id,
                geometry=region.geometry,
                properties=properties
            )

            features.append(feature)

            bounds.append(
                region.geometry.bounds
            )

        collection = {
            "type": "FeatureCollection",

            "metadata": {
                "project": "FloodVision AI",
                "task": "binary_water_segmentation",
                "source_file": Path(source_file).name,
                "model_version": self.model_version,
                "threshold": self.threshold,
                "region_count": len(features),
                "geometry_variant": (
                    "processed"
                    if is_processed
                    else "original"
                ),
                "total_water_area_m2": math.fsum(
                    feature["properties"]["area_m2"]
                    for feature in features
                )
            },

            "features": features
        }

        if is_processed:

            collection["metadata"].update({
                "simplification_tolerance_m":
                    result.tolerance_m,

                "vertex_reduction_percentage":
                    result.vertex_reduction_percentage
            })

        if bounds:

            collection["bbox"] = [
                min(b[0] for b in bounds),
                min(b[1] for b in bounds),
                max(b[2] for b in bounds),
                max(b[3] for b in bounds)
            ]

        return collection

    # =============================================
    # 5. ATOMIC GEOJSON EXPORT
    # =============================================

    def export(
        self,
        result,
        output_path,
        source_file: str,
        overwrite: bool = False
    ):

        output_path = Path(
            output_path
        ).resolve()

        if output_path.suffix.lower() != ".geojson":
            raise GeospatialError(
                "Output must use .geojson extension."
            )

        if output_path.exists() and not overwrite:
            raise GeospatialError(
                f"Output already exists: {output_path}"
            )

        collection = self.build_feature_collection(
            result,
            source_file
        )

        temporary_path = None

        try:

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                suffix=".tmp",
                delete=False
            ) as file:

                temporary_path = Path(file.name)

                json.dump(
                    collection,
                    file,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":")
                )

                file.flush()
                os.fsync(file.fileno())

            if overwrite:

                os.replace(
                    temporary_path,
                    output_path
                )

            else:

                # Atomic no-overwrite publication
                # on filesystems supporting hard links.
                os.link(
                    temporary_path,
                    output_path
                )

                temporary_path.unlink()

        except Exception as exc:

            raise GeospatialError(
                "GeoJSON export failed."
            ) from exc

        finally:

            if temporary_path is not None:

                temporary_path.unlink(
                    missing_ok=True
                )

        return output_path

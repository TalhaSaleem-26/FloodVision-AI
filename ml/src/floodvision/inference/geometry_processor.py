
from dataclasses import dataclass
import math

from pyproj import CRS, Geod, Transformer

from shapely import get_num_coordinates
from shapely.geometry.polygon import orient

from shapely import transform

from .geometry_utils import extract_polygon_parts, repair_polygonal_geometry
from .exceptions import GeospatialError
from .vectorizer import VectorizationResult


@dataclass(frozen=True)
class ProcessedRegion:

    region_id: int
    original_geometry: object
    geometry: object

    original_area_m2: float
    area_m2: float

    original_vertices: int
    processed_vertices: int

    repaired: bool
    simplified: bool

    @property
    def area_change_percentage(self):

        return (
            abs(
                self.area_m2
                - self.original_area_m2
            )
            / self.original_area_m2
            * 100
        )

    @property
    def vertex_reduction_percentage(self):

        return (
            1
            - self.processed_vertices
            / self.original_vertices
        ) * 100

    def to_dict(self):

        return {
            "region_id": self.region_id,
            "original_area_m2": self.original_area_m2,
            "area_m2": self.area_m2,
            "area_km2": self.area_m2 / 1_000_000,
            "area_hectares": self.area_m2 / 10_000,
            "original_vertices": self.original_vertices,
            "processed_vertices": self.processed_vertices,
            "area_change_percentage":
                self.area_change_percentage,
            "vertex_reduction_percentage":
                self.vertex_reduction_percentage,
            "repaired": self.repaired,
            "simplified": self.simplified
        }


@dataclass
class GeometryProcessingResult:

    regions: list[ProcessedRegion]
    tolerance_m: float
    max_area_change_percentage: float

    crs: str = "EPSG:4326"

    @property
    def region_count(self):

        return len(self.regions)

    @property
    def original_vertices(self):

        return sum(
            r.original_vertices
            for r in self.regions
        )

    @property
    def processed_vertices(self):

        return sum(
            r.processed_vertices
            for r in self.regions
        )

    @property
    def original_area_m2(self):

        return math.fsum(
            r.original_area_m2
            for r in self.regions
        )

    @property
    def processed_area_m2(self):

        return math.fsum(
            r.area_m2
            for r in self.regions
        )

    @property
    def vertex_reduction_percentage(self):

        if self.original_vertices == 0:
            return 0.0

        return (
            1
            - self.processed_vertices
            / self.original_vertices
        ) * 100

    def to_summary(self):

        return {
            "crs": self.crs,
            "tolerance_m": self.tolerance_m,
            "max_area_change_percentage":
                self.max_area_change_percentage,
            "region_count": self.region_count,

            "repaired_regions": sum(
                r.repaired for r in self.regions
            ),

            "simplified_regions": sum(
                r.simplified for r in self.regions
            ),

            "original_vertices":
                self.original_vertices,

            "processed_vertices":
                self.processed_vertices,

            "vertex_reduction_percentage":
                self.vertex_reduction_percentage,

            "original_area_m2":
                self.original_area_m2,

            "processed_area_m2":
                self.processed_area_m2
        }


class GeometryProcessor:

    def __init__(
        self,
        tolerance_m: float = 5.0,
        max_area_change_percentage: float = 1.0
    ):

        if (
            not math.isfinite(tolerance_m)
            or tolerance_m < 0
        ):

            raise GeospatialError(
                "Invalid simplification tolerance."
            )

        if (
            not math.isfinite(
                max_area_change_percentage
            )
            or not 0 <= max_area_change_percentage <= 100
        ):

            raise GeospatialError(
                "Invalid maximum area change."
            )

        self.tolerance_m = float(tolerance_m)

        self.max_area_change_percentage = float(
            max_area_change_percentage
        )

        self.geod = Geod(ellps="WGS84")

        self._transformer_cache = {}

    # =============================================
    # 1. POLYGON EXTRACTION
    # =============================================

    def _polygon_parts(self, geometry):
        return extract_polygon_parts(geometry)

    # =============================================
    # 2. GEOMETRY REPAIR
    # =============================================

    def repair_geometry(self, geometry):
        return repair_polygonal_geometry(geometry)

    # =============================================
    # 3. GEODESIC AREA
    # =============================================

    def calculate_area(self, geometry):

        if geometry.geom_type == "Polygon":

            area, _ = (
                self.geod.geometry_area_perimeter(
                    orient(geometry, sign=1.0)
                )
            )

            return abs(float(area))

        if geometry.geom_type == "MultiPolygon":

            return math.fsum(
                self.calculate_area(part)
                for part in geometry.geoms
            )

        raise GeospatialError(
            "Unsupported geometry type."
        )

    # =============================================
    # 4. LOCAL METRIC PROJECTION
    # =============================================

    def _get_transformers(self, geometry):

        center = geometry.centroid

        longitude = center.x
        latitude = center.y

        if not (
            -180 <= longitude <= 180
            and -80 <= latitude <= 84
        ):

            raise GeospatialError(
                "Geometry outside supported UTM range."
            )

        zone = min(
            60,
            max(
                1,
                int((longitude + 180) // 6) + 1
            )
        )

        epsg = (
            32600 + zone
            if latitude >= 0
            else 32700 + zone
        )

        if epsg not in self._transformer_cache:

            metric_crs = CRS.from_epsg(epsg)

            forward = Transformer.from_crs(
                "EPSG:4326",
                metric_crs,
                always_xy=True
            )

            inverse = Transformer.from_crs(
                metric_crs,
                "EPSG:4326",
                always_xy=True
            )

            self._transformer_cache[epsg] = (
                forward,
                inverse
            )

        return self._transformer_cache[epsg]

    # =============================================
    # 5. TOPOLOGY SIGNATURE
    # =============================================

    def _topology_signature(self, geometry):

        parts = self._polygon_parts(geometry)

        return (
            len(parts),
            sum(
                len(part.interiors)
                for part in parts
            )
        )

    # =============================================
    # 6. GEOMETRY SIMPLIFICATION
    # =============================================

    def simplify_geometry(self, geometry):
        if self.tolerance_m == 0:
            return geometry

        forward, inverse = self._get_transformers(geometry)
        projected = transform(
            geometry, forward.transform, interleaved=False
        )
        simplified = projected.simplify(
            self.tolerance_m, preserve_topology=True
        )
        candidate = transform(
            simplified, inverse.transform, interleaved=False
        )

        if (
            candidate.is_empty
            or not candidate.is_valid
            or candidate.geom_type not in ("Polygon", "MultiPolygon")
        ):
            return geometry

        if self._topology_signature(candidate) != self._topology_signature(geometry):
            return geometry

        if get_num_coordinates(candidate) >= get_num_coordinates(geometry):
            return geometry

        return candidate

    # =============================================
    # 7. COMPLETE REGION PROCESSING
    # =============================================

    def process_region(self, region):

        original = region.geometry

        repaired = not original.is_valid

        valid_geometry = self.repair_geometry(
            original
        )

        original_area = float(region.area_m2)

        if (
            not math.isfinite(original_area)
            or original_area <= 0
        ):

            raise GeospatialError(
                "Invalid original region area."
            )

        repaired_area = self.calculate_area(
            valid_geometry
        )

        repair_change = (
            abs(repaired_area - original_area)
            / original_area
            * 100
        )

        if (
            not math.isfinite(repaired_area)
            or repaired_area <= 0
            or repair_change
            > self.max_area_change_percentage
        ):

            raise GeospatialError(
                "Geometry repair exceeded area budget."
            )

        candidate = self.simplify_geometry(
            valid_geometry
        )

        candidate_area = self.calculate_area(
            candidate
        )

        area_change = (
            abs(candidate_area - original_area)
            / original_area
            * 100
        )

        # Retain original valid geometry if
        # simplification changes too much area.

        if (
            not math.isfinite(candidate_area)
            or candidate_area <= 0
            or area_change
            > self.max_area_change_percentage
        ):

            candidate = valid_geometry

            candidate_area = repaired_area

        simplified = (
            get_num_coordinates(candidate)
            < get_num_coordinates(valid_geometry)
        )

        return ProcessedRegion(
            region_id=region.region_id,
            original_geometry=original,
            geometry=candidate,

            original_area_m2=original_area,
            area_m2=candidate_area,

            original_vertices=int(
                get_num_coordinates(original)
            ),

            processed_vertices=int(
                get_num_coordinates(candidate)
            ),

            repaired=repaired,
            simplified=bool(simplified)
        )

    # =============================================
    # 8. PROCESS ALL WATER REGIONS
    # =============================================

    def process(
        self,
        vector_result: VectorizationResult
    ) -> GeometryProcessingResult:

        if not isinstance(
            vector_result,
            VectorizationResult
        ):

            raise GeospatialError(
                "Expected VectorizationResult."
            )

        if vector_result.crs != "EPSG:4326":

            raise GeospatialError(
                "Expected EPSG:4326 geometries."
            )

        regions = []

        for region in vector_result.regions:

            try:

                processed = self.process_region(
                    region
                )

                regions.append(processed)

            except GeospatialError:
                raise

            except Exception as exc:

                raise GeospatialError(
                    f"Geometry processing failed "
                    f"for region {region.region_id}."
                ) from exc

        return GeometryProcessingResult(
            regions=regions,
            tolerance_m=self.tolerance_m,
            max_area_change_percentage=(
                self.max_area_change_percentage
            )
        )

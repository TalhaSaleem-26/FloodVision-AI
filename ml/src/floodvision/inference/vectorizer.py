
from dataclasses import dataclass
import math
from .geometry_utils import (
    repair_polygonal_geometry
)
import numpy as np

from rasterio.features import shapes

from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.geometry.base import BaseGeometry

from pyproj import Geod

from .schemas import RasterMetadata
from .exceptions import GeospatialError
from .area_analyzer import AreaAnalyzer


@dataclass(frozen=True)
class WaterRegion:

    region_id: int
    geometry: BaseGeometry
    area_m2: float

    @property
    def area_km2(self):

        return self.area_m2 / 1_000_000

    @property
    def area_hectares(self):

        return self.area_m2 / 10_000

    def to_dict(self):

        return {
            "region_id": self.region_id,
            "geometry_type": self.geometry.geom_type,
            "area_m2": self.area_m2,
            "area_km2": self.area_km2,
            "area_hectares": self.area_hectares
        }


@dataclass
class VectorizationResult:

    regions: list[WaterRegion]

    crs: str
    connectivity: int

    @property
    def region_count(self):

        return len(self.regions)

    @property
    def total_area_m2(self):

        return math.fsum(
            region.area_m2
            for region in self.regions
        )

    @property
    def total_area_km2(self):

        return self.total_area_m2 / 1_000_000

    @property
    def largest_region(self):

        return max(
            self.regions,
            key=lambda region: region.area_m2,
            default=None
        )

    def to_summary(self):

        largest = self.largest_region

        return {
            "crs": self.crs,
            "connectivity": self.connectivity,
            "region_count": self.region_count,
            "total_area_m2": self.total_area_m2,
            "total_area_km2": self.total_area_km2,
            "largest_region": (
                largest.to_dict()
                if largest is not None
                else None
            )
        }


class WaterVectorizer:
    """
    Extract geographical polygons from
    binary water segmentation masks.

    Currently supports north-up EPSG:4326
    rasters.
    """

    def __init__(
        self,
        connectivity: int = 4,
        max_pixels: int = 4_000_000,
        max_regions: int = 100_000
    ):

        if connectivity not in (4, 8):

            raise GeospatialError(
                "Connectivity must be 4 or 8."
            )

        if (
            type(max_pixels) is not int
            or max_pixels <= 0
        ):

            raise GeospatialError(
                "Invalid maximum pixel limit."
            )

        if (
            type(max_regions) is not int
            or max_regions <= 0
        ):

            raise GeospatialError(
                "Invalid maximum region limit."
            )

        self.connectivity = connectivity
        self.max_pixels = max_pixels
        self.max_regions = max_regions

        self.geod = Geod(ellps="WGS84")

    # =============================================
    # 1. INPUT VALIDATION
    # =============================================

    def _validate_input(
        self,
        mask,
        metadata
    ):

        # Reuse existing CRS and geometry validation
        AreaAnalyzer(metadata)

        if not isinstance(mask, np.ndarray):

            raise GeospatialError(
                "Expected NumPy mask."
            )

        if mask.dtype != np.int8:

            raise GeospatialError(
                "Mask must have int8 dtype."
            )

        if mask.shape != (
            metadata.height,
            metadata.width
        ):

            raise GeospatialError(
                "Mask dimensions do not match raster."
            )

        if mask.size > self.max_pixels:

            raise GeospatialError(
                "Raster exceeds vectorization "
                "pixel limit."
            )

        for row in mask:

            if not np.isin(
                row,
                [-1, 0, 1]
            ).all():

                raise GeospatialError(
                    "Invalid segmentation labels."
                )

    # =============================================
    # 2. GEODESIC GEOMETRY AREA
    # =============================================

    def _calculate_area(
        self,
        geometry
    ):

        if geometry.geom_type == "Polygon":

            geometry = orient(
                geometry,
                sign=1.0
            )

            area, _ = (
                self.geod.geometry_area_perimeter(
                    geometry
                )
            )

            return abs(float(area))

        if geometry.geom_type == "MultiPolygon":

            return math.fsum(
                self._calculate_area(part)
                for part in geometry.geoms
            )

        raise GeospatialError(
            "Unsupported geometry type."
        )

    # =============================================
    # 3. WATER POLYGON EXTRACTION
    # =============================================

    def vectorize(
        self,
        mask: np.ndarray,
        metadata: RasterMetadata
    ) -> VectorizationResult:

        self._validate_input(
            mask,
            metadata
        )

        # Only water pixels participate
        water_mask = mask == 1

        if not water_mask.any():

            return VectorizationResult(
                regions=[],
                crs="EPSG:4326",
                connectivity=self.connectivity
            )

        source = water_mask.astype(
            np.uint8
        )

        regions = []

        try:

            polygon_iterator = shapes(
                source,
                mask=water_mask,
                connectivity=self.connectivity,
                transform=metadata.transform
            )

            for geometry_mapping, value in polygon_iterator:

                if value != 1:
                    continue

                if len(regions) >= self.max_regions:

                    raise GeospatialError(
                        "Maximum water region "
                        "limit exceeded."
                    )

                geometry = shape(
                    geometry_mapping
                )

                if geometry.is_empty:

                    raise GeospatialError(
                        "Empty water geometry."
                    )

                if geometry.geom_type not in (
                    "Polygon",
                    "MultiPolygon"
                ):

                    raise GeospatialError(
                        "Unexpected geometry type."
                    )

                if not geometry.is_valid:

                    geometry = repair_polygonal_geometry(
                         geometry)

                area = self._calculate_area(
                    geometry
                )

                if (
                    not math.isfinite(area)
                    or area <= 0
                ):

                    raise GeospatialError(
                        "Invalid water region area."
                    )

                regions.append(
                    WaterRegion(
                        region_id=len(regions) + 1,
                        geometry=geometry,
                        area_m2=area
                    )
                )

        except GeospatialError:
            raise

        except Exception as exc:

            raise GeospatialError(
                "Water polygon extraction failed."
            ) from exc

        return VectorizationResult(
            regions=regions,
            crs="EPSG:4326",
            connectivity=self.connectivity
        )

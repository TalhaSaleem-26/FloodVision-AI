
from dataclasses import dataclass
import math

import numpy as np

from pyproj import CRS, Geod

from .schemas import RasterMetadata
from .exceptions import GeospatialError


@dataclass(frozen=True)
class AreaStatistics:

    total_area_m2: float
    valid_area_m2: float
    invalid_area_m2: float
    water_area_m2: float
    non_water_area_m2: float

    method: str

    @property
    def water_area_km2(self):

        return self.water_area_m2 / 1_000_000

    @property
    def water_area_hectares(self):

        return self.water_area_m2 / 10_000

    @property
    def water_area_percentage(self):

        if self.valid_area_m2 == 0:
            return None

        return (
            self.water_area_m2
            / self.valid_area_m2
            * 100
        )

    def to_dict(self):

        return {
            "method": self.method,

            "total_area_m2": self.total_area_m2,
            "valid_area_m2": self.valid_area_m2,
            "invalid_area_m2": self.invalid_area_m2,

            "water_area_m2": self.water_area_m2,

            "non_water_area_m2":
                self.non_water_area_m2,

            "water_area_km2":
                self.water_area_km2,

            "water_area_hectares":
                self.water_area_hectares,

            "water_area_percentage":
                self.water_area_percentage
        }


class AreaAnalyzer:
    """
    Ellipsoidal area estimation for north-up
    EPSG:4326 rasters.

    Uses row-wise processing to support NumPy
    arrays and disk-backed memmaps.
    """

    def __init__(self, metadata: RasterMetadata):

        if not isinstance(metadata, RasterMetadata):

            raise GeospatialError(
                "Expected RasterMetadata."
            )

        if metadata.crs is None:

            raise GeospatialError(
                "Raster CRS is required."
            )

        self.metadata = metadata
        self.transform = metadata.transform

        try:

            self.crs = CRS.from_user_input(
                metadata.crs
            )

        except Exception as exc:

            raise GeospatialError(
                "Invalid CRS."
            ) from exc

        if self.crs.to_epsg() != 4326:

            raise GeospatialError(
                "Area analyzer currently requires "
                "EPSG:4326."
            )

        self._validate_geometry()

        self.geod = Geod(
            ellps="WGS84"
        )

        self._row_areas = None

    # =============================================
    # 1. GEOMETRY VALIDATION
    # =============================================

    def _validate_geometry(self):

        t = self.transform

        if (
            self.metadata.width <= 0
            or self.metadata.height <= 0
        ):

            raise GeospatialError(
                "Invalid raster dimensions."
            )

        coefficients = [
            t.a, t.b, t.c,
            t.d, t.e, t.f
        ]

        if not all(
            math.isfinite(v)
            for v in coefficients
        ):

            raise GeospatialError(
                "Invalid affine coefficients."
            )

        if t.b != 0 or t.d != 0:

            raise GeospatialError(
                "Rotated or sheared rasters "
                "are not supported."
            )

        if t.a <= 0 or t.e >= 0:

            raise GeospatialError(
                "Expected north-up raster."
            )

        left, top = t * (0, 0)

        right, bottom = t * (
            self.metadata.width,
            self.metadata.height
        )

        if not (
            -180 <= left < right <= 180
            and -90 <= bottom < top <= 90
        ):

            raise GeospatialError(
                "Invalid geographical bounds or "
                "antimeridian crossing."
            )

        if t.a >= 180 or abs(t.e) >= 180:

            raise GeospatialError(
                "Unsupported pixel dimensions."
            )

    # =============================================
    # 2. SINGLE PIXEL GEODESIC AREA
    # =============================================

    def pixel_area_m2(
        self,
        row: int,
        col: int = 0
    ) -> float:

        if (
            type(row) is not int
            or type(col) is not int
            or not 0 <= row < self.metadata.height
            or not 0 <= col < self.metadata.width
        ):

            raise GeospatialError(
                "Pixel outside raster."
            )

        corners = [
            self.transform * (col, row),

            self.transform * (col + 1, row),

            self.transform * (
                col + 1, row + 1
            ),

            self.transform * (col, row + 1)
        ]

        longitudes = [
            point[0] for point in corners
        ]

        latitudes = [
            point[1] for point in corners
        ]

        area, _ = (
            self.geod.polygon_area_perimeter(
                longitudes,
                latitudes
            )
        )

        area = abs(float(area))

        if (
            not math.isfinite(area)
            or area <= 0
        ):

            raise GeospatialError(
                "Invalid geodesic pixel area."
            )

        return area

    # =============================================
    # 3. ROW AREA CACHE
    # =============================================

    def get_row_areas(self):

        if self._row_areas is None:

            self._row_areas = np.array(
                [
                    self.pixel_area_m2(row)
                    for row in range(
                        self.metadata.height
                    )
                ],
                dtype=np.float64
            )

            self._row_areas.setflags(
                write=False
            )

        return self._row_areas

    # =============================================
    # 4. MASK VALIDATION
    # =============================================

    def _validate_mask(self, mask):

        if not isinstance(mask, np.ndarray):

            raise GeospatialError(
                "Expected NumPy mask."
            )

        if mask.dtype != np.int8:

            raise GeospatialError(
                "Mask must have int8 dtype."
            )

        expected_shape = (
            self.metadata.height,
            self.metadata.width
        )

        if mask.shape != expected_shape:

            raise GeospatialError(
                f"Expected mask shape "
                f"{expected_shape}."
            )

    # =============================================
    # 5. WATER AREA ANALYTICS
    # =============================================

    def analyze(
        self,
        mask: np.ndarray
    ) -> AreaStatistics:

        self._validate_mask(mask)

        row_areas = self.get_row_areas()

        water_area = 0.0
        non_water_area = 0.0
        invalid_area = 0.0

        for row_index in range(
            self.metadata.height
        ):

            row = mask[row_index]

            water_count = int(
                np.count_nonzero(row == 1)
            )

            non_water_count = int(
                np.count_nonzero(row == 0)
            )

            invalid_count = int(
                np.count_nonzero(row == -1)
            )

            if (
                water_count
                + non_water_count
                + invalid_count
                != self.metadata.width
            ):

                raise GeospatialError(
                    "Invalid segmentation labels."
                )

            pixel_area = row_areas[row_index]

            water_area += (
                water_count * pixel_area
            )

            non_water_area += (
                non_water_count * pixel_area
            )

            invalid_area += (
                invalid_count * pixel_area
            )

        valid_area = (
            water_area + non_water_area
        )

        total_area = (
            valid_area + invalid_area
        )

        return AreaStatistics(

            total_area_m2=float(total_area),

            valid_area_m2=float(valid_area),

            invalid_area_m2=float(invalid_area),

            water_area_m2=float(water_area),

            non_water_area_m2=float(
                non_water_area
            ),

            method="wgs84_geodesic_row_weighted"
        )

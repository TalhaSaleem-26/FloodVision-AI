
from dataclasses import dataclass
import math

from pyproj import CRS, Transformer

from .schemas import RasterMetadata
from .exceptions import GeospatialError


@dataclass(frozen=True)
class GeospatialInfo:

    width: int
    height: int

    crs: str
    crs_type: str
    coordinate_unit: str

    bounds: tuple[float, float, float, float]

    resolution_x: float
    resolution_y: float

    center_x: float
    center_y: float

    center_longitude: float
    center_latitude: float

    affine_transform: tuple

    def to_dict(self) -> dict:

        return {
            "width": self.width,
            "height": self.height,

            "crs": self.crs,
            "crs_type": self.crs_type,
            "coordinate_unit": self.coordinate_unit,

            "bounds": {
                "left": self.bounds[0],
                "bottom": self.bounds[1],
                "right": self.bounds[2],
                "top": self.bounds[3]
            },

            "resolution": {
                "x": self.resolution_x,
                "y": self.resolution_y,
                "unit": self.coordinate_unit
            },

            "center": {
                "x": self.center_x,
                "y": self.center_y,
                "longitude": self.center_longitude,
                "latitude": self.center_latitude
            },

            "affine_transform": list(
                self.affine_transform
            )
        }


class GeospatialAnalyzer:

    def __init__(self, metadata: RasterMetadata):

        if not isinstance(metadata, RasterMetadata):

            raise GeospatialError(
                "Expected RasterMetadata object."
            )

        if metadata.crs is None:

            raise GeospatialError(
                "Raster CRS is required."
            )

        if metadata.width <= 0 or metadata.height <= 0:

            raise GeospatialError(
                "Invalid raster dimensions."
            )

        self.metadata = metadata
        self.transform = metadata.transform

        coefficients = (
            self.transform.a,
            self.transform.b,
            self.transform.c,
            self.transform.d,
            self.transform.e,
            self.transform.f
        )

        if not all(
            math.isfinite(v) for v in coefficients
        ):

            raise GeospatialError(
                "Non-finite affine transformation."
            )

        determinant = (
            self.transform.a * self.transform.e
            - self.transform.b * self.transform.d
        )

        if determinant == 0:

            raise GeospatialError(
                "Non-invertible affine transformation."
            )

        try:

            self.crs = CRS.from_user_input(
                metadata.crs
            )

            if not (
                self.crs.is_geographic
                or self.crs.is_projected
            ):

                raise GeospatialError(
                    "Unsupported CRS type."
                )

            self.to_wgs84 = Transformer.from_crs(
                self.crs,
                CRS.from_epsg(4326),
                always_xy=True
            )

        except GeospatialError:
            raise

        except Exception as exc:

            raise GeospatialError(
                "Unable to initialize CRS."
            ) from exc

    # =============================================
    # PIXEL TO NATIVE CRS COORDINATES
    # =============================================

    def pixel_to_coordinates(
        self,
        row: int,
        col: int
    ) -> tuple[float, float]:

        if (
            type(row) is not int
            or type(col) is not int
            or not 0 <= row < self.metadata.height
            or not 0 <= col < self.metadata.width
        ):

            raise GeospatialError(
                "Pixel coordinates outside raster."
            )

        x, y = self.transform * (
            col + 0.5,
            row + 0.5
        )

        return float(x), float(y)

    # =============================================
    # PIXEL TO LONGITUDE AND LATITUDE
    # =============================================

    def pixel_to_lonlat(
        self,
        row: int,
        col: int
    ) -> tuple[float, float]:

        x, y = self.pixel_to_coordinates(
            row,
            col
        )

        return self._transform_to_wgs84(x, y)

    def _transform_to_wgs84(self, x, y):

        try:

            lon, lat = self.to_wgs84.transform(
                x,
                y,
                errcheck=True
            )

        except Exception as exc:

            raise GeospatialError(
                "Coordinate transformation failed."
            ) from exc

        if (
            not math.isfinite(lon)
            or not math.isfinite(lat)
            or not -90 <= lat <= 90
        ):

            raise GeospatialError(
                "Invalid geographical coordinates."
            )

        return float(lon), float(lat)

    # =============================================
    # RASTER BOUNDS
    # =============================================

    def get_bounds(self):

        width = self.metadata.width
        height = self.metadata.height

        corners = [
            self.transform * (0, 0),
            self.transform * (width, 0),
            self.transform * (0, height),
            self.transform * (width, height)
        ]

        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]

        return (
            float(min(xs)),
            float(min(ys)),
            float(max(xs)),
            float(max(ys))
        )

    # =============================================
    # PIXEL RESOLUTION
    # =============================================

    def get_resolution(self):

        resolution_x = math.hypot(
            self.transform.a,
            self.transform.d
        )

        resolution_y = math.hypot(
            self.transform.b,
            self.transform.e
        )

        return resolution_x, resolution_y

    # =============================================
    # COMPLETE GEOSPATIAL SUMMARY
    # =============================================

    def analyze(self) -> GeospatialInfo:

        resolution_x, resolution_y = (
            self.get_resolution()
        )

        center_x, center_y = self.transform * (
            self.metadata.width / 2,
            self.metadata.height / 2
        )

        lon, lat = self._transform_to_wgs84(
            center_x,
            center_y
        )

        axis_units = {
            axis.unit_name
            for axis in self.crs.axis_info
        }

        coordinate_unit = (
            next(iter(axis_units))
            if len(axis_units) == 1
            else "mixed_or_unknown"
        )

        return GeospatialInfo(
            width=self.metadata.width,
            height=self.metadata.height,

            crs=self.crs.to_string(),

            crs_type=(
                "geographic"
                if self.crs.is_geographic
                else "projected"
            ),

            coordinate_unit=coordinate_unit,

            bounds=self.get_bounds(),

            resolution_x=resolution_x,
            resolution_y=resolution_y,

            center_x=float(center_x),
            center_y=float(center_y),

            center_longitude=lon,
            center_latitude=lat,

            affine_transform=(
                self.transform.a,
                self.transform.b,
                self.transform.c,
                self.transform.d,
                self.transform.e,
                self.transform.f
            )
        )

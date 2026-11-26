
from shapely import make_valid
from shapely.ops import unary_union

from .exceptions import GeospatialError


def extract_polygon_parts(geometry):

    if geometry.is_empty:
        return []

    if geometry.geom_type == "Polygon":
        return [geometry]

    if geometry.geom_type in (
        "MultiPolygon",
        "GeometryCollection"
    ):

        parts = []

        for part in geometry.geoms:

            parts.extend(
                extract_polygon_parts(part)
            )

        return parts

    return []


def repair_polygonal_geometry(geometry):

    if geometry.is_empty:
        raise GeospatialError(
            "Cannot repair empty geometry."
        )

    if geometry.geom_type not in (
        "Polygon",
        "MultiPolygon"
    ):
        raise GeospatialError(
            "Expected polygonal geometry."
        )

    if geometry.is_valid:
        return geometry

    repaired = make_valid(geometry)

    parts = extract_polygon_parts(repaired)

    if not parts:
        raise GeospatialError(
            "No polygonal components remain."
        )

    repaired = unary_union(parts)

    if (
        repaired.is_empty
        or not repaired.is_valid
        or repaired.geom_type not in (
            "Polygon",
            "MultiPolygon"
        )
    ):
        raise GeospatialError(
            "Polygon repair failed."
        )

    return repaired

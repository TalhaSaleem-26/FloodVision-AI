
import math

import numpy as np

from rasterio.features import rasterize

from shapely.ops import unary_union
from shapely import transform

from pyproj import Transformer

from .exceptions import GeospatialError


class GeometryFidelityValidator:

    def __init__(
        self,
        min_iou=1.0,
        max_invalid_encroachment=0,
        max_overlap_m2=0.01
    ):

        self.min_iou = min_iou

        self.max_invalid_encroachment = (
            max_invalid_encroachment
        )

        self.max_overlap_m2 = max_overlap_m2

    def calculate_overlap(self, geometries):

        if not geometries:
            return 0.0

        bounds = unary_union(geometries).bounds

        longitude = (bounds[0] + bounds[2]) / 2
        latitude = (bounds[1] + bounds[3]) / 2

        if bounds[2] - bounds[0] > 6:
            raise GeospatialError(
                "Geometry extent exceeds local "
                "projection validation scope."
            )

        if not -80 <= latitude <= 84:
            raise GeospatialError(
                "Geometry outside supported UTM latitude."
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

        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{epsg}",
            always_xy=True
        )

        projected = [
            transform(
                geometry,
                transformer.transform,
                interleaved=False
            )
            for geometry in geometries
        ]

        summed_area = math.fsum(
            geometry.area
            for geometry in projected
        )

        union_area = unary_union(projected).area

        return max(
            0.0,
            summed_area - union_area
        )

    def validate(
        self,
        vector_result,
        mask,
        metadata
    ):

        if mask.shape != (
            metadata.height,
            metadata.width
        ):
            raise GeospatialError(
                "Mask dimensions do not match metadata."
            )

        if metadata.crs is None or metadata.crs.to_epsg() != 4326:
            raise GeospatialError(
                "Geometry fidelity requires WGS84."
            )

        geometries = [
            region.geometry
            for region in vector_result.regions
        ]

        if any(
            geometry.is_empty or not geometry.is_valid
            for geometry in geometries
        ):
            raise GeospatialError(
                "Invalid geometry in vector result."
            )

        if geometries:

            reconstructed = rasterize(
                [
                    (geometry, 1)
                    for geometry in geometries
                ],
                out_shape=mask.shape,
                transform=metadata.transform,
                fill=0,
                all_touched=False,
                dtype="uint8"
            ).astype(bool)

        else:

            reconstructed = np.zeros(
                mask.shape,
                dtype=bool
            )

        original = mask == 1

        intersection = int(
            np.count_nonzero(
                original & reconstructed
            )
        )

        union = int(
            np.count_nonzero(
                original | reconstructed
            )
        )

        iou = (
            intersection / union
            if union > 0
            else 1.0
        )

        changed_pixels = int(
            np.count_nonzero(
                original != reconstructed
            )
        )

        invalid_encroachment = int(
            np.count_nonzero(
                reconstructed & (mask == -1)
            )
        )

        overlap_m2 = self.calculate_overlap(
            geometries
        )

        passed = (
            iou >= self.min_iou
            and invalid_encroachment
                <= self.max_invalid_encroachment
            and overlap_m2 <= self.max_overlap_m2
        )

        return {
            "passed": bool(passed),
            "geometry_iou": float(iou),
            "changed_pixels": changed_pixels,
            "invalid_pixel_encroachment":
                invalid_encroachment,
            "overlap_area_m2": float(overlap_m2),
            "original_water_pixels": int(
                original.sum()
            ),
            "reconstructed_water_pixels": int(
                reconstructed.sum()
            )
        }

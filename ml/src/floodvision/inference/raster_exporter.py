
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio

from rasterio.windows import Window

from .schemas import RasterMetadata
from .exceptions import GeospatialError


class RasterExporter:
    """
    Export segmentation masks and probability maps
    as georeferenced GeoTIFF files.

    Supports NumPy arrays and disk-backed memmaps.
    """

    def __init__(self, block_size: int = 256):

        if (
            type(block_size) is not int
            or block_size <= 0
            or block_size % 16 != 0
        ):

            raise ValueError(
                "Block size must be a positive "
                "multiple of 16."
            )

        self.block_size = block_size

    # =============================================
    # 1. INPUT VALIDATION
    # =============================================

    def _validate_input(
        self,
        array,
        metadata
    ):

        if not isinstance(
            metadata,
            RasterMetadata
        ):

            raise GeospatialError(
                "Expected RasterMetadata."
            )

        if metadata.crs is None:

            raise GeospatialError(
                "Raster CRS is required."
            )

        if not isinstance(array, np.ndarray):

            raise GeospatialError(
                "Expected NumPy array."
            )

        expected_shape = (
            metadata.height,
            metadata.width
        )

        if array.shape != expected_shape:

            raise GeospatialError(
                f"Expected shape {expected_shape}, "
                f"received {array.shape}."
            )

    # =============================================
    # 2. GEOTIFF PROFILE
    # =============================================

    def _build_profile(
        self,
        metadata,
        dtype,
        nodata
    ):

        return {
            "driver": "GTiff",

            "height": metadata.height,
            "width": metadata.width,

            "count": 1,
            "dtype": dtype,

            "crs": metadata.crs,
            "transform": metadata.transform,

            "nodata": nodata,

            "compress": "deflate",

            "tiled": True,

            "blockxsize": self.block_size,
            "blockysize": self.block_size,

            "BIGTIFF": "IF_SAFER"
        }

    # =============================================
    # 3. OUTPUT PATH VALIDATION
    # =============================================

    def _prepare_path(
        self,
        output_path,
        metadata,
        overwrite
    ):

        path = Path(output_path).resolve()

        if path.suffix.lower() not in {
            ".tif", ".tiff"
        }:

            raise GeospatialError(
                "Output must be a GeoTIFF file."
            )

        if path == Path(metadata.path).resolve():

            raise GeospatialError(
                "Cannot overwrite source raster."
            )

        if path.exists() and not overwrite:

            raise GeospatialError(
                f"Output already exists: {path}"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        return path

    # =============================================
    # 4. BLOCK VALIDATION
    # =============================================

    def _validate_block(
        self,
        block,
        output_type
    ):

        if output_type == "mask":

            if not np.isin(
                block,
                [-1, 0, 1]
            ).all():

                raise GeospatialError(
                    "Invalid segmentation labels."
                )

        elif output_type == "probability":

            if np.isinf(block).any():

                raise GeospatialError(
                    "Infinite probabilities detected."
                )

            valid = ~np.isnan(block)

            values = block[valid]

            if (
                np.any(values < 0)
                or np.any(values > 1)
            ):

                raise GeospatialError(
                    "Probabilities must be "
                    "between 0 and 1."
                )

    # =============================================
    # 5. WINDOW-BASED RASTER WRITING
    # =============================================

    def _write_raster(
        self,
        array,
        metadata,
        output_path,
        dtype,
        nodata,
        output_type,
        model_version,
        threshold,
        overwrite
    ):

        self._validate_input(
            array,
            metadata
        )

        path = self._prepare_path(
            output_path,
            metadata,
            overwrite
        )

        profile = self._build_profile(
            metadata,
            dtype,
            nodata
        )

        temporary_path = path.with_name(
            f".{path.stem}.{uuid4().hex}.tmp.tif"
        )

        try:

            with rasterio.open(
                temporary_path,
                "w",
                **profile
            ) as dst:

                for row in range(
                    0,
                    metadata.height,
                    self.block_size
                ):

                    for col in range(
                        0,
                        metadata.width,
                        self.block_size
                    ):

                        height = min(
                            self.block_size,
                            metadata.height - row
                        )

                        width = min(
                            self.block_size,
                            metadata.width - col
                        )

                        block = array[
                            row:row + height,
                            col:col + width
                        ]

                        self._validate_block(
                            block,
                            output_type
                        )

                        window = Window(
                            col_off=col,
                            row_off=row,
                            width=width,
                            height=height
                        )

                        dst.write(
                            block.astype(
                                dtype,
                                copy=False
                            ),
                            indexes=1,
                            window=window
                        )

                description = (
                    "Water segmentation"
                    if output_type == "mask"
                    else "Water probability"
                )

                dst.set_band_description(
                    1,
                    description
                )

                dst.update_tags(
                    PROJECT="FloodVision AI",
                    MODEL_VERSION=str(model_version),
                    OUTPUT_TYPE=output_type,
                    THRESHOLD=str(threshold),
                    TASK="binary_water_segmentation",
                    SOURCE_FILE=metadata.path.name
                )

                if output_type == "mask":

                    dst.update_tags(
                        LABEL_INVALID="-1",
                        LABEL_NON_WATER="0",
                        LABEL_WATER="1"
                    )

            if path.exists() and not overwrite:

                raise GeospatialError(
                    "Output file already exists."
                )

            temporary_path.replace(path)

        except Exception as exc:

            temporary_path.unlink(
                missing_ok=True
            )

            if isinstance(exc, GeospatialError):
                raise

            raise GeospatialError(
                f"GeoTIFF export failed: {path}"
            ) from exc

        return path

    # =============================================
    # 6. SEGMENTATION EXPORT
    # =============================================

    def export_mask(
        self,
        mask: np.ndarray,
        metadata: RasterMetadata,
        output_path: str | Path,
        model_version: str = "1.0.0",
        threshold: float = 0.5,
        overwrite: bool = False
    ) -> Path:

        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.int8
        ):

            raise GeospatialError(
                "Mask must have int8 dtype."
            )

        return self._write_raster(
            array=mask,
            metadata=metadata,
            output_path=output_path,
            dtype="int8",
            nodata=-1,
            output_type="mask",
            model_version=model_version,
            threshold=threshold,
            overwrite=overwrite
        )

    # =============================================
    # 7. PROBABILITY EXPORT
    # =============================================

    def export_probabilities(
        self,
        probabilities: np.ndarray,
        metadata: RasterMetadata,
        output_path: str | Path,
        model_version: str = "1.0.0",
        threshold: float = 0.5,
        overwrite: bool = False
    ) -> Path:

        if (
            not isinstance(probabilities, np.ndarray)
            or probabilities.dtype != np.float32
        ):

            raise GeospatialError(
                "Probabilities must have "
                "float32 dtype."
            )

        return self._write_raster(
            array=probabilities,
            metadata=metadata,
            output_path=output_path,
            dtype="float32",
            nodata=np.nan,
            output_type="probability",
            model_version=model_version,
            threshold=threshold,
            overwrite=overwrite
        )

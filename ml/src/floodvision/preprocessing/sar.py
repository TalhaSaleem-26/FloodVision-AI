from pathlib import Path

import numpy as np
import rasterio


def read_sar_image(path: str | Path):
    """
    Read a Sentinel-1 SAR image and identify
    valid input pixels.
    """

    with rasterio.open(path) as src:

        image = src.read().astype(np.float32)

        valid_mask = (
            (src.read_masks() > 0).all(axis=0)
            & np.isfinite(image).all(axis=0)
        )

        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width
        }

    if image.shape[0] != 2:
        raise ValueError(
            f"Expected 2 SAR channels, got {image.shape[0]}"
        )

    return image, valid_mask, metadata


def read_target_mask(path: str | Path):
    """
    Read a segmentation mask.
    """

    with rasterio.open(path) as src:
        mask = src.read(1)

    if not np.isin(mask, [-1, 0, 1]).all():
        raise ValueError("Unexpected mask labels")

    return mask.astype(np.float32)
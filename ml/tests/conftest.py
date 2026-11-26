from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import yaml

from rasterio.transform import from_origin


ML_DIR = Path(__file__).resolve().parents[1]


# ==========================================
# Project paths
# ==========================================

@pytest.fixture(scope="session")
def ml_dir():
    return ML_DIR


@pytest.fixture(scope="session")
def production_bundle_dir(ml_dir):

    path = ml_dir / "artifacts" / "production"

    required_files = (
        "floodvision_mobilenetv3_v1.pt",
        "preprocessing.yaml",
        "metadata.json",
        "MODEL_CARD.md",
        "manifest.json"
    )

    if not all(
        (path / name).is_file()
        for name in required_files
    ):
        pytest.skip("Production bundle unavailable.")

    return path


# ==========================================
# Random generator
# ==========================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ==========================================
# Synthetic segmentation mask
# ==========================================

@pytest.fixture
def synthetic_mask():

    mask = np.zeros(
        (512, 512),
        dtype=np.int8
    )

    # Central water region
    mask[128:384, 128:384] = 1

    # Invalid upper-left region
    mask[:64, :64] = -1

    return mask


@pytest.fixture
def synthetic_valid_mask(synthetic_mask):

    return synthetic_mask != -1


# ==========================================
# Synthetic SAR image
# ==========================================

@pytest.fixture
def synthetic_sar(rng, synthetic_mask):

    image = np.empty(
        (2, 512, 512),
        dtype=np.float32
    )

    water = synthetic_mask == 1
    invalid = synthetic_mask == -1

    # Non-water VV and VH
    image[0] = rng.normal(
        -10, 2, (512, 512)
    )

    image[1] = rng.normal(
        -17, 2, (512, 512)
    )

    # Simulated water backscatter
    image[0, water] = rng.normal(
        -20, 1, water.sum()
    )

    image[1, water] = rng.normal(
        -27, 1, water.sum()
    )

    # Invalid pixels
    image[:, invalid] = np.nan

    return image


# ==========================================
# Model tensors
# ==========================================

@pytest.fixture
def model_input():

    generator = torch.Generator()
    generator.manual_seed(42)

    return torch.randn(
        (1, 2, 64, 64),
        generator=generator,
        dtype=torch.float32
    )


@pytest.fixture
def target_tensor(synthetic_mask):

    return (
        torch.from_numpy(
            synthetic_mask.copy()
        )
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )


# ==========================================
# Controlled prediction logits
# ==========================================

@pytest.fixture
def perfect_logits(synthetic_mask):

    logits = np.where(
        synthetic_mask == 1,
        10.0,
        -10.0
    ).astype(np.float32)

    return (
        torch.from_numpy(logits)
        .unsqueeze(0)
        .unsqueeze(0)
    )


# ==========================================
# Synthetic GeoTIFF factory
# ==========================================

@pytest.fixture
def raster_factory(tmp_path):

    def create_raster(
        image,
        filename="synthetic_sar.tif",
        crs="EPSG:4326",
        band_descriptions=("VV", "VH"),
        nodata=np.nan
    ):

        path = tmp_path / filename

        channels, height, width = image.shape

        transform = from_origin(
            73.0,
            33.6,
            0.0001,
            0.0001
        )

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=channels,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=nodata
        ) as dst:

            dst.write(
                image.astype(np.float32)
            )

            for index, description in enumerate(
                band_descriptions,
                start=1
            ):

                if index <= channels:
                    dst.set_band_description(
                        index,
                        description
                    )

        return path

    return create_raster


@pytest.fixture
def synthetic_raster_path(
    raster_factory,
    synthetic_sar
):

    return raster_factory(synthetic_sar)


@pytest.fixture
def all_invalid_raster_path(raster_factory):

    image = np.full(
        (2, 512, 512),
        np.nan,
        dtype=np.float32
    )

    return raster_factory(
        image,
        filename="all_invalid.tif"
    )


# ==========================================
# Synthetic preprocessing configuration
# ==========================================

@pytest.fixture
def preprocessing_config_path(tmp_path):

    config = {
        "preprocessing": {
            "method": "percentile_zscore",
            "fit_split": "geographical/train",
            "clip_percentiles": [1, 99],
            "sampling_seed": 42,
            "max_pixels_per_image": 5000,
            "epsilon": 1e-8,
            "invalid_fill_value": 0.0,
            "ignore_index": -1,

            "channels": {
                "VV": {
                    "clip_min": -30.0,
                    "clip_max": 0.0,
                    "mean": -10.0,
                    "std": 4.0
                },
                "VH": {
                    "clip_min": -40.0,
                    "clip_max": -5.0,
                    "mean": -17.0,
                    "std": 5.0
                }
            }
        }
    }

    path = tmp_path / "preprocessing.yaml"

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(config, file)

    return path
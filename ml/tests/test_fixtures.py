import numpy as np
import pytest
import rasterio
import torch
import yaml


pytestmark = pytest.mark.unit


def test_synthetic_mask_shape(synthetic_mask):

    assert synthetic_mask.shape == (512, 512)

    assert synthetic_mask.dtype == np.int8


def test_synthetic_mask_labels(synthetic_mask):

    assert set(
        np.unique(synthetic_mask)
    ) == {-1, 0, 1}


def test_synthetic_mask_counts(synthetic_mask):

    assert np.sum(synthetic_mask == 1) == 65536

    assert np.sum(synthetic_mask == -1) == 4096

    assert np.sum(synthetic_mask == 0) == 192512


def test_synthetic_sar_shape(synthetic_sar):

    assert synthetic_sar.shape == (
        2, 512, 512
    )

    assert synthetic_sar.dtype == np.float32


def test_invalid_pixels_preserved(
    synthetic_sar,
    synthetic_valid_mask
):

    assert np.isnan(
        synthetic_sar[:, ~synthetic_valid_mask]
    ).all()

    assert np.isfinite(
        synthetic_sar[:, synthetic_valid_mask]
    ).all()


def test_model_input(model_input):

    assert model_input.shape == (
        1, 2, 64, 64
    )

    assert model_input.dtype == torch.float32

    assert torch.isfinite(model_input).all()


def test_target_tensor(target_tensor):

    assert target_tensor.shape == (
        1, 1, 512, 512
    )

    assert target_tensor.dtype == torch.float32

    assert set(
        target_tensor.unique().tolist()
    ) == {-1.0, 0.0, 1.0}


def test_perfect_logits(
    perfect_logits,
    target_tensor
):

    valid = target_tensor != -1

    predictions = (
        torch.sigmoid(perfect_logits) >= 0.5
    ).float()

    assert torch.equal(
        predictions[valid],
        target_tensor[valid]
    )


def test_synthetic_geotiff(
    synthetic_raster_path,
    synthetic_sar
):

    assert synthetic_raster_path.exists()

    with rasterio.open(
        synthetic_raster_path
    ) as src:

        assert src.count == 2

        assert src.width == 512

        assert src.height == 512

        assert src.crs.to_epsg() == 4326

        assert src.descriptions == ("VV", "VH")

        assert np.isnan(src.nodata)

        np.testing.assert_allclose(
            src.read(),
            synthetic_sar,
            rtol=0,
            atol=0,
            equal_nan=True
        )


def test_all_invalid_raster(
    all_invalid_raster_path
):

    with rasterio.open(
        all_invalid_raster_path
    ) as src:

        assert np.isnan(src.read()).all()

        assert np.all(src.read_masks() == 0)


def test_preprocessing_config(
    preprocessing_config_path
):

    with preprocessing_config_path.open(
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    preprocessing = config["preprocessing"]

    assert preprocessing["method"] == (
        "percentile_zscore"
    )

    assert preprocessing["ignore_index"] == -1

    assert set(
        preprocessing["channels"]
    ) == {"VV", "VH"}


def test_random_seed_reproducibility(rng):

    actual = rng.normal(size=10)

    expected = np.random.default_rng(
        42
    ).normal(size=10)

    np.testing.assert_array_equal(
        actual,
        expected
    )
import numpy as np
import pytest
import rasterio

from floodvision.inference.exceptions import GeospatialError
from floodvision.inference.raster_exporter import RasterExporter
from floodvision.inference.validator import InputValidator


@pytest.fixture
def source_metadata(raster_factory):
    path = raster_factory(np.zeros((2, 17, 19), dtype=np.float32))
    return InputValidator().validate_raster(path, require_tile_size=False)


def test_mask_export_round_trips_across_partial_blocks(source_metadata, tmp_path):
    mask = (np.arange(17 * 19) % 3 - 1).astype(np.int8).reshape(17, 19)
    output = tmp_path / "results" / "mask.tif"

    assert RasterExporter(block_size=16).export_mask(
        mask, source_metadata, output, model_version="test-model", threshold=0.65
    ) == output

    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read(1), mask)
        np.testing.assert_array_equal(dataset.read_masks(1) != 0, mask != -1)
        assert dataset.crs == source_metadata.crs
        assert dataset.transform == source_metadata.transform
        assert dataset.dtypes == ("int8",)
        assert dataset.nodata == -1
        assert dataset.descriptions == ("Water segmentation",)
        tags = dataset.tags()
        assert tags["MODEL_VERSION"] == "test-model"
        assert tags["THRESHOLD"] == "0.65"
        assert tags["SOURCE_FILE"] == source_metadata.path.name
        assert [tags[key] for key in ("LABEL_INVALID", "LABEL_NON_WATER", "LABEL_WATER")] == [
            "-1", "0", "1"
        ]


def test_probability_export_round_trips_nan_and_range_limits(source_metadata, tmp_path):
    probabilities = np.linspace(0, 1, 17 * 19, dtype=np.float32).reshape(17, 19)
    probabilities[4, 7] = np.nan
    output = tmp_path / "probabilities.tiff"

    RasterExporter(block_size=16).export_probabilities(probabilities, source_metadata, output)

    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read(1), probabilities)
        np.testing.assert_array_equal(dataset.read_masks(1) != 0, np.isfinite(probabilities))
        assert dataset.dtypes == ("float32",)
        assert np.isnan(dataset.nodata)
        assert dataset.tags()["OUTPUT_TYPE"] == "probability"


@pytest.mark.parametrize(
    ("invalid", "message"),
    [(np.inf, "Infinite probabilities"), (-0.1, "between 0 and 1"), (1.1, "between 0 and 1")],
)
def test_invalid_final_block_preserves_existing_output(
    source_metadata, tmp_path, invalid, message
):
    probabilities = np.full((17, 19), 0.5, dtype=np.float32)
    output = tmp_path / "probabilities.tif"
    exporter = RasterExporter(block_size=16)
    exporter.export_probabilities(probabilities, source_metadata, output)
    original = output.read_bytes()
    probabilities[-1, -1] = invalid

    with pytest.raises(GeospatialError, match=message):
        exporter.export_probabilities(probabilities, source_metadata, output, overwrite=True)

    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp.tif"))


def test_mask_export_protects_source_even_with_overwrite(source_metadata):
    source = source_metadata.path
    original = source.read_bytes()
    mask = np.zeros((17, 19), dtype=np.int8)

    with pytest.raises(GeospatialError, match="Cannot overwrite source raster"):
        RasterExporter().export_mask(mask, source_metadata, source, overwrite=True)

    assert source.read_bytes() == original


@pytest.mark.parametrize(
    ("mask", "filename", "message"),
    [
        (np.zeros((17, 19)), "mask.tif", "int8 dtype"),
        (np.zeros((19, 17), dtype=np.int8), "mask.tif", "Expected shape"),
        (np.full((17, 19), 2, dtype=np.int8), "mask.tif", "Invalid segmentation labels"),
        (np.zeros((17, 19), dtype=np.int8), "mask.png", "must be a GeoTIFF"),
    ],
)
def test_mask_export_rejects_invalid_payload_or_path(source_metadata, tmp_path, mask, filename, message):
    output = tmp_path / filename
    with pytest.raises(GeospatialError, match=message):
        RasterExporter(block_size=16).export_mask(mask, source_metadata, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp.tif"))

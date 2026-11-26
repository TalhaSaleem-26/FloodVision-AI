import numpy as np
import pytest
import rasterio
from affine import Affine

from floodvision.inference.exceptions import InputValidationError
from floodvision.inference.tile_reader import RasterTileReader
from floodvision.inference.tiling import TileWindow


def test_small_raster_masks_invalid_bands_and_padding(raster_factory):
    image = np.full((2, 4, 7), -15, dtype=np.float32)
    image[0, 0, 0] = -9999
    image[1, 1, 2] = np.inf
    path = raster_factory(image, nodata=-9999)
    reader = RasterTileReader()

    (tile,) = reader.iter_tiles(path)

    assert tile.original_shape == (4, 7)
    assert tile.padding_pixels == 512**2 - 28
    assert tile.valid_pixels == 26
    assert not tile.is_empty
    expected_valid = np.zeros((512, 512), dtype=bool)
    expected_valid[:4, :7] = True
    expected_valid[0, 0] = expected_valid[1, 2] = False
    np.testing.assert_array_equal(tile.sar_input.valid_mask, expected_valid)
    expected_band = np.where(expected_valid, -15, 0)
    np.testing.assert_array_equal(tile.sar_input.image, np.stack([expected_band, expected_band]))


def test_right_edge_tile_preserves_georeferencing_and_pixel_values(raster_factory):
    image = np.zeros((2, 3, 600), dtype=np.float32)
    image[0] = np.arange(600)
    path = raster_factory(image)
    reader = RasterTileReader()
    metadata = reader.get_metadata(path)

    first, right = reader.iter_tiles(path)

    assert reader.count_tiles(path) == 2
    assert first.window.col_off == 0
    assert right.window.col_off == 88
    expected_transform = metadata.transform @ Affine.translation(88, 0)
    assert right.sar_input.metadata.transform == expected_transform
    assert right.sar_input.metadata.crs == metadata.crs
    np.testing.assert_array_equal(right.sar_input.image[:, :3, :], image[:, :, 88:])


def test_all_invalid_raster_yields_empty_tile(all_invalid_raster_path):
    (tile,) = RasterTileReader().iter_tiles(all_invalid_raster_path)

    assert tile.is_empty
    assert tile.valid_pixels == 0
    assert tile.padding_pixels == 0
    assert not tile.sar_input.image.any()


@pytest.mark.parametrize(
    ("window", "message"),
    [
        (TileWindow(0, 0, 0, 2, 2, 256), "tile size mismatch"),
        (TileWindow(0, -1, 0, 2, 2, 512), "exceeds raster boundaries"),
        (TileWindow(0, 0, 0, 2, 5, 512), "exceeds raster boundaries"),
    ],
)
def test_reader_rejects_invalid_windows(raster_factory, window, message):
    path = raster_factory(np.zeros((2, 3, 4), dtype=np.float32))
    reader = RasterTileReader()
    metadata = reader.get_metadata(path)

    with rasterio.open(path) as source:
        with pytest.raises(InputValidationError, match=message):
            reader.read_window(source, metadata, window)

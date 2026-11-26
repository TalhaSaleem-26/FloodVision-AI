import numpy as np
import pytest

from floodvision.inference.tiling import (
    SlidingWindowGenerator,
    TileWeightGenerator,
    TilingConfig,
)


@pytest.mark.parametrize(
    ("length", "expected"),
    [(1, [0]), (512, [0]), (896, [0, 384]), (1000, [0, 384, 488])],
)
def test_positions_anchor_final_tile_at_raster_edge(length, expected):
    assert SlidingWindowGenerator().generate_positions(length) == expected


@pytest.mark.parametrize("length", [0, -1, 512.0, True])
def test_positions_reject_invalid_dimensions(length):
    with pytest.raises(ValueError, match="positive integer"):
        SlidingWindowGenerator().generate_positions(length)


def test_windows_cover_rectangular_raster_in_row_major_order():
    generator = SlidingWindowGenerator()
    windows = list(generator.generate(height=900, width=700))

    assert len(windows) == generator.count(height=900, width=700) == 6
    assert [(w.index, w.row_off, w.col_off) for w in windows] == [
        (0, 0, 0),
        (1, 0, 188),
        (2, 384, 0),
        (3, 384, 188),
        (4, 388, 0),
        (5, 388, 188),
    ]
    covered = np.zeros((900, 700), dtype=bool)
    for window in windows:
        assert window.row_end <= 900
        assert window.col_end <= 700
        covered[window.row_off:window.row_end, window.col_off:window.col_end] = True
    assert covered.all()


def test_blend_weights_crop_without_sharing_storage():
    weights = TileWeightGenerator()
    window = next(SlidingWindowGenerator().generate(height=23, width=35))
    full = weights.get_weights()
    cropped = weights.get_weights(window)

    assert full.shape == (512, 512)
    assert full.dtype == np.float32
    assert full[0, 0] == pytest.approx(0.001)
    assert full[256, 256] == pytest.approx(1.0)
    assert cropped.shape == (23, 35)
    np.testing.assert_array_equal(cropped, full[:23, :35])
    cropped[0, 0] = 0.75
    assert weights.get_weights(window)[0, 0] == pytest.approx(0.001)


def test_nonoverlapping_tiles_use_uniform_weights():
    weights = TileWeightGenerator(TilingConfig(overlap=0)).get_weights()
    np.testing.assert_array_equal(weights, np.ones((512, 512), dtype=np.float32))

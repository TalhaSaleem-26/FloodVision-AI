import numpy as np
import pytest

from floodvision.inference.accumulator import WeightedProbabilityAccumulator
from floodvision.inference.tiling import TileWindow


@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_overlap_is_weighted_and_uncovered_pixels_stay_invalid(tmp_path, storage):
    first = TileWindow(0, 0, 0, 1, 2, 512)
    second = TileWindow(1, 0, 1, 1, 2, 512)
    with WeightedProbabilityAccumulator(1, 3, storage=storage, work_dir=tmp_path) as acc:
        assert acc.add(first, np.array([[0.25, 0.25]]), np.ones((1, 2), bool),
                       np.array([[1.0, 0.25]]))
        assert acc.add(second, np.array([[0.75, np.nan]]), np.array([[True, False]]),
                       np.array([[0.75, 1.0]]))

        mask, probabilities, stats = acc.finalize(chunk_rows=1)

        np.testing.assert_array_equal(mask, [[0, 1, -1]])
        np.testing.assert_allclose(probabilities, [[0.25, 0.625, np.nan]])
        assert stats["tiles_added"] == 2
        assert stats["invalid_pixels"] == 1
        assert stats["water_coverage"] == 50
        with pytest.raises(RuntimeError, match="already finalized"):
            acc.finalize()

    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="already closed"):
        acc.finalize()


def test_empty_tile_does_not_create_valid_pixels():
    with WeightedProbabilityAccumulator(1, 1) as acc:
        added = acc.add(TileWindow(0, 0, 0, 1, 1, 512), np.array([[np.nan]]),
                        np.zeros((1, 1), bool), np.ones((1, 1)))
        assert added is False
        mask, probabilities, stats = acc.finalize()
        assert mask.item() == -1
        assert np.isnan(probabilities.item())
        assert stats["water_coverage"] is None

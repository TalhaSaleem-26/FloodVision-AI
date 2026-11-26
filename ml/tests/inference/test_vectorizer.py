import numpy as np
import pytest

from floodvision.inference.exceptions import GeospatialError
from floodvision.inference.validator import InputValidator
from floodvision.inference.vectorizer import WaterVectorizer


@pytest.fixture
def diagonal_water(raster_factory):
    mask = np.array([[1, -1], [0, 1]], dtype=np.int8)
    path = raster_factory(np.zeros((2, *mask.shape), dtype=np.float32))
    return mask, InputValidator().validate_raster(path, require_tile_size=False)


@pytest.mark.parametrize(("connectivity", "count"), [(4, 2), (8, 1)])
def test_diagonal_water_regions_follow_configured_connectivity(diagonal_water, connectivity, count):
    mask, metadata = diagonal_water
    result = WaterVectorizer(connectivity=connectivity).vectorize(mask, metadata)

    assert result.crs == "EPSG:4326"
    assert result.region_count == count
    assert [region.region_id for region in result.regions] == list(range(1, count + 1))
    assert all(region.geometry.is_valid and region.area_m2 > 0 for region in result.regions)


def test_vectorizer_rejects_region_limit(diagonal_water):
    mask, metadata = diagonal_water
    with pytest.raises(GeospatialError, match="Maximum water region limit"):
        WaterVectorizer(max_regions=1).vectorize(mask, metadata)


def test_vectorizer_returns_empty_result_for_no_water(diagonal_water):
    mask, metadata = diagonal_water
    result = WaterVectorizer().vectorize(np.minimum(mask, 0), metadata)
    assert result.regions == []
    assert result.total_area_m2 == 0

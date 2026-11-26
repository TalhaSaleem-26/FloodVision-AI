import numpy as np
import pytest
from pyproj import Geod

from floodvision.inference.area_analyzer import AreaAnalyzer
from floodvision.inference.exceptions import GeospatialError
from floodvision.inference.validator import InputValidator


def test_area_statistics_partition_valid_and_invalid_geodesic_area(raster_factory):
    path = raster_factory(np.zeros((2, 1, 3), dtype=np.float32))
    metadata = InputValidator().validate_raster(path, require_tile_size=False)
    corners = [metadata.transform @ xy for xy in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    longitude, latitude = zip(*corners)
    area, _ = Geod(ellps="WGS84").polygon_area_perimeter(longitude, latitude)

    stats = AreaAnalyzer(metadata).analyze(np.array([[1, 0, -1]], dtype=np.int8))

    assert stats.water_area_m2 == pytest.approx(abs(area))
    assert stats.non_water_area_m2 == pytest.approx(abs(area))
    assert stats.invalid_area_m2 == pytest.approx(abs(area))
    assert stats.valid_area_m2 == pytest.approx(2 * abs(area))
    assert stats.total_area_m2 == pytest.approx(3 * abs(area))
    assert stats.water_area_percentage == pytest.approx(50)


def test_area_analyzer_rejects_unknown_labels(raster_factory):
    path = raster_factory(np.zeros((2, 1, 1), dtype=np.float32))
    metadata = InputValidator().validate_raster(path, require_tile_size=False)
    with pytest.raises(GeospatialError, match="Invalid segmentation labels"):
        AreaAnalyzer(metadata).analyze(np.array([[2]], dtype=np.int8))

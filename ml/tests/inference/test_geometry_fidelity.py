import numpy as np

from floodvision.inference.geometry_fidelity import GeometryFidelityValidator
from floodvision.inference.validator import InputValidator
from floodvision.inference.vectorizer import WaterVectorizer


def test_vector_round_trip_and_invalid_pixel_encroachment(raster_factory):
    mask = np.array([[1, 1, -1], [1, 1, 0]], dtype=np.int8)
    path = raster_factory(np.zeros((2, *mask.shape), dtype=np.float32))
    metadata = InputValidator().validate_raster(path, require_tile_size=False)
    vectorizer = WaterVectorizer()
    validator = GeometryFidelityValidator(min_iou=0.8)

    exact = validator.validate(vectorizer.vectorize(mask, metadata), mask, metadata)
    assert exact["passed"]
    assert exact["geometry_iou"] == 1
    assert exact["changed_pixels"] == 0

    expanded = mask.copy()
    expanded[0, 2] = 1
    result = validator.validate(vectorizer.vectorize(expanded, metadata), mask, metadata)
    assert result["geometry_iou"] == 0.8
    assert result["invalid_pixel_encroachment"] == 1
    assert result["changed_pixels"] == 1
    assert result["passed"] is False

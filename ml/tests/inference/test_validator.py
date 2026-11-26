import numpy as np
import pytest

from floodvision.inference.exceptions import InputValidationError
from floodvision.inference.validator import InputValidator


def test_direct_inference_rejects_small_raster_that_tiling_accepts(raster_factory):
    path = raster_factory(np.zeros((2, 12, 15), dtype=np.float32))
    validator = InputValidator()

    with pytest.raises(InputValidationError, match="512 x 512"):
        validator.validate_raster(path, require_tile_size=True)

    metadata = validator.validate_raster(path, require_tile_size=False)
    assert (metadata.height, metadata.width) == (12, 15)
    assert metadata.band_order_verified


def test_explicit_band_override_accepts_missing_labels_but_rejects_reversed_bands():
    validator = InputValidator(allow_unverified_band_order=True)
    assert validator.validate_band_order((" vv ", "vh"))
    assert validator.validate_band_order((None, None)) is False
    with pytest.raises(InputValidationError, match="Incorrect band order"):
        validator.validate_band_order(("VH", "VV"))


def test_missing_crs_is_rejected_by_default(raster_factory):
    path = raster_factory(np.zeros((2, 512, 512), dtype=np.float32), crs=None)
    with pytest.raises(InputValidationError, match="CRS is missing"):
        InputValidator().validate_raster(path)
    assert InputValidator(require_crs=False).validate_raster(path).crs is None

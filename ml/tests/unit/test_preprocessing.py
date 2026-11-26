import numpy as np
import pytest

from floodvision.preprocessing.normalization import SARNormalizer


def test_normalizer_clips_per_band_and_fills_invalid_pixels(preprocessing_config_path):
    image = np.array([[[-40, 10, -10, -12]], [[-50, 0, np.nan, -20]]])
    valid = np.array([[True, True, True, False]])

    normalized = SARNormalizer(preprocessing_config_path)(image, valid)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [[[-5, 2.5, 0, 0]], [[-4.6, 2.4, 0, 0]]])


@pytest.mark.parametrize(
    ("shape", "mask_shape", "message"),
    [((1, 2, 3), (2, 3), "Expected image shape"),
     ((2, 2, 3), (3, 2), "Invalid mask dimensions")],
)
def test_normalizer_rejects_incompatible_shapes(
    preprocessing_config_path, shape, mask_shape, message
):
    with pytest.raises(ValueError, match=message):
        SARNormalizer(preprocessing_config_path)(np.zeros(shape), np.ones(mask_shape, bool))

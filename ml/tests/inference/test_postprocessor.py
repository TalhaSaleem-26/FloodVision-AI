import numpy as np
import pytest
import torch

from floodvision.inference.exceptions import PostprocessingError
from floodvision.inference.postprocessor import SegmentationPostprocessor
from floodvision.inference.schemas import RawPrediction
from floodvision.inference.validator import InputValidator


def test_postprocessing_restores_invalid_pixels_and_excludes_them_from_statistics(
    synthetic_raster_path, synthetic_valid_mask
):
    metadata = InputValidator().validate_raster(synthetic_raster_path)
    logits = torch.zeros((1, 1, 512, 512))
    logits[:, :, :, 256:] = -2
    raw = RawPrediction(logits, synthetic_valid_mask, metadata, "test", "cpu", 2.0)

    result = SegmentationPostprocessor()(raw)

    np.testing.assert_array_equal(result.mask == -1, ~synthetic_valid_mask)
    np.testing.assert_array_equal(np.isnan(result.probabilities), ~synthetic_valid_mask)
    assert result.statistics.water_pixels == 512 * 256 - 64**2
    assert result.statistics.non_water_pixels == 512 * 256
    assert result.statistics.valid_pixels == 512**2 - 64**2
    assert result.water_coverage == pytest.approx(100 * (512 * 256 - 64**2) / (512**2 - 64**2))
    assert result.to_summary()["source_file"] == synthetic_raster_path.name


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_postprocessor_rejects_invalid_threshold(threshold):
    with pytest.raises(PostprocessingError, match="[Tt]hreshold"):
        SegmentationPostprocessor(threshold)


@pytest.mark.parametrize(
    ("logits", "valid_mask", "message"),
    [
        (torch.zeros(1, 1, 2, 2), np.ones((512, 512), bool), "Invalid logits shape"),
        (torch.full((1, 1, 512, 512), float("nan")), np.ones((512, 512), bool), "NaN or Inf"),
        (torch.zeros(1, 1, 512, 512), np.zeros((512, 512), bool), "No valid prediction pixels"),
    ],
)
def test_postprocessor_rejects_malformed_predictions(synthetic_raster_path, logits, valid_mask, message):
    metadata = InputValidator().validate_raster(synthetic_raster_path)
    raw = RawPrediction(logits, valid_mask, metadata, "test", "cpu", 0.0)
    with pytest.raises(PostprocessingError, match=message):
        SegmentationPostprocessor().process(raw)

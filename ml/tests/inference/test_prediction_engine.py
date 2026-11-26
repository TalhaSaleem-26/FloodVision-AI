from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from floodvision.inference.exceptions import PredictionError, PreprocessingError
from floodvision.inference.model_loader import ModelLoader
from floodvision.inference.predictor import FloodPredictor
from floodvision.inference.preprocessor import SARPreprocessor
from floodvision.inference.tiled_predictor import TiledInferenceEngine


class EchoVV(nn.Module):
    def forward(self, tensor):
        assert not torch.is_grad_enabled()
        return tensor[:, :1]


@pytest.fixture
def lightweight_loader(preprocessing_config_path):
    model = EchoVV().eval()
    return SimpleNamespace(
        load=lambda device: model,
        device=torch.device("cpu"),
        inference_config=ModelLoader.EXPECTED_INFERENCE_CONFIG.copy(),
        metadata={"model_version": "fixture"},
        bundle_dir=preprocessing_config_path.parent,
    )


def test_preprocessor_and_predictor_preserve_mask_and_emit_raw_logits(
    lightweight_loader, preprocessing_config_path, raster_factory
):
    image = np.full((2, 512, 512), -18, np.float32)
    image[1, 0, 0] = np.nan
    source = raster_factory(image)
    processed = SARPreprocessor(preprocessing_config_path)(source)

    assert processed.tensor.shape == (1, 2, 512, 512)
    assert processed.tensor.dtype == torch.float32
    assert processed.tensor.is_contiguous()
    assert processed.invalid_pixels == 1
    assert processed.valid_pixels == 512**2 - 1
    torch.testing.assert_close(processed.tensor[0, :, 0, 0], torch.zeros(2))
    torch.testing.assert_close(processed.tensor[0, :, 1, 1], torch.tensor([-2., -0.2]))

    predictor = FloodPredictor(lightweight_loader, device="cpu")
    result = predictor(processed)
    torch.testing.assert_close(result.logits, processed.tensor[:, :1])
    np.testing.assert_array_equal(result.valid_mask, processed.valid_mask)
    assert result.metadata.path == source
    assert result.model_version == "fixture"
    assert result.device == "cpu"

    with pytest.raises(PredictionError, match="Expected float32 input"):
        predictor(replace(processed, tensor=processed.tensor.double()))
    with pytest.raises(PredictionError, match="no valid pixels"):
        predictor(replace(processed, valid_mask=np.zeros_like(processed.valid_mask)))


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (None, "output must be a tensor"),
        (torch.zeros(1, 1, 2, 2), "Unexpected output shape"),
        (torch.zeros(1, 1, 512, 512, dtype=torch.int8), "floating-point logits"),
        (torch.full((1, 1, 512, 512), float("inf")), "contains NaN or Inf"),
    ],
)
def test_predictor_rejects_malformed_model_outputs(lightweight_loader, output, message):
    with pytest.raises(PredictionError, match=message):
        FloodPredictor(lightweight_loader).validate_output(output)


@pytest.mark.parametrize(
    ("contents", "message"),
    [(None, "Configuration not found"), ("[foo", "Unable to read"),
     ("dataset: {}", "Missing preprocessing"), ("preprocessing: {}", "Unsupported preprocessing")],
)
def test_preprocessor_rejects_invalid_configuration(tmp_path, contents, message):
    path = tmp_path / "preprocessing.yaml"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    with pytest.raises(PreprocessingError, match=message):
        SARPreprocessor(path)


def test_tiled_memory_and_disk_match_expected_overlap_and_padding(
    lightweight_loader, raster_factory, tmp_path
):
    image = np.full((2, 8, 550), -18, dtype=np.float32)
    image[0, :, 100:400] = -6
    image[:, 2, 380] = np.nan
    source = raster_factory(image)
    engine = TiledInferenceEngine(lightweight_loader, device="cpu")
    work = tmp_path / "disk"
    expected = np.where(image[0] == -6, torch.sigmoid(torch.tensor(1.)).item(),
                        torch.sigmoid(torch.tensor(-2.)).item())
    expected[2, 380] = np.nan

    outcomes = []
    for storage in ("memory", "disk"):
        with engine.predict(source, storage=storage, work_dir=work) as result:
            np.testing.assert_allclose(result.probabilities, expected, rtol=1e-6)
            assert result.to_summary()["tiling"]["processed_tiles"] == 2
            assert result.statistics["invalid_pixels"] == 1
            outcomes.append((result.mask.copy(), dict(result.statistics)))
    np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1] == outcomes[1][1]
    assert not list(work.iterdir())


@pytest.mark.parametrize(
    ("limits", "storage", "message"),
    [
        ({"max_pixels": 1}, "auto", "maximum supported pixel count"),
        ({"max_tiles": 1}, "auto", "maximum supported tile count"),
        ({"memory_budget_mib": 1}, "memory", "accumulator memory budget"),
    ],
)
def test_tiled_resource_limits_fail_before_creating_scratch(
    lightweight_loader, raster_factory, tmp_path, limits, storage, message
):
    source = raster_factory(np.zeros((2, 512, 513), dtype=np.float32))
    scratch = tmp_path / "scratch"
    engine = TiledInferenceEngine(lightweight_loader, **limits)

    with pytest.raises(PredictionError, match=message):
        with engine.predict(source, storage=storage, work_dir=scratch):
            pytest.fail("Raster bypassed the configured limit")
    assert not scratch.exists()


def test_auto_storage_honors_memory_budget_boundary(lightweight_loader):
    engine = TiledInferenceEngine(lightweight_loader, memory_budget_mib=1)
    max_pixels = 1024**2 // 13

    assert engine.resolve_storage(1, max_pixels) == "memory"
    assert engine.resolve_storage(1, max_pixels + 1) == "disk"
    with pytest.raises(ValueError, match="Storage must be"):
        engine.resolve_storage(1, 1, "unknown")

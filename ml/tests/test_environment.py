import pytest
import torch
import numpy as np

from floodvision.inference.pipeline_config import (
    PipelineConfig
)

from floodvision.inference.exceptions import (
    InputValidationError
)


@pytest.mark.unit
def test_pytorch_available():

    tensor = torch.tensor([1.0, 2.0, 3.0])

    assert tensor.sum().item() == 6.0


@pytest.mark.unit
def test_numpy_available():

    array = np.array([1, 2, 3])

    assert array.sum() == 6


@pytest.mark.unit
def test_pipeline_configuration():

    config = PipelineConfig(device="cpu")

    assert config.device == "cpu"
    assert config.tile_size == 512
    assert config.overlap == 128


@pytest.mark.unit
def test_invalid_storage_rejected():

    with pytest.raises(
        InputValidationError,
        match="Unsupported storage mode"
    ):
        PipelineConfig(storage="invalid")


@pytest.mark.unit
def test_invalid_overlap_rejected():

    with pytest.raises(
        InputValidationError,
        match="Invalid tile overlap"
    ):
        PipelineConfig(overlap=512)
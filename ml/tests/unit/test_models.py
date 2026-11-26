import io

import pytest
import torch
import torch.nn as nn

from floodvision.models.factory import build_model


pytestmark = pytest.mark.unit


# ==========================================
# Model configurations
# ==========================================

MODEL_CONFIGS = {
    "baseline_cnn": {
        "name": "baseline_cnn",
        "in_channels": 2,
        "out_channels": 1,
        "base_channels": 16
    },

    "unet": {
        "name": "unet",
        "in_channels": 2,
        "out_channels": 1,
        "base_channels": 16
    },

    "resnet_unet": {
        "name": "resnet_unet",
        "in_channels": 2,
        "out_channels": 1,
        "pretrained": False
    },

    "mobilenet_unet": {
        "name": "mobilenet_unet",
        "in_channels": 2,
        "out_channels": 1,
        "pretrained": False
    }
}


EXPECTED_PARAMETERS = {
    "baseline_cnn": 434209,
    "unet": 482593,
    "resnet_unet": 14238145,
    "mobilenet_unet": 1034057
}


# ==========================================
# Parameterized model fixture
# ==========================================

@pytest.fixture(params=list(MODEL_CONFIGS))
def model_name(request):

    return request.param


@pytest.fixture
def model(model_name):

    with torch.random.fork_rng(devices=[]):

        torch.manual_seed(42)

        instance = build_model(
            MODEL_CONFIGS[model_name].copy()
        )

    return instance.cpu()


# ==========================================
# 1. Model initialization
# ==========================================

def test_model_initialization(model):

    assert isinstance(model, nn.Module)

    assert len(
        list(model.parameters())
    ) > 0


# ==========================================
# 2. Forward pass
# ==========================================

def test_forward_pass(model, model_input):

    model.eval()

    with torch.inference_mode():

        output = model(model_input)

    assert isinstance(output, torch.Tensor)

    assert output.dtype == torch.float32

    assert output.device.type == "cpu"


# ==========================================
# 3. Output dimensions
# ==========================================

def test_output_shape(model, model_input):

    model.eval()

    with torch.inference_mode():

        output = model(model_input)

    assert output.shape == (1, 1, 64, 64)


# ==========================================
# 4. Numerical stability
# ==========================================

def test_output_is_finite(model, model_input):

    model.eval()

    with torch.inference_mode():

        output = model(model_input)

    assert torch.isfinite(output).all()


# ==========================================
# 5. Batch processing
# ==========================================

def test_batch_processing(model, model_input):

    batch = model_input.repeat(
        2, 1, 1, 1
    )

    model.eval()

    with torch.inference_mode():

        output = model(batch)

    assert output.shape == (2, 1, 64, 64)

    assert torch.isfinite(output).all()


# ==========================================
# 6. Backward pass and gradients
# ==========================================

def test_backward_pass(model, model_input):

    model.train()

    # Batch size 2 for BatchNorm stability
    inputs = model_input.repeat(
        2, 1, 1, 1
    )

    model.zero_grad(set_to_none=True)

    output = model(inputs)

    target = torch.ones_like(output)

    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(output, target)

    assert torch.isfinite(loss)

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(gradients) > 0

    assert all(
        gradient is not None
        for gradient in gradients
    )

    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
    )


# ==========================================
# 7. Optimizer weight update
# ==========================================

def test_optimizer_updates_weights(
    model,
    model_input
):

    model.train()

    inputs = model_input.repeat(
        2, 1, 1, 1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0
    )

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer.zero_grad(set_to_none=True)

    output = model(inputs)

    target = torch.ones_like(output)

    loss = nn.BCEWithLogitsLoss()(
        output,
        target
    )

    loss.backward()

    optimizer.step()

    parameters_after = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after
        )
    )

    assert all(
        torch.isfinite(parameter).all()
        for parameter in parameters_after
    )


# ==========================================
# 8. Parameter count regression
# ==========================================

def test_parameter_count(model, model_name):

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert total_parameters == (
        EXPECTED_PARAMETERS[model_name]
    )


# ==========================================
# 9. Evaluation consistency
# ==========================================

def test_evaluation_consistency(
    model,
    model_input
):

    model.eval()

    with torch.inference_mode():

        first_output = model(model_input)

        second_output = model(model_input)

    torch.testing.assert_close(
        first_output,
        second_output,
        rtol=1e-5,
        atol=1e-6
    )


# ==========================================
# 10. Model serialization
# ==========================================

def test_state_dict_round_trip(
    model,
    model_name,
    model_input
):

    model.eval()

    with torch.inference_mode():

        original_output = model(model_input)

    buffer = io.BytesIO()

    torch.save(
        model.state_dict(),
        buffer
    )

    buffer.seek(0)

    restored_model = build_model(
        MODEL_CONFIGS[model_name].copy()
    )

    state_dict = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True
    )

    restored_model.load_state_dict(
        state_dict,
        strict=True
    )

    restored_model.eval()

    with torch.inference_mode():

        restored_output = restored_model(
            model_input
        )

    torch.testing.assert_close(
        original_output,
        restored_output,
        rtol=1e-5,
        atol=1e-6
    )


# ==========================================
# 11. Factory configuration preservation
# ==========================================

def test_factory_preserves_config(model_name):

    config = MODEL_CONFIGS[
        model_name
    ].copy()

    original_config = config.copy()

    build_model(config)

    assert config == original_config


# ==========================================
# 12. Invalid model name
# ==========================================

def test_unknown_model_rejected():

    with pytest.raises(
        (ValueError, KeyError)
    ):

        build_model({
            "name": "unknown_model"
        })
import math

import pytest
import torch

from floodvision.losses.factory import build_loss


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"name": "masked_bce", "pos_weight": 3}, 3 * math.log(2)),
        ({"name": "dice", "smooth": 0.5}, 0.25),
        ({"name": "bce_dice", "bce_weight": 2, "dice_weight": 0}, 2 * math.log(2)),
    ],
)
def test_factory_applies_configuration_to_single_valid_pixel(config, expected):
    logits = torch.tensor([0., float("nan")], requires_grad=True)
    result = build_loss(config)(logits, torch.tensor([1., -1.]))

    assert result.item() == pytest.approx(expected)
    result.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1] == 0


@pytest.mark.parametrize("name", ["masked_bce", "dice", "bce_dice"])
def test_extreme_wrong_logits_have_finite_loss_and_gradients(name):
    logits = torch.tensor([-1000., 1000.], requires_grad=True)
    loss = build_loss({"name": name})(logits, torch.tensor([1., 0.]))

    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_factory_rejects_unknown_loss():
    with pytest.raises(ValueError, match="Unsupported loss function: unknown"):
        build_loss({"name": "unknown"})

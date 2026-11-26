import math

import pytest
import torch
import torch.nn.functional as F

from floodvision.losses.masked_bce import MaskedBCELoss
from floodvision.losses.dice import MaskedDiceLoss
from floodvision.losses.combined import BCEDiceLoss
from floodvision.losses.factory import build_loss


pytestmark = pytest.mark.unit


# ==========================================
# Controlled test data
# ==========================================

@pytest.fixture
def loss_inputs():

    logits = torch.tensor(
        [[[[2.0, -2.0, 0.5],
           [-1.0, 3.0, -0.5],
           [1.0, -3.0, 0.0]]]],
        dtype=torch.float32,
        requires_grad=True
    )

    targets = torch.tensor(
        [[[[1.0, 0.0, -1.0],
           [0.0, 1.0, 1.0],
           [-1.0, 0.0, 1.0]]]],
        dtype=torch.float32
    )

    return logits, targets


LOSS_FACTORIES = [
    pytest.param(
        lambda: MaskedBCELoss(),
        id="masked_bce"
    ),

    pytest.param(
        lambda: MaskedBCELoss(pos_weight=3.0),
        id="weighted_bce"
    ),

    pytest.param(
        lambda: MaskedDiceLoss(),
        id="dice"
    ),

    pytest.param(
        lambda: BCEDiceLoss(),
        id="bce_dice"
    ),

    pytest.param(
        lambda: BCEDiceLoss(pos_weight=3.0),
        id="weighted_bce_dice"
    )
]


@pytest.fixture(params=LOSS_FACTORIES)
def criterion(request):

    return request.param()


# ==========================================
# 1. BCE mathematical equivalence
# ==========================================

def test_masked_bce_equivalence(loss_inputs):

    logits, targets = loss_inputs

    valid = targets != -1

    expected = F.binary_cross_entropy_with_logits(
        logits[valid],
        targets[valid]
    )

    actual = MaskedBCELoss()(
        logits,
        targets
    )

    torch.testing.assert_close(
        actual,
        expected
    )


# ==========================================
# 2. BCE known mathematical value
# ==========================================

def test_bce_zero_logits():

    logits = torch.zeros((1, 1, 2, 2))

    targets = torch.tensor(
        [[[[1.0, 0.0],
           [0.0, 1.0]]]]
    )

    loss = MaskedBCELoss()(
        logits,
        targets
    )

    assert loss.item() == pytest.approx(
        math.log(2),
        abs=1e-6
    )


# ==========================================
# 3. Weighted BCE equivalence
# ==========================================

def test_weighted_bce_equivalence(loss_inputs):

    logits, targets = loss_inputs

    valid = targets != -1

    expected = F.binary_cross_entropy_with_logits(
        logits[valid],
        targets[valid],
        pos_weight=torch.tensor(3.0)
    )

    actual = MaskedBCELoss(
        pos_weight=3.0
    )(logits, targets)

    torch.testing.assert_close(
        actual,
        expected
    )


# ==========================================
# 4. Positive class weighting
# ==========================================

def test_positive_class_weighting():

    logits = torch.zeros((1, 1, 2, 2))

    targets = torch.ones_like(logits)

    standard = MaskedBCELoss()(
        logits,
        targets
    )

    weighted = MaskedBCELoss(
        pos_weight=3.0
    )(logits, targets)

    torch.testing.assert_close(
        weighted,
        standard * 3.0
    )


# ==========================================
# 5. Negative class remains unchanged
# ==========================================

def test_negative_class_weighting():

    logits = torch.zeros((1, 1, 2, 2))

    targets = torch.zeros_like(logits)

    standard = MaskedBCELoss()(
        logits,
        targets
    )

    weighted = MaskedBCELoss(
        pos_weight=3.0
    )(logits, targets)

    torch.testing.assert_close(
        weighted,
        standard
    )


# ==========================================
# 6. Dice mathematical equivalence
# ==========================================

def test_dice_equivalence(loss_inputs):

    logits, targets = loss_inputs

    valid = targets != -1

    probabilities = torch.sigmoid(
        logits[valid]
    )

    labels = targets[valid]

    intersection = (
        probabilities * labels
    ).sum()

    smooth = 1e-6

    expected = 1.0 - (
        (2.0 * intersection + smooth)
        /
        (
            probabilities.sum()
            + labels.sum()
            + smooth
        )
    )

    actual = MaskedDiceLoss(
        smooth=smooth
    )(logits, targets)

    torch.testing.assert_close(
        actual,
        expected
    )


# ==========================================
# 7. Dice known mathematical value
# ==========================================

def test_dice_known_value():

    logits = torch.zeros((1, 1, 2, 2))

    targets = torch.tensor(
        [[[[1.0, 0.0],
           [1.0, 0.0]]]]
    )

    loss = MaskedDiceLoss()(
        logits,
        targets
    )

    assert loss.item() == pytest.approx(
        0.5,
        abs=1e-6
    )


# ==========================================
# 8. Combined loss equivalence
# ==========================================

@pytest.mark.parametrize(
    "bce_weight,dice_weight",
    [
        (0.5, 0.5),
        (0.7, 0.3),
        (1.0, 0.0),
        (0.0, 1.0)
    ]
)
def test_combined_loss_equivalence(
    loss_inputs,
    bce_weight,
    dice_weight
):

    logits, targets = loss_inputs

    bce = MaskedBCELoss()(
        logits,
        targets
    )

    dice = MaskedDiceLoss()(
        logits,
        targets
    )

    expected = (
        bce_weight * bce
        + dice_weight * dice
    )

    actual = BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight
    )(logits, targets)

    torch.testing.assert_close(
        actual,
        expected
    )


# ==========================================
# 9. Weighted combined loss
# ==========================================

def test_weighted_combined_loss(loss_inputs):

    logits, targets = loss_inputs

    bce = MaskedBCELoss(
        pos_weight=3.0
    )(logits, targets)

    dice = MaskedDiceLoss()(
        logits,
        targets
    )

    expected = 0.5 * bce + 0.5 * dice

    actual = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=3.0
    )(logits, targets)

    torch.testing.assert_close(
        actual,
        expected
    )


# ==========================================
# 10. Scalar and finite loss
# ==========================================

def test_loss_is_scalar_and_finite(
    criterion,
    loss_inputs
):

    logits, targets = loss_inputs

    loss = criterion(logits, targets)

    assert loss.ndim == 0

    assert torch.isfinite(loss)

    assert loss.item() >= 0


# ==========================================
# 11. Invalid logits must not affect loss
# ==========================================

def test_invalid_pixel_exclusion(
    criterion,
    loss_inputs
):

    logits, targets = loss_inputs

    modified_logits = logits.detach().clone()

    modified_logits[targets == -1] = 1000.0

    original = criterion(
        logits,
        targets
    )

    modified = criterion(
        modified_logits,
        targets
    )

    torch.testing.assert_close(
        original,
        modified
    )


# ==========================================
# 12. Gradient correctness
# ==========================================

def test_loss_gradients(
    criterion,
    loss_inputs
):

    logits, targets = loss_inputs

    loss = criterion(logits, targets)

    loss.backward()

    assert logits.grad is not None

    assert torch.isfinite(
        logits.grad
    ).all()

    valid = targets != -1

    invalid = targets == -1

    assert torch.all(
        logits.grad[invalid] == 0
    )

    assert torch.any(
        logits.grad[valid] != 0
    )


# ==========================================
# 13. All-invalid target handling
# ==========================================

def test_all_invalid_targets(criterion):

    logits = torch.randn(
        (1, 1, 4, 4),
        requires_grad=True
    )

    targets = torch.full_like(
        logits,
        -1.0
    )

    loss = criterion(logits, targets)

    assert torch.isfinite(loss)

    assert loss.item() == pytest.approx(
        0.0,
        abs=1e-7
    )

    loss.backward()

    assert logits.grad is not None

    assert torch.all(
        logits.grad == 0
    )


# ==========================================
# 14. Good predictions vs bad predictions
# ==========================================

def test_good_predictions_have_lower_loss(
    criterion,
    perfect_logits,
    target_tensor
):

    good_loss = criterion(
        perfect_logits,
        target_tensor
    )

    bad_loss = criterion(
        -perfect_logits,
        target_tensor
    )

    assert good_loss.item() < bad_loss.item()
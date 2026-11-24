from floodvision.losses.masked_bce import MaskedBCELoss
from floodvision.losses.dice import MaskedDiceLoss
from floodvision.losses.combined import BCEDiceLoss


def build_loss(config):

    name = config["name"]

    pos_weight = config.get("pos_weight")

    if name == "masked_bce":

        return MaskedBCELoss(
            pos_weight=pos_weight
        )

    elif name == "dice":

        return MaskedDiceLoss(
            smooth=config.get("smooth", 1e-6)
        )

    elif name == "bce_dice":

        return BCEDiceLoss(
            bce_weight=config.get("bce_weight", 0.5),
            dice_weight=config.get("dice_weight", 0.5),
            pos_weight=pos_weight
        )

    raise ValueError(
        f"Unsupported loss function: {name}"
    )
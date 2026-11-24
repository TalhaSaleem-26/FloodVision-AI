import torch
import torch.nn as nn


class MaskedDiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):

        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        if logits.shape != targets.shape:
            raise ValueError("Logits and targets must match")

        valid = (targets == 0) | (targets == 1)

        if not valid.any():
            return logits[valid].sum()

        probabilities = torch.sigmoid(logits[valid])

        targets = targets[valid]

        intersection = (
            probabilities * targets
        ).sum()

        denominator = (
            probabilities.sum() + targets.sum()
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice
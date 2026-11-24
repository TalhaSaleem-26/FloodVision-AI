import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCELoss(nn.Module):

    def __init__(self, pos_weight=None):

        super().__init__()

        weight = (
            None if pos_weight is None
            else torch.tensor(
                float(pos_weight),
                dtype=torch.float32
            )
        )

        self.register_buffer("pos_weight", weight)

    def forward(self, logits, targets):

        if logits.shape != targets.shape:
            raise ValueError("Logits and targets must match")

        valid = (targets == 0) | (targets == 1)

        if not valid.any():
            return logits[valid].sum()

        valid_logits = logits[valid]
        valid_targets = targets[valid]

        loss = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=self.pos_weight,
            reduction="mean"
        )

        return loss
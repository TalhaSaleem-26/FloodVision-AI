import torch


class SegmentationMetrics:

    def __init__(self, threshold=0.5):

        self.threshold = threshold
        self.reset()

    def reset(self):

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, logits, targets):

        if logits.shape != targets.shape:
            raise ValueError("Logits and targets must match")

        valid = (targets == 0) | (targets == 1)

        predictions = (
            torch.sigmoid(logits[valid])
            >= self.threshold
        )

        targets = targets[valid] == 1

        self.tp += (
            predictions & targets
        ).sum().item()

        self.tn += (
            ~predictions & ~targets
        ).sum().item()

        self.fp += (
            predictions & ~targets
        ).sum().item()

        self.fn += (
            ~predictions & targets
        ).sum().item()

    @staticmethod
    def safe_divide(numerator, denominator):

        if denominator == 0:
            return float("nan")

        return numerator / denominator

    def compute(self):

        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn

        divide = self.safe_divide

        return {
            "accuracy": divide(
                tp + tn, tp + tn + fp + fn
            ),

            "precision": divide(tp, tp + fp),

            "recall": divide(tp, tp + fn),

            "iou": divide(tp, tp + fp + fn),

            "dice": divide(
                2 * tp, 2 * tp + fp + fn
            ),

            "specificity": divide(tn, tn + fp)
        }
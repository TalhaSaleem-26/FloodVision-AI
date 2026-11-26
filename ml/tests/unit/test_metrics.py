import math

import pytest
import torch

from floodvision.metrics.segmentation import SegmentationMetrics


def test_metrics_accumulate_counts_and_ignore_invalid_targets():
    metrics = SegmentationMetrics()
    metrics.update(torch.tensor([0., -2., 2.]), torch.tensor([1, 0, -1]))
    metrics.update(torch.tensor([2., -2., 2.]), torch.tensor([0, 1, 1]))

    assert metrics.compute() == pytest.approx({
        "accuracy": 3 / 5, "precision": 2 / 3, "recall": 2 / 3,
        "iou": 1 / 2, "dice": 2 / 3, "specificity": 1 / 2,
    })
    metrics.reset()
    assert all(math.isnan(value) for value in metrics.compute().values())


def test_metrics_reject_different_tensor_shapes():
    with pytest.raises(ValueError, match="must match"):
        SegmentationMetrics().update(torch.zeros(2), torch.zeros(1, 2))

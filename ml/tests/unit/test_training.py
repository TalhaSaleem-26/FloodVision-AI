import csv
import math
import random

import numpy as np
import pytest
import torch
from torch import nn

from floodvision.losses.masked_bce import MaskedBCELoss
from floodvision.training.checkpoint import save_checkpoint
from floodvision.training.early_stopping import EarlyStopping
from floodvision.training.engine import train_one_epoch, validate_one_epoch
from floodvision.training.trainer import Trainer
from floodvision.utils.config import load_config
from floodvision.utils.seed import seed_everything


def make_batch(values, labels):
    return {
        "image": torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1),
        "mask": torch.tensor(labels, dtype=torch.float32).reshape(1, 1, 1, -1),
    }


@pytest.fixture
def training_components():
    model = nn.Conv2d(1, 1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    return model, optimizer, MaskedBCELoss()


def test_training_updates_weights_and_skips_fully_invalid_batches(training_components):
    model, optimizer, criterion = training_components
    invalid = make_batch([float("nan")], [-1])
    valid = make_batch([1, -1], [1, 0])

    results = train_one_epoch(model, [invalid, valid], criterion, optimizer, "cpu")

    assert model.training
    assert model.weight.item() == pytest.approx(0.25)
    assert results["loss"] == pytest.approx(math.log(2))
    assert results["accuracy"] == 0.5


def test_validation_uses_eval_mode_and_weights_loss_by_valid_pixels(training_components):
    model, _, criterion = training_components
    with torch.no_grad():
        model.weight.fill_(1)
    first = make_batch([2, float("nan")], [1, -1])
    second = make_batch([-1, 1], [0, 0])
    hook = model.register_forward_pre_hook(
        lambda *_: check_inference_mode(model)
    )
    try:
        result = validate_one_epoch(model, [make_batch([1], [-1]), first, second], criterion, "cpu")
    finally:
        hook.remove()

    assert result["loss"] == pytest.approx(
        (math.log1p(math.exp(-2)) + math.log1p(math.exp(-1)) + math.log1p(math.exp(1))) / 3
    )
    assert result["iou"] == 0.5
    assert model.weight.item() == 1
    assert model.weight.grad is None


def check_inference_mode(model):
    assert not model.training
    assert not torch.is_grad_enabled()


@pytest.mark.parametrize("stage", ["training", "validation"])
@pytest.mark.parametrize("invalid", [True, False])
def test_epoch_rejects_no_valid_pixels_or_nonfinite_loss(training_components, stage, invalid):
    model, optimizer, criterion = training_components
    loader = [make_batch([float("nan")], [-1 if invalid else 1])]
    message = f"No valid {stage} pixels" if invalid else f"Non-finite {stage} loss"

    with pytest.raises(RuntimeError, match=message):
        if stage == "training":
            train_one_epoch(model, loader, criterion, optimizer, "cpu")
        else:
            validate_one_epoch(model, loader, criterion, "cpu")
    assert model.weight.item() == 0


def test_early_stopping_resets_on_improvement_and_rejects_nan():
    stopping = EarlyStopping(patience=2, min_delta=0.05)
    assert not stopping(0.5)
    assert not stopping(0.52)
    assert not stopping(0.6)
    assert stopping.counter == 0
    assert not stopping(0.61)
    assert stopping(0.62)
    assert stopping.best_score == 0.6
    with pytest.raises(ValueError, match="Invalid validation score"):
        stopping(float("nan"))


def test_checkpoint_round_trip_restores_model_optimizer_and_scheduler(
    training_components, tmp_path
):
    model, optimizer, criterion = training_components
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    train_one_epoch(model, [make_batch([1, -1], [1, 0])], criterion, optimizer, "cpu")
    scheduler.step(0.4)
    path = tmp_path / "checkpoints" / "test.pt"
    save_checkpoint(path, model, optimizer, scheduler, 3, 0.4, {"tag": "synthetic"})

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = nn.Conv2d(1, 1, 1, bias=False)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=1, momentum=0.9)
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer)
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restored_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    torch.testing.assert_close(restored.weight, model.weight)
    torch.testing.assert_close(
        restored_optimizer.state[restored.weight]["momentum_buffer"],
        optimizer.state[model.weight]["momentum_buffer"],
    )
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert checkpoint["epoch"] == 3
    assert checkpoint["best_score"] == 0.4
    assert checkpoint["config"] == {"tag": "synthetic"}
    assert not path.with_suffix(".pt.tmp").exists()


def test_trainer_keeps_best_checkpoint_and_logs_each_completed_epoch(
    training_components, tmp_path
):
    model, optimizer, criterion = training_components
    batch = make_batch([1, -1], [1, 0])
    config = {
        "training": {"epochs": 4}, "metrics": {"threshold": 0.5, "monitor": "iou"},
        "early_stopping": {"patience": 1, "min_delta": 0},
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=0, factor=0.5
    )
    trainer = Trainer(model, [batch], [batch], criterion, optimizer, scheduler, "cpu", config, tmp_path)

    history = trainer.fit()

    assert [row["val_iou"] for row in history] == [1, 1]
    assert optimizer.param_groups[0]["lr"] == 0.25
    best = torch.load(tmp_path / "checkpoints/best.pt", weights_only=True)
    last = torch.load(tmp_path / "checkpoints/last.pt", weights_only=True)
    assert best["epoch"] == 1
    assert last["epoch"] == 2
    assert best["best_score"] == last["best_score"] == 1
    assert best["model_state_dict"]["weight"].item() < last["model_state_dict"]["weight"].item()
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert [float(row["val_iou"]) for row in rows] == [1, 1]


def test_trainer_rejects_undefined_monitor_before_writing_metrics(training_components, tmp_path):
    model, optimizer, criterion = training_components
    train = make_batch([1, -1], [1, 0])
    dry = make_batch([-1], [0])
    config = {
        "training": {"epochs": 1}, "metrics": {"threshold": 0.5, "monitor": "recall"},
        "early_stopping": {"patience": 1, "min_delta": 0},
    }
    trainer = Trainer(model, [train], [dry], criterion, optimizer, None, "cpu", config, tmp_path)
    with pytest.raises(RuntimeError, match="Invalid validation recall"):
        trainer.fit()
    assert not list(tmp_path.iterdir())


def test_seed_reproduces_python_numpy_and_torch_sequences():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        with torch.random.fork_rng(devices=[]):
            with torch.backends.cudnn.flags():
                seed_everything(7)
                first = random.random(), np.random.random(), torch.rand(3)
                seed_everything(7)
                second = random.random(), np.random.random(), torch.rand(3)
                assert first[:2] == second[:2]
                torch.testing.assert_close(first[2], second[2])
                assert torch.backends.cudnn.deterministic
                assert not torch.backends.cudnn.benchmark
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def test_load_config_accepts_mapping_and_rejects_missing_or_nonmapping(tmp_path):
    path = tmp_path / "config.yaml"
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_config(path)
    path.write_text("training: {epochs: 2}", encoding="utf-8")
    assert load_config(path) == {"training": {"epochs": 2}}
    path.write_text("- epochs", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_config(path)

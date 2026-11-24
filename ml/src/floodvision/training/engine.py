import torch

from tqdm.auto import tqdm

from floodvision.metrics.segmentation import (
    SegmentationMetrics
)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    threshold=0.5
):

    model.train()

    metrics = SegmentationMetrics(
        threshold=threshold
    )

    total_loss = 0.0
    total_valid_pixels = 0

    progress = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for batch in progress:

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        valid = (masks == 0) | (masks == 1)

        valid_pixels = valid.sum().item()

        if valid_pixels == 0:
            continue

        # Clear previous gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward propagation
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, masks)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")

        # Backward propagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track loss
        total_loss += loss.item() * valid_pixels

        total_valid_pixels += valid_pixels

        # Track segmentation metrics
        metrics.update(logits.detach(), masks)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    if total_valid_pixels == 0:
        raise RuntimeError("No valid training pixels")

    results = metrics.compute()

    results["loss"] = (
        total_loss / total_valid_pixels
    )

    return results


@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    threshold=0.5
):

    model.eval()

    metrics = SegmentationMetrics(
        threshold=threshold
    )

    total_loss = 0.0
    total_valid_pixels = 0

    progress = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    for batch in progress:

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        valid = (masks == 0) | (masks == 1)

        valid_pixels = valid.sum().item()

        if valid_pixels == 0:
            continue

        logits = model(images)

        loss = criterion(logits, masks)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite validation loss")

        total_loss += loss.item() * valid_pixels

        total_valid_pixels += valid_pixels

        metrics.update(logits, masks)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    if total_valid_pixels == 0:
        raise RuntimeError("No valid validation pixels")

    results = metrics.compute()

    results["loss"] = (
        total_loss / total_valid_pixels
    )

    return results
from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_score,
    config
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "best_score": best_score,

        "config": config
    }

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(checkpoint, temporary_path)

    temporary_path.replace(path)
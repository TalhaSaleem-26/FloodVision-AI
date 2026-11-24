
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch

from torch.utils.data import DataLoader, Subset

from floodvision.datasets.flood_dataset import FloodDataset

from floodvision.preprocessing.transforms import (
    get_train_transforms,
    get_val_transforms
)

from floodvision.models.factory import build_model
from floodvision.losses.factory import build_loss

from floodvision.training.trainer import Trainer

from floodvision.utils.config import load_config
from floodvision.utils.seed import seed_everything


BASE_DIR = Path(__file__).resolve().parents[1]


# ==================================================
# Device Selection
# ==================================================

def resolve_device(name):

    if name == "auto":

        return torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    if name.startswith("cuda") and not torch.cuda.is_available():

        raise RuntimeError("CUDA is not available")

    return torch.device(name)


# ==================================================
# Main Training Pipeline
# ==================================================

def main():

    # --------------------------------------------------
    # 1. Command Line Arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser(
        description="FloodVision AI Training Pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration"
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small training pipeline validation"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 2. Load Configuration
    # --------------------------------------------------

    config_path = BASE_DIR / args.config

    config = load_config(config_path)

    # --------------------------------------------------
    # 3. Smoke Test Configuration
    # --------------------------------------------------

    if args.smoke_test:

        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 1
        config["training"]["num_workers"] = 0

        config["experiment"]["name"] += "_smoke"

        print("Smoke test mode enabled!")

    # --------------------------------------------------
    # 4. Reproducibility
    # --------------------------------------------------

    seed = config["experiment"]["seed"]

    seed_everything(seed)

    # --------------------------------------------------
    # 5. Device Setup
    # --------------------------------------------------

    device = resolve_device(
        config["training"]["device"]
    )

    print("Training device:", device)

    # --------------------------------------------------
    # 6. Dataset Configuration
    # --------------------------------------------------

    data_config = config["data"]

    dataset_config = (
        BASE_DIR / data_config["dataset_config"]
    )

    preprocessing_config = (
        BASE_DIR / data_config["preprocessing_config"]
    )

    # --------------------------------------------------
    # 7. Training Dataset
    # --------------------------------------------------

    train_dataset = FloodDataset(
        dataset_config_path=dataset_config,
        preprocessing_config_path=preprocessing_config,
        split_path=BASE_DIR / data_config["train_split"],
        transforms=get_train_transforms(seed=seed)
    )

    # --------------------------------------------------
    # 8. Validation Dataset
    # --------------------------------------------------

    val_dataset = FloodDataset(
        dataset_config_path=dataset_config,
        preprocessing_config_path=preprocessing_config,
        split_path=BASE_DIR / data_config["val_split"],
        transforms=get_val_transforms()
    )

    # --------------------------------------------------
    # 9. Smoke Test Dataset Subsets
    # --------------------------------------------------

    if args.smoke_test:

        train_dataset = Subset(
            train_dataset,
            range(min(4, len(train_dataset)))
        )

        val_dataset = Subset(
            val_dataset,
            range(min(2, len(val_dataset)))
        )

    if len(train_dataset) == 0:

        raise ValueError("Training dataset is empty")

    if len(val_dataset) == 0:

        raise ValueError("Validation dataset is empty")

    # --------------------------------------------------
    # 10. Training Configuration
    # --------------------------------------------------

    training_config = config["training"]

    batch_size = training_config["batch_size"]

    num_workers = training_config["num_workers"]

    generator = torch.Generator()

    generator.manual_seed(seed)

    # --------------------------------------------------
    # 11. Training DataLoader
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator
    )

    # --------------------------------------------------
    # 12. Validation DataLoader
    # --------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    )

    # --------------------------------------------------
    # 13. Model Factory
    # --------------------------------------------------

    model = build_model(
        config["model"]
    ).to(device)

    # --------------------------------------------------
    # 14. Loss Factory
    # --------------------------------------------------

    criterion = build_loss(
        config["loss"]
    ).to(device)

    # --------------------------------------------------
    # 15. Optimizer
    # --------------------------------------------------

    optimizer_config = config["optimizer"]

    if optimizer_config["name"] != "adamw":

        raise ValueError(
            "Unsupported optimizer"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"]
    )

    # --------------------------------------------------
    # 16. Learning Rate Scheduler
    # --------------------------------------------------

    scheduler_config = config["scheduler"]

    if scheduler_config["name"] != "reduce_on_plateau":

        raise ValueError(
            "Unsupported scheduler"
        )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config["mode"],
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            min_lr=scheduler_config["min_lr"]
        )
    )

    # --------------------------------------------------
    # 17. Unique Experiment Directory
    # --------------------------------------------------

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    experiment_name = config["experiment"]["name"]

    output_dir = (
        BASE_DIR
        / "experiments"
        / f"{experiment_name}_{run_id}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    # --------------------------------------------------
    # 18. Save Actual Experiment Configuration
    # --------------------------------------------------

    with open(
        output_dir / "config.yaml",
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            config,
            file,
            sort_keys=False
        )

    # --------------------------------------------------
    # 19. Model Parameter Information
    # --------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # --------------------------------------------------
    # 20. Experiment Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)

    print("FLOODVISION AI - EXPERIMENT SUMMARY")

    print("=" * 60)

    print("Experiment:", experiment_name)

    print("Model:", config["model"]["name"])

    print("Device:", device)

    print("Seed:", seed)

    print("Total parameters:", f"{total_params:,}")

    print("Trainable parameters:", f"{trainable_params:,}")

    print("Training samples:", len(train_dataset))

    print("Validation samples:", len(val_dataset))

    print("Training batches:", len(train_loader))

    print("Validation batches:", len(val_loader))

    print("Batch size:", batch_size)

    print("Epochs:", training_config["epochs"])

    print("Loss:", config["loss"]["name"])

    print("Optimizer:", optimizer_config["name"])

    print("Learning rate:", optimizer_config["learning_rate"])

    print("Output directory:", output_dir)

    print("=" * 60)

    # --------------------------------------------------
    # 21. Initialize Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        output_dir=output_dir
    )

    # --------------------------------------------------
    # 22. Start Training
    # --------------------------------------------------

    history = trainer.fit()

    # --------------------------------------------------
    # 23. Final Experiment Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)

    print("EXPERIMENT COMPLETED SUCCESSFULLY")

    print("=" * 60)

    print("Completed epochs:", len(history))

    print("Best validation score:", trainer.best_score)

    print("Results:", output_dir / "metrics.csv")

    print(
        "Best checkpoint:",
        output_dir / "checkpoints/best.pt"
    )

    print(
        "Last checkpoint:",
        output_dir / "checkpoints/last.pt"
    )

    print("=" * 60)


# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":

    main()

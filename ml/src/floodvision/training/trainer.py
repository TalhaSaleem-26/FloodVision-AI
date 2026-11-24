import math
import time
from pathlib import Path

from floodvision.training.engine import (
    train_one_epoch,
    validate_one_epoch
)

from floodvision.training.checkpoint import (
    save_checkpoint
)

from floodvision.training.early_stopping import (
    EarlyStopping
)

from floodvision.training.logger import (
    ExperimentLogger
)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
        output_dir
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.config = config

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.checkpoint_dir = (
            self.output_dir / "checkpoints"
        )

        self.logger = ExperimentLogger(
            self.output_dir / "metrics.csv"
        )

        self.early_stopping = EarlyStopping(
            patience=config["early_stopping"]["patience"],
            min_delta=config["early_stopping"]["min_delta"]
        )

        self.best_score = float("-inf")

        self.history = []

    def fit(self):

        epochs = self.config["training"]["epochs"]

        threshold = self.config["metrics"]["threshold"]

        monitor = self.config["metrics"]["monitor"]

        for epoch in range(1, epochs + 1):

            start_time = time.perf_counter()

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            print(f"\nEpoch {epoch}/{epochs}")

            print("-" * 50)

            # Training
            train_results = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                threshold=threshold
            )

            # Validation
            val_results = validate_one_epoch(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                threshold=threshold
            )

            score = val_results[monitor]

            if not math.isfinite(score):

                raise RuntimeError(
                    f"Invalid validation {monitor}: {score}"
                )

            # Learning rate scheduling
            if self.scheduler is not None:

                self.scheduler.step(score)

            # Best model checkpoint
            if score > self.best_score:

                self.best_score = score

                self._save_checkpoint(
                    "best.pt",
                    epoch
                )

                print("New best model saved!")

            # Latest checkpoint
            self._save_checkpoint(
                "last.pt",
                epoch
            )

            elapsed = time.perf_counter() - start_time

            # Experiment logging
            row = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "epoch_time": elapsed
            }

            for name, value in train_results.items():

                row[f"train_{name}"] = value

            for name, value in val_results.items():

                row[f"val_{name}"] = value

            self.logger.log(row)

            self.history.append(row)

            self._print_results(
                train_results,
                val_results,
                elapsed
            )

            # Early stopping
            if self.early_stopping(score):

                print("\nEarly stopping triggered!")

                break

        print("\nTraining completed!")

        print(f"Best validation {monitor}: {self.best_score:.4f}")

        return self.history

    def _save_checkpoint(self, filename, epoch):

        save_checkpoint(
            path=self.checkpoint_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_score=self.best_score,
            config=self.config
        )

    @staticmethod
    def _print_results(
        train_results,
        val_results,
        elapsed
    ):

        print(
            f"Train Loss: {train_results['loss']:.4f}"
        )

        print(
            f"Val Loss: {val_results['loss']:.4f}"
        )

        print(
            f"Train IoU: {train_results['iou']:.4f}"
        )

        print(
            f"Val IoU: {val_results['iou']:.4f}"
        )

        print(
            f"Val Dice: {val_results['dice']:.4f}"
        )

        print(f"Epoch Time: {elapsed:.2f}s")
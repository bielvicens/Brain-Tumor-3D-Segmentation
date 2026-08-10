"""Training entry point for the BraTS 3D U-Net project."""

from __future__ import annotations

from pathlib import Path

import torch

from src.builders import (
    build_dataloader,
    build_datasets,
    build_loss,
    build_model,
    build_optimizer,
    build_pipeline,
)
from src.models import Trainer, TrainingHistory
from torch.utils.data import DataLoader
from src.utils import (
    EarlyStopping,
    ProjectConfig,
)


def train(config: ProjectConfig) -> None:
    """Run a complete training experiment."""

    # ------------------------------------------------------------------
    # Build preprocessing pipelines
    # ------------------------------------------------------------------
    train_pipeline = build_pipeline(
        config,
        training=True,
    )

    validation_pipeline = build_pipeline(
        config,
        training=False,
    )

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    train_dataset, validation_dataset = build_datasets(
        config=config,
        train_pipeline=train_pipeline,
        validation_pipeline=validation_pipeline,
    )

    # ------------------------------------------------------------------
    # Build dataloaders
    # ------------------------------------------------------------------
    train_loader = build_dataloader(
        train_dataset,
        config,
    )

    val_loader = build_dataloader(
        validation_dataset,
        config,
    )

    # ------------------------------------------------------------------
    # Build training components
    # ------------------------------------------------------------------
    model = build_model(config)

    criterion = build_loss(config)

    optimizer = build_optimizer(
        model,
        config,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config.training.device,
    )

    early_stopping = EarlyStopping(
        patience=config.early_stopping.patience,
        min_delta=config.early_stopping.min_delta,
        mode=config.early_stopping.mode,
    )

    checkpoint_dir = (
        Path(config.checkpoint.directory)
        / config.experiment.name
    )

    last_checkpoint = checkpoint_dir / "last.pt"

    start_epoch = 0
    history = None

    if last_checkpoint.exists():

        print(f"Resuming from {last_checkpoint}")

        checkpoint = torch.load(
            last_checkpoint,
            map_location=config.training.device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        h = checkpoint["history"]

        history = TrainingHistory(
            train_loss=h["train_loss"],
            val_loss=h["val_loss"],

            train_dice=h["train_dice"],
            val_dice=h["val_dice"],

            train_ncr_dice=h["train_ncr_dice"],
            train_ed_dice=h["train_ed_dice"],
            train_et_dice=h["train_et_dice"],

            val_ncr_dice=h["val_ncr_dice"],
            val_ed_dice=h["val_ed_dice"],
            val_et_dice=h["val_et_dice"],
        )

        print(f"Resuming at epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        start_epoch=start_epoch,
        history=history,
        checkpoint_dir=checkpoint_dir,
        early_stopping=early_stopping,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("Training finished.")
    print(f"Epochs completed: {history.epochs}")

    if history.train_loss:
        print(
            f"Final training loss: "
            f"{history.train_loss[-1]:.6f}"
        )

    if history.val_loss:
        print(
            f"Final validation loss: "
            f"{history.val_loss[-1]:.6f}"
        )


def main() -> None:
    """Program entry point."""

    config = ProjectConfig()

    train(config)


if __name__ == "__main__":
    main()
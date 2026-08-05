"""Training entry point for the BraTS 3D U-Net project."""

from __future__ import annotations

from pathlib import Path

from src.builders import (
    build_dataloader,
    build_datasets,
    build_loss,
    build_model,
    build_optimizer,
    build_pipeline,
)
from src.models import Trainer
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

    validation_loader = build_dataloader(
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

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=validation_loader,
        epochs=config.training.epochs,
        early_stopping=early_stopping,
        checkpoint_dir=checkpoint_dir,
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
"""Training entry point for the BraTS 3D U-Net project."""

from __future__ import annotations

from pathlib import Path

import torch

from src.data import BraTSDataset, create_dataloader
from src.data import train_validation_split
from src.models import DiceCrossEntropyLoss, Trainer, UNet3D
from src.preprocessing import PreprocessingPipeline
from src.utils import (
    EarlyStopping,
    ProjectConfig,
)


def build_model(config: ProjectConfig) -> UNet3D:
    """Build the segmentation model."""
    return UNet3D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
    )


def build_loss() -> DiceCrossEntropyLoss:
    """Build the training loss."""
    return DiceCrossEntropyLoss()


def build_optimizer(
    model: UNet3D,
    config: ProjectConfig,
) -> torch.optim.Optimizer:
    """Build the optimizer."""
    return torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def build_dataset(
    config: ProjectConfig,
) -> BraTSDataset:
    """Build the training dataset."""

    pipeline = PreprocessingPipeline()

    return BraTSDataset(
        dataset_root=config.data.dataset_root,
        pipeline=pipeline,
    )


def build_dataloader(
    dataset: BraTSDataset,
    config: ProjectConfig,
):
    """Build a DataLoader."""

    return create_dataloader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.experiment.seed,
    )


def train(config: ProjectConfig) -> None:
    """Run a complete training experiment."""

    reader_dataset = build_dataset(config)

    train_ids, val_ids = train_validation_split(
        reader_dataset.patient_ids,
        validation_fraction=config.data.validation_split,
        seed=config.experiment.seed,
    )

    train_dataset = BraTSDataset(
        dataset_root=config.data.dataset_root,
        pipeline=reader_dataset.pipeline,
        patient_ids=train_ids,
    )

    val_dataset = BraTSDataset(
        dataset_root=config.data.dataset_root,
        pipeline=reader_dataset.pipeline,
        patient_ids=val_ids,
    )

    train_loader = build_dataloader(train_dataset, config)

    val_loader = build_dataloader(val_dataset, config)

    model = build_model(config)

    criterion = build_loss()

    optimizer = build_optimizer(model, config)

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

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        early_stopping=early_stopping,
        checkpoint_dir=checkpoint_dir,
    )

    print()
    print("Training finished.")
    print(f"Epochs completed: {history.epochs}")

    if history.train_loss:
        print(f"Final training loss: {history.train_loss[-1]:.6f}")

    if history.val_loss:
        print(f"Final validation loss: {history.val_loss[-1]:.6f}")


def main() -> None:
    """Program entry point."""

    config = ProjectConfig()

    train(config)


if __name__ == "__main__":
    main()
"""Factory functions for constructing the main project components.

Centralising object construction avoids duplicating configuration logic
between training, inference and future scripts. The builders create fully
configured objects but contain no application logic.
"""

from __future__ import annotations

import torch

from src.models import DiceCrossEntropyLoss, UNet3D
from src.preprocessing import (
    PreprocessingPipeline,
    ResamplingTransform,
    ZScoreNormalization,
    RandomFlip,
    RandomRotation90,
    RandomGaussianNoise,
    RandomGamma,
    RandomIntensityShift,
)
from src.utils import ProjectConfig
from src.data import BraTSDataset, train_validation_split


def build_model(config: ProjectConfig) -> UNet3D:
    """Build the segmentation model from the project configuration."""
    return UNet3D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
    )


def build_pipeline(
    config: ProjectConfig,
    training: bool = False,
) -> PreprocessingPipeline:
    """Build the preprocessing pipeline.

    Args:
        config:
            Project configuration.
        training:
            Whether the pipeline will be used for training. If True,
            data augmentations are appended after preprocessing.
    """
    _ = config

    transforms = [
        ZScoreNormalization(),
        ResamplingTransform(
            target_spacing=(1.0, 1.0, 1.0),
        ),
    ]

    if training:
        transforms.extend(
            [
                RandomFlip(),
                RandomRotation90(),
                RandomGaussianNoise(),
                RandomGamma(),
                RandomIntensityShift(),
            ]
        )

    return PreprocessingPipeline(transforms)


def build_loss(config: ProjectConfig) -> DiceCrossEntropyLoss:
    """Build the segmentation loss."""
    _ = config
    return DiceCrossEntropyLoss()


def build_optimizer(
    model: UNet3D,
    config: ProjectConfig,
) -> torch.optim.Optimizer:
    """Build the optimizer used during training."""
    return torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

def build_datasets(
    config: ProjectConfig,
    train_pipeline: PreprocessingPipeline,
    validation_pipeline: PreprocessingPipeline,
) -> tuple[BraTSDataset, BraTSDataset]:
    """Build the training and validation datasets."""

    full_dataset = BraTSDataset(
        dataset_root=config.data.dataset_root,
        pipeline=train_pipeline,
    )

    train_ids, val_ids = train_validation_split(
        full_dataset.patient_ids,
        validation_fraction=config.data.validation_split,
        seed=config.experiment.seed,
    )

    train_dataset = BraTSDataset(
        dataset_root=config.data.dataset_root,
        pipeline=train_pipeline,
        patient_ids=train_ids,
    )

    validation_dataset = BraTSDataset(
        dataset_root=config.data.dataset_root,
        pipeline=validation_pipeline,
        patient_ids=val_ids,
    )

    return train_dataset, validation_dataset


def build_dataloader(
    dataset: BraTSDataset,
    config: ProjectConfig,
):
    """Build a DataLoader for a dataset."""

    from src.data import create_dataloader

    return create_dataloader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.experiment.seed,
    )
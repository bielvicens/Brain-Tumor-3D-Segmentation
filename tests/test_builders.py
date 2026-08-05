"""Tests for builder helper functions."""

from __future__ import annotations

import torch

from src.builders import (
    build_loss,
    build_model,
    build_optimizer,
    build_pipeline,
)
from src.models import (
    DiceCrossEntropyLoss,
    UNet3D,
)
from src.preprocessing import (
    PreprocessingPipeline,
    ResamplingTransform,
    ZScoreNormalization,
)
from src.utils import ProjectConfig


def test_build_model_returns_unet3d() -> None:
    config = ProjectConfig()

    model = build_model(config)

    assert isinstance(model, UNet3D)


def test_build_model_uses_configuration() -> None:
    config = ProjectConfig()

    config.model.in_channels = 2
    config.model.out_channels = 5
    config.model.base_channels = 16

    model = build_model(config)

    assert model.in_channels == 2
    assert model.out_channels == 5


def test_build_loss_returns_dice_cross_entropy_loss() -> None:
    config = ProjectConfig()

    loss = build_loss(config)

    assert isinstance(loss, DiceCrossEntropyLoss)


def test_build_optimizer_returns_adam() -> None:
    config = ProjectConfig()

    model = build_model(config)

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )


def test_build_optimizer_uses_configuration() -> None:
    config = ProjectConfig()

    config.training.learning_rate = 5e-4
    config.training.weight_decay = 1e-3

    model = build_model(config)

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    group = optimizer.param_groups[0]

    assert group["lr"] == config.training.learning_rate
    assert group["weight_decay"] == config.training.weight_decay


def test_build_pipeline_returns_pipeline() -> None:
    config = ProjectConfig()

    pipeline = build_pipeline(config)

    assert isinstance(
        pipeline,
        PreprocessingPipeline,
    )


def test_build_pipeline_contains_expected_transforms() -> None:
    config = ProjectConfig()

    pipeline = build_pipeline(config)

    transforms = list(pipeline)

    assert len(transforms) == 3
    assert isinstance(transforms[0], ZScoreNormalization)
    assert isinstance(transforms[1], ResamplingTransform)
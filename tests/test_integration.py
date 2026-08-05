"""Integration tests for the complete training pipeline."""

from __future__ import annotations

import pytest
import torch

from src.builders import (
    build_dataloader,
    build_datasets,
    build_loss,
    build_model,
    build_optimizer,
    build_pipeline,
)
from src.models import Trainer
from src.utils import ProjectConfig

pytestmark = pytest.mark.slow


def test_complete_training_pipeline() -> None:
    """Run one complete training epoch."""

    config = ProjectConfig()

    config.training.batch_size = 1
    config.training.epochs = 1
    config.data.num_workers = 0
    config.data.pin_memory = False

    train_pipeline = build_pipeline(
        config,
        training=True,
    )

    validation_pipeline = build_pipeline(
        config,
        training=False,
    )

    train_dataset, validation_dataset = build_datasets(
        config=config,
        train_pipeline=train_pipeline,
        validation_pipeline=validation_pipeline,
    )

    assert len(train_dataset) > 0
    assert len(validation_dataset) > 0

    # Fem servir només un pacient per accelerar el test.
    train_dataset.patient_ids = train_dataset.patient_ids[:1]
    validation_dataset.patient_ids = validation_dataset.patient_ids[:1]

    train_loader = build_dataloader(
        train_dataset,
        config,
    )

    validation_loader = build_dataloader(
        validation_dataset,
        config,
    )

    model = build_model(config)

    optimizer = build_optimizer(
        model,
        config,
    )

    criterion = build_loss(config)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=validation_loader,
        epochs=1,
    )

    assert len(history.train_loss) == 1
    assert len(history.val_loss) == 1

    assert history.train_loss[0] > 0
    assert history.val_loss[0] > 0

    assert torch.isfinite(torch.tensor(history.train_loss[0]))
    assert torch.isfinite(torch.tensor(history.val_loss[0]))
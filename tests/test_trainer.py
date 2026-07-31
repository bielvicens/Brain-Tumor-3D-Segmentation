from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.trainer import Trainer, TrainingHistory


class TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def make_loader() -> DataLoader:
    images = torch.randn(4, 1, 4, 4, 4)
    masks = torch.randint(0, 2, (4, 4, 4, 4))
    return DataLoader(TensorDataset(images, masks), batch_size=2)


def make_trainer() -> Trainer:
    model = TinySegmentationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
    )


def test_train_epoch_returns_finite_loss() -> None:
    trainer = make_trainer()

    loss = trainer.train_epoch(make_loader())

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_train_epoch_updates_model_parameters() -> None:
    trainer = make_trainer()
    before = {
        key: value.detach().clone()
        for key, value in trainer.model.state_dict().items()
    }

    trainer.train_epoch(make_loader())

    after = trainer.model.state_dict()

    assert any(not torch.equal(before[key], after[key]) for key in before)


def test_validate_epoch_returns_finite_loss() -> None:
    trainer = make_trainer()

    loss = trainer.validate_epoch(make_loader())

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_validate_does_not_update_model_parameters() -> None:
    trainer = make_trainer()
    before = {
        key: value.detach().clone()
        for key, value in trainer.model.state_dict().items()
    }

    trainer.validate_epoch(make_loader())

    after = trainer.model.state_dict()

    for key in before:
        assert torch.equal(before[key], after[key])


def test_fit_returns_training_history() -> None:
    trainer = make_trainer()

    history = trainer.fit(
        train_loader=make_loader(),
        val_loader=make_loader(),
        epochs=2,
    )

    assert isinstance(history, TrainingHistory)
    assert history.epochs == 2
    assert len(history.train_loss) == 2
    assert len(history.val_loss) == 2


def test_fit_without_validation() -> None:
    trainer = make_trainer()

    history = trainer.fit(
        train_loader=make_loader(),
        epochs=2,
    )

    assert history.epochs == 2
    assert len(history.train_loss) == 2
    assert history.val_loss == []


def test_fit_rejects_invalid_epoch_count() -> None:
    trainer = make_trainer()

    try:
        trainer.fit(make_loader(), epochs=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for epochs=0")


def test_empty_train_loader_raises() -> None:
    trainer = make_trainer()

    empty_loader = DataLoader(
        TensorDataset(
            torch.empty(0, 1, 4, 4, 4),
            torch.empty(0, 4, 4, 4, dtype=torch.long),
        ),
        batch_size=2,
    )

    try:
        trainer.train_epoch(empty_loader)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty dataloader")


def test_checkpoint_can_be_saved(tmp_path) -> None:
    trainer = make_trainer()

    history = TrainingHistory(
        train_loss=[1.0],
        val_loss=[0.9],
    )

    path = trainer.save_checkpoint(
        tmp_path / "checkpoint.pt",
        epoch=1,
        history=history,
    )

    assert path.exists()

    checkpoint = torch.load(path, map_location="cpu")

    assert checkpoint["epoch"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["history"]["train_loss"] == [1.0]
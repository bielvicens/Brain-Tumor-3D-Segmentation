"""Tests for src.utils.checkpoints (save_checkpoint / load_checkpoint).

Uses tiny torch models, optimizers and schedulers defined here - never the
real U-Net - since this module must work with any standard PyTorch
model/optimizer/scheduler.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from src.utils.checkpoints import CheckpointData, load_checkpoint, save_checkpoint


# ----------------------------------------------------------------------
# Dummy model/optimizer/scheduler factories
# ----------------------------------------------------------------------
class _TinyModel(torch.nn.Module):
    """A minimal model - just large enough to have real, comparable weights."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_model() -> _TinyModel:
    torch.manual_seed(0)
    return _TinyModel()


def _make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def _make_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


def _train_one_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Run one optimization step so the optimizer accumulates real state
    (e.g. SGD momentum buffers), not just an empty `state` dict."""
    optimizer.zero_grad()
    output = model(torch.randn(3, 4))
    loss = output.sum()
    loss.backward()
    optimizer.step()


def _assert_state_dicts_equal(a: Any, b: Any) -> None:
    """Recursively compare two (possibly nested) state dicts for exact
    equality, including tensors."""
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a.keys()) == set(b.keys())
        for key in a:
            _assert_state_dicts_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for item_a, item_b in zip(a, b):
            _assert_state_dicts_equal(item_a, item_b)
    else:
        assert a == b


# ----------------------------------------------------------------------
# Basic save / load round-trip
# ----------------------------------------------------------------------
def test_save_checkpoint_creates_a_file(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)
    assert path.is_file()


def test_load_checkpoint_returns_checkpoint_data(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path, epoch=3)

    result = load_checkpoint(path, _TinyModel())
    assert isinstance(result, CheckpointData)


def test_returned_types_are_correct(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path, epoch=2, history={"loss": [1.0]}, metadata={"note": "x"})

    result = load_checkpoint(path, _TinyModel())
    assert isinstance(result.epoch, int)
    assert isinstance(result.history, dict)
    assert isinstance(result.metadata, dict)


# ----------------------------------------------------------------------
# Model weight restoration
# ----------------------------------------------------------------------
def test_model_weights_are_restored(tmp_path: Path) -> None:
    source_model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(source_model, path)

    target_model = _TinyModel()  # different random init (no manual_seed here)
    load_checkpoint(path, target_model)

    _assert_state_dicts_equal(source_model.state_dict(), target_model.state_dict())


def test_model_restored_exactly_matches_source_values() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "ckpt.pt"
        source_model = _make_model()
        # Perturb weights to known, non-default values.
        with torch.no_grad():
            for param in source_model.parameters():
                param.fill_(3.1415)
        save_checkpoint(source_model, path)

        target_model = _TinyModel()
        load_checkpoint(path, target_model)

        for param in target_model.parameters():
            assert torch.all(param == 3.1415)


# ----------------------------------------------------------------------
# Optimizer restoration
# ----------------------------------------------------------------------
def test_optimizer_state_is_restored(tmp_path: Path) -> None:
    source_model = _make_model()
    source_optimizer = _make_optimizer(source_model)
    _train_one_step(source_model, source_optimizer)  # populate momentum buffers

    path = tmp_path / "ckpt.pt"
    save_checkpoint(source_model, path, optimizer=source_optimizer)

    target_model = _TinyModel()
    target_optimizer = _make_optimizer(target_model)
    load_checkpoint(path, target_model, optimizer=target_optimizer)

    _assert_state_dicts_equal(source_optimizer.state_dict(), target_optimizer.state_dict())


def test_optimizer_not_restored_when_not_requested(tmp_path: Path) -> None:
    source_model = _make_model()
    source_optimizer = _make_optimizer(source_model)
    _train_one_step(source_model, source_optimizer)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(source_model, path, optimizer=source_optimizer)

    target_model = _TinyModel()
    # optimizer=None -> checkpoint's optimizer_state_dict is simply ignored.
    result = load_checkpoint(path, target_model)
    assert isinstance(result, CheckpointData)


# ----------------------------------------------------------------------
# Scheduler restoration
# ----------------------------------------------------------------------
def test_scheduler_state_is_restored(tmp_path: Path) -> None:
    source_model = _make_model()
    source_optimizer = _make_optimizer(source_model)
    source_scheduler = _make_scheduler(source_optimizer)
    for _ in range(3):
        source_optimizer.step()
        source_scheduler.step()  # advance internal state (last_epoch, LR)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        source_model, path, optimizer=source_optimizer, scheduler=source_scheduler
    )

    target_model = _TinyModel()
    target_optimizer = _make_optimizer(target_model)
    target_scheduler = _make_scheduler(target_optimizer)
    load_checkpoint(
        path, target_model, optimizer=target_optimizer, scheduler=target_scheduler
    )

    _assert_state_dicts_equal(source_scheduler.state_dict(), target_scheduler.state_dict())


def test_scheduler_restored_learning_rate_matches_source(tmp_path: Path) -> None:
    source_model = _make_model()
    source_optimizer = _make_optimizer(source_model)
    source_scheduler = _make_scheduler(source_optimizer)
    for _ in range(2):
        source_optimizer.step()
        source_scheduler.step()
    expected_lr = source_optimizer.param_groups[0]["lr"]

    path = tmp_path / "ckpt.pt"
    save_checkpoint(source_model, path, optimizer=source_optimizer, scheduler=source_scheduler)

    target_model = _TinyModel()
    target_optimizer = _make_optimizer(target_model)
    target_scheduler = _make_scheduler(target_optimizer)
    load_checkpoint(path, target_model, optimizer=target_optimizer, scheduler=target_scheduler)

    assert target_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


# ----------------------------------------------------------------------
# epoch / history / metadata preservation
# ----------------------------------------------------------------------
def test_epoch_is_preserved(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path, epoch=17)

    result = load_checkpoint(path, _TinyModel())
    assert result.epoch == 17


def test_epoch_defaults_to_none_when_not_given(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)

    result = load_checkpoint(path, _TinyModel())
    assert result.epoch is None


def test_history_is_preserved(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    history = {"train_loss": [0.9, 0.7, 0.5], "val_dice": [0.5, 0.6, 0.65]}
    save_checkpoint(model, path, history=history)

    result = load_checkpoint(path, _TinyModel())
    assert result.history == history


def test_metadata_is_preserved(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    metadata = {"git_commit": "abc123", "num_classes": 4, "config": {"lr": 0.001}}
    save_checkpoint(model, path, metadata=metadata)

    result = load_checkpoint(path, _TinyModel())
    assert result.metadata == metadata


def test_history_and_metadata_default_to_empty_dict(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)

    result = load_checkpoint(path, _TinyModel())
    assert result.history == {}
    assert result.metadata == {}


# ----------------------------------------------------------------------
# Directory creation / overwrite
# ----------------------------------------------------------------------
def test_save_checkpoint_creates_missing_parent_directories(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "nested" / "sub" / "dir" / "ckpt.pt"
    assert not path.parent.exists()

    save_checkpoint(model, path)

    assert path.is_file()


def test_save_checkpoint_overwrites_existing_file(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(model, path, epoch=1)
    save_checkpoint(model, path, epoch=2)  # same path, should overwrite

    result = load_checkpoint(path, _TinyModel())
    assert result.epoch == 2


# ----------------------------------------------------------------------
# map_location
# ----------------------------------------------------------------------
def test_load_checkpoint_supports_map_location_string(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)

    result = load_checkpoint(path, _TinyModel(), map_location="cpu")
    assert isinstance(result, CheckpointData)


def test_load_checkpoint_supports_map_location_device(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)

    result = load_checkpoint(path, _TinyModel(), map_location=torch.device("cpu"))
    assert isinstance(result, CheckpointData)


# ----------------------------------------------------------------------
# File / content errors
# ----------------------------------------------------------------------
def test_load_checkpoint_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt", _TinyModel())


def test_load_checkpoint_corrupted_file_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupted.pt"
    path.write_bytes(b"this is definitely not a valid torch checkpoint")

    with pytest.raises(ValueError, match="corrupted"):
        load_checkpoint(path, _TinyModel())


def test_load_checkpoint_missing_model_state_dict_raises(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.pt"
    torch.save({"epoch": 1, "history": {}, "metadata": {}}, path)  # no model_state_dict

    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(path, _TinyModel())


def test_load_checkpoint_non_dict_content_raises(tmp_path: Path) -> None:
    path = tmp_path / "not_a_dict.pt"
    torch.save(torch.randn(3, 3), path)  # a bare tensor, not a checkpoint dict

    with pytest.raises(ValueError):
        load_checkpoint(path, _TinyModel())


def test_load_checkpoint_missing_optimizer_state_raises_when_optimizer_requested(
    tmp_path: Path,
) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)  # no optimizer saved

    target_model = _TinyModel()
    target_optimizer = _make_optimizer(target_model)
    with pytest.raises(ValueError, match="optimizer_state_dict"):
        load_checkpoint(path, target_model, optimizer=target_optimizer)


def test_load_checkpoint_missing_scheduler_state_raises_when_scheduler_requested(
    tmp_path: Path,
) -> None:
    model = _make_model()
    optimizer = _make_optimizer(model)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path, optimizer=optimizer)  # no scheduler saved

    target_model = _TinyModel()
    target_optimizer = _make_optimizer(target_model)
    target_scheduler = _make_scheduler(target_optimizer)
    with pytest.raises(ValueError, match="scheduler_state_dict"):
        load_checkpoint(
            path, target_model, optimizer=target_optimizer, scheduler=target_scheduler
        )


# ----------------------------------------------------------------------
# save_checkpoint must never modify the objects it's given
# ----------------------------------------------------------------------
def test_save_checkpoint_does_not_modify_model_weights(tmp_path: Path) -> None:
    model = _make_model()
    weights_before = copy.deepcopy(model.state_dict())

    save_checkpoint(model, tmp_path / "ckpt.pt")

    _assert_state_dicts_equal(weights_before, model.state_dict())


def test_save_checkpoint_does_not_modify_optimizer_state(tmp_path: Path) -> None:
    model = _make_model()
    optimizer = _make_optimizer(model)
    _train_one_step(model, optimizer)
    state_before = copy.deepcopy(optimizer.state_dict())

    save_checkpoint(model, tmp_path / "ckpt.pt", optimizer=optimizer)

    _assert_state_dicts_equal(state_before, optimizer.state_dict())


def test_save_checkpoint_does_not_modify_history_or_metadata_dicts(tmp_path: Path) -> None:
    model = _make_model()
    history = {"loss": [1.0, 0.5]}
    metadata = {"note": "original"}
    history_copy = copy.deepcopy(history)
    metadata_copy = copy.deepcopy(metadata)

    save_checkpoint(model, tmp_path / "ckpt.pt", history=history, metadata=metadata)

    assert history == history_copy
    assert metadata == metadata_copy


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
def test_save_checkpoint_rejects_invalid_model(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        save_checkpoint("not a model", tmp_path / "ckpt.pt")  # type: ignore[arg-type]


def test_save_checkpoint_rejects_invalid_path_type() -> None:
    with pytest.raises(TypeError, match="path"):
        save_checkpoint(_make_model(), 12345)  # type: ignore[arg-type]


def test_save_checkpoint_rejects_invalid_optimizer_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="optimizer"):
        save_checkpoint(_make_model(), tmp_path / "ckpt.pt", optimizer="not an optimizer")  # type: ignore[arg-type]


def test_save_checkpoint_rejects_invalid_scheduler_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="scheduler"):
        save_checkpoint(_make_model(), tmp_path / "ckpt.pt", scheduler="not a scheduler")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_epoch", [-1, 1.5, "3", True])
def test_save_checkpoint_rejects_invalid_epoch(tmp_path: Path, bad_epoch) -> None:
    with pytest.raises(TypeError, match="epoch"):
        save_checkpoint(_make_model(), tmp_path / "ckpt.pt", epoch=bad_epoch)


def test_save_checkpoint_rejects_invalid_history_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="history"):
        save_checkpoint(_make_model(), tmp_path / "ckpt.pt", history=["not", "a", "dict"])  # type: ignore[arg-type]


def test_save_checkpoint_rejects_invalid_metadata_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="metadata"):
        save_checkpoint(_make_model(), tmp_path / "ckpt.pt", metadata="not a dict")  # type: ignore[arg-type]


def test_load_checkpoint_rejects_invalid_model(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)

    with pytest.raises(TypeError, match="torch.nn.Module"):
        load_checkpoint(path, "not a model")  # type: ignore[arg-type]


def test_load_checkpoint_rejects_invalid_path_type() -> None:
    with pytest.raises(TypeError, match="path"):
        load_checkpoint(12345, _TinyModel())  # type: ignore[arg-type]


def test_load_checkpoint_rejects_invalid_optimizer_type(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)

    with pytest.raises(TypeError, match="optimizer"):
        load_checkpoint(path, _TinyModel(), optimizer="not an optimizer")  # type: ignore[arg-type]


def test_load_checkpoint_rejects_invalid_scheduler_type(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)

    with pytest.raises(TypeError, match="scheduler"):
        load_checkpoint(path, _TinyModel(), scheduler="not a scheduler")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Full round-trip integration (model + optimizer + scheduler + bookkeeping)
# ----------------------------------------------------------------------
def test_full_round_trip_restores_everything_together(tmp_path: Path) -> None:
    source_model = _make_model()
    source_optimizer = _make_optimizer(source_model)
    source_scheduler = _make_scheduler(source_optimizer)
    for _ in range(2):
        _train_one_step(source_model, source_optimizer)
        source_scheduler.step()

    path = tmp_path / "full.pt"
    save_checkpoint(
        source_model,
        path,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        epoch=10,
        history={"loss": [1.0, 0.8]},
        metadata={"run_name": "test-run"},
    )

    target_model = _TinyModel()
    target_optimizer = _make_optimizer(target_model)
    target_scheduler = _make_scheduler(target_optimizer)
    result = load_checkpoint(
        path, target_model, optimizer=target_optimizer, scheduler=target_scheduler
    )

    _assert_state_dicts_equal(source_model.state_dict(), target_model.state_dict())
    _assert_state_dicts_equal(source_optimizer.state_dict(), target_optimizer.state_dict())
    _assert_state_dicts_equal(source_scheduler.state_dict(), target_scheduler.state_dict())
    assert result.epoch == 10
    assert result.history == {"loss": [1.0, 0.8]}
    assert result.metadata == {"run_name": "test-run"}

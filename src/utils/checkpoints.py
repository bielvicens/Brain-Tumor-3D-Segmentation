"""Checkpoint management: saving and restoring PyTorch training state.

This module centralizes everything needed to persist and later restore a
model's (and optionally an optimizer's and scheduler's) state, plus
lightweight training bookkeeping (``epoch``, ``history``, ``metadata``).

It is intentionally independent of the Trainer, the Dataset, and the
Predictor - it has no training loop, no data loading, and no inference
logic. Its only job is to serialize/deserialize state with
``torch.save``/``torch.load``. Early stopping, learning-rate scheduling
policy, automatic best-checkpoint selection, multi-checkpoint management,
and export formats other than PyTorch's own (ONNX, TorchScript) are all
out of scope here; each is a natural extension point for a later module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)

#: Keys always present in every checkpoint written by save_checkpoint.
_REQUIRED_CHECKPOINT_KEYS = ("model_state_dict",)


@dataclass
class CheckpointData:
    """Information recovered from a loaded checkpoint.

    Attributes:
        epoch: The epoch the checkpoint was saved at, if any.
        history: Training history (e.g. per-epoch loss/metric lists), as
            stored by whoever called :func:`save_checkpoint`.
        metadata: Free-form additional information, as stored by whoever
            called :func:`save_checkpoint`.
    """

    epoch: Optional[int]
    history: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Validation helpers (private)
# ----------------------------------------------------------------------
def _validate_model(model: torch.nn.Module) -> None:
    """Check that `model` is a real torch.nn.Module."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}.")


def _validate_path_type(path: Union[str, Path]) -> Path:
    """Check that `path` is a str or Path, and return it as a Path."""
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}.")
    return Path(path)


def _validate_optimizer(optimizer: Optional[torch.optim.Optimizer]) -> None:
    """Check that `optimizer`, if given, is a real torch.optim.Optimizer."""
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            f"optimizer must be a torch.optim.Optimizer or None, got {type(optimizer).__name__}."
        )


def _validate_scheduler(scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]) -> None:
    """Check that `scheduler`, if given, is a real LRScheduler."""
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        raise TypeError(
            "scheduler must be a torch.optim.lr_scheduler.LRScheduler or None, "
            f"got {type(scheduler).__name__}."
        )


def _validate_epoch(epoch: Optional[int]) -> None:
    """Check that `epoch`, if given, is a non-negative integer."""
    if epoch is None:
        return
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise TypeError(f"epoch must be a non-negative integer or None, got {epoch!r}.")


def _validate_dict_arg(value: Optional[Dict[str, Any]], name: str) -> None:
    """Check that `value`, if given, is a dict."""
    if value is not None and not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict or None, got {type(value).__name__}.")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def save_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    epoch: Optional[int] = None,
    history: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint with ``torch.save``.

    Always saves ``model_state_dict``, ``epoch``, ``history`` and
    ``metadata``; additionally saves ``optimizer_state_dict`` and/or
    ``scheduler_state_dict`` when the corresponding object is given. Any
    existing file at ``path`` is overwritten, and parent directories are
    created automatically if they don't exist yet.

    Args:
        model: The model whose weights to save.
        path: Destination file path.
        optimizer: Optimizer whose state to save, if any.
        scheduler: LR scheduler whose state to save, if any.
        epoch: The epoch this checkpoint corresponds to, if any.
        history: Training history to store alongside the weights (e.g.
            per-epoch loss/metric values). Stored as-is (defaults to an
            empty dict if not given).
        metadata: Free-form additional information to store (e.g. config,
            git commit hash, dataset version). Stored as-is (defaults to
            an empty dict if not given).

    Raises:
        TypeError: If any argument has the wrong type.

    Note:
        Only ``model.state_dict()`` / ``optimizer.state_dict()`` /
        ``scheduler.state_dict()`` are read here - ``model``,
        ``optimizer``, ``scheduler``, ``history`` and ``metadata`` are
        never modified.
    """
    _validate_model(model)
    resolved_path = _validate_path_type(path)
    _validate_optimizer(optimizer)
    _validate_scheduler(scheduler)
    _validate_epoch(epoch)
    _validate_dict_arg(history, "history")
    _validate_dict_arg(metadata, "metadata")

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "history": history if history is not None else {},
        "metadata": metadata if metadata is not None else {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, resolved_path)

    logger.info(
        "Saved checkpoint to '%s' (epoch=%s, optimizer=%s, scheduler=%s).",
        resolved_path,
        epoch,
        optimizer is not None,
        scheduler is not None,
    )


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> CheckpointData:
    """Load a training checkpoint with ``torch.load`` and restore state.

    Restores ``model``'s weights unconditionally. Also restores
    ``optimizer``'s and/or ``scheduler``'s state if the corresponding
    object is given (and the checkpoint actually has state saved for it).

    Args:
        path: Path to a checkpoint file previously written by
            :func:`save_checkpoint`.
        model: Model to restore weights into.
        optimizer: Optimizer to restore state into, if any.
        scheduler: LR scheduler to restore state into, if any.
        map_location: Passed straight through to ``torch.load`` - e.g.
            ``"cpu"`` to load a GPU-trained checkpoint on a CPU-only
            machine.

    Returns:
        A :class:`CheckpointData` with the checkpoint's ``epoch``,
        ``history`` and ``metadata``.

    Raises:
        TypeError: If any argument has the wrong type.
        FileNotFoundError: If ``path`` doesn't point to an existing file.
        ValueError: If the file can't be unpickled as a checkpoint, isn't
            a dict, is missing ``model_state_dict``, or ``optimizer``/
            ``scheduler`` was given but the checkpoint has no matching
            saved state.

    Note:
        Loads with ``weights_only=False``, so arbitrary Python objects in
        ``history``/``metadata`` round-trip correctly (the ``Any`` in
        their type hints is taken literally). Only load checkpoint files
        you trust, the same way you'd treat any other pickle file.
    """
    _validate_model(model)
    resolved_path = _validate_path_type(path)
    _validate_optimizer(optimizer)
    _validate_scheduler(scheduler)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: '{resolved_path}'.")

    try:
        checkpoint = torch.load(resolved_path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"Failed to load checkpoint from '{resolved_path}': the file appears to be "
            "corrupted or is not a valid torch checkpoint."
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint at '{resolved_path}' does not contain a dict "
            f"(got {type(checkpoint).__name__}); it wasn't saved with save_checkpoint()."
        )

    missing_keys = [key for key in _REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            f"Checkpoint at '{resolved_path}' is missing required key(s): {missing_keys}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint at '{resolved_path}' has no 'optimizer_state_dict', but an "
                "optimizer was provided to restore state into."
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        if "scheduler_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint at '{resolved_path}' has no 'scheduler_state_dict', but a "
                "scheduler was provided to restore state into."
            )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        "Loaded checkpoint from '%s' (epoch=%s, optimizer_restored=%s, scheduler_restored=%s).",
        resolved_path,
        checkpoint.get("epoch"),
        optimizer is not None,
        scheduler is not None,
    )

    return CheckpointData(
        epoch=checkpoint.get("epoch"),
        history=checkpoint.get("history", {}),
        metadata=checkpoint.get("metadata", {}),
    )

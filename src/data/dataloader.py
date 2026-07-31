"""DataLoader factory for :class:`BraTSDataset`.

This module contains only PyTorch DataLoader configuration. Dataset loading and
preprocessing remain the responsibility of ``BraTSDataset`` and its pipeline.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import BraTSDataset


DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 0


def create_dataloader(
    dataset: BraTSDataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: Optional[int] = None,
) -> DataLoader:
    """Create a PyTorch DataLoader for a ``BraTSDataset``.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of patients per batch.
        shuffle: Whether to shuffle patient order each epoch.
        num_workers: Number of worker processes used by PyTorch.
        pin_memory: Whether batches are allocated in pinned host memory.
        drop_last: Drop the final incomplete batch when ``True``.
        seed: Optional seed for the DataLoader's generator. This controls
            deterministic shuffling when ``shuffle=True``.

    Returns:
        Configured PyTorch ``DataLoader``.

    Raises:
        TypeError: If ``dataset`` is not a ``BraTSDataset``.
        ValueError: If ``batch_size`` or ``num_workers`` is invalid.
    """
    if not isinstance(dataset, BraTSDataset):
        raise TypeError(
            f"dataset must be a BraTSDataset, got {type(dataset).__name__}."
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be an integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )

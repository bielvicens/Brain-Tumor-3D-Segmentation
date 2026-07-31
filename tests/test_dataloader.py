from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataloader import create_dataloader
from src.data.dataset import BraTSDataset


class _TinyDataset(BraTSDataset):
    """Test-only dataset avoiding filesystem access."""

    def __init__(self, length: int = 5) -> None:
        self.patient_ids = [f"patient-{i}" for i in range(length)]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int):
        return torch.tensor([index]), torch.tensor(index)


def test_create_dataloader_returns_pytorch_dataloader() -> None:
    loader = create_dataloader(_TinyDataset())
    assert isinstance(loader, DataLoader)


def test_dataloader_batch_size_and_length() -> None:
    loader = create_dataloader(_TinyDataset(5), batch_size=2)
    assert loader.batch_size == 2
    assert len(loader) == 3


def test_dataloader_preserves_dataset_order_when_shuffle_disabled() -> None:
    loader = create_dataloader(_TinyDataset(4), batch_size=2, shuffle=False)
    values = torch.cat([batch[0].flatten() for batch in loader]).tolist()
    assert values == [0, 1, 2, 3]


def test_dataloader_shuffle_is_reproducible_with_seed() -> None:
    loader_a = create_dataloader(_TinyDataset(10), batch_size=1, shuffle=True, seed=42)
    loader_b = create_dataloader(_TinyDataset(10), batch_size=1, shuffle=True, seed=42)

    order_a = [int(batch[0].item()) for batch in loader_a]
    order_b = [int(batch[0].item()) for batch in loader_b]
    assert order_a == order_b


def test_drop_last_drops_incomplete_batch() -> None:
    loader = create_dataloader(_TinyDataset(5), batch_size=2, drop_last=True)
    assert len(loader) == 2


def test_loader_options_are_forwarded() -> None:
    loader = create_dataloader(
        _TinyDataset(),
        batch_size=3,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    assert loader.batch_size == 3
    assert loader.drop_last is True
    assert loader.pin_memory is True


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_raises(batch_size: int) -> None:
    with pytest.raises(ValueError):
        create_dataloader(_TinyDataset(), batch_size=batch_size)


@pytest.mark.parametrize("num_workers", [-1])
def test_invalid_num_workers_raises(num_workers: int) -> None:
    with pytest.raises(ValueError):
        create_dataloader(_TinyDataset(), num_workers=num_workers)


def test_non_dataset_input_raises() -> None:
    with pytest.raises(TypeError):
        create_dataloader([])  # type: ignore[arg-type]

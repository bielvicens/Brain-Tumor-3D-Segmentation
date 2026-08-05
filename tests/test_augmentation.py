from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing import PreprocessingSample
from src.preprocessing.augmentation import RandomFlip


def _build_sample() -> PreprocessingSample:
    volume = np.arange(27).reshape(3, 3, 3).astype(np.float32)

    modalities = {
        modality: volume.copy()
        for modality in (
            Modality.T1N,
            Modality.T1C,
            Modality.T2W,
            Modality.T2F,
        )
    }

    segmentation = np.arange(27).reshape(3, 3, 3).astype(np.int16)

    return PreprocessingSample(
        patient_id="patient",
        modalities=modalities,
        segmentation=segmentation,
    )


def test_random_flip_probability_zero_returns_original() -> None:
    sample = _build_sample()

    transform = RandomFlip(
        probability=0.0,
        axis=0,
    )

    result = transform(sample)

    assert result is sample


def test_random_flip_flips_modalities() -> None:
    sample = _build_sample()

    transform = RandomFlip(
        probability=1.0,
        axis=0,
    )

    result = transform(sample)

    expected = np.flip(
        sample.modalities[Modality.T1N],
        axis=0,
    )

    assert np.array_equal(
        result.modalities[Modality.T1N],
        expected,
    )


def test_random_flip_flips_segmentation() -> None:
    sample = _build_sample()

    transform = RandomFlip(
        probability=1.0,
        axis=2,
    )

    result = transform(sample)

    expected = np.flip(
        sample.segmentation,
        axis=2,
    )

    assert np.array_equal(
        result.segmentation,
        expected,
    )


def test_random_flip_preserves_original_sample() -> None:
    sample = _build_sample()

    original = sample.modalities[Modality.T1N].copy()

    transform = RandomFlip(
        probability=1.0,
        axis=1,
    )

    _ = transform(sample)

    assert np.array_equal(
        sample.modalities[Modality.T1N],
        original,
    )


def test_random_flip_preserves_metadata() -> None:
    sample = _build_sample()

    sample.metadata["source"] = "BraTS"

    transform = RandomFlip(
        probability=1.0,
        axis=0,
    )

    result = transform(sample)

    assert result.metadata["source"] == "BraTS"


def test_random_flip_registers_metadata() -> None:
    sample = _build_sample()

    transform = RandomFlip(
        probability=1.0,
        axis=1,
    )

    result = transform(sample)

    assert "augmentations" in result.metadata

    augmentation = result.metadata["augmentations"][0]

    assert augmentation["name"] == "RandomFlip"
    assert augmentation["axis"] == 1


def test_random_flip_with_seed_is_reproducible() -> None:
    sample = _build_sample()

    transform1 = RandomFlip(
        probability=1.0,
        seed=42,
    )

    transform2 = RandomFlip(
        probability=1.0,
        seed=42,
    )

    result1 = transform1(sample)
    result2 = transform2(sample)

    assert np.array_equal(
        result1.modalities[Modality.T1N],
        result2.modalities[Modality.T1N],
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_random_flip_all_axes(axis: int) -> None:
    sample = _build_sample()

    transform = RandomFlip(
        probability=1.0,
        axis=axis,
    )

    result = transform(sample)

    expected = np.flip(
        sample.modalities[Modality.T1N],
        axis=axis,
    )

    assert np.array_equal(
        result.modalities[Modality.T1N],
        expected,
    )


def test_random_flip_invalid_probability() -> None:
    with pytest.raises(ValueError):
        RandomFlip(probability=1.5)


def test_random_flip_invalid_axis() -> None:
    with pytest.raises(ValueError):
        RandomFlip(axis=4)
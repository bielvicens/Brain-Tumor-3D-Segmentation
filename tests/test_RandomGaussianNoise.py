from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing import (
    PreprocessingSample,
    RandomGaussianNoise,
)


def create_sample() -> PreprocessingSample:
    """Create a simple preprocessing sample for testing."""

    modalities = {
        modality: np.ones((4, 4, 4), dtype=np.float32)
        for modality in (
            Modality.T1N,
            Modality.T1C,
            Modality.T2W,
            Modality.T2F,
        )
    }

    segmentation = np.ones(
        (4, 4, 4),
        dtype=np.int16,
    )

    return PreprocessingSample(
        patient_id="patient",
        modalities=modalities,
        segmentation=segmentation,
    )


# ---------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------


def test_invalid_probability_negative() -> None:
    with pytest.raises(ValueError):
        RandomGaussianNoise(probability=-0.1)


def test_invalid_probability_greater_than_one() -> None:
    with pytest.raises(ValueError):
        RandomGaussianNoise(probability=1.1)


def test_invalid_std_negative() -> None:
    with pytest.raises(ValueError):
        RandomGaussianNoise(std=-0.5)


# ---------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------


def test_probability_zero_returns_same_sample() -> None:
    sample = create_sample()

    transform = RandomGaussianNoise(
        probability=0.0,
        seed=42,
    )

    result = transform(sample)

    for modality in sample.modalities:
        np.testing.assert_array_equal(
            sample.modalities[modality],
            result.modalities[modality],
        )

    np.testing.assert_array_equal(
        sample.segmentation,
        result.segmentation,
    )


def test_probability_one_changes_modalities() -> None:
    sample = create_sample()

    transform = RandomGaussianNoise(
        probability=1.0,
        std=0.2,
        seed=42,
    )

    result = transform(sample)

    changed = False

    for modality in sample.modalities:
        if not np.array_equal(
            sample.modalities[modality],
            result.modalities[modality],
        ):
            changed = True

    assert changed


# ---------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------


def test_segmentation_is_not_modified() -> None:
    sample = create_sample()

    transform = RandomGaussianNoise(
        probability=1.0,
        seed=42,
    )

    result = transform(sample)

    np.testing.assert_array_equal(
        sample.segmentation,
        result.segmentation,
    )


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------


def test_metadata_contains_noise_information() -> None:
    sample = create_sample()

    transform = RandomGaussianNoise(
        probability=1.0,
        mean=0.1,
        std=0.25,
        seed=42,
    )

    result = transform(sample)

    assert "augmentations" in result.metadata

    augmentation = result.metadata["augmentations"][0]

    assert augmentation["name"] == "RandomGaussianNoise"
    assert augmentation["mean"] == 0.1
    assert augmentation["std"] == 0.25


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def test_same_seed_produces_same_result() -> None:
    sample = create_sample()

    transform1 = RandomGaussianNoise(
        probability=1.0,
        std=0.2,
        seed=7,
    )

    transform2 = RandomGaussianNoise(
        probability=1.0,
        std=0.2,
        seed=7,
    )

    result1 = transform1(sample)
    result2 = transform2(sample)

    for modality in sample.modalities:
        np.testing.assert_array_equal(
            result1.modalities[modality],
            result2.modalities[modality],
        )

    np.testing.assert_array_equal(
        result1.segmentation,
        result2.segmentation,
    )


# ---------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------


def test_original_sample_is_not_modified() -> None:
    sample = create_sample()

    original_modalities = {
        modality: volume.copy()
        for modality, volume in sample.modalities.items()
    }

    original_segmentation = sample.segmentation.copy()

    transform = RandomGaussianNoise(
        probability=1.0,
        std=0.2,
        seed=42,
    )

    _ = transform(sample)

    for modality in sample.modalities:
        np.testing.assert_array_equal(
            sample.modalities[modality],
            original_modalities[modality],
        )

    np.testing.assert_array_equal(
        sample.segmentation,
        original_segmentation,
    )


# ---------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------


def test_modalities_remain_float32() -> None:
    sample = create_sample()

    transform = RandomGaussianNoise(
        probability=1.0,
        seed=42,
    )

    result = transform(sample)

    for modality in result.modalities.values():
        assert modality.dtype == np.float32
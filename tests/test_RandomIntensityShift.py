from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing import (
    PreprocessingSample,
    RandomIntensityShift,
)


def create_sample() -> PreprocessingSample:
    """Create a simple preprocessing sample for testing."""

    values = np.linspace(
        -2.0,
        2.0,
        64,
        dtype=np.float32,
    ).reshape(4, 4, 4)

    modalities = {
        modality: values.copy()
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
        RandomIntensityShift(probability=-0.1)


def test_invalid_probability_greater_than_one() -> None:
    with pytest.raises(ValueError):
        RandomIntensityShift(probability=1.1)


def test_invalid_shift_range_order() -> None:
    with pytest.raises(ValueError):
        RandomIntensityShift(
            shift_range=(1.0, -1.0),
        )


# ---------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------


def test_probability_zero_returns_same_sample() -> None:
    sample = create_sample()

    transform = RandomIntensityShift(
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

    transform = RandomIntensityShift(
        probability=1.0,
        shift_range=(0.5, 0.5),
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
# Shift correctness
# ---------------------------------------------------------------------


def test_constant_shift_is_applied() -> None:
    sample = create_sample()

    transform = RandomIntensityShift(
        probability=1.0,
        shift_range=(0.25, 0.25),
        seed=42,
    )

    result = transform(sample)

    for modality in sample.modalities:
        np.testing.assert_allclose(
            result.modalities[modality],
            sample.modalities[modality] + 0.25,
        )


# ---------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------


def test_segmentation_is_not_modified() -> None:
    sample = create_sample()

    transform = RandomIntensityShift(
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


def test_metadata_contains_shift_information() -> None:
    sample = create_sample()

    transform = RandomIntensityShift(
        probability=1.0,
        shift_range=(0.2, 0.2),
        seed=42,
    )

    result = transform(sample)

    assert "augmentations" in result.metadata

    augmentation = result.metadata["augmentations"][0]

    assert augmentation["name"] == "RandomIntensityShift"
    assert augmentation["shift"] == pytest.approx(0.2)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def test_same_seed_produces_same_result() -> None:
    sample = create_sample()

    transform1 = RandomIntensityShift(
        probability=1.0,
        seed=7,
    )

    transform2 = RandomIntensityShift(
        probability=1.0,
        seed=7,
    )

    result1 = transform1(sample)
    result2 = transform2(sample)

    for modality in sample.modalities:
        np.testing.assert_array_equal(
            result1.modalities[modality],
            result2.modalities[modality],
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

    transform = RandomIntensityShift(
        probability=1.0,
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

    transform = RandomIntensityShift(
        probability=1.0,
        seed=42,
    )

    result = transform(sample)

    for modality in result.modalities.values():
        assert modality.dtype == np.float32
from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing import (
    PreprocessingSample,
    RandomRotation90,
)


def create_sample() -> PreprocessingSample:
    """Create a simple preprocessing sample for testing."""

    modalities = {
        modality: np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        for modality in (
            Modality.T1N,
            Modality.T1C,
            Modality.T2W,
            Modality.T2F,
        )
    }

    segmentation = np.arange(
        27,
        dtype=np.int16,
    ).reshape(3, 3, 3)

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
        RandomRotation90(probability=-0.1)


def test_invalid_probability_greater_than_one() -> None:
    with pytest.raises(ValueError):
        RandomRotation90(probability=1.5)


# ---------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------


def test_probability_zero_returns_same_sample() -> None:
    sample = create_sample()

    transform = RandomRotation90(
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


def test_probability_one_applies_rotation() -> None:
    sample = create_sample()

    transform = RandomRotation90(
        probability=1.0,
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
# Consistency
# ---------------------------------------------------------------------


def test_all_modalities_receive_same_rotation() -> None:
    sample = create_sample()

    transform = RandomRotation90(
        probability=1.0,
        seed=123,
    )

    result = transform(sample)

    reference_difference = (
        result.modalities[Modality.T1N]
        - result.segmentation.astype(np.float32)
    )

    for modality in (
        Modality.T1C,
        Modality.T2W,
        Modality.T2F,
    ):
        difference = (
            result.modalities[modality]
            - result.segmentation.astype(np.float32)
        )

        np.testing.assert_array_equal(
            difference,
            reference_difference,
        )


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------


def test_metadata_contains_rotation_information() -> None:
    sample = create_sample()

    transform = RandomRotation90(
        probability=1.0,
        seed=42,
    )

    result = transform(sample)

    assert "augmentations" in result.metadata

    augmentation = result.metadata["augmentations"][0]

    assert augmentation["name"] == "RandomRotation90"
    assert augmentation["k"] in (1, 2, 3)
    assert augmentation["axes"] in (
        (0, 1),
        (0, 2),
        (1, 2),
    )


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def test_same_seed_produces_same_result() -> None:
    sample = create_sample()

    transform1 = RandomRotation90(
        probability=1.0,
        seed=7,
    )

    transform2 = RandomRotation90(
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

    transform = RandomRotation90(
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
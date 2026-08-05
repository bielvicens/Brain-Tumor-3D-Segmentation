from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing import (
    PreprocessingSample,
    RandomGamma,
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
        RandomGamma(probability=-0.1)


def test_invalid_probability_greater_than_one() -> None:
    with pytest.raises(ValueError):
        RandomGamma(probability=1.1)


def test_invalid_gamma_minimum() -> None:
    with pytest.raises(ValueError):
        RandomGamma(gamma_range=(0.0, 1.5))


def test_invalid_gamma_range_order() -> None:
    with pytest.raises(ValueError):
        RandomGamma(gamma_range=(2.0, 1.0))


# ---------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------


def test_probability_zero_returns_same_sample() -> None:
    sample = create_sample()

    transform = RandomGamma(
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

    transform = RandomGamma(
        probability=1.0,
        gamma_range=(0.5, 0.5),
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

    transform = RandomGamma(
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


def test_metadata_contains_gamma_information() -> None:
    sample = create_sample()

    transform = RandomGamma(
        probability=1.0,
        gamma_range=(1.2, 1.2),
        seed=42,
    )

    result = transform(sample)

    assert "augmentations" in result.metadata

    augmentation = result.metadata["augmentations"][0]

    assert augmentation["name"] == "RandomGamma"
    assert augmentation["gamma"] == pytest.approx(1.2)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def test_same_seed_produces_same_result() -> None:
    sample = create_sample()

    transform1 = RandomGamma(
        probability=1.0,
        seed=7,
    )

    transform2 = RandomGamma(
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
# Constant images
# ---------------------------------------------------------------------


def test_constant_image_is_left_unchanged() -> None:
    constant = np.ones(
        (4, 4, 4),
        dtype=np.float32,
    )

    sample = PreprocessingSample(
        patient_id="patient",
        modalities={
            modality: constant.copy()
            for modality in (
                Modality.T1N,
                Modality.T1C,
                Modality.T2W,
                Modality.T2F,
            )
        },
        segmentation=np.ones(
            (4, 4, 4),
            dtype=np.int16,
        ),
    )

    transform = RandomGamma(
        probability=1.0,
        seed=42,
    )

    result = transform(sample)

    for modality in sample.modalities:
        np.testing.assert_array_equal(
            sample.modalities[modality],
            result.modalities[modality],
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

    transform = RandomGamma(
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

    transform = RandomGamma(
        probability=1.0,
        seed=42,
    )

    result = transform(sample)

    for modality in result.modalities.values():
        assert modality.dtype == np.float32
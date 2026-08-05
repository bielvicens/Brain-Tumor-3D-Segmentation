from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    CenterCrop3D,
    PreprocessingSample,
)
from src.preprocessing.exceptions import InvalidVolumeError
from src.data import MRI_MODALITIES


@pytest.fixture
def sample() -> PreprocessingSample:
    """Create a synthetic preprocessing sample."""

    volume = np.random.rand(
        240,
        240,
        155,
    ).astype(np.float32)

    segmentation = np.random.randint(
        0,
        4,
        size=(240, 240, 155),
        dtype=np.uint8,
    )

    return PreprocessingSample(
        patient_id="patient",
        modalities={
            modality: volume.copy()
            for modality in MRI_MODALITIES
        },
        segmentation=segmentation,
        voxel_spacing=(1.0, 1.0, 1.0),
    )


def test_center_crop_preserves_crop_shape(
    sample: PreprocessingSample,
) -> None:

    transform = CenterCrop3D(
        crop_size=(128, 128, 128),
    )

    result = transform(sample)

    for volume in result.modalities.values():
        assert volume.shape == (
            128,
            128,
            128,
        )

    assert result.segmentation is not None
    assert result.segmentation.shape == (
        128,
        128,
        128,
    )


def test_center_crop_is_deterministic(
    sample: PreprocessingSample,
) -> None:

    transform = CenterCrop3D(
        crop_size=(128, 128, 128),
    )

    result1 = transform(sample)
    result2 = transform(sample)

    for modality in result1.modalities:

        np.testing.assert_array_equal(
            result1.modalities[modality],
            result2.modalities[modality],
        )

    np.testing.assert_array_equal(
        result1.segmentation,
        result2.segmentation,
    )


def test_center_crop_larger_than_volume_raises(
    sample: PreprocessingSample,
) -> None:

    transform = CenterCrop3D(
        crop_size=(300, 300, 300),
    )

    with pytest.raises(InvalidVolumeError):
        transform(sample)


@pytest.mark.parametrize(
    "crop_size",
    [
        (),
        (128,),
        (128, 128),
        (128, 128, 128, 128),
        (-1, 128, 128),
        (0, 128, 128),
    ],
)
def test_center_crop_invalid_crop_size_raises(
    crop_size,
) -> None:

    with pytest.raises(ValueError):
        CenterCrop3D(
            crop_size=crop_size,
        )


def test_center_crop_preserves_alignment() -> None:
    """All modalities and segmentation must receive the same crop."""

    volume = np.arange(
        240 * 240 * 155,
        dtype=np.float32,
    ).reshape(
        240,
        240,
        155,
    )

    sample = PreprocessingSample(
        patient_id="patient",
        modalities={
            modality: volume.copy()
            for modality in MRI_MODALITIES
        },
        segmentation=volume.astype(np.uint8),
        voxel_spacing=(1.0, 1.0, 1.0),
    )

    transform = CenterCrop3D(
        crop_size=(128, 128, 128),
    )

    result = transform(sample)

    reference = result.modalities[MRI_MODALITIES[0]]

    for modality in result.modalities.values():

        np.testing.assert_array_equal(
            modality,
            reference,
        )

    np.testing.assert_array_equal(
        result.segmentation,
        reference.astype(np.uint8),
    )
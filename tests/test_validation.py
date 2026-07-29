"""Unit tests for src.preprocessing.validation.

Every check is pure, so tests build PreprocessingSample instances
directly with small in-memory numpy arrays - no filesystem, no reader.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing.exceptions import InvalidVolumeError
from src.preprocessing.transforms import PreprocessingSample
from src.preprocessing.validation import (
    validate_arrays_are_numpy,
    validate_modalities_present,
    validate_no_nan_or_inf,
    validate_non_empty_arrays,
    validate_sample,
    validate_shapes_consistent,
    validate_voxel_spacing,
)


def _sample(**overrides) -> PreprocessingSample:
    defaults = dict(
        patient_id="patient-0",
        modalities={Modality.T1N: np.ones((2, 2, 2), dtype=np.float32)},
    )
    defaults.update(overrides)
    return PreprocessingSample(**defaults)


# ----------------------------------------------------------------------
# validate_modalities_present
# ----------------------------------------------------------------------
def test_validate_modalities_present_passes_with_modalities() -> None:
    validate_modalities_present(_sample())  # should not raise


def test_validate_modalities_present_raises_when_empty() -> None:
    with pytest.raises(InvalidVolumeError, match="no modalities"):
        validate_modalities_present(_sample(modalities={}))


# ----------------------------------------------------------------------
# validate_arrays_are_numpy
# ----------------------------------------------------------------------
def test_validate_arrays_are_numpy_raises_for_non_ndarray_modality() -> None:
    bad_sample = _sample(modalities={Modality.T1N: [1, 2, 3]})  # a plain list
    with pytest.raises(InvalidVolumeError, match="not a numpy array"):
        validate_arrays_are_numpy(bad_sample)


def test_validate_arrays_are_numpy_raises_for_non_ndarray_segmentation() -> None:
    bad_sample = _sample(segmentation=[0, 1, 1])
    with pytest.raises(InvalidVolumeError, match="segmentation"):
        validate_arrays_are_numpy(bad_sample)


# ----------------------------------------------------------------------
# validate_non_empty_arrays
# ----------------------------------------------------------------------
def test_validate_non_empty_arrays_raises_for_empty_modality() -> None:
    bad_sample = _sample(modalities={Modality.T1N: np.empty((0,), dtype=np.float32)})
    with pytest.raises(InvalidVolumeError, match="empty"):
        validate_non_empty_arrays(bad_sample)


# ----------------------------------------------------------------------
# validate_shapes_consistent
# ----------------------------------------------------------------------
def test_validate_shapes_consistent_passes_when_shapes_match() -> None:
    sample = _sample(
        modalities={
            Modality.T1N: np.ones((4, 4, 4), dtype=np.float32),
            Modality.T1C: np.ones((4, 4, 4), dtype=np.float32),
        },
        segmentation=np.zeros((4, 4, 4), dtype=np.int16),
    )
    validate_shapes_consistent(sample)  # should not raise


def test_validate_shapes_consistent_raises_on_mismatch_between_modalities() -> None:
    sample = _sample(
        modalities={
            Modality.T1N: np.ones((4, 4, 4), dtype=np.float32),
            Modality.T1C: np.ones((5, 5, 5), dtype=np.float32),
        }
    )
    with pytest.raises(InvalidVolumeError, match="mismatched volume shapes"):
        validate_shapes_consistent(sample)


def test_validate_shapes_consistent_raises_on_mismatch_with_segmentation() -> None:
    sample = _sample(
        modalities={Modality.T1N: np.ones((4, 4, 4), dtype=np.float32)},
        segmentation=np.zeros((3, 3, 3), dtype=np.int16),
    )
    with pytest.raises(InvalidVolumeError, match="mismatched volume shapes"):
        validate_shapes_consistent(sample)


# ----------------------------------------------------------------------
# validate_no_nan_or_inf
# ----------------------------------------------------------------------
def test_validate_no_nan_or_inf_passes_for_clean_data() -> None:
    validate_no_nan_or_inf(_sample())  # should not raise


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_validate_no_nan_or_inf_raises_for_bad_values(bad_value: float) -> None:
    volume = np.ones((2, 2, 2), dtype=np.float32)
    volume[0, 0, 0] = bad_value
    sample = _sample(modalities={Modality.T1N: volume})
    with pytest.raises(InvalidVolumeError, match="NaN or infinite"):
        validate_no_nan_or_inf(sample)


# ----------------------------------------------------------------------
# validate_voxel_spacing
# ----------------------------------------------------------------------
def test_validate_voxel_spacing_passes_when_none() -> None:
    validate_voxel_spacing(_sample(voxel_spacing=None))  # should not raise


def test_validate_voxel_spacing_passes_for_valid_spacing() -> None:
    validate_voxel_spacing(_sample(voxel_spacing=(1.0, 1.0, 1.0)))  # should not raise


def test_validate_voxel_spacing_raises_for_wrong_length() -> None:
    with pytest.raises(InvalidVolumeError, match="3 components"):
        validate_voxel_spacing(_sample(voxel_spacing=(1.0, 1.0)))


@pytest.mark.parametrize("bad_spacing", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (np.nan, 1.0, 1.0)])
def test_validate_voxel_spacing_raises_for_non_positive_or_non_finite(bad_spacing) -> None:
    with pytest.raises(InvalidVolumeError, match="strictly positive and finite"):
        validate_voxel_spacing(_sample(voxel_spacing=bad_spacing))


# ----------------------------------------------------------------------
# validate_sample (orchestration)
# ----------------------------------------------------------------------
def test_validate_sample_passes_for_a_valid_sample() -> None:
    sample = _sample(
        modalities={
            Modality.T1N: np.ones((4, 4, 4), dtype=np.float32),
            Modality.T1C: np.ones((4, 4, 4), dtype=np.float32),
        },
        segmentation=np.zeros((4, 4, 4), dtype=np.int16),
        voxel_spacing=(1.0, 1.0, 1.0),
    )
    validate_sample(sample)  # should not raise


def test_validate_sample_stops_at_first_failing_check() -> None:
    # No modalities at all - the very first check should fail with its
    # own specific message, not some later, less relevant one.
    with pytest.raises(InvalidVolumeError, match="no modalities"):
        validate_sample(_sample(modalities={}))


def test_validate_sample_can_run_a_custom_subset_of_checks() -> None:
    # A sample with mismatched shapes would normally fail
    # validate_shapes_consistent, but we only run one unrelated check.
    sample = _sample(
        modalities={
            Modality.T1N: np.ones((4, 4, 4), dtype=np.float32),
            Modality.T1C: np.ones((5, 5, 5), dtype=np.float32),
        }
    )
    validate_sample(sample, checks=[validate_modalities_present])  # should not raise

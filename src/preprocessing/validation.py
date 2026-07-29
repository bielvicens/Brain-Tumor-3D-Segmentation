"""Validation checks run on a PreprocessingSample before preprocessing.

Every function here is pure (no I/O, no dependency on the reader or on
any Transform) and raises :class:`~src.preprocessing.exceptions.InvalidVolumeError`
with a specific, actionable message on failure. This mirrors the
``statistics`` module from the EDA package (Module 2): small, composable,
independently-testable functions rather than one monolithic check.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Sequence

import numpy as np

from .exceptions import InvalidVolumeError
from .transforms import PreprocessingSample

logger = logging.getLogger(__name__)


def validate_modalities_present(sample: PreprocessingSample) -> None:
    """Check that the sample has at least one MRI modality."""
    if not sample.modalities:
        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}' has no modalities to preprocess."
        )


def validate_arrays_are_numpy(sample: PreprocessingSample) -> None:
    """Check that every modality (and the segmentation, if present) is a
    real numpy array - not e.g. a list or a lazily-loaded nibabel image."""
    for modality, volume in sample.modalities.items():
        if not isinstance(volume, np.ndarray):
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': modality '{modality.value}' "
                f"is a {type(volume).__name__}, not a numpy array."
            )
    if sample.segmentation is not None and not isinstance(sample.segmentation, np.ndarray):
        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}': segmentation is a "
            f"{type(sample.segmentation).__name__}, not a numpy array."
        )


def validate_non_empty_arrays(sample: PreprocessingSample) -> None:
    """Check that no modality (or the segmentation) is an empty array."""
    for modality, volume in sample.modalities.items():
        if volume.size == 0:
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': modality '{modality.value}' is empty."
            )
    if sample.segmentation is not None and sample.segmentation.size == 0:
        raise InvalidVolumeError(f"Patient '{sample.patient_id}': segmentation array is empty.")


def validate_shapes_consistent(sample: PreprocessingSample) -> None:
    """Check every modality (and the segmentation, if present) shares the
    same shape - required for any per-voxel operation across modalities."""
    shapes = {modality.value: volume.shape for modality, volume in sample.modalities.items()}
    if sample.segmentation is not None:
        shapes["segmentation"] = sample.segmentation.shape

    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        details = ", ".join(f"{key}={shape}" for key, shape in shapes.items())
        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}' has mismatched volume shapes: {details}."
        )


def validate_no_nan_or_inf(sample: PreprocessingSample) -> None:
    """Check that no modality contains NaN or infinite values."""
    for modality, volume in sample.modalities.items():
        if not np.all(np.isfinite(volume)):
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': modality '{modality.value}' "
                "contains NaN or infinite values."
            )


def validate_voxel_spacing(sample: PreprocessingSample) -> None:
    """Check that voxel spacing, if provided, is a positive, finite 3-tuple."""
    if sample.voxel_spacing is None:
        return
    if len(sample.voxel_spacing) != 3:
        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}': voxel_spacing must have 3 "
            f"components, got {sample.voxel_spacing}."
        )
    if any((value <= 0 or not np.isfinite(value)) for value in sample.voxel_spacing):
        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}': voxel_spacing must be strictly "
            f"positive and finite, got {sample.voxel_spacing}."
        )


#: Default checks run by `validate_sample`, in order. Each takes the
#: sample and raises `InvalidVolumeError` on failure.
DEFAULT_CHECKS: List[Callable[[PreprocessingSample], None]] = [
    validate_modalities_present,
    validate_arrays_are_numpy,
    validate_non_empty_arrays,
    validate_shapes_consistent,
    validate_no_nan_or_inf,
    validate_voxel_spacing,
]


def validate_sample(
    sample: PreprocessingSample,
    checks: Sequence[Callable[[PreprocessingSample], None]] = DEFAULT_CHECKS,
) -> None:
    """Run validation checks against a sample, stopping at the first failure.

    Args:
        sample: The sample to validate.
        checks: Ordered sequence of validation functions. Defaults to
            :data:`DEFAULT_CHECKS`; override to run a subset (e.g. skip
            the NaN/Inf check on data already known to be clean, for
            speed on a large dataset).

    Raises:
        InvalidVolumeError: On the first check that fails, with a message
            identifying exactly what's wrong.
    """
    for check in checks:
        logger.debug(
            "Running validation check '%s' for patient '%s'.", check.__name__, sample.patient_id
        )
        check(sample)
    logger.debug("Patient '%s' passed all validation checks.", sample.patient_id)

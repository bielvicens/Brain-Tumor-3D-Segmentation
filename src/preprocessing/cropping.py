"""Center-cropping transform for BraTS preprocessing.

This module implements the concrete :class:`CroppingTransform` on top of the
Module 3.1 preprocessing contract. It crops every modality and the
segmentation with exactly the same spatial slices, preserving voxel
alignment.

Cropping is intentionally limited to volumes that are at least as large as
the requested target shape. Padding belongs to the following preprocessing
module and is never performed implicitly here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .exceptions import InvalidVolumeError
from .transforms import PreprocessingSample, Transform
from .validation import validate_shapes_consistent


@dataclass(frozen=True)
class CropInfo:
    """Record of one center-cropping operation.

    Stored in ``sample.metadata["cropping"]``.
    """

    original_shape: Tuple[int, ...]
    target_shape: Tuple[int, ...]
    start_indices: Tuple[int, ...]
    end_indices: Tuple[int, ...]


class CroppingTransform(Transform):
    """Center-crop all arrays in a preprocessing sample to a fixed shape.

    The same spatial crop is applied to every MRI modality and to the
    segmentation mask, guaranteeing voxel-wise alignment.

    If the requested target shape equals the current shape, the operation is
    a no-op on values but still returns a new sample and records crop
    metadata. If any target dimension is larger than the corresponding input
    dimension, the transform raises ``InvalidVolumeError`` rather than
    silently padding; padding is the responsibility of ``PaddingTransform``.

    The crop is centered. For an odd number of voxels to remove, the extra
    voxel is removed from the end of the axis (e.g. 10 -> 7 gives start=1,
    end=8).
    """

    def __init__(self, target_shape: Sequence[int]) -> None:
        self.target_shape = self._validate_target_shape(target_shape)

    @staticmethod
    def _validate_target_shape(target_shape: Sequence[int]) -> Tuple[int, ...]:
        try:
            values = tuple(int(v) for v in target_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"target_shape must be an iterable of positive integers, got {target_shape!r}."
            ) from exc

        if len(values) != 3:
            raise ValueError(
                f"target_shape must have exactly 3 dimensions, got {values!r}."
            )

        if any(value <= 0 for value in values):
            raise ValueError(
                f"target_shape values must be strictly positive, got {values!r}."
            )

        # Reject values such as 3.5 instead of silently truncating them via int().
        try:
            original_values = tuple(target_shape)
            if any(float(v) != int(v) for v in original_values):
                raise ValueError(
                    f"target_shape values must be integers, got {target_shape!r}."
                )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "must be integers" in str(exc):
                raise
            raise ValueError(
                f"target_shape must contain integer values, got {target_shape!r}."
            ) from exc

        return values

    def validate_input(self, sample: PreprocessingSample) -> None:
        """Validate that all arrays exist, agree in shape, and can be cropped."""
        validate_shapes_consistent(sample)

        if not sample.modalities and sample.segmentation is None:
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': cannot crop a sample with "
                "no modalities and no segmentation."
            )

        original_shape = self._reference_shape(sample)

        if len(original_shape) != len(self.target_shape):
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': target shape {self.target_shape} "
                f"has {len(self.target_shape)} dimensions but input shape "
                f"{original_shape} has {len(original_shape)}."
            )

        too_small = [
            (axis, old, target)
            for axis, (old, target) in enumerate(
                zip(original_shape, self.target_shape)
            )
            if target > old
        ]
        if too_small:
            details = ", ".join(
                f"axis {axis}: input={old}, target={target}"
                for axis, old, target in too_small
            )
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': target crop shape "
                f"{self.target_shape} is larger than input shape "
                f"{original_shape} ({details}). Use PaddingTransform before "
                "cropping when the requested output is larger than the input."
            )

    def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
        """Return a new sample containing the centered crop.

        The input sample and its metadata are never mutated. Array data is
        copied into new arrays, including the no-op case, so callers can
        safely modify the returned arrays without modifying the original
        sample.
        """
        original_shape = self._reference_shape(sample)
        start_indices = tuple(
            (old - target) // 2
            for old, target in zip(original_shape, self.target_shape)
        )
        end_indices = tuple(
            start + target
            for start, target in zip(start_indices, self.target_shape)
        )
        slices = tuple(
            slice(start, end)
            for start, end in zip(start_indices, end_indices)
        )

        cropped_modalities: Dict = {
            modality: np.array(volume[slices], copy=True)
            for modality, volume in sample.modalities.items()
        }

        cropped_segmentation = (
            np.array(sample.segmentation[slices], copy=True)
            if sample.segmentation is not None
            else None
        )

        crop_info = CropInfo(
            original_shape=tuple(original_shape),
            target_shape=self.target_shape,
            start_indices=start_indices,
            end_indices=end_indices,
        )

        updated_metadata = dict(sample.metadata)
        updated_metadata["cropping"] = crop_info

        return sample.replace(
            modalities=cropped_modalities,
            segmentation=cropped_segmentation,
            metadata=updated_metadata,
        )

    @staticmethod
    def _reference_shape(sample: PreprocessingSample) -> Tuple[int, ...]:
        if sample.modalities:
            return tuple(next(iter(sample.modalities.values())).shape)
        if sample.segmentation is not None:
            return tuple(sample.segmentation.shape)
        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}': cannot crop a sample with "
            "no modalities and no segmentation."
        )

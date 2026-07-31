"""Center-padding transform for BraTS preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .exceptions import InvalidVolumeError
from .transforms import PreprocessingSample, Transform
from .validation import validate_shapes_consistent


@dataclass(frozen=True)
class PaddingInfo:
    """Record of one center-padding operation."""

    original_shape: Tuple[int, ...]
    target_shape: Tuple[int, ...]
    pad_before: Tuple[int, ...]
    pad_after: Tuple[int, ...]


class PaddingTransform(Transform):
    """Center-pad all arrays in a sample to a fixed spatial shape.

    The same padding is applied to every MRI modality and the segmentation,
    preserving voxel-wise alignment. Padding values are zero.

    If the requested target shape equals the current shape, values are
    preserved and new arrays are returned. Padding never crops: a target
    dimension smaller than the input raises ``InvalidVolumeError``.

    For an odd number of voxels to add, the extra voxel is added to the end
    of the corresponding axis.
    """

    def __init__(self, target_shape: Sequence[int]) -> None:
        self.target_shape = self._validate_target_shape(target_shape)

    @staticmethod
    def _validate_target_shape(target_shape: Sequence[int]) -> Tuple[int, ...]:
        try:
            raw = tuple(target_shape)
        except TypeError as exc:
            raise ValueError(
                f"target_shape must be an iterable of positive integers, got {target_shape!r}."
            ) from exc

        if len(raw) != 3:
            raise ValueError(
                f"target_shape must have exactly 3 dimensions, got {raw!r}."
            )

        values = []
        for value in raw:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"target_shape must contain integer values, got {target_shape!r}."
                ) from exc
            if not numeric.is_integer():
                raise ValueError(
                    f"target_shape must contain integer values, got {target_shape!r}."
                )
            integer = int(numeric)
            if integer <= 0:
                raise ValueError(
                    f"target_shape values must be strictly positive, got {target_shape!r}."
                )
            values.append(integer)

        return tuple(values)

    def validate_input(self, sample: PreprocessingSample) -> None:
        """Validate that the sample can be padded to the requested shape."""
        validate_shapes_consistent(sample)

        if not sample.modalities and sample.segmentation is None:
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': cannot pad a sample with "
                "no modalities and no segmentation."
            )

        original_shape = self._reference_shape(sample)

        if len(original_shape) != len(self.target_shape):
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': target shape {self.target_shape} "
                f"has {len(self.target_shape)} dimensions but input shape "
                f"{original_shape} has {len(original_shape)}."
            )

        too_large = [
            (axis, old, target)
            for axis, (old, target) in enumerate(
                zip(original_shape, self.target_shape)
            )
            if old > target
        ]
        if too_large:
            details = ", ".join(
                f"axis {axis}: input={old}, target={target}"
                for axis, old, target in too_large
            )
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': target padding shape "
                f"{self.target_shape} is smaller than input shape "
                f"{original_shape} ({details}). Use CroppingTransform first "
                "when a smaller output is required."
            )

    def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
        """Return a new sample containing the centered zero-padded arrays."""
        original_shape = self._reference_shape(sample)

        pad_before = tuple(
            (target - old) // 2
            for old, target in zip(original_shape, self.target_shape)
        )
        pad_after = tuple(
            target - old - before
            for old, target, before in zip(
                original_shape, self.target_shape, pad_before
            )
        )

        pad_width = tuple(zip(pad_before, pad_after))

        padded_modalities: Dict = {
            modality: np.pad(
                volume,
                pad_width,
                mode="constant",
                constant_values=0,
            )
            for modality, volume in sample.modalities.items()
        }

        padded_segmentation = (
            np.pad(
                sample.segmentation,
                pad_width,
                mode="constant",
                constant_values=0,
            )
            if sample.segmentation is not None
            else None
        )

        padding_info = PaddingInfo(
            original_shape=tuple(original_shape),
            target_shape=self.target_shape,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        updated_metadata = dict(sample.metadata)
        updated_metadata["padding"] = padding_info

        return sample.replace(
            modalities=padded_modalities,
            segmentation=padded_segmentation,
            metadata=updated_metadata,
        )

    @staticmethod
    def _reference_shape(sample: PreprocessingSample) -> Tuple[int, ...]:
        if sample.modalities:
            return tuple(next(iter(sample.modalities.values())).shape)
        if sample.segmentation is not None:
            return tuple(sample.segmentation.shape)
        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}': cannot pad a sample with "
            "no modalities and no segmentation."
        )

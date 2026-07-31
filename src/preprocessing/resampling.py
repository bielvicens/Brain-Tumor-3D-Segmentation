"""Spatial resampling / voxel-spacing standardization for BraTS volumes.

This module holds a single concrete ``Transform``: :class:`ResamplingTransform`,
which converts a :class:`~src.preprocessing.transforms.PreprocessingSample`
from its current voxel spacing to a fixed ``target_spacing``, recomputing
the volume shape and interpolating every array accordingly. It depends
only on the architecture established in Module 3.1 - adding it required
no change to ``pipeline.py``, ``transforms.py`` or ``validation.py``.

Resampling uses ``scipy.ndimage.zoom``, which was already a transitive
dependency of this project's scientific-Python stack (installed alongside
numpy/nibabel) - no new dependency was needed. It is a mature, widely used
N-dimensional interpolation routine, so no manual interpolation code is
implemented here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .exceptions import InvalidVolumeError
from .transforms import PreprocessingSample, Transform
from .validation import validate_shapes_consistent, validate_voxel_spacing

logger = logging.getLogger(__name__)

#: Spline order used for MRI modalities: 1 = (tri)linear interpolation.
_MRI_INTERPOLATION_ORDER = 1

#: Spline order used for segmentation masks: 0 = nearest-neighbor.
#: This is mandatory, not a tunable default - any higher order would blend
#: label values at boundaries and fabricate labels that never existed in
#: the original mask.
_SEGMENTATION_INTERPOLATION_ORDER = 0


@dataclass(frozen=True)
class ResamplingInfo:
    """Record of what one ResamplingTransform run actually did.

    Stored verbatim in ``sample.metadata["resampling"]``.
    """

    original_spacing: Tuple[float, float, float]
    target_spacing: Tuple[float, float, float]
    original_shape: Tuple[int, ...]
    new_shape: Tuple[int, ...]


def compute_resampled_shape(
    original_shape: Sequence[int],
    original_spacing: Sequence[float],
    target_spacing: Sequence[float],
) -> Tuple[int, ...]:
    """Derive the resampled shape for a given spacing change.

    ``new_size = original_size * (original_spacing / target_spacing)``,
    computed independently per axis.

    Rounding convention: each axis is rounded to the nearest integer with
    ties rounding *up* (``floor(x + 0.5)``, i.e. classic "round half up" -
    deliberately not Python's built-in banker's rounding, so the behavior
    at exact .5 boundaries is unambiguous and easy to state). The result
    is clamped to a minimum of 1 voxel per axis, so an extreme downsample
    can never collapse an axis to size 0.

    Args:
        original_shape: Shape of the volume before resampling.
        original_spacing: Current voxel spacing (mm), same axis order as
            ``original_shape``.
        target_spacing: Desired voxel spacing (mm), same axis order.

    Returns:
        The new shape, with the same number of dimensions as the inputs.
    """
    new_shape = []
    for size, original_sp, target_sp in zip(original_shape, original_spacing, target_spacing):
        scaled_size = size * (original_sp / target_sp)
        rounded_size = int(math.floor(scaled_size + 0.5))
        new_shape.append(max(1, rounded_size))
    return tuple(new_shape)


def _resample_array(
    volume: np.ndarray,
    new_shape: Tuple[int, ...],
    order: int,
    output_dtype: np.dtype,
) -> np.ndarray:
    """Resample a single array to ``new_shape`` with the given spline order.

    The zoom factor passed to ``scipy.ndimage.zoom`` is derived directly
    from ``new_shape / volume.shape`` (rather than the theoretical
    ``original_spacing / target_spacing`` ratio), so scipy's own internal
    shape rounding reproduces ``new_shape`` exactly - this is what
    guarantees every modality and the segmentation end up with identical
    shapes, since :func:`compute_resampled_shape` is only ever computed
    once per sample and reused for every array.
    """
    if tuple(volume.shape) == tuple(new_shape):
        # Nothing to resample: skips the interpolation pass entirely,
        # which is both faster and numerically exact for the common
        # "spacing already matches the target" case.
        logger.debug("Shape %s already matches target; skipping interpolation.", new_shape)
        return volume.astype(output_dtype, copy=True)

    zoom_factors = tuple(new / old for new, old in zip(new_shape, volume.shape))
    resampled = ndimage.zoom(volume, zoom=zoom_factors, order=order)

    if tuple(resampled.shape) != tuple(new_shape):
        # Defensive: should be unreachable given the exact zoom factors
        # above, but a silent shape mismatch here would corrupt every
        # downstream batching/collation step, so fail loudly instead.
        raise InvalidVolumeError(
            f"Resampling produced shape {resampled.shape}, expected {new_shape}."
        )
    return resampled.astype(output_dtype, copy=False)


class ResamplingTransform(Transform):
    """Resamples every array in a sample to a fixed target voxel spacing.

    Applies to all modalities present in the sample (each resampled
    independently, but sharing one computed target shape) and to the
    segmentation mask, if present:

    - MRI modalities use (tri)linear interpolation (spline order 1).
    - The segmentation mask uses nearest-neighbor interpolation (spline
      order 0) - mandatory, to avoid fabricating label values that never
      existed in the original mask.

    Example:
        >>> transform = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0))
        >>> result = transform.apply(sample)  # sample had spacing (1.0, 1.0, 2.0)
        >>> result.voxel_spacing
        (1.0, 1.0, 1.0)

    Note:
        This transform does not modify ``sample.affine``. Correctly
        updating an affine matrix for a new voxel spacing requires
        preserving its rotation/direction-cosine components, not just
        rescaling the diagonal, which is out of scope here - the affine
        is carried over unchanged and may no longer describe the
        resampled array's true geometry. Flagged here rather than guessed
        at, since it wasn't part of this module's requirements.
    """

    def __init__(self, target_spacing: Tuple[float, float, float]) -> None:
        """
        Args:
            target_spacing: Desired voxel spacing in mm, as ``(sx, sy, sz)``.
                Every value must be strictly positive and finite.

        Raises:
            ValueError: If ``target_spacing`` doesn't have exactly 3
                components, or any component is not a positive, finite
                number.
        """
        self.target_spacing = self._validate_target_spacing(target_spacing)

    @staticmethod
    def _validate_target_spacing(target_spacing: Sequence[float]) -> Tuple[float, float, float]:
        try:
            values = tuple(float(v) for v in target_spacing)
        except TypeError as exc:
            raise ValueError(
                f"target_spacing must be an iterable of 3 numbers, got {target_spacing!r}."
            ) from exc

        if len(values) != 3:
            raise ValueError(
                f"target_spacing must have exactly 3 components (sx, sy, sz), got {values!r}."
            )
        if any(value <= 0 or not math.isfinite(value) for value in values):
            raise ValueError(
                f"target_spacing values must be strictly positive and finite, got {values!r}."
            )
        return values  # type: ignore[return-value]

    def validate_input(self, sample: PreprocessingSample) -> None:
        """Check the sample can actually be resampled.

        Raises:
            InvalidVolumeError: If ``sample.voxel_spacing`` is missing or
                invalid, if the sample has neither modalities nor a
                segmentation, or if modalities/segmentation don't already
                share a common shape (required so a single target shape
                can be computed once and reused for every array).
        """
        if sample.voxel_spacing is None:
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': resampling requires "
                "sample.voxel_spacing, but it is None."
            )
        validate_voxel_spacing(sample)
        validate_shapes_consistent(sample)

        if not sample.modalities and sample.segmentation is None:
            raise InvalidVolumeError(
                f"Patient '{sample.patient_id}': cannot resample a sample "
                "with no modalities and no segmentation."
            )

    def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
        """Resample every array in ``sample`` to ``self.target_spacing``.

        Args:
            sample: The input sample. Not modified - a new sample is
                always returned, with new arrays for every modality and
                (if present) the segmentation.

        Returns:
            A new :class:`~src.preprocessing.transforms.PreprocessingSample`
            with:

            - Every modality resampled (linear interpolation, ``float32``).
            - The segmentation, if present, resampled (nearest-neighbor,
              original integer dtype preserved).
            - ``voxel_spacing`` set to exactly ``self.target_spacing``.
            - ``metadata["resampling"]`` set to a :class:`ResamplingInfo`
              describing what changed; all other metadata keys preserved.
        """
        original_spacing = sample.voxel_spacing
        original_shape = self._reference_shape(sample)
        new_shape = compute_resampled_shape(original_shape, original_spacing, self.target_spacing)

        logger.info(
            "Resampling patient '%s': spacing %s -> %s, shape %s -> %s.",
            sample.patient_id,
            original_spacing,
            self.target_spacing,
            original_shape,
            new_shape,
        )

        resampled_modalities: Dict = {}
        for modality, volume in sample.modalities.items():
            resampled_modalities[modality] = _resample_array(
                volume, new_shape, _MRI_INTERPOLATION_ORDER, np.float32
            )
            logger.debug(
                "Patient '%s': modality '%s' resampled to shape %s.",
                sample.patient_id,
                modality.value,
                new_shape,
            )

        resampled_segmentation = None
        if sample.segmentation is not None:
            resampled_segmentation = _resample_array(
                sample.segmentation,
                new_shape,
                _SEGMENTATION_INTERPOLATION_ORDER,
                sample.segmentation.dtype,
            )
            logger.debug(
                "Patient '%s': segmentation resampled to shape %s (nearest-neighbor).",
                sample.patient_id,
                new_shape,
            )

        resampling_info = ResamplingInfo(
            original_spacing=tuple(original_spacing),
            target_spacing=self.target_spacing,
            original_shape=tuple(original_shape),
            new_shape=new_shape,
        )
        updated_metadata = dict(sample.metadata)
        updated_metadata["resampling"] = resampling_info

        return sample.replace(
            modalities=resampled_modalities,
            segmentation=resampled_segmentation,
            voxel_spacing=self.target_spacing,
            metadata=updated_metadata,
        )

    @staticmethod
    def _reference_shape(sample: PreprocessingSample) -> Tuple[int, ...]:
        """The shape shared by every array in the sample.

        ``validate_input`` (called by the pipeline before ``apply``)
        already guarantees this via ``validate_shapes_consistent``; this
        defensive fallback only matters if ``apply`` is called directly,
        bypassing the pipeline.
        """
        if sample.modalities:
            return tuple(next(iter(sample.modalities.values())).shape)
        if sample.segmentation is not None:
            return tuple(sample.segmentation.shape)
        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}': cannot resample a sample with "
            "no modalities and no segmentation."
        )

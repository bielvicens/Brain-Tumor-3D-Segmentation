"""Data augmentation transforms for BraTS preprocessing.

These transforms are intended for training only. They inherit from the
standard ``Transform`` interface, so they integrate seamlessly with the
existing ``PreprocessingPipeline``.

Geometric augmentations are always applied consistently to every MRI
modality and to the segmentation mask. Intensity augmentations only
modify the MRI modalities.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.preprocessing.transforms import (
    PreprocessingSample,
    Transform,
)
from .validation import validate_shapes_consistent
from .exceptions import InvalidVolumeError

logger = logging.getLogger(__name__)


class RandomFlip(Transform):
    """Randomly flip all volumes along one spatial axis.

    The same flip is applied to every MRI modality and to the
    segmentation mask so voxel correspondence is preserved.

    Args:
        probability:
            Probability of applying the transform.
        axis:
            Axis to flip. If ``None``, one of the three spatial axes is
            chosen uniformly at random.
        seed:
            Optional random seed for reproducibility.
    """

    def __init__(
        self,
        probability: float = 0.5,
        axis: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1."
            )

        if axis is not None and axis not in (0, 1, 2):
            raise ValueError(
                "axis must be 0, 1 or 2."
            )

        self.probability = probability
        self.axis = axis
        self._rng = np.random.default_rng(seed)

    def apply(
        self,
        sample: PreprocessingSample,
    ) -> PreprocessingSample:

        if self._rng.random() >= self.probability:
            return sample

        axis = (
            self.axis
            if self.axis is not None
            else int(self._rng.integers(0, 3))
        )

        logger.debug(
            "Applying RandomFlip(axis=%d) to patient '%s'.",
            axis,
            sample.patient_id,
        )

        modalities = {
            modality: np.flip(volume, axis=axis).copy()
            for modality, volume in sample.modalities.items()
        }

        segmentation = None
        if sample.segmentation is not None:
            segmentation = np.flip(
                sample.segmentation,
                axis=axis,
            ).copy()

        metadata = dict(sample.metadata)
        metadata.setdefault("augmentations", []).append(
            {
                "name": "RandomFlip",
                "axis": axis,
            }
        )

        return sample.replace(
            modalities=modalities,
            segmentation=segmentation,
            metadata=metadata,
        )
class RandomRotation90(Transform):
    """Randomly rotate all volumes by multiples of 90 degrees.

    The same rotation is applied to every MRI modality and to the
    segmentation mask so voxel correspondence is preserved.

    Rotations are performed using ``numpy.rot90`` and therefore do not
    require interpolation.

    Args:
        probability:
            Probability of applying the transform.
        seed:
            Optional random seed for reproducibility.
    """

    def __init__(
        self,
        probability: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1."
            )

        self.probability = probability
        self._rng = np.random.default_rng(seed)

    def apply(
        self,
        sample: PreprocessingSample,
    ) -> PreprocessingSample:

        if self._rng.random() >= self.probability:
            return sample

        k = int(self._rng.integers(1, 4))

        axes = (0, 1)

        logger.debug(
            (
                "Applying RandomRotation90(k=%d, axes=%s) "
                "to patient '%s'."
            ),
            k,
            axes,
            sample.patient_id,
        )

        modalities = {
            modality: np.rot90(
                volume,
                k=k,
                axes=axes,
            ).copy()
            for modality, volume in sample.modalities.items()
        }

        segmentation = None
        if sample.segmentation is not None:
            segmentation = np.rot90(
                sample.segmentation,
                k=k,
                axes=axes,
            ).copy()

        metadata = dict(sample.metadata)
        metadata.setdefault(
            "augmentations",
            [],
        ).append(
            {
                "name": "RandomRotation90",
                "k": k,
                "axes": axes,
            }
        )

        return sample.replace(
            modalities=modalities,
            segmentation=segmentation,
            metadata=metadata,
        )
    
class RandomGaussianNoise(Transform):
    """Randomly add Gaussian noise to every MRI modality.

    Gaussian noise is independently sampled for every voxel of every MRI
    modality. The segmentation mask is left unchanged.

    Args:
        probability:
            Probability of applying the transform.
        mean:
            Mean of the Gaussian distribution.
        std:
            Standard deviation of the Gaussian distribution.
        seed:
            Optional random seed for reproducibility.
    """

    def __init__(
        self,
        probability: float = 0.5,
        mean: float = 0.0,
        std: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1."
            )

        if std < 0.0:
            raise ValueError(
                "std must be non-negative."
            )

        self.probability = probability
        self.mean = mean
        self.std = std
        self._rng = np.random.default_rng(seed)

    def apply(
        self,
        sample: PreprocessingSample,
    ) -> PreprocessingSample:

        if self._rng.random() >= self.probability:
            return sample

        logger.debug(
            (
                "Applying RandomGaussianNoise("
                "mean=%.4f, std=%.4f) "
                "to patient '%s'."
            ),
            self.mean,
            self.std,
            sample.patient_id,
        )

        modalities = {}

        for modality, volume in sample.modalities.items():

            noise = self._rng.normal(
                loc=self.mean,
                scale=self.std,
                size=volume.shape,
            ).astype(np.float32)

            modalities[modality] = (
                volume.astype(np.float32) + noise
            )

        metadata = dict(sample.metadata)
        metadata.setdefault(
            "augmentations",
            [],
        ).append(
            {
                "name": "RandomGaussianNoise",
                "mean": self.mean,
                "std": self.std,
            }
        )

        return sample.replace(
            modalities=modalities,
            segmentation=sample.segmentation,
            metadata=metadata,
        )
class RandomGamma(Transform):
    """Randomly apply gamma intensity correction to every MRI modality.

    Gamma correction is applied independently to every MRI modality while
    preserving the segmentation mask.

    Since MRI intensities may contain negative values after
    normalization, each modality is temporarily normalized to the range
    [0, 1], gamma correction is applied, and the result is mapped back to
    the original intensity range.

    Args:
        probability:
            Probability of applying the transform.
        gamma_range:
            Inclusive range from which the gamma value is sampled.
        seed:
            Optional random seed for reproducibility.
    """

    def __init__(
        self,
        probability: float = 0.5,
        gamma_range: tuple[float, float] = (0.7, 1.5),
        seed: Optional[int] = None,
    ) -> None:

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1."
            )

        gamma_min, gamma_max = gamma_range

        if gamma_min <= 0.0:
            raise ValueError(
                "gamma values must be positive."
            )

        if gamma_min > gamma_max:
            raise ValueError(
                "gamma_range must satisfy min <= max."
            )

        self.probability = probability
        self.gamma_range = gamma_range
        self._rng = np.random.default_rng(seed)

    def apply(
        self,
        sample: PreprocessingSample,
    ) -> PreprocessingSample:

        if self._rng.random() >= self.probability:
            return sample

        gamma = float(
            self._rng.uniform(
                self.gamma_range[0],
                self.gamma_range[1],
            )
        )

        logger.debug(
            (
                "Applying RandomGamma("
                "gamma=%.3f) "
                "to patient '%s'."
            ),
            gamma,
            sample.patient_id,
        )

        modalities = {}

        for modality, volume in sample.modalities.items():

            volume = volume.astype(np.float32)

            vmin = float(volume.min())
            vmax = float(volume.max())

            if np.isclose(vmin, vmax):
                modalities[modality] = volume.copy()
                continue

            normalized = (volume - vmin) / (vmax - vmin)

            corrected = np.power(
                normalized,
                gamma,
                dtype=np.float32,
            )

            corrected = corrected * (vmax - vmin) + vmin

            modalities[modality] = corrected.astype(np.float32)

        metadata = dict(sample.metadata)
        metadata.setdefault(
            "augmentations",
            [],
        ).append(
            {
                "name": "RandomGamma",
                "gamma": gamma,
            }
        )

        return sample.replace(
            modalities=modalities,
            segmentation=sample.segmentation,
            metadata=metadata,
        )
    
class RandomIntensityShift(Transform):
    """Randomly shift the intensity of every MRI modality.

    A constant value is added to every voxel of every MRI modality.
    The segmentation mask is left unchanged.

    Args:
        probability:
            Probability of applying the transform.
        shift_range:
            Inclusive range from which the intensity shift is sampled.
        seed:
            Optional random seed for reproducibility.
    """

    def __init__(
        self,
        probability: float = 0.5,
        shift_range: tuple[float, float] = (-0.1, 0.1),
        seed: Optional[int] = None,
    ) -> None:

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1."
            )

        shift_min, shift_max = shift_range

        if shift_min > shift_max:
            raise ValueError(
                "shift_range must satisfy min <= max."
            )

        self.probability = probability
        self.shift_range = shift_range
        self._rng = np.random.default_rng(seed)

    def apply(
        self,
        sample: PreprocessingSample,
    ) -> PreprocessingSample:

        if self._rng.random() >= self.probability:
            return sample

        shift = float(
            self._rng.uniform(
                self.shift_range[0],
                self.shift_range[1],
            )
        )

        logger.debug(
            (
                "Applying RandomIntensityShift("
                "shift=%.4f) "
                "to patient '%s'."
            ),
            shift,
            sample.patient_id,
        )

        modalities = {
            modality: (
                volume.astype(np.float32) + shift
            ).astype(np.float32)
            for modality, volume in sample.modalities.items()
        }

        metadata = dict(sample.metadata)
        metadata.setdefault(
            "augmentations",
            [],
        ).append(
            {
                "name": "RandomIntensityShift",
                "shift": shift,
            }
        )

        return sample.replace(
            modalities=modalities,
            segmentation=sample.segmentation,
            metadata=metadata,
        )

class RandomCrop3D(Transform):
    """Randomly crop a fixed-size 3D patch.

    The crop can optionally be biased towards tumor regions to increase
    exposure to underrepresented tumor classes such as NCR.

    Args:
        crop_size:
            Desired crop size as (depth, height, width).
        probability:
            Probability of applying the transform.
        tumor_probability:
            Probability that a crop is sampled around tumor voxels instead
            of being completely random.
        ncr_probability:
            Conditional probability, among tumor-aware crops, of sampling
            specifically around NCR voxels.

            For example:
                tumor_probability=0.60
                ncr_probability=0.20

            means:
                40% -> completely random crop
                48% -> general tumor crop
                12% -> NCR-focused crop

        seed:
            Optional random seed for reproducibility.

    BraTS labels:
        NCR = 1
        ED  = 2
        ET  = 3
    """

    NCR_LABEL = 1
    TUMOR_LABELS = (1, 2, 3)

    def __init__(
        self,
        crop_size: tuple[int, int, int],
        probability: float = 1.0,
        tumor_probability: float = 0.0,
        ncr_probability: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:

        self.crop_size = self._validate_crop_size(crop_size)

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1."
            )

        if not 0.0 <= tumor_probability <= 1.0:
            raise ValueError(
                "tumor_probability must be between 0 and 1."
            )

        if not 0.0 <= ncr_probability <= 1.0:
            raise ValueError(
                "ncr_probability must be between 0 and 1."
            )

        self.probability = probability
        self.tumor_probability = tumor_probability
        self.ncr_probability = ncr_probability

        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_crop_size(
        crop_size: Sequence[int],
    ) -> tuple[int, int, int]:
        """Validate the requested crop size."""

        try:
            values = tuple(int(v) for v in crop_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "crop_size must be an iterable of three integers."
            ) from exc

        if len(values) != 3:
            raise ValueError(
                "crop_size must contain exactly three dimensions."
            )

        if any(v <= 0 for v in values):
            raise ValueError(
                "crop_size values must be strictly positive."
            )

        return values

    def validate_input(
        self,
        sample: PreprocessingSample,
    ) -> None:
        """Ensure the crop fits inside the volume."""

        validate_shapes_consistent(sample)

        shape = self._reference_shape(sample)

        if len(shape) != 3:
            raise InvalidVolumeError(
                f"Expected 3D volumes, got shape {shape}."
            )

        for crop, dim in zip(self.crop_size, shape):
            if crop > dim:
                raise InvalidVolumeError(
                    f"Crop size {self.crop_size} exceeds image shape {shape}."
                )

    def apply(
        self,
        sample: PreprocessingSample,
    ) -> PreprocessingSample:
        """Apply a random or tumor-aware 3D crop."""

        if self._rng.random() >= self.probability:
            return sample

        shape = self._reference_shape(sample)

        crop_mode = "random"

        # --------------------------------------------------------------
        # Decide whether this crop should be tumor-aware.
        # --------------------------------------------------------------
        tumor_aware = (
            sample.segmentation is not None
            and self._rng.random() < self.tumor_probability
        )

        if tumor_aware:
            use_ncr = (
                self._rng.random() < self.ncr_probability
            )

            if use_ncr:
                start = self._tumor_aware_start_indices(
                    sample.segmentation,
                    shape,
                    labels=(self.NCR_LABEL,),
                )

                if start is not None:
                    crop_mode = "ncr"
                else:
                    # NCR may be absent in this patient.
                    # Fall back to general tumor sampling.
                    start = self._tumor_aware_start_indices(
                        sample.segmentation,
                        shape,
                        labels=self.TUMOR_LABELS,
                    )

                    if start is not None:
                        crop_mode = "tumor"
                    else:
                        start = self._random_start_indices(shape)
                        crop_mode = "random_fallback"

            else:
                start = self._tumor_aware_start_indices(
                    sample.segmentation,
                    shape,
                    labels=self.TUMOR_LABELS,
                )

                if start is not None:
                    crop_mode = "tumor"
                else:
                    start = self._random_start_indices(shape)
                    crop_mode = "random_fallback"

        else:
            start = self._random_start_indices(shape)

        slices = tuple(
            slice(s, s + size)
            for s, size in zip(start, self.crop_size)
        )

        logger.debug(
            (
                "Applying RandomCrop3D("
                "crop_size=%s, start=%s, mode=%s, "
                "tumor_probability=%.2f, ncr_probability=%.2f"
                ") to patient '%s'."
            ),
            self.crop_size,
            start,
            crop_mode,
            self.tumor_probability,
            self.ncr_probability,
            sample.patient_id,
        )

        cropped_modalities = {
            modality: volume[slices].copy()
            for modality, volume in sample.modalities.items()
        }

        cropped_segmentation = None

        if sample.segmentation is not None:
            cropped_segmentation = sample.segmentation[
                slices
            ].copy()

        metadata = dict(sample.metadata)

        metadata.setdefault(
            "augmentations",
            [],
        ).append(
            {
                "name": "RandomCrop3D",
                "crop_size": self.crop_size,
                "start": start,
                "mode": crop_mode,
                "tumor_probability": self.tumor_probability,
                "ncr_probability": self.ncr_probability,
            }
        )

        return sample.replace(
            modalities=cropped_modalities,
            segmentation=cropped_segmentation,
            metadata=metadata,
        )

    def _tumor_aware_start_indices(
        self,
        segmentation: np.ndarray,
        shape: tuple[int, int, int],
        labels: tuple[int, ...],
    ) -> Optional[tuple[int, int, int]]:
        """Generate crop coordinates around a target tumor region.

        A random voxel belonging to one of ``labels`` is selected as the
        crop anchor. The crop is then positioned so that the anchor lies
        somewhere inside the resulting patch.

        Returns ``None`` when the requested labels are not present.
        """

        if segmentation is None:
            return None

        mask = np.isin(
            segmentation,
            labels,
        )

        coordinates = np.argwhere(mask)

        if coordinates.size == 0:
            return None

        anchor = coordinates[
            self._rng.integers(
                0,
                len(coordinates),
            )
        ]

        starts = []

        for axis, (anchor_position, dim, crop) in enumerate(
            zip(anchor, shape, self.crop_size)
        ):
            max_start = dim - crop

            if max_start <= 0:
                starts.append(0)
                continue

            # Allow the anchor to appear anywhere inside the crop.
            min_start = max(
                0,
                int(anchor_position) - crop + 1,
            )

            max_valid_start = min(
                int(anchor_position),
                max_start,
            )

            if min_start > max_valid_start:
                starts.append(
                    min(
                        max(
                            int(anchor_position) - crop // 2,
                            0,
                        ),
                        max_start,
                    )
                )
            else:
                starts.append(
                    int(
                        self._rng.integers(
                            min_start,
                            max_valid_start + 1,
                        )
                    )
                )

        return tuple(starts)

    def _random_start_indices(
        self,
        shape: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Generate completely random crop start coordinates."""

        starts: list[int] = []

        for dim, crop in zip(shape, self.crop_size):

            if crop > dim:
                raise InvalidVolumeError(
                    f"Crop size {self.crop_size} exceeds image shape {shape}."
                )

            if dim == crop:
                starts.append(0)
            else:
                starts.append(
                    int(
                        self._rng.integers(
                            0,
                            dim - crop + 1,
                        )
                    )
                )

        return tuple(starts)

    @staticmethod
    def _reference_shape(
        sample: PreprocessingSample,
    ) -> tuple[int, int, int]:
        """Return the common volume shape."""

        if sample.modalities:
            return tuple(
                next(
                    iter(sample.modalities.values())
                ).shape
            )

        if sample.segmentation is not None:
            return tuple(sample.segmentation.shape)

        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}' contains no image data."
        )
    
class CenterCrop3D(Transform):
    """Crop a fixed-size 3D patch from the centre of a preprocessing sample.

    The same crop is applied to every MRI modality and to the segmentation
    mask, preserving voxel correspondence.

    This transform is intended for validation and inference, where
    deterministic preprocessing is required.

    Args:
        crop_size:
            Desired crop size as ``(depth, height, width)``.
    """

    def __init__(
        self,
        crop_size: tuple[int, int, int],
    ) -> None:

        self.crop_size = self._validate_crop_size(crop_size)

    @staticmethod
    def _validate_crop_size(
        crop_size: Sequence[int],
    ) -> tuple[int, int, int]:
        """Validate the requested crop size."""

        try:
            values = tuple(int(v) for v in crop_size)
        except TypeError as exc:
            raise ValueError(
                "crop_size must be an iterable of three integers."
            ) from exc

        if len(values) != 3:
            raise ValueError(
                "crop_size must contain exactly three dimensions."
            )

        if any(v <= 0 for v in values):
            raise ValueError(
                "crop_size values must be strictly positive."
            )

        return values

    def validate_input(
        self,
        sample: PreprocessingSample,
    ) -> None:
        """Ensure the crop fits inside the volume."""

        validate_shapes_consistent(sample)

        shape = self._reference_shape(sample)

        if len(shape) != 3:
            raise InvalidVolumeError(
                f"Expected 3D volumes, got shape {shape}."
            )

        for crop, dim in zip(self.crop_size, shape):
            if crop > dim:
                raise InvalidVolumeError(
                    "Crop size "
                    f"{self.crop_size} exceeds image shape {shape}."
                )

    def apply(
        self,
        sample: PreprocessingSample,
    ) -> PreprocessingSample:
        """Crop the centre of the sample."""

        shape = self._reference_shape(sample)

        start = tuple(
            (dim - crop) // 2
            for dim, crop in zip(shape, self.crop_size)
        )

        slices = tuple(
            slice(s, s + crop)
            for s, crop in zip(start, self.crop_size)
        )

        logger.debug(
            (
                "Applying CenterCrop3D(crop_size=%s) "
                "to patient '%s'."
            ),
            self.crop_size,
            sample.patient_id,
        )

        cropped_modalities = {
            modality: volume[slices].copy()
            for modality, volume in sample.modalities.items()
        }

        cropped_segmentation = None
        if sample.segmentation is not None:
            cropped_segmentation = sample.segmentation[slices].copy()

        metadata = dict(sample.metadata)
        metadata.setdefault(
            "preprocessing",
            [],
        ).append(
            {
                "name": "CenterCrop3D",
                "crop_size": self.crop_size,
            }
        )

        return sample.replace(
            modalities=cropped_modalities,
            segmentation=cropped_segmentation,
            metadata=metadata,
        )

    @staticmethod
    def _reference_shape(
        sample: PreprocessingSample,
    ) -> tuple[int, int, int]:
        """Return the common volume shape."""

        if sample.modalities:
            return tuple(
                next(
                    iter(
                        sample.modalities.values()
                    )
                ).shape
            )

        if sample.segmentation is not None:
            return tuple(sample.segmentation.shape)

        raise InvalidVolumeError(
            f"Patient '{sample.patient_id}' contains no image data."
        )
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
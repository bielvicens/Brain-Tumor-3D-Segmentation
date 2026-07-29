"""Intensity normalization transforms for BraTS MRI volumes.

This module holds concrete, math-implementing ``Transform`` subclasses,
starting with :class:`ZScoreNormalization`. It depends only on the
architecture established in Module 3.1
(:class:`~src.preprocessing.transforms.Transform`,
:class:`~src.preprocessing.transforms.PreprocessingSample`) - adding it
required no change to ``pipeline.py`` or ``validation.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.data import Modality

from .transforms import PreprocessingSample, Transform

logger = logging.getLogger(__name__)

#: Default minimum std used in place of a (near-)zero foreground std, to
#: avoid dividing by zero for a degenerate, constant-intensity volume.
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True)
class NormalizationStats:
    """Statistics used to z-score-normalize one modality's foreground voxels.

    Recorded verbatim in ``sample.metadata["normalization"]`` so the exact
    numbers used are always inspectable afterwards - e.g. to invert the
    transform for visualization, or to report them in the clinical report.
    """

    mean: float
    std: float
    epsilon_used: bool
    foreground_voxel_count: int


class ZScoreNormalization(Transform):
    """Per-modality, foreground-only z-score intensity normalization.

    BraTS MRI volumes are skull-stripped, so voxels outside the brain are
    exactly 0. Including them when computing mean/std would corrupt the
    statistics with a huge spike at 0, and normalizing them would turn
    "background" into an arbitrary non-zero value - not what downstream
    models expect. This transform therefore, independently per modality:

    - Treats every voxel <= 0 as background.
    - Computes mean and std using only voxels strictly greater than 0.
    - Applies ``(x - mean) / std`` to those foreground voxels only.
    - Leaves every background voxel at exactly ``0.0`` in the output.

    The segmentation mask, voxel spacing, affine and any pre-existing
    metadata are carried over unchanged - this transform only touches
    ``sample.modalities`` and adds one key to ``sample.metadata``.
    """

    def __init__(self, epsilon: float = DEFAULT_EPSILON) -> None:
        """
        Args:
            epsilon: Minimum std used when normalizing. If a modality's
                foreground std falls below this value (e.g. a constant-
                intensity volume), ``epsilon`` is used instead, to avoid
                dividing by zero.

        Raises:
            ValueError: If ``epsilon`` is not strictly positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be strictly positive, got {epsilon}.")
        self.epsilon = epsilon

    def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
        """Normalize every modality in ``sample`` independently.

        Args:
            sample: The input sample. Not modified - a new sample is
                always returned, with new arrays for every modality.

        Returns:
            A new :class:`PreprocessingSample` with normalized modalities
            and per-modality statistics recorded in
            ``metadata["normalization"]`` (keyed by ``modality.value``).
        """
        normalized_modalities: Dict[Modality, np.ndarray] = {}
        stats_by_modality: Dict[str, NormalizationStats] = {}

        for modality, volume in sample.modalities.items():
            normalized_volume, stats = self._normalize_volume(volume, modality, sample.patient_id)
            normalized_modalities[modality] = normalized_volume
            stats_by_modality[modality.value] = stats

        updated_metadata = dict(sample.metadata)
        updated_metadata["normalization"] = stats_by_modality

        logger.info(
            "Applied z-score normalization to %d modality(ies) for patient '%s'.",
            len(normalized_modalities),
            sample.patient_id,
        )
        return sample.replace(modalities=normalized_modalities, metadata=updated_metadata)

    def _normalize_volume(
        self, volume: np.ndarray, modality: Modality, patient_id: str
    ) -> Tuple[np.ndarray, NormalizationStats]:
        """Normalize a single modality's volume. Never mutates ``volume``.

        Uses ``where=`` for the mean/std reductions, and ``out=``/``where=``
        on the ufuncs that apply the normalization, so no separate
        foreground-only copy of the volume is ever materialized: the only
        new allocations are the boolean mask and the zero-initialized
        output buffer.
        """
        foreground_mask = volume > 0
        foreground_voxel_count = int(np.count_nonzero(foreground_mask))

        normalized = np.zeros(volume.shape, dtype=np.float32)

        if foreground_voxel_count == 0:
            logger.warning(
                "Patient '%s': modality '%s' has no foreground voxels "
                "(nothing > 0); returning an all-zero volume unchanged.",
                patient_id,
                modality.value,
            )
            return normalized, NormalizationStats(
                mean=0.0, std=0.0, epsilon_used=False, foreground_voxel_count=0
            )

        mean = float(np.mean(volume, where=foreground_mask))
        std = float(np.std(volume, where=foreground_mask))

        epsilon_used = std < self.epsilon
        effective_std = self.epsilon if epsilon_used else std
        if epsilon_used:
            logger.warning(
                "Patient '%s': modality '%s' has (near-)zero foreground std "
                "(%.3e); using epsilon=%.3e instead to avoid division by zero.",
                patient_id,
                modality.value,
                std,
                self.epsilon,
            )

        # Background positions are never touched by either call below, so
        # they stay at the 0.0 that `normalized` was initialized with.
        np.subtract(volume, mean, out=normalized, where=foreground_mask)
        np.divide(normalized, effective_std, out=normalized, where=foreground_mask)

        logger.debug(
            "Patient '%s': modality '%s' normalized (mean=%.4f, std=%.4f, "
            "foreground_voxels=%d).",
            patient_id,
            modality.value,
            mean,
            std,
            foreground_voxel_count,
        )

        return normalized, NormalizationStats(
            mean=mean,
            std=std,
            epsilon_used=epsilon_used,
            foreground_voxel_count=foreground_voxel_count,
        )

"""Pure statistical computations for BraTS dataset EDA.

This module has no dependency on the reader, matplotlib, or the
filesystem. Every function operates on plain numpy arrays / Python
collections, which keeps it trivially unit-testable and reusable outside
the analysis pipeline (e.g. directly from a notebook).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional

import numpy as np

from src.data import Modality


@dataclass(frozen=True)
class DistributionSummary:
    """Descriptive statistics for a 1-D collection of numeric values."""

    count: int
    minimum: float
    maximum: float
    mean: float
    std: float
    median: float
    percentile_5: float
    percentile_95: float


def summarize_distribution(values: Sequence[float]) -> DistributionSummary:
    """Compute descriptive statistics for a sequence of scalar values.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if len(values) == 0:
        raise ValueError("Cannot summarize an empty sequence of values.")
    arr = np.asarray(values, dtype=np.float64)
    return DistributionSummary(
        count=int(arr.size),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        median=float(np.median(arr)),
        percentile_5=float(np.percentile(arr, 5)),
        percentile_95=float(np.percentile(arr, 95)),
    )


@dataclass(frozen=True)
class IntensityStatistics:
    """Voxel-intensity statistics for a single MRI volume."""

    modality: Modality
    minimum: float
    maximum: float
    mean: float
    std: float
    median: float
    percentile_1: float
    percentile_99: float
    voxels_considered: int


def compute_intensity_statistics(
    volume: np.ndarray,
    modality: Modality,
    exclude_background: bool = True,
) -> IntensityStatistics:
    """Compute intensity statistics for one MRI volume.

    Args:
        volume: The raw voxel array.
        modality: Which modality this volume belongs to (kept alongside the
            stats so downstream code doesn't need a separate lookup).
        exclude_background: BraTS MRI volumes are skull-stripped, so voxels
            outside the brain are exactly 0. Including them would dominate
            the statistics with a huge spike at zero and hide the actual
            tissue-intensity distribution. When ``True`` (default), only
            strictly positive voxels are considered.

    Raises:
        ValueError: If no voxels remain after filtering (e.g. an all-zero
            volume).
    """
    values = volume[volume > 0] if exclude_background else volume.ravel()
    if values.size == 0:
        raise ValueError(
            f"No non-background voxels found for modality '{modality.value}'."
        )
    return IntensityStatistics(
        modality=modality,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        median=float(np.median(values)),
        percentile_1=float(np.percentile(values, 1)),
        percentile_99=float(np.percentile(values, 99)),
        voxels_considered=int(values.size),
    )


@dataclass(frozen=True)
class MaskStatistics:
    """Statistics describing one patient's segmentation mask."""

    labels_present: Tuple[int, ...]
    voxel_counts_by_label: Dict[int, int]
    total_voxels: int
    tumor_voxel_count: int
    tumor_volume_ratio: float
    tumor_volume_mm3: Optional[float]


def compute_mask_statistics(
    mask: np.ndarray,
    voxel_spacing: Optional[Tuple[float, ...]] = None,
) -> MaskStatistics:
    """Compute label distribution and tumor volume for a segmentation mask.

    Args:
        mask: Integer-labeled segmentation array (0 = background).
        voxel_spacing: Physical size of one voxel in mm, if known. Used to
            convert the tumor voxel count into a volume in mm^3.
    """
    labels, counts = np.unique(mask, return_counts=True)
    voxel_counts_by_label = {int(label): int(count) for label, count in zip(labels, counts)}
    total_voxels = int(mask.size)
    tumor_voxel_count = total_voxels - voxel_counts_by_label.get(0, 0)
    tumor_volume_ratio = tumor_voxel_count / total_voxels if total_voxels else 0.0

    tumor_volume_mm3: Optional[float] = None
    if voxel_spacing is not None:
        voxel_volume = float(np.prod(voxel_spacing))
        tumor_volume_mm3 = tumor_voxel_count * voxel_volume

    return MaskStatistics(
        labels_present=tuple(int(label) for label in labels),
        voxel_counts_by_label=voxel_counts_by_label,
        total_voxels=total_voxels,
        tumor_voxel_count=tumor_voxel_count,
        tumor_volume_ratio=tumor_volume_ratio,
        tumor_volume_mm3=tumor_volume_mm3,
    )


@dataclass(frozen=True)
class ShapeStatistics:
    """Distribution of volume shapes across the dataset."""

    shape_counts: Dict[Tuple[int, ...], int]
    most_common_shape: Tuple[int, ...]
    unique_shape_count: int


def compute_shape_statistics(shapes: Sequence[Tuple[int, ...]]) -> ShapeStatistics:
    """Summarize how many distinct volume shapes occur in the dataset.

    Raises:
        ValueError: If ``shapes`` is empty.
    """
    if not shapes:
        raise ValueError("Cannot compute shape statistics for an empty dataset.")
    counts: Dict[Tuple[int, ...], int] = {}
    for shape in shapes:
        counts[shape] = counts.get(shape, 0) + 1
    most_common = max(counts.items(), key=lambda item: item[1])[0]
    return ShapeStatistics(
        shape_counts=counts,
        most_common_shape=most_common,
        unique_shape_count=len(counts),
    )


_AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class SpacingStatistics:
    """Distribution of voxel spacing across the dataset."""

    spacing_counts: Dict[Tuple[float, ...], int]
    most_common_spacing: Tuple[float, ...]
    per_axis: Dict[str, DistributionSummary]


def compute_spacing_statistics(spacings: Sequence[Tuple[float, ...]]) -> SpacingStatistics:
    """Summarize voxel spacing across the dataset, per-axis and as tuples.

    Spacing tuples are rounded to 4 decimal places before counting so that
    floating-point noise (e.g. 1.0000001 vs 1.0) doesn't create spurious
    "unique" spacings.

    Raises:
        ValueError: If ``spacings`` is empty.
    """
    if not spacings:
        raise ValueError("Cannot compute spacing statistics for an empty dataset.")

    rounded = [tuple(round(v, 4) for v in s) for s in spacings]
    counts: Dict[Tuple[float, ...], int] = {}
    for spacing in rounded:
        counts[spacing] = counts.get(spacing, 0) + 1
    most_common = max(counts.items(), key=lambda item: item[1])[0]

    n_axes = len(spacings[0])
    per_axis: Dict[str, DistributionSummary] = {}
    for axis_index in range(n_axes):
        axis_name = _AXIS_NAMES[axis_index] if axis_index < len(_AXIS_NAMES) else f"axis_{axis_index}"
        axis_values = [s[axis_index] for s in spacings]
        per_axis[axis_name] = summarize_distribution(axis_values)

    return SpacingStatistics(
        spacing_counts=counts,
        most_common_spacing=most_common,
        per_axis=per_axis,
    )


@dataclass(frozen=True)
class OutlierResult:
    """Patients flagged as outliers for one criterion."""

    method: str
    flagged_patient_ids: Tuple[str, ...]
    details: Dict[str, str]


def detect_shape_outliers(
    shapes_by_patient: Dict[str, Tuple[int, ...]],
    reference_shape: Tuple[int, ...],
) -> OutlierResult:
    """Flag patients whose volume shape differs from ``reference_shape``.

    Volume shape in a properly preprocessed BraTS dataset is constant
    across patients (every volume is resampled to the same grid), so any
    deviation is treated as an outlier directly rather than via a
    statistical threshold.
    """
    flagged = tuple(
        patient_id for patient_id, shape in shapes_by_patient.items() if shape != reference_shape
    )
    return OutlierResult(
        method=f"shape != {reference_shape}",
        flagged_patient_ids=flagged,
        details={pid: str(shapes_by_patient[pid]) for pid in flagged},
    )


def _iqr_bounds(values: Sequence[float], k: float) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def detect_spacing_outliers(
    spacings_by_patient: Dict[str, Tuple[float, ...]],
    k: float = 1.5,
) -> OutlierResult:
    """Flag patients whose voxel spacing is an IQR-based outlier on any axis."""
    if not spacings_by_patient:
        return OutlierResult(method=f"iqr(k={k}) per axis", flagged_patient_ids=(), details={})

    n_axes = len(next(iter(spacings_by_patient.values())))
    flagged_ids: set = set()
    details: Dict[str, str] = {}

    for axis_index in range(n_axes):
        axis_values = {pid: s[axis_index] for pid, s in spacings_by_patient.items()}
        low, high = _iqr_bounds(list(axis_values.values()), k)
        for pid, value in axis_values.items():
            if value < low or value > high:
                flagged_ids.add(pid)
                note = f"axis_{axis_index}={value:.4f} (bounds {low:.4f}..{high:.4f}); "
                details[pid] = details.get(pid, "") + note

    return OutlierResult(
        method=f"iqr(k={k}) per axis",
        flagged_patient_ids=tuple(sorted(flagged_ids)),
        details=details,
    )


def detect_value_outliers_iqr(values_by_patient: Dict[str, float], k: float = 1.5) -> OutlierResult:
    """Generic IQR-based outlier detection over one scalar value per patient.

    Used for per-modality intensity outliers (on the per-patient mean
    intensity), but works for any scalar metric.
    """
    if not values_by_patient:
        return OutlierResult(method=f"iqr(k={k})", flagged_patient_ids=(), details={})

    low, high = _iqr_bounds(list(values_by_patient.values()), k)
    flagged = tuple(
        pid for pid, value in values_by_patient.items() if value < low or value > high
    )
    details = {
        pid: f"value={values_by_patient[pid]:.4f} (bounds {low:.4f}..{high:.4f})" for pid in flagged
    }
    return OutlierResult(method=f"iqr(k={k})", flagged_patient_ids=flagged, details=details)

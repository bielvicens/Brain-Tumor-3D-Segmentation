"""Unit tests for src.analysis.statistics.

These tests use plain numpy arrays and dicts only - no BraTSReader, no
filesystem - since the module under test is pure.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.statistics import (
    compute_intensity_statistics,
    compute_mask_statistics,
    compute_shape_statistics,
    compute_spacing_statistics,
    detect_shape_outliers,
    detect_spacing_outliers,
    detect_value_outliers_iqr,
    summarize_distribution,
)
from src.data import Modality


# ----------------------------------------------------------------------
# summarize_distribution
# ----------------------------------------------------------------------
def test_summarize_distribution_basic_values() -> None:
    summary = summarize_distribution([1, 2, 3, 4, 5])
    assert summary.count == 5
    assert summary.minimum == 1
    assert summary.maximum == 5
    assert summary.mean == 3
    assert summary.median == 3


def test_summarize_distribution_empty_raises() -> None:
    with pytest.raises(ValueError):
        summarize_distribution([])


# ----------------------------------------------------------------------
# compute_intensity_statistics
# ----------------------------------------------------------------------
def test_compute_intensity_statistics_excludes_background_by_default() -> None:
    volume = np.array([0, 0, 0, 10, 20, 30], dtype=np.float32)
    stats = compute_intensity_statistics(volume, Modality.T1N)
    assert stats.voxels_considered == 3
    assert stats.minimum == 10
    assert stats.maximum == 30
    assert stats.mean == 20


def test_compute_intensity_statistics_can_include_background() -> None:
    volume = np.array([0, 0, 0, 10, 20, 30], dtype=np.float32)
    stats = compute_intensity_statistics(volume, Modality.T1N, exclude_background=False)
    assert stats.voxels_considered == 6
    assert stats.minimum == 0


def test_compute_intensity_statistics_all_zero_raises() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_intensity_statistics(volume, Modality.T1N)


# ----------------------------------------------------------------------
# compute_mask_statistics
# ----------------------------------------------------------------------
def test_compute_mask_statistics_counts_labels_and_tumor_ratio() -> None:
    mask = np.array([0, 0, 0, 0, 1, 1, 2, 4], dtype=np.int16)
    stats = compute_mask_statistics(mask)
    assert stats.labels_present == (0, 1, 2, 4)
    assert stats.total_voxels == 8
    assert stats.tumor_voxel_count == 4  # everything that isn't label 0
    assert stats.tumor_volume_ratio == pytest.approx(0.5)
    assert stats.tumor_volume_mm3 is None


def test_compute_mask_statistics_with_voxel_spacing() -> None:
    mask = np.array([0, 0, 1, 1], dtype=np.int16)
    stats = compute_mask_statistics(mask, voxel_spacing=(2.0, 2.0, 2.0))
    # 2 tumor voxels * (2*2*2 mm^3 per voxel) = 16 mm^3
    assert stats.tumor_volume_mm3 == pytest.approx(16.0)


# ----------------------------------------------------------------------
# compute_shape_statistics
# ----------------------------------------------------------------------
def test_compute_shape_statistics_finds_most_common_shape() -> None:
    shapes = [(240, 240, 155), (240, 240, 155), (240, 240, 154)]
    stats = compute_shape_statistics(shapes)
    assert stats.most_common_shape == (240, 240, 155)
    assert stats.unique_shape_count == 2
    assert stats.shape_counts[(240, 240, 155)] == 2


def test_compute_shape_statistics_empty_raises() -> None:
    with pytest.raises(ValueError):
        compute_shape_statistics([])


# ----------------------------------------------------------------------
# compute_spacing_statistics
# ----------------------------------------------------------------------
def test_compute_spacing_statistics_per_axis_summary() -> None:
    spacings = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 2.0)]
    stats = compute_spacing_statistics(spacings)
    assert stats.most_common_spacing == (1.0, 1.0, 1.0)
    assert set(stats.per_axis.keys()) == {"x", "y", "z"}
    assert stats.per_axis["z"].maximum == 2.0


def test_compute_spacing_statistics_empty_raises() -> None:
    with pytest.raises(ValueError):
        compute_spacing_statistics([])


# ----------------------------------------------------------------------
# Outlier detection
# ----------------------------------------------------------------------
def test_detect_shape_outliers_flags_non_matching_shapes() -> None:
    shapes_by_patient = {
        "p1": (240, 240, 155),
        "p2": (240, 240, 155),
        "p3": (128, 128, 128),
    }
    result = detect_shape_outliers(shapes_by_patient, reference_shape=(240, 240, 155))
    assert result.flagged_patient_ids == ("p3",)


def test_detect_shape_outliers_no_outliers_when_all_match() -> None:
    shapes_by_patient = {"p1": (240, 240, 155), "p2": (240, 240, 155)}
    result = detect_shape_outliers(shapes_by_patient, reference_shape=(240, 240, 155))
    assert result.flagged_patient_ids == ()


def test_detect_spacing_outliers_flags_extreme_axis_value() -> None:
    # 4 identical points -> IQR collapses to a single value, so the 5th,
    # clearly different point is unambiguously flagged.
    spacings_by_patient = {
        "p1": (1.0, 1.0, 1.0),
        "p2": (1.0, 1.0, 1.0),
        "p3": (1.0, 1.0, 1.0),
        "p4": (1.0, 1.0, 1.0),
        "p5": (5.0, 1.0, 1.0),
    }
    result = detect_spacing_outliers(spacings_by_patient)
    assert result.flagged_patient_ids == ("p5",)


def test_detect_value_outliers_iqr_flags_extreme_value() -> None:
    values_by_patient = {"p1": 10.0, "p2": 10.0, "p3": 10.0, "p4": 10.0, "p5": 1000.0}
    result = detect_value_outliers_iqr(values_by_patient)
    assert result.flagged_patient_ids == ("p5",)


def test_detect_value_outliers_iqr_empty_input() -> None:
    result = detect_value_outliers_iqr({})
    assert result.flagged_patient_ids == ()

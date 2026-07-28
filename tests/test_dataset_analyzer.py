"""Integration tests for DatasetAnalyzer.

Reuses the `synthetic_dataset` fixture from tests/conftest.py (the same
one used by test_brats_reader.py) to keep the two test suites consistent,
and adds one dedicated local fixture for exercising the positive outlier-
detection path end to end.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.data import BraTSReader, Modality
from src.analysis.dataset_analyzer import DatasetAnalyzer


def test_analyze_reports_correct_valid_and_invalid_counts(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    result = DatasetAnalyzer(reader).analyze()

    assert result.total_patients == 3
    assert result.valid_patient_count == 2
    assert result.invalid_patient_count == 1


def test_analyze_shape_and_spacing_statistics(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    result = DatasetAnalyzer(reader).analyze()

    assert result.shape_statistics.unique_shape_count == 1
    assert result.shape_statistics.most_common_shape == (4, 4, 4)
    assert result.spacing_statistics.most_common_spacing == (1.0, 1.0, 1.0)


def test_analyze_intensity_summary_matches_synthetic_fill_values(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    result = DatasetAnalyzer(reader).analyze()

    expected_means = {Modality.T1N: 1.0, Modality.T1C: 2.0, Modality.T2W: 3.0, Modality.T2F: 4.0}
    for modality, expected_mean in expected_means.items():
        assert modality in result.intensity_summary
        assert result.intensity_summary[modality].mean_distribution.mean == pytest.approx(expected_mean)


def test_analyze_mask_summary(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    result = DatasetAnalyzer(reader).analyze()

    assert result.mask_summary is not None
    assert result.mask_summary.label_frequency == {1: 2}
    assert result.mask_summary.tumor_ratio_distribution.mean == pytest.approx(1.0)


def test_analyze_flags_no_outliers_on_a_clean_dataset(synthetic_dataset: Path) -> None:
    """The two valid patients are identical in every measured quantity, so
    no outlier category should fire any false positives."""
    reader = BraTSReader(synthetic_dataset)
    result = DatasetAnalyzer(reader).analyze()

    assert result.outliers.shape_outliers.flagged_patient_ids == ()
    assert result.outliers.spacing_outliers.flagged_patient_ids == ()
    for outlier_result in result.outliers.intensity_outliers.values():
        assert outlier_result.flagged_patient_ids == ()


def test_analyze_respects_max_patients(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    result = DatasetAnalyzer(reader).analyze(max_patients=1)
    assert result.total_patients == 1


# ----------------------------------------------------------------------
# Dedicated fixture: a genuine intensity outlier
# ----------------------------------------------------------------------
def _write_nifti(path: Path, fill_value: float, dtype=np.float32) -> None:
    data = np.full((4, 4, 4), fill_value, dtype=dtype)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(path))


@pytest.fixture
def dataset_with_intensity_outlier(tmp_path: Path) -> Path:
    """4 'normal' patients with identical T1N intensity, plus 1 patient
    with a dramatically different T1N intensity. Shape and spacing are
    kept identical across all 5 so only the intensity signal differs.
    """
    root = tmp_path / "BraTS"
    root.mkdir()

    t1n_fill_values = {
        "patient-0": 1.0,
        "patient-1": 1.0,
        "patient-2": 1.0,
        "patient-3": 1.0,
        "patient-4": 1000.0,  # the outlier
    }
    for patient_id, t1n_value in t1n_fill_values.items():
        case_dir = root / patient_id
        case_dir.mkdir()
        _write_nifti(case_dir / f"{patient_id}-t1n.nii.gz", fill_value=t1n_value)
        _write_nifti(case_dir / f"{patient_id}-t1c.nii.gz", fill_value=2.0)
        _write_nifti(case_dir / f"{patient_id}-t2w.nii.gz", fill_value=3.0)
        _write_nifti(case_dir / f"{patient_id}-t2f.nii.gz", fill_value=4.0)
        _write_nifti(case_dir / f"{patient_id}-seg.nii.gz", fill_value=1.0, dtype=np.int16)

    return root


def test_analyze_detects_genuine_intensity_outlier(dataset_with_intensity_outlier: Path) -> None:
    reader = BraTSReader(dataset_with_intensity_outlier)
    result = DatasetAnalyzer(reader).analyze()

    assert result.valid_patient_count == 5
    t1n_outliers = result.outliers.intensity_outliers[Modality.T1N]
    assert t1n_outliers.flagged_patient_ids == ("patient-4",)

    # The other modalities were identical across all 5 patients, so they
    # should report no outliers.
    for modality in (Modality.T1C, Modality.T2W, Modality.T2F):
        assert result.outliers.intensity_outliers[modality].flagged_patient_ids == ()

    # Shape/spacing were identical across all 5 patients too.
    assert result.outliers.shape_outliers.flagged_patient_ids == ()
    assert result.outliers.spacing_outliers.flagged_patient_ids == ()

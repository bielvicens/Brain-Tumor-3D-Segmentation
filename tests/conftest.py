"""Shared pytest fixtures for the BraTSReader and analysis test suites.

`synthetic_dataset` builds a tiny, fully synthetic dataset on disk (real,
valid NIfTI files written via nibabel) so the reader and the analyzer can
be tested without requiring access to the actual BraTS dataset.

`fake_analysis_result` is a hand-built DatasetAnalysisResult (no reader
involved at all) used to test visualization and report generation in
isolation from dataset loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import pytest

from src.analysis.dataset_analyzer import (
    DatasetAnalysisResult,
    MaskSummary,
    ModalityIntensitySummary,
    OutlierReport,
    PatientAnalysisResult,
)
from src.analysis.statistics import (
    IntensityStatistics,
    MaskStatistics,
    OutlierResult,
    ShapeStatistics,
    SpacingStatistics,
    summarize_distribution,
)
from src.data import Modality

VOLUME_SHAPE: Tuple[int, int, int] = (4, 4, 4)


def _write_nifti(path: Path, fill_value: float, dtype=np.float32) -> None:
    data = np.full(VOLUME_SHAPE, fill_value, dtype=dtype)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Builds a small dataset with:

    - one complete patient using the BraTS 2023+ naming convention
    - one complete patient using the legacy BraTS 2020/2021 naming convention
    - one incomplete patient (missing the T2-FLAIR volume)
    """
    root = tmp_path / "BraTS"
    root.mkdir()

    # Patient 1: new (2023+) naming convention, fully complete.
    p1 = root / "BraTS-GLI-00000-000"
    p1.mkdir()
    _write_nifti(p1 / "BraTS-GLI-00000-000-t1n.nii.gz", fill_value=1.0)
    _write_nifti(p1 / "BraTS-GLI-00000-000-t1c.nii.gz", fill_value=2.0)
    _write_nifti(p1 / "BraTS-GLI-00000-000-t2w.nii.gz", fill_value=3.0)
    _write_nifti(p1 / "BraTS-GLI-00000-000-t2f.nii.gz", fill_value=4.0)
    _write_nifti(p1 / "BraTS-GLI-00000-000-seg.nii.gz", fill_value=1.0, dtype=np.int16)

    # Patient 2: legacy naming convention, fully complete.
    p2 = root / "BraTS20_Training_001"
    p2.mkdir()
    _write_nifti(p2 / "BraTS20_Training_001_t1.nii.gz", fill_value=1.0)
    _write_nifti(p2 / "BraTS20_Training_001_t1ce.nii.gz", fill_value=2.0)
    _write_nifti(p2 / "BraTS20_Training_001_t2.nii.gz", fill_value=3.0)
    _write_nifti(p2 / "BraTS20_Training_001_flair.nii.gz", fill_value=4.0)
    _write_nifti(p2 / "BraTS20_Training_001_seg.nii.gz", fill_value=1.0, dtype=np.int16)

    # Patient 3: incomplete - missing the T2-FLAIR volume.
    p3 = root / "BraTS-GLI-00002-000"
    p3.mkdir()
    _write_nifti(p3 / "BraTS-GLI-00002-000-t1n.nii.gz", fill_value=1.0)
    _write_nifti(p3 / "BraTS-GLI-00002-000-t1c.nii.gz", fill_value=2.0)
    _write_nifti(p3 / "BraTS-GLI-00002-000-t2w.nii.gz", fill_value=3.0)
    _write_nifti(p3 / "BraTS-GLI-00002-000-seg.nii.gz", fill_value=1.0, dtype=np.int16)

    return root


@pytest.fixture
def fake_analysis_result() -> DatasetAnalysisResult:
    """A minimal, hand-built DatasetAnalysisResult - no reader involved.

    Used by the visualization and report_generator tests, which only care
    that they render whatever DatasetAnalysisResult they're given, not
    about how such a result is produced.
    """
    intensity_stats_a = {
        Modality.T1N: IntensityStatistics(
            modality=Modality.T1N,
            minimum=1.0,
            maximum=10.0,
            mean=5.0,
            std=1.0,
            median=5.0,
            percentile_1=1.0,
            percentile_99=9.0,
            voxels_considered=100,
        )
    }
    mask_stats_a = MaskStatistics(
        labels_present=(0, 1, 2),
        voxel_counts_by_label={0: 50, 1: 30, 2: 20},
        total_voxels=100,
        tumor_voxel_count=50,
        tumor_volume_ratio=0.5,
        tumor_volume_mm3=50.0,
    )
    patient_a = PatientAnalysisResult(
        patient_id="patient-a",
        is_valid=True,
        shape=(4, 4, 4),
        voxel_spacing=(1.0, 1.0, 1.0),
        intensity_stats=intensity_stats_a,
        mask_stats=mask_stats_a,
    )
    patient_b = PatientAnalysisResult(
        patient_id="patient-b",
        is_valid=False,
        missing_modalities=["t2f"],
    )

    shape_stats = ShapeStatistics(
        shape_counts={(4, 4, 4): 1}, most_common_shape=(4, 4, 4), unique_shape_count=1
    )
    spacing_stats = SpacingStatistics(
        spacing_counts={(1.0, 1.0, 1.0): 1},
        most_common_spacing=(1.0, 1.0, 1.0),
        per_axis={
            "x": summarize_distribution([1.0]),
            "y": summarize_distribution([1.0]),
            "z": summarize_distribution([1.0]),
        },
    )
    intensity_summary = {
        Modality.T1N: ModalityIntensitySummary(
            modality=Modality.T1N,
            mean_distribution=summarize_distribution([5.0]),
            std_distribution=summarize_distribution([1.0]),
        )
    }
    mask_summary = MaskSummary(
        label_frequency={0: 1, 1: 1, 2: 1},
        tumor_ratio_distribution=summarize_distribution([0.5]),
        tumor_volume_mm3_distribution=summarize_distribution([50.0]),
    )
    outliers = OutlierReport(
        shape_outliers=OutlierResult(method="shape != (4, 4, 4)", flagged_patient_ids=(), details={}),
        spacing_outliers=OutlierResult(method="iqr(k=1.5) per axis", flagged_patient_ids=(), details={}),
        intensity_outliers={
            Modality.T1N: OutlierResult(method="iqr(k=1.5)", flagged_patient_ids=(), details={})
        },
    )

    return DatasetAnalysisResult(
        total_patients=2,
        valid_patient_count=1,
        invalid_patient_count=1,
        per_patient=[patient_a, patient_b],
        shape_statistics=shape_stats,
        spacing_statistics=spacing_stats,
        intensity_summary=intensity_summary,
        mask_summary=mask_summary,
        outliers=outliers,
    )

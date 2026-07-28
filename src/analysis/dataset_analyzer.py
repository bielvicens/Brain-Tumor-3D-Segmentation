"""Orchestrates a full exploratory data analysis pass over a BraTS dataset.

This module ties the :class:`~src.data.BraTSReader` (file discovery /
loading) together with the pure statistics functions in
:mod:`src.analysis.statistics`. It contains no plotting and no report
generation - those are handled by :mod:`src.analysis.visualization` and
:mod:`src.analysis.report_generator` respectively, so each concern can be
tested and reused independently.

Design note: patients are processed one at a time and only their computed
statistics (scalars, small dicts) are retained - the raw voxel arrays are
discarded as soon as their statistics are computed. This keeps memory
usage roughly constant regardless of dataset size, which matters when
analyzing datasets with 1000+ patients on a modest machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.data import BraTSReader, Modality, MRI_MODALITIES, PatientRecord

from .statistics import (
    DistributionSummary,
    IntensityStatistics,
    MaskStatistics,
    OutlierResult,
    ShapeStatistics,
    SpacingStatistics,
    compute_intensity_statistics,
    compute_mask_statistics,
    compute_shape_statistics,
    compute_spacing_statistics,
    detect_shape_outliers,
    detect_spacing_outliers,
    detect_value_outliers_iqr,
    summarize_distribution,
)

logger = logging.getLogger(__name__)


@dataclass
class PatientAnalysisResult:
    """Per-patient outcome of the EDA pass.

    ``is_valid`` is ``False`` both for patients that failed the reader's
    modality-completeness check and for patients whose files exist but
    could not actually be loaded (e.g. a corrupt NIfTI file) - in both
    cases downstream aggregation should ignore them.
    """

    patient_id: str
    is_valid: bool
    missing_modalities: List[str] = field(default_factory=list)
    shape: Optional[Tuple[int, ...]] = None
    voxel_spacing: Optional[Tuple[float, ...]] = None
    intensity_stats: Dict[Modality, IntensityStatistics] = field(default_factory=dict)
    mask_stats: Optional[MaskStatistics] = None
    error: Optional[str] = None


@dataclass
class ModalityIntensitySummary:
    """Dataset-wide intensity summary for a single modality.

    Both distributions are computed over one value *per patient*, not over
    all voxels combined, so a large volume can't drown out the rest of the
    dataset.
    """

    modality: Modality
    mean_distribution: DistributionSummary
    std_distribution: DistributionSummary


@dataclass
class MaskSummary:
    """Dataset-wide summary of segmentation masks."""

    label_frequency: Dict[int, int]
    tumor_ratio_distribution: DistributionSummary
    tumor_volume_mm3_distribution: Optional[DistributionSummary]


@dataclass
class OutlierReport:
    """All outlier-detection results for one analysis run."""

    shape_outliers: OutlierResult
    spacing_outliers: OutlierResult
    intensity_outliers: Dict[Modality, OutlierResult]


@dataclass
class DatasetAnalysisResult:
    """Complete output of a :meth:`DatasetAnalyzer.analyze` run."""

    total_patients: int
    valid_patient_count: int
    invalid_patient_count: int
    per_patient: List[PatientAnalysisResult]
    shape_statistics: ShapeStatistics
    spacing_statistics: SpacingStatistics
    intensity_summary: Dict[Modality, ModalityIntensitySummary]
    mask_summary: Optional[MaskSummary]
    outliers: OutlierReport


class DatasetAnalyzer:
    """Runs a full exploratory data analysis pass over a BraTS dataset."""

    def __init__(self, reader: BraTSReader, exclude_background: bool = True) -> None:
        """
        Args:
            reader: A configured :class:`BraTSReader` pointing at the
                dataset to analyze.
            exclude_background: Whether intensity statistics should ignore
                zero-valued (background) voxels. See
                :func:`~src.analysis.statistics.compute_intensity_statistics`.
        """
        self.reader = reader
        self.exclude_background = exclude_background

    def analyze(self, max_patients: Optional[int] = None) -> DatasetAnalysisResult:
        """Run the full analysis and return an aggregated result.

        Args:
            max_patients: If set, only analyze the first N discovered
                patients. Useful for a quick smoke-test run before
                committing to a full pass over a large dataset.
        """
        records = self.reader.get_patients(only_valid=False)
        if max_patients is not None:
            records = records[:max_patients]

        logger.info("Analyzing %d patient(s)...", len(records))
        per_patient = [self._analyze_patient(record) for record in records]

        valid = [r for r in per_patient if r.is_valid]
        logger.info("%d of %d patient(s) analyzed successfully.", len(valid), len(per_patient))

        shapes = [r.shape for r in valid if r.shape is not None]
        spacings = [r.voxel_spacing for r in valid if r.voxel_spacing is not None]

        shape_stats = compute_shape_statistics(shapes)
        spacing_stats = compute_spacing_statistics(spacings)
        intensity_summary = self._summarize_intensity(valid)
        mask_summary = self._summarize_masks(valid)
        outliers = self._detect_outliers(valid, shape_stats)

        return DatasetAnalysisResult(
            total_patients=len(per_patient),
            valid_patient_count=len(valid),
            invalid_patient_count=len(per_patient) - len(valid),
            per_patient=per_patient,
            shape_statistics=shape_stats,
            spacing_statistics=spacing_stats,
            intensity_summary=intensity_summary,
            mask_summary=mask_summary,
            outliers=outliers,
        )

    def _analyze_patient(self, record: PatientRecord) -> PatientAnalysisResult:
        """Compute statistics for one patient, never raising to the caller.

        A single unreadable/corrupt case must not abort an analysis run
        that may cover a thousand other patients - the failure is logged
        and the patient is marked invalid with the error message attached.
        """
        if not record.is_complete:
            return PatientAnalysisResult(
                patient_id=record.patient_id,
                is_valid=False,
                missing_modalities=[m.value for m in record.missing_modalities],
            )

        try:
            metadata = self.reader.get_metadata(record, Modality.T1N)
            volumes = self.reader.load_modalities(record)
            mask = self.reader.load_segmentation(record)

            intensity_stats = {
                modality: compute_intensity_statistics(volume, modality, self.exclude_background)
                for modality, volume in volumes.items()
            }
            mask_stats = compute_mask_statistics(mask, metadata.voxel_spacing)

            return PatientAnalysisResult(
                patient_id=record.patient_id,
                is_valid=True,
                shape=metadata.shape,
                voxel_spacing=metadata.voxel_spacing,
                intensity_stats=intensity_stats,
                mask_stats=mask_stats,
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the whole run
            logger.error("Failed to analyze patient '%s': %s", record.patient_id, exc)
            return PatientAnalysisResult(
                patient_id=record.patient_id,
                is_valid=False,
                error=str(exc),
            )

    def _summarize_intensity(
        self, valid: List[PatientAnalysisResult]
    ) -> Dict[Modality, ModalityIntensitySummary]:
        summary: Dict[Modality, ModalityIntensitySummary] = {}
        for modality in MRI_MODALITIES:
            means = [r.intensity_stats[modality].mean for r in valid if modality in r.intensity_stats]
            stds = [r.intensity_stats[modality].std for r in valid if modality in r.intensity_stats]
            if not means:
                continue
            summary[modality] = ModalityIntensitySummary(
                modality=modality,
                mean_distribution=summarize_distribution(means),
                std_distribution=summarize_distribution(stds),
            )
        return summary

    def _summarize_masks(self, valid: List[PatientAnalysisResult]) -> Optional[MaskSummary]:
        mask_results = [r.mask_stats for r in valid if r.mask_stats is not None]
        if not mask_results:
            return None

        label_frequency: Dict[int, int] = {}
        for stats in mask_results:
            for label in stats.labels_present:
                label_frequency[label] = label_frequency.get(label, 0) + 1

        tumor_ratios = [stats.tumor_volume_ratio for stats in mask_results]
        volumes_mm3 = [
            stats.tumor_volume_mm3 for stats in mask_results if stats.tumor_volume_mm3 is not None
        ]

        return MaskSummary(
            label_frequency=label_frequency,
            tumor_ratio_distribution=summarize_distribution(tumor_ratios),
            tumor_volume_mm3_distribution=(
                summarize_distribution(volumes_mm3) if volumes_mm3 else None
            ),
        )

    def _detect_outliers(
        self, valid: List[PatientAnalysisResult], shape_stats: ShapeStatistics
    ) -> OutlierReport:
        shapes_by_patient = {r.patient_id: r.shape for r in valid if r.shape is not None}
        spacings_by_patient = {
            r.patient_id: r.voxel_spacing for r in valid if r.voxel_spacing is not None
        }

        shape_outliers = detect_shape_outliers(shapes_by_patient, shape_stats.most_common_shape)
        spacing_outliers = detect_spacing_outliers(spacings_by_patient)

        intensity_outliers: Dict[Modality, OutlierResult] = {}
        for modality in MRI_MODALITIES:
            means_by_patient = {
                r.patient_id: r.intensity_stats[modality].mean
                for r in valid
                if modality in r.intensity_stats
            }
            if means_by_patient:
                intensity_outliers[modality] = detect_value_outliers_iqr(means_by_patient)

        return OutlierReport(
            shape_outliers=shape_outliers,
            spacing_outliers=spacing_outliers,
            intensity_outliers=intensity_outliers,
        )

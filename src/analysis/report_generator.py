"""Generates the human-readable EDA report (Markdown) and the tabular
statistics export (CSV) for a completed dataset analysis.

This module only renders already-computed data: it does not touch the
reader, does not run any statistics itself, and does not create figures -
it only references figure paths that were already saved, typically by
:func:`~src.analysis.visualization.generate_all_figures`.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset_analyzer import DatasetAnalysisResult, PatientAnalysisResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Renders a :class:`DatasetAnalysisResult` to Markdown and CSV."""

    def __init__(self, reports_dir: Path = Path("reports")) -> None:
        """
        Args:
            reports_dir: Directory where ``dataset_report.md`` and
                ``statistics.csv`` are written. Created if it doesn't exist.
        """
        self.reports_dir = Path(reports_dir)

    def generate_markdown_report(
        self,
        result: "DatasetAnalysisResult",
        figure_paths: Optional[Dict[str, Path]] = None,
        filename: str = "dataset_report.md",
    ) -> Path:
        """Render the full EDA report as Markdown.

        Args:
            result: Output of :meth:`DatasetAnalyzer.analyze`.
            figure_paths: Mapping of figure key -> saved path, as returned
                by :func:`~src.analysis.visualization.generate_all_figures`.
                Figures are embedded using a path relative to the report
                file, so the Markdown renders correctly wherever the
                ``reports/`` folder is viewed from (e.g. on GitHub).
            filename: Output filename, relative to ``reports_dir``.
        """
        figure_paths = figure_paths or {}
        output_path = self.reports_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = self._build_markdown_lines(result, figure_paths, output_path)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote dataset report to '%s'.", output_path)
        return output_path

    def generate_statistics_csv(
        self, result: "DatasetAnalysisResult", filename: str = "statistics.csv"
    ) -> Path:
        """Export one row per patient with shape, spacing, intensity and
        mask statistics - suitable for opening in a spreadsheet or pandas.
        """
        output_path = self.reports_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = self._csv_fieldnames(result)
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for patient in result.per_patient:
                writer.writerow(self._patient_to_row(patient))

        logger.info("Wrote per-patient statistics to '%s'.", output_path)
        return output_path

    # ------------------------------------------------------------------
    # CSV internals
    # ------------------------------------------------------------------
    @staticmethod
    def _csv_fieldnames(result: "DatasetAnalysisResult") -> List[str]:
        fields = ["patient_id", "is_valid", "missing_modalities", "error", "shape", "voxel_spacing"]
        for modality in result.intensity_summary:
            prefix = modality.value
            fields += [
                f"{prefix}_mean",
                f"{prefix}_std",
                f"{prefix}_min",
                f"{prefix}_max",
                f"{prefix}_p1",
                f"{prefix}_p99",
            ]
        fields += ["tumor_voxel_count", "tumor_volume_ratio", "tumor_volume_mm3", "labels_present"]
        return fields

    @staticmethod
    def _patient_to_row(patient: "PatientAnalysisResult") -> Dict[str, str]:
        row: Dict[str, str] = {
            "patient_id": patient.patient_id,
            "is_valid": str(patient.is_valid),
            "missing_modalities": ";".join(patient.missing_modalities),
            "error": patient.error or "",
            "shape": str(patient.shape) if patient.shape else "",
            "voxel_spacing": str(patient.voxel_spacing) if patient.voxel_spacing else "",
        }
        for modality, stats in patient.intensity_stats.items():
            prefix = modality.value
            row[f"{prefix}_mean"] = f"{stats.mean:.4f}"
            row[f"{prefix}_std"] = f"{stats.std:.4f}"
            row[f"{prefix}_min"] = f"{stats.minimum:.4f}"
            row[f"{prefix}_max"] = f"{stats.maximum:.4f}"
            row[f"{prefix}_p1"] = f"{stats.percentile_1:.4f}"
            row[f"{prefix}_p99"] = f"{stats.percentile_99:.4f}"

        if patient.mask_stats is not None:
            row["tumor_voxel_count"] = str(patient.mask_stats.tumor_voxel_count)
            row["tumor_volume_ratio"] = f"{patient.mask_stats.tumor_volume_ratio:.6f}"
            row["tumor_volume_mm3"] = (
                f"{patient.mask_stats.tumor_volume_mm3:.2f}"
                if patient.mask_stats.tumor_volume_mm3 is not None
                else ""
            )
            row["labels_present"] = ";".join(str(label) for label in patient.mask_stats.labels_present)
        else:
            row["tumor_voxel_count"] = ""
            row["tumor_volume_ratio"] = ""
            row["tumor_volume_mm3"] = ""
            row["labels_present"] = ""
        return row

    # ------------------------------------------------------------------
    # Markdown internals
    # ------------------------------------------------------------------
    def _build_markdown_lines(
        self,
        result: "DatasetAnalysisResult",
        figure_paths: Dict[str, Path],
        output_path: Path,
    ) -> List[str]:
        lines: List[str] = []
        lines.append("# BraTS Dataset - Exploratory Data Analysis Report")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total patients discovered: **{result.total_patients}**")
        lines.append(
            f"- Valid patients (all modalities present and loadable): **{result.valid_patient_count}**"
        )
        lines.append(f"- Invalid / incomplete patients: **{result.invalid_patient_count}**")
        lines.append("")

        invalid = [p for p in result.per_patient if not p.is_valid]
        if invalid:
            lines.append("### Invalid patients")
            lines.append("")
            lines.append("| Patient ID | Missing modalities | Error |")
            lines.append("|---|---|---|")
            for p in invalid:
                lines.append(
                    f"| {p.patient_id} | {', '.join(p.missing_modalities) or '-'} | {p.error or '-'} |"
                )
            lines.append("")

        lines.append("## Volume shape distribution")
        lines.append("")
        lines.append(f"- Distinct shapes found: **{result.shape_statistics.unique_shape_count}**")
        lines.append(f"- Most common shape: **{result.shape_statistics.most_common_shape}**")
        lines.append("")
        for shape, count in sorted(result.shape_statistics.shape_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{shape}`: {count} patient(s)")
        lines.append("")
        self._embed_figure(lines, figure_paths, "shape_distribution", output_path)

        if result.spacing_statistics.per_axis:
            lines.append("## Voxel spacing distribution")
            lines.append("")
            lines.append(f"- Most common spacing: **{result.spacing_statistics.most_common_spacing}** mm")
            lines.append("")
            lines.append("| Axis | Mean | Std | Min | Max | P5 | P95 |")
            lines.append("|---|---|---|---|---|---|---|")
            for axis, summary in result.spacing_statistics.per_axis.items():
                lines.append(
                    f"| {axis} | {summary.mean:.3f} | {summary.std:.3f} | {summary.minimum:.3f} | "
                    f"{summary.maximum:.3f} | {summary.percentile_5:.3f} | {summary.percentile_95:.3f} |"
                )
            lines.append("")
            self._embed_figure(lines, figure_paths, "spacing_boxplot", output_path)

        if result.intensity_summary:
            lines.append("## Intensity statistics per modality")
            lines.append("")
            lines.append(
                "Computed over non-background (> 0) voxels only, then aggregated across patients."
            )
            lines.append("")
            lines.append("| Modality | Mean of means | Std of means | Mean of stds |")
            lines.append("|---|---|---|---|")
            for modality, summary in result.intensity_summary.items():
                lines.append(
                    f"| {modality.value.upper()} | {summary.mean_distribution.mean:.2f} | "
                    f"{summary.mean_distribution.std:.2f} | {summary.std_distribution.mean:.2f} |"
                )
            lines.append("")
            self._embed_figure(lines, figure_paths, "intensity_histograms", output_path)
            self._embed_figure(lines, figure_paths, "intensity_boxplots", output_path)

        if result.mask_summary is not None:
            lines.append("## Segmentation mask statistics")
            lines.append("")
            lines.append(
                f"- Labels observed across dataset: "
                f"**{sorted(result.mask_summary.label_frequency.keys())}**"
            )
            lines.append(
                f"- Mean tumor volume: "
                f"**{result.mask_summary.tumor_ratio_distribution.mean * 100:.2f}%** of brain voxels"
            )
            lines.append("")
            self._embed_figure(lines, figure_paths, "tumor_volume_distribution", output_path)
            self._embed_figure(lines, figure_paths, "label_frequency", output_path)

        lines.append("## Outlier detection")
        lines.append("")
        lines.append(
            f"- Shape outliers ({result.outliers.shape_outliers.method}): "
            f"**{len(result.outliers.shape_outliers.flagged_patient_ids)}** patient(s)"
        )
        lines.append(
            f"- Spacing outliers ({result.outliers.spacing_outliers.method}): "
            f"**{len(result.outliers.spacing_outliers.flagged_patient_ids)}** patient(s)"
        )
        for modality, outlier_result in result.outliers.intensity_outliers.items():
            lines.append(
                f"- {modality.value.upper()} intensity outliers ({outlier_result.method}): "
                f"**{len(outlier_result.flagged_patient_ids)}** patient(s)"
            )
        lines.append("")

        flagged_ids = set(result.outliers.shape_outliers.flagged_patient_ids)
        flagged_ids |= set(result.outliers.spacing_outliers.flagged_patient_ids)
        for outlier_result in result.outliers.intensity_outliers.values():
            flagged_ids |= set(outlier_result.flagged_patient_ids)

        if flagged_ids:
            lines.append("### Flagged patient IDs")
            lines.append("")
            for patient_id in sorted(flagged_ids):
                lines.append(f"- {patient_id}")
            lines.append("")

        return lines

    @staticmethod
    def _embed_figure(
        lines: List[str], figure_paths: Dict[str, Path], key: str, report_path: Path
    ) -> None:
        if key not in figure_paths:
            return
        relative = Path(
            os.path.relpath(figure_paths[key], report_path.parent)
        ).as_posix()

        lines.append(f"![{key}]({relative})")
        lines.append("")

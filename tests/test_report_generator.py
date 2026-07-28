"""Tests for src.analysis.report_generator.ReportGenerator.

Uses the shared `fake_analysis_result` fixture (see tests/conftest.py) so
these tests are independent of BraTSReader and of the analysis pipeline
itself - they only check that a given DatasetAnalysisResult renders to
sensible Markdown and CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.analysis.dataset_analyzer import DatasetAnalysisResult
from src.analysis.report_generator import ReportGenerator


def test_generate_markdown_report_contains_key_sections(
    tmp_path: Path, fake_analysis_result: DatasetAnalysisResult
) -> None:
    generator = ReportGenerator(reports_dir=tmp_path)
    report_path = generator.generate_markdown_report(fake_analysis_result)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    assert "# BraTS Dataset - Exploratory Data Analysis Report" in content
    assert "Total patients discovered" in content
    assert "Volume shape distribution" in content
    assert "Voxel spacing distribution" in content
    assert "Intensity statistics per modality" in content
    assert "Segmentation mask statistics" in content
    assert "Outlier detection" in content
    assert "patient-b" in content  # the invalid patient should be listed


def test_generate_markdown_report_embeds_figures_with_relative_paths(
    tmp_path: Path, fake_analysis_result: DatasetAnalysisResult
) -> None:
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures" / "analysis"
    figures_dir.mkdir(parents=True)
    fake_figure = figures_dir / "shape_distribution.png"
    fake_figure.write_bytes(b"not a real png, just a placeholder")

    generator = ReportGenerator(reports_dir=reports_dir)
    report_path = generator.generate_markdown_report(
        fake_analysis_result, figure_paths={"shape_distribution": fake_figure}
    )

    content = report_path.read_text(encoding="utf-8")
    assert "![shape_distribution](../figures/analysis/shape_distribution.png)" in content


def test_generate_statistics_csv_has_expected_columns_and_rows(
    tmp_path: Path, fake_analysis_result: DatasetAnalysisResult
) -> None:
    generator = ReportGenerator(reports_dir=tmp_path)
    csv_path = generator.generate_statistics_csv(fake_analysis_result)

    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert {"patient_id", "is_valid", "t1n_mean", "tumor_volume_ratio"}.issubset(rows[0].keys())

    row_by_id = {row["patient_id"]: row for row in rows}
    assert row_by_id["patient-a"]["is_valid"] == "True"
    assert row_by_id["patient-a"]["t1n_mean"] == "5.0000"
    assert row_by_id["patient-b"]["is_valid"] == "False"
    assert row_by_id["patient-b"]["missing_modalities"] == "t2f"

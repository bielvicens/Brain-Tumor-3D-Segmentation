"""Smoke tests for src.analysis.visualization.

These check that each plotting function produces a real file on disk
without raising - they do not inspect pixel content, since that would
make the tests brittle without adding much confidence.
"""

from __future__ import annotations

from pathlib import Path

from src.analysis.dataset_analyzer import DatasetAnalysisResult
from src.analysis.visualization import (
    generate_all_figures,
    plot_intensity_boxplots,
    plot_intensity_histograms,
    plot_label_frequency,
    plot_shape_distribution,
    plot_spacing_boxplot,
    plot_tumor_volume_distribution,
)
from src.data import Modality


def test_plot_shape_distribution_saves_file(tmp_path: Path) -> None:
    output_path = plot_shape_distribution({(4, 4, 4): 3, (5, 5, 5): 1}, tmp_path / "shapes.png")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_spacing_boxplot_saves_file(tmp_path: Path) -> None:
    spacings = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.2), (1.1, 1.0, 1.0)]
    output_path = plot_spacing_boxplot(spacings, tmp_path / "spacing.png")
    assert output_path.exists()


def test_plot_intensity_histograms_saves_file(tmp_path: Path) -> None:
    means = {Modality.T1N: [1.0, 2.0, 3.0], Modality.T1C: [4.0, 5.0, 6.0]}
    output_path = plot_intensity_histograms(means, tmp_path / "hist.png")
    assert output_path.exists()


def test_plot_intensity_boxplots_saves_file(tmp_path: Path) -> None:
    means = {Modality.T1N: [1.0, 2.0, 3.0], Modality.T1C: [4.0, 5.0, 6.0]}
    output_path = plot_intensity_boxplots(means, tmp_path / "box.png")
    assert output_path.exists()


def test_plot_tumor_volume_distribution_saves_file(tmp_path: Path) -> None:
    output_path = plot_tumor_volume_distribution([0.01, 0.02, 0.05], tmp_path / "tumor.png")
    assert output_path.exists()


def test_plot_label_frequency_saves_file(tmp_path: Path) -> None:
    output_path = plot_label_frequency({0: 5, 1: 5, 2: 3, 4: 2}, tmp_path / "labels.png")
    assert output_path.exists()


def test_generate_all_figures_creates_expected_keys(
    tmp_path: Path, fake_analysis_result: DatasetAnalysisResult
) -> None:
    figures = generate_all_figures(fake_analysis_result, tmp_path)

    expected_keys = {
        "shape_distribution",
        "spacing_boxplot",
        "intensity_histograms",
        "intensity_boxplots",
        "tumor_volume_distribution",
        "label_frequency",
    }
    assert expected_keys.issubset(figures.keys())
    for path in figures.values():
        assert path.exists()

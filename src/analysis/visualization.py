"""Matplotlib visualizations for BraTS dataset EDA.

Every ``plot_*`` function takes already-computed data (never touches the
reader or the filesystem beyond writing its own output figure) and returns
the path where the figure was saved. Keeping plotting separate from
statistics and from data loading means each concern can be tested and
swapped out independently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless-safe: works in scripts, servers, Colab/Kaggle

import matplotlib.pyplot as plt

from src.data import Modality

if TYPE_CHECKING:
    from .dataset_analyzer import DatasetAnalysisResult

logger = logging.getLogger(__name__)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure to '%s'.", output_path)
    return output_path


def plot_shape_distribution(shape_counts: Dict[Tuple[int, ...], int], output_path: Path) -> Path:
    """Bar chart of how many patients have each distinct volume shape."""
    items = sorted(shape_counts.items(), key=lambda kv: -kv[1])
    labels = [str(shape) for shape, _ in items]
    values = [count for _, count in items]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_title("Volume shape distribution")
    ax.set_xlabel("Shape (H, W, D)")
    ax.set_ylabel("Number of patients")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_spacing_boxplot(spacings: Sequence[Tuple[float, ...]], output_path: Path) -> Path:
    """Boxplot of voxel spacing per axis (x, y, z)."""
    n_axes = len(spacings[0]) if spacings else 0
    axis_names = ["x", "y", "z"][:n_axes]
    data = [[s[i] for s in spacings] for i in range(n_axes)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=axis_names)
    ax.set_title("Voxel spacing distribution per axis")
    ax.set_ylabel("Spacing (mm)")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_intensity_histograms(
    means_by_modality: Dict[Modality, Sequence[float]], output_path: Path
) -> Path:
    """Grid of per-modality histograms of per-patient mean intensity."""
    modalities = list(means_by_modality.keys())
    fig, axes = plt.subplots(1, len(modalities), figsize=(4 * len(modalities), 4), sharey=True)
    axes_list: List[plt.Axes] = [axes] if len(modalities) == 1 else list(axes)

    for ax, modality in zip(axes_list, modalities):
        ax.hist(means_by_modality[modality], bins=30, color="#55A868")
        ax.set_title(modality.value.upper())
        ax.set_xlabel("Mean intensity (per patient)")
    axes_list[0].set_ylabel("Number of patients")
    fig.suptitle("Per-patient mean intensity distribution")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_intensity_boxplots(
    means_by_modality: Dict[Modality, Sequence[float]], output_path: Path
) -> Path:
    """Boxplot comparing per-patient mean intensity across modalities."""
    labels = [m.value.upper() for m in means_by_modality]
    data = list(means_by_modality.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=labels)
    ax.set_title("Mean intensity by modality")
    ax.set_ylabel("Mean intensity (per patient)")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_tumor_volume_distribution(tumor_ratios: Sequence[float], output_path: Path) -> Path:
    """Histogram of tumor volume as a percentage of total brain volume."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([r * 100 for r in tumor_ratios], bins=30, color="#C44E52")
    ax.set_title("Tumor volume distribution")
    ax.set_xlabel("Tumor volume (% of total voxels)")
    ax.set_ylabel("Number of patients")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_label_frequency(label_frequency: Dict[int, int], output_path: Path) -> Path:
    """Bar chart of how many patients contain each segmentation label."""
    labels = sorted(label_frequency.keys())
    values = [label_frequency[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(label) for label in labels], values, color="#8172B2")
    ax.set_title("Segmentation label frequency")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of patients containing this label")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def generate_all_figures(result: "DatasetAnalysisResult", output_dir: Path) -> Dict[str, Path]:
    """Generate the standard set of EDA figures for a full analysis result.

    Args:
        result: Output of :meth:`~src.analysis.dataset_analyzer.DatasetAnalyzer.analyze`.
        output_dir: Directory where figures are saved (e.g. ``figures/analysis``).

    Returns:
        Mapping from a short figure key (used by
        :class:`~src.analysis.report_generator.ReportGenerator`) to the path
        where it was saved. A figure is skipped (and simply absent from the
        returned mapping) if there wasn't enough data to plot it - e.g. no
        valid patients at all.
    """
    output_dir = Path(output_dir)
    figures: Dict[str, Path] = {}

    if result.shape_statistics.shape_counts:
        figures["shape_distribution"] = plot_shape_distribution(
            result.shape_statistics.shape_counts, output_dir / "shape_distribution.png"
        )

    valid_spacings = [r.voxel_spacing for r in result.per_patient if r.voxel_spacing is not None]
    if valid_spacings:
        figures["spacing_boxplot"] = plot_spacing_boxplot(
            valid_spacings, output_dir / "spacing_boxplot.png"
        )

    means_by_modality = {
        modality: [
            r.intensity_stats[modality].mean
            for r in result.per_patient
            if modality in r.intensity_stats
        ]
        for modality in result.intensity_summary
    }
    if means_by_modality:
        figures["intensity_histograms"] = plot_intensity_histograms(
            means_by_modality, output_dir / "intensity_histograms.png"
        )
        figures["intensity_boxplots"] = plot_intensity_boxplots(
            means_by_modality, output_dir / "intensity_boxplots.png"
        )

    if result.mask_summary is not None:
        tumor_ratios = [
            r.mask_stats.tumor_volume_ratio for r in result.per_patient if r.mask_stats is not None
        ]
        figures["tumor_volume_distribution"] = plot_tumor_volume_distribution(
            tumor_ratios, output_dir / "tumor_volume_distribution.png"
        )
        figures["label_frequency"] = plot_label_frequency(
            result.mask_summary.label_frequency, output_dir / "label_frequency.png"
        )

    return figures

"""Exploratory Data Analysis (EDA) toolkit for the BraTS dataset.

Typical usage (see also the example notebook):

    from src.data import BraTSReader
    from src.analysis import run_full_eda

    reader = BraTSReader("data/raw/BraTS")
    result = run_full_eda(reader, output_root=".")

Each stage - analysis, plotting, reporting - is also usable on its own via
:class:`DatasetAnalyzer`, :func:`generate_all_figures` and
:class:`ReportGenerator`, for anyone who wants more control than the
convenience wrapper gives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from src.data import BraTSReader

from .dataset_analyzer import (
    DatasetAnalysisResult,
    DatasetAnalyzer,
    MaskSummary,
    ModalityIntensitySummary,
    OutlierReport,
    PatientAnalysisResult,
)
from .report_generator import ReportGenerator
from .statistics import (
    DistributionSummary,
    IntensityStatistics,
    MaskStatistics,
    OutlierResult,
    ShapeStatistics,
    SpacingStatistics,
)
from .visualization import generate_all_figures

__all__ = [
    "DatasetAnalyzer",
    "DatasetAnalysisResult",
    "PatientAnalysisResult",
    "ModalityIntensitySummary",
    "MaskSummary",
    "OutlierReport",
    "ReportGenerator",
    "DistributionSummary",
    "IntensityStatistics",
    "MaskStatistics",
    "OutlierResult",
    "ShapeStatistics",
    "SpacingStatistics",
    "generate_all_figures",
    "run_full_eda",
]


def run_full_eda(
    reader: BraTSReader,
    output_root: Union[str, Path] = ".",
    max_patients: Optional[int] = None,
    exclude_background: bool = True,
) -> DatasetAnalysisResult:
    """Analyze the dataset, generate figures, and write the reports.

    This is the single call the example notebook needs. Internally it is
    just three independent steps chained together:

    1. :meth:`DatasetAnalyzer.analyze`
    2. :func:`generate_all_figures`
    3. :meth:`ReportGenerator.generate_markdown_report` /
       :meth:`ReportGenerator.generate_statistics_csv`

    Args:
        reader: A configured ``BraTSReader``.
        output_root: Project root under which ``reports/`` and
            ``figures/analysis/`` will be created.
        max_patients: Optional cap on the number of patients analyzed
            (useful for a fast smoke-test run before a full pass).
        exclude_background: See :class:`DatasetAnalyzer`.

    Returns:
        The full :class:`DatasetAnalysisResult`.
    """
    output_root = Path(output_root)

    analyzer = DatasetAnalyzer(reader, exclude_background=exclude_background)
    result = analyzer.analyze(max_patients=max_patients)

    figure_paths = generate_all_figures(result, output_root / "figures" / "analysis")

    report_generator = ReportGenerator(reports_dir=output_root / "reports")
    report_generator.generate_markdown_report(result, figure_paths)
    report_generator.generate_statistics_csv(result)

    return result

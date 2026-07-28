"""Modality definitions for the BraTS dataset.

This module defines the imaging modalities present in a BraTS case and the
file-naming patterns used to identify them on disk. Keeping this mapping in
one place (instead of scattering string literals through the reader) means
new BraTS releases or sub-challenges with different naming conventions can
be supported by editing a dictionary, not the reader's logic.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class Modality(str, Enum):
    """The five volumes that make up a single BraTS patient case."""

    T1N = "t1n"  # Native (pre-contrast) T1-weighted MRI
    T1C = "t1c"  # Contrast-enhanced (post-gadolinium) T1-weighted MRI
    T2W = "t2w"  # T2-weighted MRI
    T2F = "t2f"  # T2-FLAIR MRI
    SEG = "seg"  # Expert tumor segmentation mask


#: The four MRI modalities used as model input (excludes the mask).
MRI_MODALITIES: List[Modality] = [Modality.T1N, Modality.T1C, Modality.T2W, Modality.T2F]

#: Everything a "complete" patient case must contain on disk.
REQUIRED_MODALITIES: List[Modality] = MRI_MODALITIES + [Modality.SEG]

#: Regex patterns (tried in order, case-insensitive, anchored to the end of
#: the filename) used to match a modality to a file. Covers both the
#: BraTS 2023+ naming convention, e.g. "BraTS-GLI-00000-000-t1n.nii.gz",
#: and the legacy BraTS 2020/2021 convention, e.g.
#: "BraTS20_Training_001_t1.nii.gz".
DEFAULT_MODALITY_PATTERNS: Dict[Modality, List[str]] = {
    Modality.T1N: [r"-t1n\.nii(\.gz)?$", r"_t1\.nii(\.gz)?$"],
    Modality.T1C: [r"-t1c\.nii(\.gz)?$", r"_t1ce\.nii(\.gz)?$"],
    Modality.T2W: [r"-t2w\.nii(\.gz)?$", r"_t2\.nii(\.gz)?$"],
    Modality.T2F: [r"-t2f\.nii(\.gz)?$", r"_flair\.nii(\.gz)?$"],
    Modality.SEG: [r"-seg\.nii(\.gz)?$", r"_seg\.nii(\.gz)?$"],
}

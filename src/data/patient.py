"""Data structures describing a single BraTS patient case."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .modality import Modality


@dataclass(frozen=True)
class VolumeMetadata:
    """Lightweight metadata about a single NIfTI volume.

    Reading this does not require loading the full voxel array into memory:
    nibabel exposes shape, spacing and affine straight from the file header.
    """

    shape: Tuple[int, ...]
    voxel_spacing: Tuple[float, ...]
    affine: np.ndarray
    dtype: str


@dataclass
class PatientRecord:
    """Everything the reader knows about one BraTS patient case.

    Attributes:
        patient_id: Folder name / unique identifier for the case.
        case_dir: Path to the patient's directory on disk.
        modality_paths: Mapping from modality to the file that was found for
            it. Only contains entries for modalities actually located on
            disk, so an incomplete patient simply has fewer keys here.
        missing_modalities: Required modalities that could not be located
            for this patient.
    """

    patient_id: str
    case_dir: Path
    modality_paths: Dict[Modality, Path] = field(default_factory=dict)
    missing_modalities: List[Modality] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Whether every required modality was found for this patient."""
        return len(self.missing_modalities) == 0

    @property
    def available_modalities(self) -> List[Modality]:
        """Modalities that were successfully located on disk."""
        return list(self.modality_paths.keys())

    def path_for(self, modality: Modality) -> Path:
        """Return the file path for a given modality.

        Raises:
            KeyError: If the modality is not available for this patient.
        """
        if modality not in self.modality_paths:
            raise KeyError(
                f"Modality '{modality.value}' is not available for "
                f"patient '{self.patient_id}'."
            )
        return self.modality_paths[modality]

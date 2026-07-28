"""Reader for the BraTS brain tumor MRI dataset.

This module is intentionally the *only* place that knows about the raw
on-disk layout of BraTS. It has no dependency on PyTorch and does no
preprocessing: its sole job is to discover patients, validate that each one
has the required files, and load the raw NIfTI volumes / metadata on
request. Everything downstream (PyTorch ``Dataset``, preprocessing,
augmentation, ...) should be built on top of this module rather than
re-implementing file discovery.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import nibabel as nib
import numpy as np

from .exceptions import (
    DatasetRootNotFoundError,
    IncompletePatientError,
    PatientNotFoundError,
)
from .modality import (
    DEFAULT_MODALITY_PATTERNS,
    MRI_MODALITIES,
    REQUIRED_MODALITIES,
    Modality,
)
from .patient import PatientRecord, VolumeMetadata

logger = logging.getLogger(__name__)


class BraTSReader:
    """Discovers, validates and loads cases from a BraTS dataset directory.

    Example:
        >>> reader = BraTSReader("data/raw/BraTS")
        >>> patients = reader.get_patients()
        >>> len(patients)
        1251
        >>> patient = patients[0]
        >>> patient.patient_id
        'BraTS-GLI-00000-000'
        >>> images = reader.load_modalities(patient)
        >>> mask = reader.load_segmentation(patient)
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        modality_patterns: Optional[Dict[Modality, List[str]]] = None,
    ) -> None:
        """Initialize the reader.

        Args:
            root_dir: Directory containing one sub-folder per patient
                (e.g. ``data/raw/BraTS``).
            modality_patterns: Optional override for the regex patterns used
                to match each modality to a filename. Defaults to
                :data:`DEFAULT_MODALITY_PATTERNS`, which covers both the
                BraTS 2023+ and legacy BraTS 2020/2021 naming schemes.

        Raises:
            DatasetRootNotFoundError: If ``root_dir`` does not exist or is
                not a directory.
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise DatasetRootNotFoundError(
                f"BraTS dataset root directory not found: '{self.root_dir}'"
            )

        patterns = modality_patterns or DEFAULT_MODALITY_PATTERNS
        self._compiled_patterns: Dict[Modality, List[re.Pattern]] = {
            modality: [re.compile(p, re.IGNORECASE) for p in modality_pats]
            for modality, modality_pats in patterns.items()
        }

    # ------------------------------------------------------------------
    # Discovery & validation
    # ------------------------------------------------------------------
    def discover_patient_ids(self) -> List[str]:
        """Return the sorted list of patient folder names found on disk.

        A patient is any immediate sub-directory of ``root_dir``. No
        assumptions are made about naming, so discovery works for any BraTS
        sub-challenge (GLI, MEN, MET, PED, SSA, ...) without configuration.
        """
        patient_ids = sorted(p.name for p in self.root_dir.iterdir() if p.is_dir())
        logger.info(
            "Discovered %d candidate patient folder(s) in '%s'.",
            len(patient_ids),
            self.root_dir,
        )
        return patient_ids

    def get_patients(self, only_valid: bool = True) -> List[PatientRecord]:
        """Discover and validate every patient in the dataset.

        Args:
            only_valid: If ``True`` (default), patients missing any required
                modality are excluded from the returned list, so the rest of
                the pipeline never has to worry about incomplete cases.
                Every skipped patient is still logged as an error. Set to
                ``False`` to get every patient, including invalid ones -
                useful for a dataset quality report.

        Returns:
            List of :class:`PatientRecord`, sorted by patient ID.
        """
        records = [self._build_patient_record(pid) for pid in self.discover_patient_ids()]

        valid_records = [r for r in records if r.is_complete]
        n_invalid = len(records) - len(valid_records)
        if n_invalid:
            logger.warning(
                "%d of %d patient(s) were skipped due to missing modalities.",
                n_invalid,
                len(records),
            )

        return valid_records if only_valid else records

    def get_patient(self, patient_id: str) -> PatientRecord:
        """Build the record for a single, known patient ID.

        Raises:
            PatientNotFoundError: If no folder named ``patient_id`` exists
                under ``root_dir``.
        """
        case_dir = self.root_dir / patient_id
        if not case_dir.is_dir():
            raise PatientNotFoundError(
                f"No patient folder named '{patient_id}' under '{self.root_dir}'."
            )
        return self._build_patient_record(patient_id)

    def _build_patient_record(self, patient_id: str) -> PatientRecord:
        """Inspect one patient folder and build its :class:`PatientRecord`."""
        case_dir = self.root_dir / patient_id
        modality_paths: Dict[Modality, Path] = {}
        missing: List[Modality] = []

        for modality in REQUIRED_MODALITIES:
            match = self._match_modality_file(case_dir, modality)
            if match is not None:
                modality_paths[modality] = match
            else:
                missing.append(modality)

        record = PatientRecord(
            patient_id=patient_id,
            case_dir=case_dir,
            modality_paths=modality_paths,
            missing_modalities=missing,
        )

        if missing:
            logger.error(
                "Patient '%s' is missing required modalities: %s",
                patient_id,
                [m.value for m in missing],
            )
        else:
            logger.debug("Patient '%s' validated successfully.", patient_id)

        return record

    def _match_modality_file(self, case_dir: Path, modality: Modality) -> Optional[Path]:
        """Find the file inside ``case_dir`` matching ``modality``, if any."""
        candidates = sorted(case_dir.glob("*.nii*"))

        matches: List[Path] = []
        for pattern in self._compiled_patterns[modality]:
            matches = [c for c in candidates if pattern.search(c.name)]
            if matches:
                break

        if not matches:
            return None

        if len(matches) > 1:
            logger.warning(
                "Multiple files matched modality '%s' for patient '%s': %s. "
                "Using '%s'.",
                modality.value,
                case_dir.name,
                [m.name for m in matches],
                matches[0].name,
            )

        return matches[0]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_modalities(self, patient: PatientRecord) -> Dict[Modality, np.ndarray]:
        """Load the four MRI volumes for a patient as float32 numpy arrays.

        Raises:
            IncompletePatientError: If the patient failed validation.
        """
        self._require_complete(patient)
        return {
            modality: self._load_volume(patient.path_for(modality))
            for modality in MRI_MODALITIES
        }

    def load_segmentation(self, patient: PatientRecord) -> np.ndarray:
        """Load the expert segmentation mask as an integer-label numpy array.

        Raises:
            IncompletePatientError: If the patient failed validation.
        """
        self._require_complete(patient)
        img = nib.load(str(patient.path_for(Modality.SEG)))
        return np.asarray(img.get_fdata(), dtype=np.int16)

    def get_metadata(self, patient: PatientRecord, modality: Modality) -> VolumeMetadata:
        """Return volume metadata (shape, spacing, affine, dtype).

        This does not load the full voxel array: nibabel only needs to read
        the NIfTI header for these fields, so it is cheap to call even
        across an entire dataset.
        """
        img = nib.load(str(patient.path_for(modality)))
        header = img.header
        return VolumeMetadata(
            shape=tuple(img.shape),
            voxel_spacing=tuple(float(z) for z in header.get_zooms()[:3]),
            affine=np.array(img.affine),
            dtype=str(img.get_data_dtype()),
        )

    @staticmethod
    def _load_volume(path: Path) -> np.ndarray:
        img = nib.load(str(path))
        return np.asarray(img.get_fdata(), dtype=np.float32)

    @staticmethod
    def _require_complete(patient: PatientRecord) -> None:
        if not patient.is_complete:
            raise IncompletePatientError(
                f"Patient '{patient.patient_id}' is missing modalities: "
                f"{[m.value for m in patient.missing_modalities]}. "
                "Refusing to load partial data."
            )

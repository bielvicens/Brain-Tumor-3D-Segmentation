"""Tests for BraTSReader.

Run with: pytest -v
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from src.data import (
    BraTSReader,
    DatasetRootNotFoundError,
    IncompletePatientError,
    Modality,
    PatientNotFoundError,
)


def test_root_dir_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetRootNotFoundError):
        BraTSReader(tmp_path / "does_not_exist")


def test_discover_patient_ids_finds_all_folders(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    ids = reader.discover_patient_ids()
    assert ids == sorted(
        ["BraTS-GLI-00000-000", "BraTS20_Training_001", "BraTS-GLI-00002-000"]
    )


def test_get_patients_excludes_incomplete_by_default(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    patients = reader.get_patients()
    ids = {p.patient_id for p in patients}
    assert ids == {"BraTS-GLI-00000-000", "BraTS20_Training_001"}
    assert all(p.is_complete for p in patients)


def test_get_patients_can_include_invalid(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    all_patients = reader.get_patients(only_valid=False)
    assert len(all_patients) == 3
    incomplete = [p for p in all_patients if not p.is_complete]
    assert len(incomplete) == 1
    assert incomplete[0].missing_modalities == [Modality.T2F]


def test_missing_modality_is_logged(synthetic_dataset: Path, caplog) -> None:
    reader = BraTSReader(synthetic_dataset)
    with caplog.at_level(logging.ERROR):
        reader.get_patients()
    assert any("missing required modalities" in message for message in caplog.messages)


def test_legacy_naming_convention_is_supported(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    patient = reader.get_patient("BraTS20_Training_001")
    assert patient.is_complete
    assert set(patient.modality_paths.keys()) == {
        Modality.T1N,
        Modality.T1C,
        Modality.T2W,
        Modality.T2F,
        Modality.SEG,
    }


def test_get_patient_unknown_id_raises(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    with pytest.raises(PatientNotFoundError):
        reader.get_patient("does-not-exist")


def test_load_modalities_returns_expected_shapes_and_dtype(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    patient = reader.get_patient("BraTS-GLI-00000-000")
    volumes = reader.load_modalities(patient)

    assert set(volumes.keys()) == {Modality.T1N, Modality.T1C, Modality.T2W, Modality.T2F}
    for volume in volumes.values():
        assert volume.shape == (4, 4, 4)
        assert volume.dtype == np.float32


def test_load_segmentation_returns_integer_array(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    patient = reader.get_patient("BraTS-GLI-00000-000")
    mask = reader.load_segmentation(patient)
    assert mask.shape == (4, 4, 4)
    assert np.issubdtype(mask.dtype, np.integer)


def test_load_modalities_raises_for_incomplete_patient(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    patient = reader.get_patient("BraTS-GLI-00002-000")
    assert not patient.is_complete
    with pytest.raises(IncompletePatientError):
        reader.load_modalities(patient)


def test_get_metadata_matches_synthetic_volume(synthetic_dataset: Path) -> None:
    reader = BraTSReader(synthetic_dataset)
    patient = reader.get_patient("BraTS-GLI-00000-000")
    metadata = reader.get_metadata(patient, Modality.T1N)
    assert metadata.shape == (4, 4, 4)
    assert len(metadata.voxel_spacing) == 3
    assert metadata.affine.shape == (4, 4)

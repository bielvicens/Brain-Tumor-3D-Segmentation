"""PyTorch Dataset for BraTS MRI segmentation cases.

This module bridges the raw :class:`BraTSReader` / preprocessing pipeline
and PyTorch training code. It deliberately contains no image-processing
logic: loading belongs to ``BraTSReader`` and transformations belong to
``PreprocessingPipeline``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .brats_reader import BraTSReader
from .modality import MRI_MODALITIES
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.preprocessing.pipeline import PreprocessingPipeline


class BraTSDataset(Dataset):
    """PyTorch Dataset exposing BraTS cases as ``(image, mask)`` tensors.

    ``image`` has shape ``(4, D, H, W)`` in the fixed order
    ``T1n, T1c, T2w, T2f`` and dtype ``torch.float32``.
    ``mask`` has shape ``(D, H, W)`` and dtype ``torch.long``.
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        pipeline: Optional[PreprocessingPipeline] = None,
        patient_ids: Optional[Sequence[str]] = None,
        include_segmentation: bool = True,
    ) -> None:
        self.reader = BraTSReader(dataset_root)
        self.pipeline = pipeline
        self.include_segmentation = include_segmentation

        available_ids = self.reader.discover_patient_ids()
        if patient_ids is None:
            self.patient_ids = [
                record.patient_id
                for record in self.reader.get_patients(only_valid=True)
            ]
        else:
            requested = list(patient_ids)
            unknown = sorted(set(requested) - set(available_ids))
            if unknown:
                raise ValueError(f"Unknown patient ID(s): {unknown}")
            records = {
                record.patient_id: record
                for record in self.reader.get_patients(only_valid=True)
            }
            incomplete = sorted(pid for pid in requested if pid not in records)
            if incomplete:
                raise ValueError(
                    f"Patient ID(s) are incomplete or invalid: {incomplete}"
                )
            self.patient_ids = requested

        if not include_segmentation:
            # Images can still be used for inference, but __getitem__ will
            # return a tensor and ``None`` rather than inventing a mask.
            return

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Optional[Tensor]]:
        from src.preprocessing.transforms import PreprocessingSample
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError(f"Dataset index must be an integer, got {type(index).__name__}.")
        if index < 0:
            index += len(self.patient_ids)
        if index < 0 or index >= len(self.patient_ids):
            raise IndexError(f"Dataset index out of range: {index}")

        patient = self.reader.get_patient(self.patient_ids[index])
        modalities = self.reader.load_modalities(patient)

        segmentation = (
            self.reader.load_segmentation(patient)
            if self.include_segmentation
            else None
        )

        metadata = self.reader.get_metadata(patient, MRI_MODALITIES[0])
        sample = PreprocessingSample(
            patient_id=patient.patient_id,
            modalities=modalities,
            segmentation=segmentation,
            voxel_spacing=metadata.voxel_spacing,
            affine=metadata.affine,
        )

        if self.pipeline is not None:
            sample = self.pipeline.run(sample)

        image = torch.stack(
            [
                torch.from_numpy(np.asarray(sample.modalities[modality], dtype=np.float32))
                for modality in MRI_MODALITIES
            ],
            dim=0,
        )

        if sample.segmentation is None:
            if self.include_segmentation:
                raise RuntimeError(
                    f"Patient '{sample.patient_id}' has no segmentation after preprocessing."
                )
            return image, None

        mask = torch.from_numpy(
            np.asarray(sample.segmentation, dtype=np.int64)
        ).to(dtype=torch.long)

        return image, mask

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import BraTSDataset
from src.data.modality import Modality
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.transforms import PreprocessingSample, Transform


def _make_dataset(root: Path) -> Path:
    # Reuse the real NIfTI-writing fixture from the project's conftest through
    # the synthetic_dataset fixture in the actual test suite.
    return root


def test_dataset_len_and_item(synthetic_dataset):
    dataset = BraTSDataset(synthetic_dataset)
    assert len(dataset) == 2
    image, mask = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)


def test_image_has_four_channels_in_fixed_order(synthetic_dataset):
    dataset = BraTSDataset(synthetic_dataset)
    image, _ = dataset[0]
    assert image.shape == (4, 4, 4, 4)
    assert torch.all(image[0] == 1)
    assert torch.all(image[1] == 2)
    assert torch.all(image[2] == 3)
    assert torch.all(image[3] == 4)


def test_tensor_dtypes(synthetic_dataset):
    image, mask = BraTSDataset(synthetic_dataset)[0]
    assert image.dtype == torch.float32
    assert mask.dtype == torch.long


def test_mask_shape_and_values(synthetic_dataset):
    _, mask = BraTSDataset(synthetic_dataset)[0]
    assert mask.shape == (4, 4, 4)
    assert torch.all(mask == 1)


def test_negative_indexing(synthetic_dataset):
    dataset = BraTSDataset(synthetic_dataset)
    assert torch.equal(dataset[-1][0], dataset[1][0])


def test_out_of_range_raises(synthetic_dataset):
    dataset = BraTSDataset(synthetic_dataset)
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_unknown_patient_raises(synthetic_dataset):
    with pytest.raises(ValueError, match="Unknown patient"):
        BraTSDataset(synthetic_dataset, patient_ids=["does-not-exist"])


def test_patient_subset(synthetic_dataset):
    dataset = BraTSDataset(
        synthetic_dataset,
        patient_ids=["BraTS-GLI-00000-000"],
    )
    assert len(dataset) == 1
    assert dataset.patient_ids == ["BraTS-GLI-00000-000"]


def test_default_dataset_excludes_incomplete_patient(synthetic_dataset):
    dataset = BraTSDataset(synthetic_dataset)
    assert "BraTS-GLI-00002-000" not in dataset.patient_ids


def test_pipeline_is_applied(synthetic_dataset):
    class AddOne(Transform):
        def apply(self, sample):
            return sample.replace(
                modalities={
                    modality: volume + 1
                    for modality, volume in sample.modalities.items()
                }
            )

    pipeline = PreprocessingPipeline([AddOne()])
    image, _ = BraTSDataset(synthetic_dataset, pipeline=pipeline)[0]
    assert torch.all(image[0] == 2)


def test_pipeline_can_change_shape(synthetic_dataset):
    class Crop(Transform):
        def apply(self, sample):
            return sample.replace(
                modalities={
                    modality: volume[:2, :2, :2]
                    for modality, volume in sample.modalities.items()
                },
                segmentation=sample.segmentation[:2, :2, :2],
            )

    pipeline = PreprocessingPipeline([Crop()])
    image, mask = BraTSDataset(synthetic_dataset, pipeline=pipeline)[0]
    assert image.shape == (4, 2, 2, 2)
    assert mask.shape == (2, 2, 2)


def test_no_segmentation_returns_none(synthetic_dataset):
    image, mask = BraTSDataset(
        synthetic_dataset,
        include_segmentation=False,
    )[0]
    assert image.shape == (4, 4, 4, 4)
    assert mask is None


def test_empty_patient_subset(synthetic_dataset):
    dataset = BraTSDataset(synthetic_dataset, patient_ids=[])
    assert len(dataset) == 0


def test_dataset_is_torch_dataset(synthetic_dataset):
    assert isinstance(BraTSDataset(synthetic_dataset), torch.utils.data.Dataset)


def test_dataloader_compatibility(synthetic_dataset):
    from torch.utils.data import DataLoader

    loader = DataLoader(BraTSDataset(synthetic_dataset), batch_size=2, shuffle=False)
    images, masks = next(iter(loader))
    assert images.shape == (2, 4, 4, 4, 4)
    assert masks.shape == (2, 4, 4, 4)

from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing.cropping import CropInfo, CroppingTransform
from src.preprocessing.exceptions import InvalidVolumeError
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.transforms import PreprocessingSample


def make_sample(shape=(6, 8, 10), *, with_segmentation=True):
    volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    modalities = {
        Modality.T1N: volume.copy(),
        Modality.T1C: (volume + 1000).copy(),
    }
    segmentation = (
        (volume.astype(np.int16) % 3) if with_segmentation else None
    )
    return PreprocessingSample(
        patient_id="patient-test",
        modalities=modalities,
        segmentation=segmentation,
        voxel_spacing=(1.0, 1.0, 1.0),
        metadata={"existing": "value"},
    )


def test_target_shape_must_have_three_dimensions():
    with pytest.raises(ValueError):
        CroppingTransform((4, 4))
    with pytest.raises(ValueError):
        CroppingTransform((4, 4, 4, 4))


def test_target_shape_must_be_positive():
    with pytest.raises(ValueError):
        CroppingTransform((4, 0, 4))
    with pytest.raises(ValueError):
        CroppingTransform((-1, 4, 4))


def test_crop_is_centered():
    sample = make_sample((6, 8, 10))
    result = CroppingTransform((4, 4, 6)).apply(sample)

    assert result.modalities[Modality.T1N].shape == (4, 4, 6)
    expected = sample.modalities[Modality.T1N][1:5, 2:6, 2:8]
    np.testing.assert_array_equal(result.modalities[Modality.T1N], expected)


def test_all_modalities_receive_identical_crop():
    sample = make_sample()
    result = CroppingTransform((4, 4, 6)).apply(sample)

    np.testing.assert_array_equal(
        result.modalities[Modality.T1C],
        sample.modalities[Modality.T1C][1:5, 2:6, 2:8],
    )


def test_segmentation_receives_identical_crop():
    sample = make_sample()
    result = CroppingTransform((4, 4, 6)).apply(sample)

    np.testing.assert_array_equal(
        result.segmentation,
        sample.segmentation[1:5, 2:6, 2:8],
    )


def test_odd_crop_difference_removes_extra_voxel_from_end():
    sample = make_sample((10, 10, 10))
    result = CroppingTransform((7, 7, 7)).apply(sample)

    info = result.metadata["cropping"]
    assert info.start_indices == (1, 1, 1)
    assert info.end_indices == (8, 8, 8)


def test_equal_shape_is_a_value_preserving_noop():
    sample = make_sample((6, 8, 10))
    result = CroppingTransform((6, 8, 10)).apply(sample)

    np.testing.assert_array_equal(
        result.modalities[Modality.T1N], sample.modalities[Modality.T1N]
    )
    assert result.modalities[Modality.T1N] is not sample.modalities[Modality.T1N]


def test_target_larger_than_input_raises():
    sample = make_sample((6, 8, 10))
    transform = CroppingTransform((7, 8, 10))

    with pytest.raises(InvalidVolumeError, match="larger than input"):
        transform.validate_input(sample)


def test_empty_sample_raises():
    sample = PreprocessingSample(patient_id="empty", modalities={})
    with pytest.raises(InvalidVolumeError):
        CroppingTransform((4, 4, 4)).validate_input(sample)


def test_inconsistent_shapes_are_rejected():
    sample = make_sample()
    sample.modalities[Modality.T2W] = np.zeros((5, 8, 10), dtype=np.float32)

    with pytest.raises(InvalidVolumeError):
        CroppingTransform((4, 4, 6)).validate_input(sample)


def test_segmentation_can_be_absent():
    sample = make_sample(with_segmentation=False)
    result = CroppingTransform((4, 4, 6)).apply(sample)

    assert result.segmentation is None
    assert result.modalities[Modality.T1N].shape == (4, 4, 6)


def test_metadata_is_preserved_and_extended():
    sample = make_sample()
    result = CroppingTransform((4, 4, 6)).apply(sample)

    assert result.metadata["existing"] == "value"
    assert isinstance(result.metadata["cropping"], CropInfo)
    assert result.metadata["cropping"].original_shape == (6, 8, 10)
    assert result.metadata["cropping"].target_shape == (4, 4, 6)


def test_original_sample_is_not_mutated():
    sample = make_sample()
    original_metadata = dict(sample.metadata)
    original_volume = sample.modalities[Modality.T1N].copy()
    original_segmentation = sample.segmentation.copy()

    result = CroppingTransform((4, 4, 6)).apply(sample)

    np.testing.assert_array_equal(sample.modalities[Modality.T1N], original_volume)
    np.testing.assert_array_equal(sample.segmentation, original_segmentation)
    assert sample.metadata == original_metadata
    assert result is not sample


def test_returned_arrays_are_independent_from_original():
    sample = make_sample()
    result = CroppingTransform((4, 4, 6)).apply(sample)

    result.modalities[Modality.T1N][0, 0, 0] = -999
    assert sample.modalities[Modality.T1N][1, 2, 2] != -999


def test_voxel_spacing_is_preserved():
    sample = make_sample()
    result = CroppingTransform((4, 4, 6)).apply(sample)

    assert result.voxel_spacing == sample.voxel_spacing


def test_affine_is_preserved():
    sample = make_sample()
    affine = np.eye(4)
    sample.affine = affine

    result = CroppingTransform((4, 4, 6)).apply(sample)

    assert result.affine is affine


def test_dtypes_are_preserved():
    sample = make_sample()
    sample.modalities[Modality.T1N] = sample.modalities[Modality.T1N].astype(np.float64)
    sample.segmentation = sample.segmentation.astype(np.int16)

    result = CroppingTransform((4, 4, 6)).apply(sample)

    assert result.modalities[Modality.T1N].dtype == np.float64
    assert result.segmentation.dtype == np.int16


def test_pipeline_integration():
    sample = make_sample()
    pipeline = PreprocessingPipeline([CroppingTransform((4, 4, 6))])

    result = pipeline.run(sample)

    assert result.modalities[Modality.T1N].shape == (4, 4, 6)
    assert result.segmentation.shape == (4, 4, 6)


def test_repr_uses_transform_name():
    assert repr(CroppingTransform((4, 4, 4))) == "CroppingTransform()"

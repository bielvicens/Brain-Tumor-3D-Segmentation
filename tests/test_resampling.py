"""Tests for src.preprocessing.resampling.ResamplingTransform.

Uses small, fully synthetic in-memory volumes - no BraTSReader, no
filesystem - consistent with the rest of the preprocessing test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing.exceptions import InvalidVolumeError
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.resampling import (
    ResamplingInfo,
    ResamplingTransform,
    compute_resampled_shape,
)
from src.preprocessing.transforms import PreprocessingSample


def _sample(**overrides) -> PreprocessingSample:
    defaults = dict(
        patient_id="patient-0",
        modalities={Modality.T1N: np.ones((4, 4, 8), dtype=np.float32)},
        voxel_spacing=(1.0, 1.0, 2.0),
    )
    defaults.update(overrides)
    return PreprocessingSample(**defaults)


# ----------------------------------------------------------------------
# target_spacing validation (constructor)
# ----------------------------------------------------------------------
def test_valid_target_spacing_is_accepted() -> None:
    transform = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0))
    assert transform.target_spacing == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "bad_spacing",
    [
        (1.0, 1.0),  # wrong length
        (1.0, 1.0, 1.0, 1.0),  # wrong length
        (0.0, 1.0, 1.0),  # zero
        (-1.0, 1.0, 1.0),  # negative
        (float("inf"), 1.0, 1.0),  # infinite
        (float("nan"), 1.0, 1.0),  # NaN
    ],
)
def test_invalid_target_spacing_raises_value_error(bad_spacing) -> None:
    with pytest.raises(ValueError):
        ResamplingTransform(target_spacing=bad_spacing)


def test_non_iterable_target_spacing_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ResamplingTransform(target_spacing=1.0)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# voxel_spacing validation (on the sample)
# ----------------------------------------------------------------------
def test_missing_voxel_spacing_raises() -> None:
    sample = _sample(voxel_spacing=None)
    transform = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0))
    with pytest.raises(InvalidVolumeError, match="voxel_spacing"):
        transform.validate_input(sample)


@pytest.mark.parametrize("bad_spacing", [(1.0, 1.0), (0.0, 1.0, 1.0), (-1.0, 1.0, 1.0)])
def test_invalid_sample_voxel_spacing_raises(bad_spacing) -> None:
    sample = _sample(voxel_spacing=bad_spacing)
    transform = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0))
    with pytest.raises(InvalidVolumeError):
        transform.validate_input(sample)


def test_mismatched_shapes_across_modalities_raises() -> None:
    sample = _sample(
        modalities={
            Modality.T1N: np.ones((4, 4, 8), dtype=np.float32),
            Modality.T1C: np.ones((5, 5, 5), dtype=np.float32),
        }
    )
    transform = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0))
    with pytest.raises(InvalidVolumeError):
        transform.validate_input(sample)


def test_no_modalities_and_no_segmentation_raises() -> None:
    sample = _sample(modalities={}, segmentation=None)
    transform = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0))
    with pytest.raises(InvalidVolumeError, match="no modalities and no segmentation"):
        transform.validate_input(sample)


# ----------------------------------------------------------------------
# compute_resampled_shape (pure function)
# ----------------------------------------------------------------------
def test_compute_resampled_shape_matches_the_documented_formula() -> None:
    # spacing halves on the z axis -> twice as many voxels on that axis.
    new_shape = compute_resampled_shape((4, 4, 8), (1.0, 1.0, 2.0), (1.0, 1.0, 1.0))
    assert new_shape == (4, 4, 16)


def test_compute_resampled_shape_downsamples_correctly() -> None:
    new_shape = compute_resampled_shape((8, 8, 8), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    assert new_shape == (4, 4, 4)


def test_compute_resampled_shape_identity_when_spacing_matches() -> None:
    new_shape = compute_resampled_shape((7, 9, 11), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert new_shape == (7, 9, 11)


def test_compute_resampled_shape_never_returns_zero_sized_axis() -> None:
    new_shape = compute_resampled_shape((2, 2, 2), (0.1, 0.1, 0.1), (100.0, 100.0, 100.0))
    assert all(size >= 1 for size in new_shape)


def test_compute_resampled_shape_handles_different_factors_per_axis() -> None:
    new_shape = compute_resampled_shape((10, 20, 30), (1.0, 2.0, 0.5), (2.0, 1.0, 1.0))
    # x: 10 * 1.0/2.0 = 5 | y: 20 * 2.0/1.0 = 40 | z: 30 * 0.5/1.0 = 15
    assert new_shape == (5, 40, 15)


# ----------------------------------------------------------------------
# End-to-end shape / spacing correctness
# ----------------------------------------------------------------------
def test_apply_changes_shape_correctly() -> None:
    sample = _sample(
        modalities={Modality.T1N: np.ones((4, 4, 8), dtype=np.float32)},
        voxel_spacing=(1.0, 1.0, 2.0),
    )
    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)
    assert result.modalities[Modality.T1N].shape == (4, 4, 16)


def test_apply_updates_voxel_spacing_to_exactly_the_target() -> None:
    sample = _sample(voxel_spacing=(1.0, 1.0, 2.0))
    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)
    assert result.voxel_spacing == (1.0, 1.0, 1.0)


def test_all_modalities_are_resampled() -> None:
    sample = _sample(
        modalities={
            Modality.T1N: np.ones((4, 4, 8), dtype=np.float32),
            Modality.T1C: np.full((4, 4, 8), 2.0, dtype=np.float32),
            Modality.T2W: np.full((4, 4, 8), 3.0, dtype=np.float32),
            Modality.T2F: np.full((4, 4, 8), 4.0, dtype=np.float32),
        },
        voxel_spacing=(1.0, 1.0, 2.0),
    )
    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)

    assert set(result.modalities.keys()) == {
        Modality.T1N, Modality.T1C, Modality.T2W, Modality.T2F,
    }
    for volume in result.modalities.values():
        assert volume.shape == (4, 4, 16)


def test_segmentation_is_resampled() -> None:
    seg = np.zeros((4, 4, 8), dtype=np.int16)
    seg[1:3, 1:3, 2:6] = 1
    sample = _sample(segmentation=seg, voxel_spacing=(1.0, 1.0, 2.0))

    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)

    assert result.segmentation is not None
    assert result.segmentation.shape == (4, 4, 16)


def test_modalities_and_segmentation_share_the_same_shape_after_resampling() -> None:
    seg = np.zeros((4, 4, 8), dtype=np.int16)
    seg[1:3, 1:3, 2:6] = 1
    sample = _sample(
        modalities={
            Modality.T1N: np.ones((4, 4, 8), dtype=np.float32),
            Modality.T1C: np.full((4, 4, 8), 2.0, dtype=np.float32),
        },
        segmentation=seg,
        voxel_spacing=(1.0, 1.7, 2.3),  # awkward, non-round factors on purpose
    )

    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)

    shapes = {v.shape for v in result.modalities.values()}
    shapes.add(result.segmentation.shape)
    assert len(shapes) == 1  # every array ended up with exactly the same shape


# ----------------------------------------------------------------------
# Segmentation uses nearest-neighbor: no artificial labels
# ----------------------------------------------------------------------
def test_segmentation_uses_nearest_neighbor_no_new_labels_appear() -> None:
    seg = np.zeros((6, 6, 6), dtype=np.int16)
    seg[1:3, 1:3, 1:3] = 1
    seg[3:5, 3:5, 3:5] = 2
    original_labels = set(np.unique(seg).tolist())

    sample = _sample(segmentation=seg, voxel_spacing=(1.0, 1.0, 1.7))
    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)

    resampled_labels = set(np.unique(result.segmentation).tolist())
    assert resampled_labels.issubset(original_labels)


def test_segmentation_interpolation_stays_hard_edged() -> None:
    # A hard-edged mask: nearest-neighbor must keep it hard-edged (only
    # 0s and 1s), never introducing fractional/blended values.
    seg = np.zeros((4, 4, 4), dtype=np.int16)
    seg[2:, :, :] = 1
    sample = _sample(segmentation=seg, voxel_spacing=(1.0, 1.0, 1.0))

    result = ResamplingTransform(target_spacing=(1.0, 1.0, 0.5)).apply(sample)

    assert set(np.unique(result.segmentation).tolist()).issubset({0, 1})


# ----------------------------------------------------------------------
# dtype
# ----------------------------------------------------------------------
def test_modalities_are_returned_as_float32() -> None:
    sample = _sample(modalities={Modality.T1N: np.ones((4, 4, 8), dtype=np.float64)})
    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)
    assert result.modalities[Modality.T1N].dtype == np.float32


def test_segmentation_preserves_integer_dtype() -> None:
    seg = np.zeros((4, 4, 8), dtype=np.int16)
    seg[1:3, 1:3, 2:6] = 1
    sample = _sample(segmentation=seg, voxel_spacing=(1.0, 1.0, 2.0))
    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)
    assert np.issubdtype(result.segmentation.dtype, np.integer)
    assert result.segmentation.dtype == seg.dtype


# ----------------------------------------------------------------------
# metadata
# ----------------------------------------------------------------------
def test_pre_existing_metadata_is_preserved() -> None:
    sample = _sample(metadata={"normalization": "some-earlier-value"})
    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)
    assert result.metadata["normalization"] == "some-earlier-value"


def test_resampling_metadata_is_correct() -> None:
    sample = _sample(
        modalities={Modality.T1N: np.ones((4, 4, 8), dtype=np.float32)},
        voxel_spacing=(1.0, 1.0, 2.0),
    )
    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)

    info = result.metadata["resampling"]
    assert isinstance(info, ResamplingInfo)
    assert info.original_spacing == (1.0, 1.0, 2.0)
    assert info.target_spacing == (1.0, 1.0, 1.0)
    assert info.original_shape == (4, 4, 8)
    assert info.new_shape == (4, 4, 16)


# ----------------------------------------------------------------------
# Immutability
# ----------------------------------------------------------------------
def test_original_sample_is_not_modified() -> None:
    original_volume = np.ones((4, 4, 8), dtype=np.float32)
    original = _sample(modalities={Modality.T1N: original_volume}, voxel_spacing=(1.0, 1.0, 2.0))

    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(original)

    assert original.modalities[Modality.T1N].shape == (4, 4, 8)
    assert original.voxel_spacing == (1.0, 1.0, 2.0)
    assert "resampling" not in original.metadata
    assert result is not original
    assert result.modalities[Modality.T1N] is not original.modalities[Modality.T1N]


# ----------------------------------------------------------------------
# Integration with PreprocessingPipeline
# ----------------------------------------------------------------------
def test_resampling_works_inside_a_pipeline() -> None:
    sample = _sample(voxel_spacing=(1.0, 1.0, 2.0))
    pipeline = PreprocessingPipeline([ResamplingTransform(target_spacing=(1.0, 1.0, 1.0))])

    result = pipeline.run(sample)

    assert result.modalities[Modality.T1N].shape == (4, 4, 16)
    assert result.voxel_spacing == (1.0, 1.0, 1.0)
    assert "resampling" in result.metadata


def test_resampling_chains_after_normalization_in_a_pipeline() -> None:
    from src.preprocessing.normalization import ZScoreNormalization

    volume = np.zeros((4, 4, 8), dtype=np.float32)
    volume[1:3, 1:3, 2:6] = 10.0
    sample = _sample(modalities={Modality.T1N: volume}, voxel_spacing=(1.0, 1.0, 2.0))

    pipeline = PreprocessingPipeline(
        [ZScoreNormalization(), ResamplingTransform(target_spacing=(1.0, 1.0, 1.0))]
    )
    result = pipeline.run(sample)

    assert "normalization" in result.metadata  # from step 1
    assert "resampling" in result.metadata  # from step 2
    assert result.modalities[Modality.T1N].shape == (4, 4, 16)


# ----------------------------------------------------------------------
# Spacing already equal to target
# ----------------------------------------------------------------------
def test_spacing_already_equal_to_target_is_a_no_op_on_values() -> None:
    volume = np.arange(4 * 4 * 8, dtype=np.float32).reshape((4, 4, 8))
    sample = _sample(modalities={Modality.T1N: volume}, voxel_spacing=(1.0, 1.0, 1.0))

    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)

    assert result.modalities[Modality.T1N].shape == volume.shape
    np.testing.assert_array_equal(result.modalities[Modality.T1N], volume)
    assert result.metadata["resampling"].original_shape == result.metadata["resampling"].new_shape


# ----------------------------------------------------------------------
# Different scale factors per axis
# ----------------------------------------------------------------------
def test_different_scale_factors_per_axis() -> None:
    sample = _sample(
        modalities={Modality.T1N: np.ones((4, 8, 6), dtype=np.float32)},
        voxel_spacing=(1.0, 2.0, 0.5),
    )
    # x: no change | y: upsample x2 | z: downsample x2
    result = ResamplingTransform(target_spacing=(1.0, 1.0, 1.0)).apply(sample)
    assert result.modalities[Modality.T1N].shape == (4, 16, 3)

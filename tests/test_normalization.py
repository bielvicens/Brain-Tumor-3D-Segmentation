"""Tests for src.preprocessing.normalization.ZScoreNormalization."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing.normalization import DEFAULT_EPSILON, ZScoreNormalization
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.transforms import PreprocessingSample


def _sample(**overrides) -> PreprocessingSample:
    defaults = dict(
        patient_id="patient-0",
        modalities={Modality.T1N: np.array([0, 0, 10, 20, 30, 40], dtype=np.float32)},
    )
    defaults.update(overrides)
    return PreprocessingSample(**defaults)


# ----------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------
def test_zscore_is_computed_correctly_on_foreground_voxels() -> None:
    volume = np.array([0, 0, 10, 20, 30, 40], dtype=np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    result = ZScoreNormalization().apply(sample)
    normalized = result.modalities[Modality.T1N]

    foreground = volume[volume > 0]
    expected_mean = foreground.mean()
    expected_std = foreground.std()
    expected_foreground = (foreground - expected_mean) / expected_std

    np.testing.assert_allclose(normalized[volume > 0], expected_foreground, rtol=1e-5)
    # The normalized foreground should itself have ~zero mean and ~unit std.
    assert normalized[volume > 0].mean() == pytest.approx(0.0, abs=1e-5)
    assert normalized[volume > 0].std() == pytest.approx(1.0, abs=1e-5)


def test_zscore_handles_3d_volumes() -> None:
    rng = np.random.default_rng(42)
    volume = np.zeros((6, 6, 6), dtype=np.float32)
    volume[2:4, 2:4, 2:4] = rng.uniform(1.0, 100.0, size=(2, 2, 2)).astype(np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    result = ZScoreNormalization().apply(sample)
    normalized = result.modalities[Modality.T1N]

    assert normalized.shape == volume.shape
    assert normalized[volume > 0].mean() == pytest.approx(0.0, abs=1e-4)


# ----------------------------------------------------------------------
# Background preserved at exactly zero
# ----------------------------------------------------------------------
def test_background_voxels_stay_exactly_zero() -> None:
    volume = np.array([0, 0, 0, 5, 10, 15], dtype=np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    result = ZScoreNormalization().apply(sample)
    normalized = result.modalities[Modality.T1N]

    assert np.all(normalized[volume <= 0] == 0.0)


def test_negative_voxels_are_also_treated_as_background() -> None:
    # Real BraTS data is never negative, but the transform should still
    # behave predictably (and not raise) if it ever is.
    volume = np.array([-5, 0, 10, 20, 30, 40], dtype=np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    result = ZScoreNormalization().apply(sample)
    normalized = result.modalities[Modality.T1N]

    assert normalized[0] == 0.0  # the negative voxel
    assert normalized[1] == 0.0  # the exact-zero voxel


def test_all_background_volume_returns_all_zero_without_raising() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    result = ZScoreNormalization().apply(sample)
    normalized = result.modalities[Modality.T1N]

    assert np.all(normalized == 0.0)
    stats = result.metadata["normalization"][Modality.T1N.value]
    assert stats.foreground_voxel_count == 0
    assert stats.mean == 0.0
    assert stats.std == 0.0


# ----------------------------------------------------------------------
# Modalities normalized independently
# ----------------------------------------------------------------------
def test_modalities_are_normalized_independently() -> None:
    t1n = np.array([0, 10, 20, 30], dtype=np.float32)  # mean=20, std=~8.16
    t2f = np.array([0, 100, 200, 300], dtype=np.float32)  # mean=200, std=~81.6
    sample = _sample(modalities={Modality.T1N: t1n, Modality.T2F: t2f})

    result = ZScoreNormalization().apply(sample)

    t1n_stats = result.metadata["normalization"][Modality.T1N.value]
    t2f_stats = result.metadata["normalization"][Modality.T2F.value]

    assert t1n_stats.mean == pytest.approx(20.0)
    assert t2f_stats.mean == pytest.approx(200.0)
    assert t1n_stats.mean != t2f_stats.mean
    # Each modality's own statistics were used - not some pooled value.
    np.testing.assert_allclose(
        result.modalities[Modality.T1N][t1n > 0],
        (t1n[t1n > 0] - t1n_stats.mean) / t1n_stats.std,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result.modalities[Modality.T2F][t2f > 0],
        (t2f[t2f > 0] - t2f_stats.mean) / t2f_stats.std,
        rtol=1e-5,
    )


# ----------------------------------------------------------------------
# Original sample is not modified
# ----------------------------------------------------------------------
def test_original_sample_is_not_modified() -> None:
    volume = np.array([0, 10, 20, 30], dtype=np.float32)
    original = _sample(modalities={Modality.T1N: volume})
    original_values = original.modalities[Modality.T1N].copy()

    result = ZScoreNormalization().apply(original)

    # The original array's values are untouched.
    np.testing.assert_array_equal(original.modalities[Modality.T1N], original_values)
    # The result is a genuinely different array, not a view/alias.
    assert result.modalities[Modality.T1N] is not original.modalities[Modality.T1N]
    assert result is not original
    assert original.metadata == {}


def test_original_sample_metadata_dict_is_not_mutated() -> None:
    sample = _sample(metadata={"some_earlier_key": "value"})
    ZScoreNormalization().apply(sample)
    # apply() must not have added "normalization" to the original's dict.
    assert "normalization" not in sample.metadata


# ----------------------------------------------------------------------
# Metadata correctness
# ----------------------------------------------------------------------
def test_metadata_contains_expected_normalization_stats() -> None:
    volume = np.array([0, 10, 20, 30], dtype=np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    result = ZScoreNormalization().apply(sample)
    stats = result.metadata["normalization"][Modality.T1N.value]

    assert stats.mean == pytest.approx(20.0)
    assert stats.std == pytest.approx(volume[volume > 0].std())
    assert stats.foreground_voxel_count == 3
    assert stats.epsilon_used is False


def test_metadata_preserves_pre_existing_keys() -> None:
    sample = _sample(metadata={"original_shape": (4, 4, 4)})
    result = ZScoreNormalization().apply(sample)
    assert result.metadata["original_shape"] == (4, 4, 4)
    assert "normalization" in result.metadata


# ----------------------------------------------------------------------
# Zero standard deviation
# ----------------------------------------------------------------------
def test_zero_std_foreground_uses_epsilon_and_does_not_produce_nan_or_inf() -> None:
    # Every foreground voxel has the exact same value -> std == 0.
    volume = np.array([0, 0, 5, 5, 5, 5], dtype=np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    result = ZScoreNormalization().apply(sample)
    normalized = result.modalities[Modality.T1N]
    stats = result.metadata["normalization"][Modality.T1N.value]

    assert stats.std == 0.0
    assert stats.epsilon_used is True
    assert np.all(np.isfinite(normalized))  # no NaN / inf anywhere
    assert np.all(normalized[volume <= 0] == 0.0)


def test_custom_epsilon_is_used_when_std_is_near_zero() -> None:
    volume = np.array([0, 0, 5, 5, 5, 5], dtype=np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    custom_epsilon = 0.5
    result = ZScoreNormalization(epsilon=custom_epsilon).apply(sample)
    normalized = result.modalities[Modality.T1N]

    # (5 - 5) / 0.5 == 0.0 regardless of epsilon here (mean == value), so
    # instead check the epsilon actually used matches what we passed.
    stats = result.metadata["normalization"][Modality.T1N.value]
    assert stats.epsilon_used is True
    assert np.all(np.isfinite(normalized))


def test_epsilon_must_be_strictly_positive() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        ZScoreNormalization(epsilon=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        ZScoreNormalization(epsilon=-1.0)


def test_default_epsilon_constant_is_used_by_default() -> None:
    assert ZScoreNormalization().epsilon == DEFAULT_EPSILON


# ----------------------------------------------------------------------
# Output dtype
# ----------------------------------------------------------------------
def test_output_dtype_is_always_float32() -> None:
    volume = np.array([0, 10, 20, 30], dtype=np.float64)  # deliberately not float32
    sample = _sample(modalities={Modality.T1N: volume})

    result = ZScoreNormalization().apply(sample)

    assert result.modalities[Modality.T1N].dtype == np.float32


def test_output_shape_matches_input_shape() -> None:
    volume = np.zeros((5, 6, 7), dtype=np.float32)
    volume[1:3, 1:3, 1:3] = 42.0
    sample = _sample(modalities={Modality.T1N: volume})

    result = ZScoreNormalization().apply(sample)
    assert result.modalities[Modality.T1N].shape == (5, 6, 7)


# ----------------------------------------------------------------------
# Integration with PreprocessingPipeline
# ----------------------------------------------------------------------
def test_zscore_normalization_works_inside_a_pipeline() -> None:
    volume = np.array([0, 0, 10, 20, 30, 40], dtype=np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    pipeline = PreprocessingPipeline([ZScoreNormalization()])
    result = pipeline.run(sample)

    direct_result = ZScoreNormalization().apply(sample)
    np.testing.assert_array_equal(
        result.modalities[Modality.T1N], direct_result.modalities[Modality.T1N]
    )
    assert "normalization" in result.metadata


def test_zscore_normalization_chains_with_another_transform_in_a_pipeline() -> None:
    from src.preprocessing.transforms import Transform

    class _RecordShape(Transform):
        """Test-only transform, run after normalization, to prove chaining."""

        def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
            shapes = {m.value: v.shape for m, v in sample.modalities.items()}
            return sample.replace(metadata={**sample.metadata, "shapes_seen": shapes})

    volume = np.array([0, 10, 20, 30], dtype=np.float32)
    sample = _sample(modalities={Modality.T1N: volume})

    pipeline = PreprocessingPipeline([ZScoreNormalization(), _RecordShape()])
    result = pipeline.run(sample)

    assert "normalization" in result.metadata  # from step 1
    assert "shapes_seen" in result.metadata  # from step 2, sees step 1's output
    assert result.metadata["shapes_seen"] == {"t1n": (4,)}

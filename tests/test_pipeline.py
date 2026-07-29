"""Tests for src.preprocessing.pipeline.PreprocessingPipeline.

Uses small local dummy Transform subclasses (not real preprocessing) to
exercise the pipeline's orchestration logic in isolation: ordering, error
wrapping, and configuration validation.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing.exceptions import (
    InvalidVolumeError,
    PipelineConfigurationError,
    TransformError,
)
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.transforms import PreprocessingSample, Transform


def _sample(**overrides) -> PreprocessingSample:
    defaults = dict(
        patient_id="patient-0",
        modalities={Modality.T1N: np.zeros((2, 2, 2), dtype=np.float32)},
    )
    defaults.update(overrides)
    return PreprocessingSample(**defaults)


class _AddConstant(Transform):
    """Test-only transform: adds a constant to every modality."""

    def __init__(self, amount: float) -> None:
        self.amount = amount

    def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
        new_modalities = {m: v + self.amount for m, v in sample.modalities.items()}
        return sample.replace(modalities=new_modalities)


class _RecordingTransform(Transform):
    """Test-only transform: records that it ran, to assert call order."""

    def __init__(self, log: List[str]) -> None:
        self.log = log

    def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
        self.log.append(self.name)
        return sample


class _FailingTransform(Transform):
    """Test-only transform: always raises, to test pipeline error wrapping."""

    def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
        raise RuntimeError("something went wrong inside the transform")


class _FailingValidateInput(Transform):
    """Test-only transform whose pre-condition check fails, not its apply."""

    def validate_input(self, sample: PreprocessingSample) -> None:
        raise InvalidVolumeError("this transform's pre-condition was not met")

    def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
        return sample


# ----------------------------------------------------------------------
# Construction / configuration
# ----------------------------------------------------------------------
def test_pipeline_requires_at_least_one_transform() -> None:
    with pytest.raises(PipelineConfigurationError, match="at least one"):
        PreprocessingPipeline([])


def test_pipeline_rejects_non_transform_items() -> None:
    with pytest.raises(PipelineConfigurationError, match="not a Transform instance"):
        PreprocessingPipeline([_AddConstant(1.0), "not a transform"])  # type: ignore[list-item]


def test_pipeline_len_and_iter_expose_the_transforms() -> None:
    transforms = [_AddConstant(1.0), _AddConstant(2.0)]
    pipeline = PreprocessingPipeline(transforms)

    assert len(pipeline) == 2
    assert list(pipeline) == transforms


def test_pipeline_repr_shows_step_order() -> None:
    pipeline = PreprocessingPipeline([_AddConstant(1.0), _AddConstant(2.0)])
    assert repr(pipeline) == "PreprocessingPipeline([_AddConstant -> _AddConstant])"


# ----------------------------------------------------------------------
# Running the pipeline
# ----------------------------------------------------------------------
def test_pipeline_applies_transforms_in_order() -> None:
    log: List[str] = []
    first = _RecordingTransform(log)
    second = _RecordingTransform(log)
    pipeline = PreprocessingPipeline([first, second])

    pipeline.run(_sample())

    assert log == ["_RecordingTransform", "_RecordingTransform"]


def test_pipeline_chains_output_of_one_transform_into_the_next() -> None:
    pipeline = PreprocessingPipeline([_AddConstant(1.0), _AddConstant(10.0)])
    result = pipeline.run(_sample())
    assert np.all(result.modalities[Modality.T1N] == 11.0)


def test_pipeline_call_is_equivalent_to_run() -> None:
    pipeline = PreprocessingPipeline([_AddConstant(5.0)])
    result = pipeline(_sample())
    assert np.all(result.modalities[Modality.T1N] == 5.0)


def test_pipeline_does_not_mutate_the_original_sample() -> None:
    pipeline = PreprocessingPipeline([_AddConstant(1.0)])
    original = _sample()
    pipeline.run(original)
    assert np.all(original.modalities[Modality.T1N] == 0.0)


# ----------------------------------------------------------------------
# Validation before running
# ----------------------------------------------------------------------
def test_pipeline_validates_input_by_default() -> None:
    pipeline = PreprocessingPipeline([_AddConstant(1.0)])
    invalid_sample = _sample(modalities={})  # fails validate_modalities_present
    with pytest.raises(InvalidVolumeError):
        pipeline.run(invalid_sample)


def test_pipeline_can_skip_validation() -> None:
    pipeline = PreprocessingPipeline([_AddConstant(1.0)], validate_input=False)
    invalid_sample = _sample(modalities={})
    # Without validation, the pipeline runs the transform straight away -
    # _AddConstant on an empty modalities dict is a no-op, so no error.
    result = pipeline.run(invalid_sample)
    assert result.modalities == {}


# ----------------------------------------------------------------------
# Error wrapping
# ----------------------------------------------------------------------
def test_pipeline_wraps_apply_failures_in_transform_error() -> None:
    pipeline = PreprocessingPipeline([_FailingTransform()])
    with pytest.raises(TransformError, match="_FailingTransform.*step 1/1.*patient-0"):
        pipeline.run(_sample())


def test_pipeline_preserves_original_exception_as_cause() -> None:
    pipeline = PreprocessingPipeline([_FailingTransform()])
    with pytest.raises(TransformError) as exc_info:
        pipeline.run(_sample())
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "something went wrong" in str(exc_info.value.__cause__)


def test_pipeline_wraps_validate_input_failures_too() -> None:
    pipeline = PreprocessingPipeline([_FailingValidateInput()])
    with pytest.raises(TransformError, match="pre-condition was not met"):
        pipeline.run(_sample())


def test_pipeline_reports_correct_step_number_on_failure() -> None:
    pipeline = PreprocessingPipeline([_AddConstant(1.0), _FailingTransform(), _AddConstant(2.0)])
    with pytest.raises(TransformError, match=r"step 2/3"):
        pipeline.run(_sample())


def test_pipeline_stops_before_later_transforms_on_failure() -> None:
    log: List[str] = []
    recorder_before = _RecordingTransform(log)
    recorder_after = _RecordingTransform(log)
    pipeline = PreprocessingPipeline([recorder_before, _FailingTransform(), recorder_after])

    with pytest.raises(TransformError):
        pipeline.run(_sample())

    # Only the transform before the failure should have run.
    assert log == ["_RecordingTransform"]

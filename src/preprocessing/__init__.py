"""Preprocessing pipeline architecture for the BraTS dataset.

This package provides a chainable transform system
(:class:`Transform`, :class:`PreprocessingSample`), a pipeline that runs
transforms in order (:class:`PreprocessingPipeline`), validation that
runs before any transform (:func:`validate_sample`), and a growing set of
concrete transforms: :class:`ZScoreNormalization` (Module 3.2) and
:class:`ResamplingTransform` (Module 3.3).

Cropping and padding are not implemented yet; they will be added as
further ``Transform`` subclasses, without requiring any change to the
pipeline or validation logic defined here.
"""

from .exceptions import (
    InvalidVolumeError,
    PipelineConfigurationError,
    PreprocessingError,
    TransformError,
)
from .normalization import DEFAULT_EPSILON, NormalizationStats, ZScoreNormalization
from .pipeline import PreprocessingPipeline
from .resampling import ResamplingInfo, ResamplingTransform, compute_resampled_shape
from .transforms import PreprocessingSample, Transform
from .validation import DEFAULT_CHECKS, validate_sample

__all__ = [
    "PreprocessingPipeline",
    "PreprocessingSample",
    "Transform",
    "ZScoreNormalization",
    "NormalizationStats",
    "DEFAULT_EPSILON",
    "ResamplingTransform",
    "ResamplingInfo",
    "compute_resampled_shape",
    "PreprocessingError",
    "InvalidVolumeError",
    "TransformError",
    "PipelineConfigurationError",
    "validate_sample",
    "DEFAULT_CHECKS",
]

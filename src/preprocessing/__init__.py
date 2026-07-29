"""Preprocessing pipeline architecture for the BraTS dataset.

This package defines the *architecture* only: a chainable transform
system (:class:`Transform`, :class:`PreprocessingSample`), a pipeline that
runs transforms in order (:class:`PreprocessingPipeline`), and validation
that runs before any transform (:func:`validate_sample`).

No preprocessing math (normalization, resampling, cropping, padding) is
implemented here yet - those will be added in a later module as concrete
``Transform`` subclasses, without requiring any change to the pipeline or
validation logic defined here.
"""

from .exceptions import (
    InvalidVolumeError,
    PipelineConfigurationError,
    PreprocessingError,
    TransformError,
)
from .pipeline import PreprocessingPipeline
from .transforms import PreprocessingSample, Transform
from .validation import DEFAULT_CHECKS, validate_sample

__all__ = [
    "PreprocessingPipeline",
    "PreprocessingSample",
    "Transform",
    "PreprocessingError",
    "InvalidVolumeError",
    "TransformError",
    "PipelineConfigurationError",
    "validate_sample",
    "DEFAULT_CHECKS",
]

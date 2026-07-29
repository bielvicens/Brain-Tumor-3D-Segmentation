"""Custom exceptions for the preprocessing module."""

from __future__ import annotations


class PreprocessingError(Exception):
    """Base class for all errors raised by the preprocessing module."""


class InvalidVolumeError(PreprocessingError):
    """Raised when a sample fails validation before preprocessing.

    See :mod:`src.preprocessing.validation`.
    """


class TransformError(PreprocessingError):
    """Raised when a Transform fails while being applied.

    The pipeline wraps the original exception in one of these so the
    traceback always makes clear which transform, which step, and which
    patient failed - the original exception is preserved as
    ``__cause__`` (via ``raise ... from exc``), so no information is lost.
    """


class PipelineConfigurationError(PreprocessingError):
    """Raised when a PreprocessingPipeline itself is misconfigured.

    E.g. an empty transform list, or an item that isn't a ``Transform``.
    This is distinct from ``InvalidVolumeError`` (bad *data*) and from
    ``TransformError`` (a transform failed while *running*): this one
    means the pipeline was built incorrectly, before any data was involved.
    """

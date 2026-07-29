"""Chains preprocessing transforms into a single, ordered pipeline.

This module contains no preprocessing logic itself (no normalization,
resampling, cropping or padding math) - it only orchestrates: validate
the incoming sample, then apply each registered :class:`~src.preprocessing.transforms.Transform`
in order, attaching clear context to any failure. Concrete transforms are
added in a later module; this pipeline works with any object implementing
the ``Transform`` interface, so no change here will be needed when they
arrive.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Sequence

from .exceptions import PipelineConfigurationError, TransformError
from .transforms import PreprocessingSample, Transform
from .validation import validate_sample

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Applies an ordered sequence of Transforms to a PreprocessingSample.

    Example:
        >>> pipeline = PreprocessingPipeline([SomeTransform(), AnotherTransform()])
        >>> result = pipeline.run(sample)

    Transforms run strictly in the order given to the constructor; each
    receives the sample returned by the previous one.
    """

    def __init__(self, transforms: Sequence[Transform], validate_input: bool = True) -> None:
        """
        Args:
            transforms: Ordered sequence of ``Transform`` instances. Must
                contain at least one transform.
            validate_input: Whether to validate the sample (see
                :func:`~src.preprocessing.validation.validate_sample`)
                before running any transform. Disable only for trusted,
                already-validated input.

        Raises:
            PipelineConfigurationError: If ``transforms`` is empty or
                contains anything that isn't a ``Transform`` instance.
        """
        self._transforms: List[Transform] = list(transforms)
        self._validate_input = validate_input
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not self._transforms:
            raise PipelineConfigurationError(
                "PreprocessingPipeline requires at least one Transform."
            )
        for index, transform in enumerate(self._transforms):
            if not isinstance(transform, Transform):
                raise PipelineConfigurationError(
                    f"Item {index} ('{transform!r}') is not a Transform instance."
                )

    def run(self, sample: PreprocessingSample) -> PreprocessingSample:
        """Validate (optionally) and transform one sample.

        Args:
            sample: The input sample, typically built from
                ``BraTSReader.load_modalities`` / ``load_segmentation``.

        Returns:
            The sample after every transform has been applied, in order.

        Raises:
            InvalidVolumeError: If ``validate_input`` is enabled and the
                sample fails validation.
            TransformError: If any transform's ``validate_input`` or
                ``apply`` raises. The original exception is preserved as
                ``__cause__``.
        """
        if self._validate_input:
            validate_sample(sample)

        logger.info(
            "Running %d-step pipeline on patient '%s'.", len(self._transforms), sample.patient_id
        )

        current = sample
        for step_index, transform in enumerate(self._transforms):
            logger.debug(
                "Step %d/%d: applying '%s' to patient '%s'.",
                step_index + 1,
                len(self._transforms),
                transform.name,
                sample.patient_id,
            )
            current = self._run_step(transform, current, step_index)

        logger.info("Pipeline completed for patient '%s'.", sample.patient_id)
        return current

    def _run_step(
        self, transform: Transform, sample: PreprocessingSample, step_index: int
    ) -> PreprocessingSample:
        """Run one transform, wrapping any failure with clear step context."""
        try:
            transform.validate_input(sample)
            return transform.apply(sample)
        except Exception as exc:
            raise TransformError(
                f"Transform '{transform.name}' (step {step_index + 1}/"
                f"{len(self._transforms)}) failed for patient "
                f"'{sample.patient_id}': {exc}"
            ) from exc

    def __call__(self, sample: PreprocessingSample) -> PreprocessingSample:
        return self.run(sample)

    def __len__(self) -> int:
        return len(self._transforms)

    def __iter__(self) -> Iterator[Transform]:
        return iter(self._transforms)

    def __repr__(self) -> str:
        steps = " -> ".join(transform.name for transform in self._transforms)
        return f"PreprocessingPipeline([{steps}])"

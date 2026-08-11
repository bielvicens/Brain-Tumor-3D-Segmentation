"""Chainable transform architecture for BraTS preprocessing.

This module defines the *contract* between pipeline steps, not any actual
preprocessing math. Two things live here:

- :class:`PreprocessingSample`: the data container that flows through the
  pipeline - one patient's modalities, segmentation and metadata.
- :class:`Transform`: the base class every preprocessing step (future
  normalization, resampling, cropping, padding, ...) must implement.

Concrete transforms are added in a later module as ``Transform``
subclasses; neither the pipeline nor the validation logic need to change
when that happens.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.data.modality import Modality


@dataclass
class PreprocessingSample:
    """A single patient case as it flows through the preprocessing pipeline.

    This is the common data contract between :class:`~src.data.BraTSReader`
    output (Module 1), every :class:`Transform`, and eventually the
    PyTorch ``Dataset``: whatever a transform receives, it must return an
    object shaped exactly like this.

    Attributes:
        patient_id: Identifier carried through for logging/error messages.
        modalities: The MRI volumes for this patient, keyed by modality.
        segmentation: The expert segmentation mask, if available (absent
            at inference time on unlabeled data).
        voxel_spacing: Physical voxel size in mm (x, y, z), if known.
        affine: The NIfTI affine matrix, if known.
        metadata: Free-form dict for transforms to record what they did
            (e.g. ``"original_shape"``, ``"crop_bounds"``) - useful later
            for undoing a transform or for the clinical report, without
            requiring any change to this dataclass.
    """

    patient_id: str
    modalities: Dict[Modality, np.ndarray]
    segmentation: Optional[np.ndarray] = None
    voxel_spacing: Optional[Tuple[float, float, float]] = None
    affine: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes: Any) -> "PreprocessingSample":
        """Return a new sample with the given fields replaced.

        Transforms must not mutate a sample in place (see
        :meth:`Transform.apply`) - this is the supported way to derive a
        new sample from an existing one, e.g.::

            return sample.replace(modalities=normalized_modalities)
        """
        return dataclasses.replace(self, **changes)


class Transform(ABC):
    """Base class for a single preprocessing step.

    Concrete transforms (normalization, resampling, cropping, padding -
    implemented in a later module) subclass this and implement
    :meth:`apply`. A :class:`~src.preprocessing.pipeline.PreprocessingPipeline`
    guarantees transforms run in the order they were registered, each
    receiving the output of the previous one.

    Example:
        A future concrete transform would look like::

            class ZScoreNormalize(Transform):
                def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
                    normalized = {
                        modality: (volume - volume.mean()) / volume.std()
                        for modality, volume in sample.modalities.items()
                    }
                    return sample.replace(modalities=normalized)

        No change to this base class, to ``PreprocessingPipeline``, or to
        ``validation.py`` would be needed to add it.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier, used in logs and error messages."""
        return self.__class__.__name__

    def validate_input(self, sample: PreprocessingSample) -> None:
        """Optional transform-specific pre-condition check.

        Called by the pipeline immediately before :meth:`apply`. The
        default implementation does nothing. Override this when a
        transform needs a guarantee stronger than the pipeline-wide
        validation in :mod:`src.preprocessing.validation` (for example,
        "this transform requires exactly one modality") and raise
        :class:`~src.preprocessing.exceptions.InvalidVolumeError` with a
        clear, transform-specific message if it isn't met.
        """
        return None

    @abstractmethod
    def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
        """Apply this transform and return a new, transformed sample.

        Implementations must NOT mutate ``sample`` in place - they must
        return a new :class:`PreprocessingSample` (typically via
        :meth:`PreprocessingSample.replace`). This keeps transforms free
        of hidden side effects: reusing a transform, retrying a failed
        step, or running transforms out of a testing harness never risks
        corrupting data the caller still holds a reference to.
        """
        raise NotImplementedError

    def __call__(
        self,
        sample: PreprocessingSample,
    ) -> PreprocessingSample:
        self.validate_input(sample)
        return self.apply(sample)

    def __repr__(self) -> str:
        return f"{self.name}()"

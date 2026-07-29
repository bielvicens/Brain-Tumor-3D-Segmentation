"""Tests for src.preprocessing.transforms.

Since Transform is abstract and no concrete transform exists yet (Module
3.1 is architecture-only), tests define small local dummy subclasses just
to exercise the base-class contract - they are not meant to represent
real preprocessing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import Modality
from src.preprocessing.transforms import PreprocessingSample, Transform


def _sample(**overrides) -> PreprocessingSample:
    defaults = dict(
        patient_id="patient-0",
        modalities={Modality.T1N: np.zeros((2, 2, 2), dtype=np.float32)},
    )
    defaults.update(overrides)
    return PreprocessingSample(**defaults)


# ----------------------------------------------------------------------
# PreprocessingSample
# ----------------------------------------------------------------------
def test_sample_replace_returns_new_instance_without_mutating_original() -> None:
    original = _sample()
    new_modalities = {Modality.T1N: np.ones((2, 2, 2), dtype=np.float32)}

    updated = original.replace(modalities=new_modalities)

    assert updated is not original
    assert updated.modalities is new_modalities
    # The original must be untouched.
    assert np.all(original.modalities[Modality.T1N] == 0)
    assert updated.patient_id == original.patient_id


def test_sample_metadata_defaults_to_empty_dict() -> None:
    sample = _sample()
    assert sample.metadata == {}


def test_sample_replace_can_update_metadata() -> None:
    sample = _sample()
    updated = sample.replace(metadata={"original_shape": (2, 2, 2)})
    assert updated.metadata == {"original_shape": (2, 2, 2)}
    assert sample.metadata == {}


# ----------------------------------------------------------------------
# Transform (base class contract)
# ----------------------------------------------------------------------
def test_transform_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Transform()  # type: ignore[abstract]


def test_transform_default_name_is_class_name() -> None:
    class MyTransform(Transform):
        def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
            return sample

    assert MyTransform().name == "MyTransform"


def test_transform_call_delegates_to_apply() -> None:
    class AddOne(Transform):
        def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
            new_modalities = {m: v + 1.0 for m, v in sample.modalities.items()}
            return sample.replace(modalities=new_modalities)

    sample = _sample()
    result = AddOne()(sample)  # uses __call__

    assert np.all(result.modalities[Modality.T1N] == 1.0)
    # __call__ must not mutate the input sample.
    assert np.all(sample.modalities[Modality.T1N] == 0.0)


def test_transform_repr_includes_name() -> None:
    class MyTransform(Transform):
        def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
            return sample

    assert repr(MyTransform()) == "MyTransform()"


def test_transform_default_validate_input_is_a_noop() -> None:
    class MyTransform(Transform):
        def apply(self, sample: PreprocessingSample) -> PreprocessingSample:
            return sample

    # Should not raise for any sample - the default hook does nothing.
    MyTransform().validate_input(_sample())

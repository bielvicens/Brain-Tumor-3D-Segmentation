"""Tests for dataset splitting utilities."""

from __future__ import annotations

import pytest

from src.data.split import train_validation_split


def test_split_returns_lists() -> None:
    patient_ids = [f"patient_{i}" for i in range(10)]

    train_ids, val_ids = train_validation_split(patient_ids)

    assert isinstance(train_ids, list)
    assert isinstance(val_ids, list)


def test_all_patients_are_preserved() -> None:
    patient_ids = [f"patient_{i}" for i in range(20)]

    train_ids, val_ids = train_validation_split(patient_ids)

    assert sorted(train_ids + val_ids) == sorted(patient_ids)


def test_no_duplicate_patients_between_sets() -> None:
    patient_ids = [f"patient_{i}" for i in range(20)]

    train_ids, val_ids = train_validation_split(patient_ids)

    assert set(train_ids).isdisjoint(val_ids)


def test_original_sequence_is_not_modified() -> None:
    patient_ids = [f"patient_{i}" for i in range(10)]
    original = patient_ids.copy()

    train_validation_split(patient_ids)

    assert patient_ids == original


def test_split_is_reproducible_with_same_seed() -> None:
    patient_ids = [f"patient_{i}" for i in range(20)]

    split1 = train_validation_split(patient_ids, seed=42)
    split2 = train_validation_split(patient_ids, seed=42)

    assert split1 == split2


def test_different_seed_changes_split() -> None:
    patient_ids = [f"patient_{i}" for i in range(20)]

    split1 = train_validation_split(patient_ids, seed=1)
    split2 = train_validation_split(patient_ids, seed=2)

    assert split1 != split2


def test_validation_fraction_point_two() -> None:
    patient_ids = [f"patient_{i}" for i in range(10)]

    train_ids, val_ids = train_validation_split(
        patient_ids,
        validation_fraction=0.2,
        seed=0,
    )

    assert len(train_ids) == 8
    assert len(val_ids) == 2


def test_validation_fraction_half() -> None:
    patient_ids = [f"patient_{i}" for i in range(10)]

    train_ids, val_ids = train_validation_split(
        patient_ids,
        validation_fraction=0.5,
    )

    assert len(train_ids) == 5
    assert len(val_ids) == 5


@pytest.mark.parametrize(
    "fraction",
    [
        -0.1,
        0.0,
        1.0,
        1.5,
    ],
)
def test_invalid_validation_fraction_raises(
    fraction: float,
) -> None:
    with pytest.raises(ValueError):
        train_validation_split(
            ["a", "b"],
            validation_fraction=fraction,
        )


def test_duplicate_patient_ids_raise_error() -> None:
    with pytest.raises(ValueError):
        train_validation_split(
            [
                "patient1",
                "patient2",
                "patient1",
            ]
        )


def test_requires_at_least_two_patients() -> None:
    with pytest.raises(ValueError):
        train_validation_split(["patient1"])


def test_non_sequence_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        train_validation_split(123)  # type: ignore[arg-type]


def test_validation_contains_at_least_one_patient() -> None:
    patient_ids = ["a", "b"]

    train_ids, val_ids = train_validation_split(
        patient_ids,
        validation_fraction=0.1,
    )

    assert len(train_ids) == 1
    assert len(val_ids) == 1


def test_training_contains_at_least_one_patient() -> None:
    patient_ids = ["a", "b"]

    train_ids, val_ids = train_validation_split(
        patient_ids,
        validation_fraction=0.9,
    )

    assert len(train_ids) == 1
    assert len(val_ids) == 1


def test_all_ids_are_unique_after_split() -> None:
    patient_ids = [f"id_{i}" for i in range(50)]

    train_ids, val_ids = train_validation_split(
        patient_ids,
        seed=123,
    )

    combined = train_ids + val_ids

    assert len(combined) == len(set(combined))


def test_split_sizes_sum_to_original_size() -> None:
    patient_ids = [f"id_{i}" for i in range(37)]

    train_ids, val_ids = train_validation_split(patient_ids)

    assert len(train_ids) + len(val_ids) == len(patient_ids)


@pytest.mark.parametrize(
    "size",
    [
        5,
        10,
        25,
        100,
    ],
)
def test_split_preserves_number_of_patients(
    size: int,
) -> None:
    patient_ids = [f"id_{i}" for i in range(size)]

    train_ids, val_ids = train_validation_split(patient_ids)

    assert len(train_ids) + len(val_ids) == size


def test_seed_none_is_supported() -> None:
    patient_ids = [f"id_{i}" for i in range(20)]

    train_ids, val_ids = train_validation_split(patient_ids)

    assert len(train_ids) + len(val_ids) == 20


def test_returns_new_lists() -> None:
    patient_ids = [f"id_{i}" for i in range(10)]

    train_ids, val_ids = train_validation_split(patient_ids)

    assert train_ids is not patient_ids
    assert val_ids is not patient_ids


def test_validation_fraction_rounding() -> None:
    patient_ids = [f"id_{i}" for i in range(9)]

    train_ids, val_ids = train_validation_split(
        patient_ids,
        validation_fraction=0.2,
    )

    assert len(train_ids) == 7
    assert len(val_ids) == 2
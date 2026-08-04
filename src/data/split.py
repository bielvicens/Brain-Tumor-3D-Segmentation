"""Utilities for splitting BraTS patient identifiers into train and validation sets."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Optional


def train_validation_split(
    patient_ids: Sequence[str],
    *,
    validation_fraction: float = 0.2,
    seed: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """Split patient identifiers into training and validation subsets.

    The split is random but reproducible when a seed is provided. Every
    patient appears exactly once in one of the returned subsets.

    Args:
        patient_ids:
            Sequence of unique patient identifiers.
        validation_fraction:
            Fraction of patients assigned to the validation set.
            Must satisfy ``0 < validation_fraction < 1``.
        seed:
            Optional random seed for reproducibility.

    Returns:
        Tuple ``(train_ids, validation_ids)``.

    Raises:
        TypeError:
            If ``patient_ids`` is not a sequence.
        ValueError:
            If the validation fraction is invalid, there are duplicated
            identifiers or there are fewer than two patients.
    """
    if not isinstance(patient_ids, Sequence):
        raise TypeError(
            "patient_ids must be a sequence of patient identifiers."
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be strictly between 0 and 1."
        )

    patient_ids = list(patient_ids)

    if len(patient_ids) < 2:
        raise ValueError(
            "At least two patients are required to create a split."
        )

    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError(
            "patient_ids must not contain duplicated identifiers."
        )

    shuffled = patient_ids.copy()

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    validation_size = max(
        1,
        min(
            len(shuffled) - 1,
            round(len(shuffled) * validation_fraction),
        ),
    )

    validation_ids = shuffled[:validation_size]
    train_ids = shuffled[validation_size:]

    return train_ids, validation_ids
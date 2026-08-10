"""Early stopping based on a watched metric.

This module is independent of PyTorch and operates only on Python
numeric values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EarlyStopping:
    """Stop training when a watched metric stops improving.

    The metric can be minimized or maximized depending on ``mode``.

    For this project, when monitoring NCR Dice, use::

        EarlyStopping(
            patience=15,
            min_delta=0.001,
            mode="max",
        )

    This means:
        - higher NCR Dice is better;
        - an improvement must be greater than 0.001;
        - training stops after 15 consecutive epochs without
          sufficient NCR Dice improvement.

    Args:
        patience:
            Number of consecutive epochs without sufficient improvement
            before stopping.

        min_delta:
            Minimum improvement required to reset the patience counter.

        mode:
            ``"min"`` when lower values are better, such as loss.
            ``"max"`` when higher values are better, such as Dice.

    """

    patience: int
    min_delta: float = 0.0
    mode: str = "min"

    _best_score: Optional[float] = field(
        default=None,
        init=False,
    )

    _num_bad_epochs: int = field(
        default=0,
        init=False,
    )

    _should_stop: bool = field(
        default=False,
        init=False,
    )

    def __post_init__(self) -> None:
        self._validate_patience(self.patience)
        self._validate_min_delta(self.min_delta)
        self._validate_mode(self.mode)

        self.min_delta = float(self.min_delta)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_patience(patience: int) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise TypeError(
                f"patience must be an int, "
                f"got {type(patience).__name__}."
            )

        if patience < 0:
            raise ValueError(
                f"patience must be non-negative, got {patience}."
            )

    @staticmethod
    def _validate_min_delta(min_delta: float) -> None:
        if isinstance(min_delta, bool) or not isinstance(
            min_delta,
            (int, float),
        ):
            raise TypeError(
                f"min_delta must be a number, "
                f"got {type(min_delta).__name__}."
            )

        if min_delta < 0:
            raise ValueError(
                f"min_delta must be non-negative, got {min_delta}."
            )

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if not isinstance(mode, str):
            raise TypeError(
                f"mode must be a str, got {type(mode).__name__}."
            )

        if mode not in ("min", "max"):
            raise ValueError(
                f"mode must be 'min' or 'max', got {mode!r}."
            )

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    @property
    def best_score(self) -> Optional[float]:
        """Best metric observed so far."""
        return self._best_score

    @property
    def num_bad_epochs(self) -> int:
        """Number of consecutive epochs without sufficient improvement."""
        return self._num_bad_epochs

    @property
    def should_stop(self) -> bool:
        """Whether early stopping has been triggered."""
        return self._should_stop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, metric: float) -> bool:
        """Record one epoch's metric.

        Returns:
            True if training should stop, otherwise False.
        """

        if isinstance(metric, bool) or not isinstance(
            metric,
            (int, float),
        ):
            raise TypeError(
                f"metric must be a real number, "
                f"got {type(metric).__name__}."
            )

        metric = float(metric)

        if self._should_stop:
            return True

        # First metric is always considered the current best.
        if self._best_score is None:
            self._best_score = metric
            self._num_bad_epochs = 0

            return False

        # Improvement.
        if self._is_improvement(metric):
            self._best_score = metric
            self._num_bad_epochs = 0

            return False

        # No sufficient improvement.
        self._num_bad_epochs += 1

        if self._num_bad_epochs >= self.patience:
            self._should_stop = True

        return self._should_stop

    def reset(self) -> None:
        """Reset the internal state."""
        self._best_score = None
        self._num_bad_epochs = 0
        self._should_stop = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_improvement(self, metric: float) -> bool:
        """Return whether ``metric`` sufficiently improves the best score."""

        if self._best_score is None:
            return True

        if self.mode == "min":
            return metric < self._best_score - self.min_delta

        return metric > self._best_score + self.min_delta

    def __repr__(self) -> str:
        return (
            f"EarlyStopping("
            f"patience={self.patience}, "
            f"min_delta={self.min_delta}, "
            f"mode={self.mode!r}, "
            f"best_score={self._best_score}, "
            f"num_bad_epochs={self._num_bad_epochs}, "
            f"should_stop={self._should_stop}"
            f")"
        )
"""Early stopping: decide when to stop training based on a watched metric.

This module has a single responsibility: given a stream of metric values
(typically a validation loss or score, one per epoch), decide whether
training should stop because the metric has stopped improving. It knows
nothing about a model, a Dataset, or a Trainer - it doesn't save
checkpoints, doesn't load anything, and doesn't log anything. A training
loop is expected to call :meth:`EarlyStopping.step` once per epoch and
check the returned bool.

Deliberately has no dependency on PyTorch (or anything else beyond the
standard library) - it operates purely on plain Python numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EarlyStopping:
    """Stops training when a watched metric stops improving.

    Call :meth:`step` once per epoch with the metric to watch (e.g. a
    validation loss or score). It returns ``True`` once training should
    stop - once ``patience`` consecutive epochs have passed without an
    improvement of more than ``min_delta``.

    Args:
        patience: Number of consecutive non-improving epochs to tolerate
            before stopping. Must be a non-negative integer.
        min_delta: Minimum change required to count as an improvement -
            an epoch only counts as an improvement if it beats the best
            score by *more* than this amount. Must be non-negative.
            Defaults to ``0.0`` (any strict improvement counts).
        mode: ``"min"`` if a lower metric is better (e.g. a loss), or
            ``"max"`` if a higher metric is better (e.g. a Dice score).

    Example (``mode="min"``, a decreasing loss)::

        >>> stopper = EarlyStopping(patience=2, mode="min")
        >>> stopper.step(0.50)  # first value, always "improves"
        False
        >>> stopper.step(0.45)  # improved
        False
        >>> stopper.step(0.43)  # improved
        False

    Example (``mode="max"``, an increasing score)::

        >>> stopper = EarlyStopping(patience=2, mode="max")
        >>> stopper.step(0.70)
        False
        >>> stopper.step(0.73)  # improved
        False
        >>> stopper.step(0.81)  # improved
        False

    Raises:
        TypeError: If any constructor argument has the wrong type.
        ValueError: If ``patience``/``min_delta`` is negative, or ``mode``
            isn't ``"min"``/``"max"``.
    """

    patience: int
    min_delta: float = 0.0
    mode: str = "min"

    _best_score: Optional[float] = field(default=None, init=False)
    _num_bad_epochs: int = field(default=0, init=False)
    _should_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._validate_patience(self.patience)
        self._validate_min_delta(self.min_delta)
        self._validate_mode(self.mode)
        # Normalize to float once, so every later comparison is float-vs-float.
        self.min_delta = float(self.min_delta)

    # ------------------------------------------------------------------
    # Validation (private)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_patience(patience: int) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise TypeError(f"patience must be an int, got {type(patience).__name__}.")
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}.")

    @staticmethod
    def _validate_min_delta(min_delta: float) -> None:
        if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)):
            raise TypeError(f"min_delta must be a number, got {type(min_delta).__name__}.")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}.")

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if not isinstance(mode, str):
            raise TypeError(f"mode must be a str, got {type(mode).__name__}.")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------
    @property
    def best_score(self) -> Optional[float]:
        """The best metric value observed so far, or ``None`` before the
        first call to :meth:`step`."""
        return self._best_score

    @property
    def num_bad_epochs(self) -> int:
        """Consecutive epochs since the last improvement."""
        return self._num_bad_epochs

    @property
    def should_stop(self) -> bool:
        """Whether early stopping has been triggered."""
        return self._should_stop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(self, metric: float) -> bool:
        """Record one epoch's metric and decide whether to stop.

        Args:
            metric: The metric value for the current epoch (e.g. a
                validation loss or score).

        Returns:
            ``True`` if training should stop, ``False`` otherwise. Once
            ``True`` has been returned, every subsequent call also
            returns ``True`` immediately without changing any internal
            state further - calling ``step`` again after stopping is
            safe, but has no effect.

        Raises:
            TypeError: If ``metric`` is not a real number.
        """
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise TypeError(f"metric must be a real number, got {type(metric).__name__}.")
        metric = float(metric)

        if self._should_stop:
            return True

        if self._best_score is None or self._is_improvement(metric):
            self._best_score = metric
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1
            if self._num_bad_epochs >= self.patience:
                self._should_stop = True

        return self._should_stop

    def reset(self) -> None:
        """Reset all internal state, as if the object had just been constructed.

        Configuration (``patience``, ``min_delta``, ``mode``) is left untouched.
        """
        self._best_score = None
        self._num_bad_epochs = 0
        self._should_stop = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _is_improvement(self, metric: float) -> bool:
        """Whether `metric` beats the current best by more than `min_delta`."""
        if self.mode == "min":
            return metric < self._best_score - self.min_delta
        return metric > self._best_score + self.min_delta

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
            f"mode={self.mode!r}, best_score={self._best_score}, "
            f"num_bad_epochs={self._num_bad_epochs}, should_stop={self._should_stop})"
        )

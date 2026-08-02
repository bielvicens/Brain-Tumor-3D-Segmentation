"""Tests for src.utils.early_stopping.EarlyStopping.

Uses only small numeric values - no PyTorch, no filesystem, no other
project module - since EarlyStopping is a pure, standalone utility.
"""

from __future__ import annotations

import pytest

from src.utils.early_stopping import EarlyStopping


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------
def test_constructor_accepts_valid_arguments() -> None:
    stopper = EarlyStopping(patience=5, min_delta=0.01, mode="min")
    assert stopper.patience == 5
    assert stopper.min_delta == 0.01
    assert stopper.mode == "min"


def test_constructor_defaults_min_delta_and_mode() -> None:
    stopper = EarlyStopping(patience=3)
    assert stopper.min_delta == 0.0
    assert stopper.mode == "min"


def test_constructor_accepts_patience_zero() -> None:
    stopper = EarlyStopping(patience=0)
    assert stopper.patience == 0


def test_constructor_accepts_mode_max() -> None:
    stopper = EarlyStopping(patience=3, mode="max")
    assert stopper.mode == "max"


@pytest.mark.parametrize("bad_patience", [-1, -5])
def test_constructor_rejects_negative_patience(bad_patience) -> None:
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=bad_patience)


@pytest.mark.parametrize("bad_patience", [1.5, "3", None, True])
def test_constructor_rejects_non_int_patience(bad_patience) -> None:
    with pytest.raises(TypeError, match="patience"):
        EarlyStopping(patience=bad_patience)


def test_constructor_rejects_negative_min_delta() -> None:
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopping(patience=3, min_delta=-0.1)


def test_constructor_rejects_non_numeric_min_delta() -> None:
    with pytest.raises(TypeError, match="min_delta"):
        EarlyStopping(patience=3, min_delta="0.1")  # type: ignore[arg-type]


def test_constructor_rejects_invalid_mode_string() -> None:
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(patience=3, mode="minimum")


def test_constructor_rejects_non_string_mode() -> None:
    with pytest.raises(TypeError, match="mode"):
        EarlyStopping(patience=3, mode=1)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# First value
# ----------------------------------------------------------------------
def test_first_step_never_triggers_stopping() -> None:
    stopper = EarlyStopping(patience=0, mode="min")
    assert stopper.step(0.5) is False


def test_first_step_sets_best_score() -> None:
    stopper = EarlyStopping(patience=3, mode="min")
    stopper.step(0.5)
    assert stopper.best_score == 0.5


def test_first_step_leaves_num_bad_epochs_at_zero() -> None:
    stopper = EarlyStopping(patience=3, mode="min")
    stopper.step(0.5)
    assert stopper.num_bad_epochs == 0


# ----------------------------------------------------------------------
# Continuous improvement (mode="min", exact example from the spec)
# ----------------------------------------------------------------------
def test_mode_min_continuous_improvement_never_stops() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper.step(0.50) is False
    assert stopper.step(0.45) is False
    assert stopper.step(0.43) is False
    assert stopper.should_stop is False
    assert stopper.num_bad_epochs == 0
    assert stopper.best_score == 0.43


# ----------------------------------------------------------------------
# Continuous improvement (mode="max", exact example from the spec)
# ----------------------------------------------------------------------
def test_mode_max_continuous_improvement_never_stops() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(0.70) is False
    assert stopper.step(0.73) is False
    assert stopper.step(0.81) is False
    assert stopper.should_stop is False
    assert stopper.num_bad_epochs == 0
    assert stopper.best_score == 0.81


# ----------------------------------------------------------------------
# No improvement (plateau)
# ----------------------------------------------------------------------
def test_no_improvement_increments_bad_epochs() -> None:
    stopper = EarlyStopping(patience=5, mode="min")
    stopper.step(0.5)
    stopper.step(0.5)  # no improvement (equal)
    stopper.step(0.6)  # worse
    assert stopper.num_bad_epochs == 2
    assert stopper.best_score == 0.5


def test_improvement_after_bad_epochs_resets_the_counter() -> None:
    stopper = EarlyStopping(patience=5, mode="min")
    stopper.step(0.5)
    stopper.step(0.6)  # bad epoch #1
    stopper.step(0.55)  # bad epoch #2 (still worse than 0.5)
    stopper.step(0.4)  # improvement -> counter resets
    assert stopper.num_bad_epochs == 0
    assert stopper.best_score == 0.4


# ----------------------------------------------------------------------
# Activation after patience is exhausted
# ----------------------------------------------------------------------
def test_stops_exactly_after_patience_bad_epochs() -> None:
    stopper = EarlyStopping(patience=3, mode="min")
    stopper.step(0.5)  # sets best_score, not a "bad" epoch
    assert stopper.step(0.6) is False  # bad epoch #1
    assert stopper.step(0.6) is False  # bad epoch #2
    assert stopper.step(0.6) is True  # bad epoch #3 -> patience reached
    assert stopper.should_stop is True


def test_patience_zero_stops_on_first_bad_epoch() -> None:
    stopper = EarlyStopping(patience=0, mode="min")
    assert stopper.step(0.5) is False  # first value, always "improves"
    assert stopper.step(0.6) is True  # any non-improving epoch stops immediately


def test_does_not_stop_before_patience_is_reached() -> None:
    stopper = EarlyStopping(patience=3, mode="min")
    stopper.step(0.5)
    stopper.step(0.6)
    stopper.step(0.6)
    assert stopper.should_stop is False  # only 2 bad epochs so far, patience=3


# ----------------------------------------------------------------------
# reset()
# ----------------------------------------------------------------------
def test_reset_restores_initial_state() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.step(0.5)
    stopper.step(0.6)
    stopper.step(0.7)  # triggers should_stop

    stopper.reset()

    assert stopper.best_score is None
    assert stopper.num_bad_epochs == 0
    assert stopper.should_stop is False


def test_reset_preserves_configuration() -> None:
    stopper = EarlyStopping(patience=4, min_delta=0.05, mode="max")
    stopper.step(0.5)
    stopper.reset()

    assert stopper.patience == 4
    assert stopper.min_delta == 0.05
    assert stopper.mode == "max"


def test_behavior_after_reset_matches_a_fresh_instance() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.step(0.5)
    stopper.step(0.4)
    stopper.step(0.3)
    stopper.reset()

    fresh = EarlyStopping(patience=2, mode="min")

    # Feed the same sequence into both and expect identical outcomes.
    sequence = [0.9, 0.8, 0.8, 0.8]
    results_after_reset = [stopper.step(value) for value in sequence]
    results_fresh = [fresh.step(value) for value in sequence]
    assert results_after_reset == results_fresh
    assert stopper.best_score == fresh.best_score
    assert stopper.num_bad_epochs == fresh.num_bad_epochs


# ----------------------------------------------------------------------
# min_delta
# ----------------------------------------------------------------------
def test_min_delta_rejects_improvement_smaller_than_threshold() -> None:
    stopper = EarlyStopping(patience=5, min_delta=0.1, mode="min")
    stopper.step(0.50)
    stopper.step(0.45)  # improved by 0.05, less than min_delta=0.1 -> not counted
    assert stopper.num_bad_epochs == 1
    assert stopper.best_score == 0.50


def test_min_delta_accepts_improvement_larger_than_threshold() -> None:
    stopper = EarlyStopping(patience=5, min_delta=0.1, mode="min")
    stopper.step(0.50)
    stopper.step(0.30)  # improved by 0.2, more than min_delta=0.1 -> counted
    assert stopper.num_bad_epochs == 0
    assert stopper.best_score == 0.30


def test_min_delta_boundary_exactly_equal_does_not_count() -> None:
    # Improvement of exactly min_delta must NOT count ("superior a min_delta").
    stopper = EarlyStopping(patience=5, min_delta=0.1, mode="min")
    stopper.step(0.50)
    stopper.step(0.40)  # improved by exactly 0.1
    assert stopper.num_bad_epochs == 1
    assert stopper.best_score == 0.50


def test_min_delta_with_mode_max() -> None:
    stopper = EarlyStopping(patience=5, min_delta=0.05, mode="max")
    stopper.step(0.70)
    stopper.step(0.72)  # improved by 0.02, less than min_delta=0.05 -> not counted
    assert stopper.num_bad_epochs == 1
    stopper.step(0.80)  # improved by 0.10 over 0.70 -> counted
    assert stopper.num_bad_epochs == 0
    assert stopper.best_score == 0.80


# ----------------------------------------------------------------------
# Multiple calls after stopping
# ----------------------------------------------------------------------
def test_step_keeps_returning_true_after_stopping() -> None:
    stopper = EarlyStopping(patience=1, mode="min")
    stopper.step(0.5)
    stopper.step(0.6)  # triggers should_stop
    assert stopper.should_stop is True

    assert stopper.step(0.6) is True
    assert stopper.step(0.1) is True  # even a dramatic "improvement" - still stopped


def test_state_is_frozen_once_stopped() -> None:
    stopper = EarlyStopping(patience=1, mode="min")
    stopper.step(0.5)
    stopper.step(0.6)  # triggers should_stop
    best_before = stopper.best_score
    bad_epochs_before = stopper.num_bad_epochs

    stopper.step(0.0001)  # would be a huge improvement if it were still active
    stopper.step(999.0)

    assert stopper.best_score == best_before
    assert stopper.num_bad_epochs == bad_epochs_before


# ----------------------------------------------------------------------
# Negative metrics
# ----------------------------------------------------------------------
def test_negative_metrics_mode_min() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper.step(-0.5) is False
    assert stopper.step(-0.8) is False  # more negative is "lower" -> improvement
    assert stopper.step(-0.6) is False  # bad epoch #1 (worse than -0.8)
    assert stopper.best_score == -0.8


def test_negative_metrics_mode_max() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(-0.5) is False
    assert stopper.step(-0.2) is False  # less negative is "higher" -> improvement
    assert stopper.best_score == -0.2


# ----------------------------------------------------------------------
# Equal metrics
# ----------------------------------------------------------------------
def test_equal_metrics_count_as_no_improvement() -> None:
    stopper = EarlyStopping(patience=3, mode="min")
    stopper.step(0.5)
    stopper.step(0.5)
    stopper.step(0.5)
    assert stopper.num_bad_epochs == 2
    assert stopper.best_score == 0.5


# ----------------------------------------------------------------------
# Float / int metrics
# ----------------------------------------------------------------------
def test_step_accepts_float_metrics() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper.step(0.123456) is False
    assert isinstance(stopper.best_score, float)


def test_step_accepts_int_metrics_and_stores_as_float() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.step(5)
    assert stopper.best_score == 5.0
    assert isinstance(stopper.best_score, float)


def test_step_rejects_non_numeric_metric() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    with pytest.raises(TypeError, match="metric"):
        stopper.step("0.5")  # type: ignore[arg-type]


def test_step_rejects_bool_metric() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    with pytest.raises(TypeError, match="metric"):
        stopper.step(True)  # type: ignore[arg-type]


def test_step_rejects_none_metric() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    with pytest.raises(TypeError, match="metric"):
        stopper.step(None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Properties: correctness and read-only enforcement
# ----------------------------------------------------------------------
def test_properties_reflect_state_at_each_point_in_time() -> None:
    stopper = EarlyStopping(patience=2, mode="min")

    assert stopper.best_score is None
    assert stopper.num_bad_epochs == 0
    assert stopper.should_stop is False

    stopper.step(0.5)
    assert stopper.best_score == 0.5
    assert stopper.num_bad_epochs == 0
    assert stopper.should_stop is False

    stopper.step(0.6)  # bad epoch #1
    assert stopper.best_score == 0.5
    assert stopper.num_bad_epochs == 1
    assert stopper.should_stop is False

    stopper.step(0.7)  # bad epoch #2 -> patience reached
    assert stopper.best_score == 0.5
    assert stopper.num_bad_epochs == 2
    assert stopper.should_stop is True


def test_best_score_property_is_read_only() -> None:
    stopper = EarlyStopping(patience=2)
    with pytest.raises(AttributeError):
        stopper.best_score = 0.1  # type: ignore[misc]


def test_num_bad_epochs_property_is_read_only() -> None:
    stopper = EarlyStopping(patience=2)
    with pytest.raises(AttributeError):
        stopper.num_bad_epochs = 5  # type: ignore[misc]


def test_should_stop_property_is_read_only() -> None:
    stopper = EarlyStopping(patience=2)
    with pytest.raises(AttributeError):
        stopper.should_stop = True  # type: ignore[misc]


# ----------------------------------------------------------------------
# Independence from PyTorch / other project modules
# ----------------------------------------------------------------------
def test_module_has_no_external_imports() -> None:
    import inspect
    import src.utils.early_stopping as module

    source = inspect.getsource(module)
    import_lines = [
        line for line in source.splitlines() if line.strip().startswith(("import ", "from "))
    ]
    # Only the standard library (dataclasses, typing) should appear.
    for line in import_lines:
        assert "torch" not in line
        assert "src.data" not in line
        assert "src.preprocessing" not in line

"""Tests for src.utils.metrics (Dice / IoU segmentation metrics).

Uses small, fully synthetic torch tensors throughout - these metric
functions are pure and have no dependency on BraTSReader, the
preprocessing pipeline, or any model/Trainer.
"""

from __future__ import annotations

import pytest
import torch

from src.utils.metrics import (
    DEFAULT_SMOOTH,
    dice_per_class,
    dice_score,
    iou_per_class,
    iou_score,
    mean_dice,
    mean_iou,
)

# ----------------------------------------------------------------------
# Hand-crafted multiclass example, worked out manually (see PR description
# / assistant response for the arithmetic), reused across several tests.
#
# target:     [0,0,1, 1,1,2, 2,2,0]
# prediction: [0,1,1, 1,1,2, 2,0,0]
#
# Per class (intersection, pred_count, target_count):
#   class 0: intersection=2, pred=3, target=3 -> dice=4/6,    iou=2/4
#   class 1: intersection=3, pred=4, target=3 -> dice=6/7,    iou=3/4
#   class 2: intersection=2, pred=2, target=3 -> dice=4/5,    iou=2/3
# ----------------------------------------------------------------------
_TARGET = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 0], dtype=torch.long)
_PREDICTION = torch.tensor([0, 1, 1, 1, 1, 2, 2, 0, 0], dtype=torch.long)

_EXPECTED_DICE = [4 / 6, 6 / 7, 4 / 5]
_EXPECTED_IOU = [2 / 4, 3 / 4, 2 / 3]


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
def test_dice_score_rejects_non_tensor_prediction() -> None:
    with pytest.raises(TypeError):
        dice_score([1, 0, 1], torch.tensor([1, 0, 1]))  # type: ignore[arg-type]


def test_dice_score_rejects_non_tensor_target() -> None:
    with pytest.raises(TypeError):
        dice_score(torch.tensor([1, 0, 1]), [1, 0, 1])  # type: ignore[arg-type]


def test_dice_score_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        dice_score(torch.ones(4), torch.ones(5))


def test_dice_score_rejects_empty_tensors() -> None:
    with pytest.raises(ValueError, match="empty"):
        dice_score(torch.empty(0), torch.empty(0))


@pytest.mark.parametrize("bad_class_id", [-1, 1.5, "1", True])
def test_dice_score_rejects_invalid_class_id(bad_class_id) -> None:
    with pytest.raises(ValueError, match="class_id"):
        dice_score(torch.zeros(4), torch.zeros(4), class_id=bad_class_id)


def test_dice_score_rejects_negative_smooth() -> None:
    with pytest.raises(ValueError, match="smooth"):
        dice_score(torch.zeros(4), torch.zeros(4), smooth=-1.0)


def test_dice_score_binary_mode_rejects_non_binary_values() -> None:
    prediction = torch.tensor([0, 1, 2])
    target = torch.tensor([0, 1, 1])
    with pytest.raises(ValueError, match="values 0 and 1"):
        dice_score(prediction, target)


def test_iou_score_shares_the_same_validation() -> None:
    with pytest.raises(ValueError, match="same shape"):
        iou_score(torch.ones(4), torch.ones(5))


@pytest.mark.parametrize("bad_num_classes", [0, -1, 2.5, "3"])
def test_dice_per_class_rejects_invalid_num_classes(bad_num_classes) -> None:
    with pytest.raises(ValueError, match="num_classes"):
        dice_per_class(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), num_classes=bad_num_classes)


def test_mean_dice_exclude_background_with_single_class_raises() -> None:
    with pytest.raises(ValueError, match="background"):
        mean_dice(
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            num_classes=1,
            include_background=False,
        )


def test_mean_iou_exclude_background_with_single_class_raises() -> None:
    with pytest.raises(ValueError, match="background"):
        mean_iou(
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            num_classes=1,
            include_background=False,
        )


# ----------------------------------------------------------------------
# Perfect match
# ----------------------------------------------------------------------
def test_dice_score_perfect_match_binary() -> None:
    mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)
    assert dice_score(mask, mask.clone()).item() == pytest.approx(1.0)


def test_iou_score_perfect_match_binary() -> None:
    mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)
    assert iou_score(mask, mask.clone()).item() == pytest.approx(1.0)


def test_dice_score_perfect_match_multiclass() -> None:
    labels = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    assert dice_score(labels, labels.clone(), class_id=1).item() == pytest.approx(1.0)


def test_iou_score_perfect_match_multiclass() -> None:
    labels = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    assert iou_score(labels, labels.clone(), class_id=2).item() == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Zero overlap
# ----------------------------------------------------------------------
def test_dice_score_zero_overlap_binary() -> None:
    prediction = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
    target = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)
    assert dice_score(prediction, target, smooth=0.0).item() == pytest.approx(0.0)


def test_iou_score_zero_overlap_binary() -> None:
    prediction = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
    target = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)
    assert iou_score(prediction, target, smooth=0.0).item() == pytest.approx(0.0)


def test_dice_score_zero_overlap_is_near_zero_with_default_smooth() -> None:
    prediction = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
    target = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)
    score = dice_score(prediction, target).item()
    assert 0.0 <= score < 1e-3  # not exactly 0 because of the default smooth term


# ----------------------------------------------------------------------
# Absent class (both prediction and target)
# ----------------------------------------------------------------------
def test_dice_score_absent_class_returns_one_with_default_smooth() -> None:
    prediction = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    target = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    assert dice_score(prediction, target, class_id=3).item() == pytest.approx(1.0)


def test_dice_score_absent_class_returns_one_with_zero_smooth() -> None:
    # Exercises the explicit smooth=0 / denominator=0 branch directly.
    prediction = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    target = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    assert dice_score(prediction, target, class_id=3, smooth=0.0).item() == pytest.approx(1.0)


def test_iou_score_absent_class_returns_one_with_zero_smooth() -> None:
    prediction = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    target = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    assert iou_score(prediction, target, class_id=3, smooth=0.0).item() == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Per-class computation, hand-verified values
# ----------------------------------------------------------------------
def test_dice_score_matches_manual_calculation_per_class() -> None:
    for class_id, expected in enumerate(_EXPECTED_DICE):
        score = dice_score(_PREDICTION, _TARGET, class_id=class_id, smooth=0.0).item()
        assert score == pytest.approx(expected, rel=1e-5)


def test_iou_score_matches_manual_calculation_per_class() -> None:
    for class_id, expected in enumerate(_EXPECTED_IOU):
        score = iou_score(_PREDICTION, _TARGET, class_id=class_id, smooth=0.0).item()
        assert score == pytest.approx(expected, rel=1e-5)


def test_dice_per_class_matches_manual_calculation() -> None:
    result = dice_per_class(_PREDICTION, _TARGET, num_classes=3, smooth=0.0)
    assert result.tolist() == pytest.approx(_EXPECTED_DICE, rel=1e-5)


def test_iou_per_class_matches_manual_calculation() -> None:
    result = iou_per_class(_PREDICTION, _TARGET, num_classes=3, smooth=0.0)
    assert result.tolist() == pytest.approx(_EXPECTED_IOU, rel=1e-5)


def test_dice_per_class_shape_and_dtype() -> None:
    result = dice_per_class(_PREDICTION, _TARGET, num_classes=3)
    assert result.shape == (3,)
    assert result.dtype == torch.float32


def test_iou_per_class_shape_and_dtype() -> None:
    result = iou_per_class(_PREDICTION, _TARGET, num_classes=3)
    assert result.shape == (3,)
    assert result.dtype == torch.float32


# ----------------------------------------------------------------------
# Mean metrics, with and without background
# ----------------------------------------------------------------------
def test_mean_dice_includes_background_by_default() -> None:
    expected = sum(_EXPECTED_DICE) / 3
    result = mean_dice(_PREDICTION, _TARGET, num_classes=3, smooth=0.0)
    assert result.item() == pytest.approx(expected, rel=1e-5)


def test_mean_dice_excludes_background_when_requested() -> None:
    expected = sum(_EXPECTED_DICE[1:]) / 2
    result = mean_dice(_PREDICTION, _TARGET, num_classes=3, smooth=0.0, include_background=False)
    assert result.item() == pytest.approx(expected, rel=1e-5)


def test_mean_iou_includes_background_by_default() -> None:
    expected = sum(_EXPECTED_IOU) / 3
    result = mean_iou(_PREDICTION, _TARGET, num_classes=3, smooth=0.0)
    assert result.item() == pytest.approx(expected, rel=1e-5)


def test_mean_iou_excludes_background_when_requested() -> None:
    expected = sum(_EXPECTED_IOU[1:]) / 2
    result = mean_iou(_PREDICTION, _TARGET, num_classes=3, smooth=0.0, include_background=False)
    assert result.item() == pytest.approx(expected, rel=1e-5)


def test_mean_dice_excluding_background_differs_from_including_it() -> None:
    with_bg = mean_dice(_PREDICTION, _TARGET, num_classes=3, smooth=0.0)
    without_bg = mean_dice(_PREDICTION, _TARGET, num_classes=3, smooth=0.0, include_background=False)
    assert with_bg.item() != pytest.approx(without_bg.item())


# ----------------------------------------------------------------------
# Return type / dtype
# ----------------------------------------------------------------------
def test_dice_score_returns_a_tensor() -> None:
    result = dice_score(torch.ones(4), torch.ones(4))
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.ndim == 0


def test_iou_score_returns_a_tensor() -> None:
    result = iou_score(torch.ones(4), torch.ones(4))
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.ndim == 0


def test_mean_dice_returns_a_scalar_tensor() -> None:
    result = mean_dice(_PREDICTION, _TARGET, num_classes=3)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert result.dtype == torch.float32


def test_mean_iou_returns_a_scalar_tensor() -> None:
    result = mean_iou(_PREDICTION, _TARGET, num_classes=3)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert result.dtype == torch.float32


def test_metrics_use_float32_regardless_of_input_dtype() -> None:
    # int64 label maps are the typical case (e.g. BraTSDataset's mask
    # tensor), but the computation itself must happen in float32.
    prediction = torch.tensor([0, 1, 1], dtype=torch.int64)
    target = torch.tensor([0, 1, 0], dtype=torch.int64)
    result = dice_score(prediction, target, class_id=1)
    assert result.dtype == torch.float32


# ----------------------------------------------------------------------
# Multiclass 3D tensors (the actual target use case: segmentation volumes)
# ----------------------------------------------------------------------
def test_metrics_work_with_3d_volumes() -> None:
    target = torch.zeros((4, 4, 4), dtype=torch.long)
    target[1:3, 1:3, 1:3] = 1
    prediction = target.clone()
    prediction[1, 1, 1] = 0  # one voxel disagreement

    dice = dice_score(prediction, target, class_id=1)
    assert 0.0 < dice.item() < 1.0

    per_class = dice_per_class(prediction, target, num_classes=2)
    assert per_class.shape == (2,)


def test_metrics_work_with_batched_4d_tensors() -> None:
    # (B, D, H, W): treated as one flattened set of voxels (a "global"
    # score), not averaged per batch item.
    target = torch.zeros((2, 4, 4, 4), dtype=torch.long)
    target[:, 1:3, 1:3, 1:3] = 1
    prediction = target.clone()

    result = mean_dice(prediction, target, num_classes=2)
    assert result.item() == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Binary masks: bool tensors accepted directly
# ----------------------------------------------------------------------
def test_dice_score_accepts_boolean_tensors() -> None:
    prediction = torch.tensor([True, False, True, True])
    target = torch.tensor([True, False, True, False])
    result = dice_score(prediction, target)
    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0


# ----------------------------------------------------------------------
# smooth is configurable and changes the result away from the edge cases
# ----------------------------------------------------------------------
def test_smooth_parameter_is_configurable_and_affects_result() -> None:
    prediction = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    target = torch.tensor([1, 0, 0, 0], dtype=torch.float32)

    score_small_smooth = dice_score(prediction, target, smooth=1e-6).item()
    score_large_smooth = dice_score(prediction, target, smooth=10.0).item()

    assert score_small_smooth != pytest.approx(score_large_smooth)

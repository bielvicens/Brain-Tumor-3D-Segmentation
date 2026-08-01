"""Segmentation evaluation metrics: Dice and IoU for 3D multiclass masks.

These functions are pure and framework-agnostic beyond depending on
PyTorch tensors: no Trainer, model, or dataset dependency, so they can be
used identically in a training loop, a standalone evaluation script, or a
notebook.

Two calling conventions are supported by every function that accepts
``class_id``:

- **Binary masks** (``class_id=None``): ``prediction`` and ``target`` are
  already binary (values in ``{0, 1}``, or boolean) - the metric is
  computed directly on them.
- **Multiclass label maps** (``class_id=<int>``): ``prediction`` and
  ``target`` hold integer class labels (e.g. the argmax of a model's
  output over the BraTS classes); the metric first extracts the binary
  mask for ``class_id`` from each (``tensor == class_id``), then computes
  the metric on those two binary masks.

``prediction`` and ``target`` may have any matching shape - a single
volume ``(D, H, W)``, a batch ``(B, D, H, W)``, or anything else. Every
element is treated as one flattened set of voxels for the intersection /
union sums: this is a single "global" score, not averaged per batch item.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

#: Default Laplace-smoothing term added to numerator and denominator to
#: avoid division by zero. Kept small (rather than e.g. 1.0, sometimes
#: used for training losses) so it never meaningfully biases a reported
#: evaluation metric, even for a small region of interest.
DEFAULT_SMOOTH = 1e-6


# ----------------------------------------------------------------------
# Validation helpers (private)
# ----------------------------------------------------------------------
def _validate_tensor_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    """Check that prediction/target are tensors of the same, non-empty shape."""
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(f"prediction must be a torch.Tensor, got {type(prediction).__name__}.")
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"target must be a torch.Tensor, got {type(target).__name__}.")
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.numel() == 0:
        raise ValueError("prediction and target must not be empty tensors.")


def _validate_class_id(class_id: Optional[int]) -> None:
    """Check that class_id, if given, is a non-negative integer."""
    if class_id is None:
        return
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
        raise ValueError(f"class_id must be a non-negative integer or None, got {class_id!r}.")


def _validate_smooth(smooth: float) -> None:
    """Check that smooth is non-negative."""
    if smooth < 0:
        raise ValueError(f"smooth must be non-negative, got {smooth}.")


def _validate_num_classes(num_classes: int) -> None:
    """Check that num_classes is a positive integer."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}.")


def _validate_binary_values(tensor: torch.Tensor, name: str) -> None:
    """Check that `tensor` contains only 0/1 (or boolean) values."""
    if not torch.all((tensor == 0) | (tensor == 1)):
        raise ValueError(
            f"{name} must contain only values 0 and 1 when class_id is None "
            "(binary mode). Pass class_id to extract a specific class from a "
            "multiclass label map instead."
        )


# ----------------------------------------------------------------------
# Core computation helpers (private)
# ----------------------------------------------------------------------
def _binarize_pair(
    prediction: torch.Tensor, target: torch.Tensor, class_id: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce prediction/target to a matching pair of float32 binary masks.

    If `class_id` is given, extracts that class from each tensor (assumed
    to be an integer label map) via equality comparison. Otherwise,
    validates that both tensors are already binary and casts them to
    float32.
    """
    if class_id is not None:
        pred_bin = (prediction == class_id).to(torch.float32)
        target_bin = (target == class_id).to(torch.float32)
    else:
        _validate_binary_values(prediction, "prediction")
        _validate_binary_values(target, "target")
        pred_bin = prediction.to(torch.float32)
        target_bin = target.to(torch.float32)
    return pred_bin, target_bin


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, smooth: float) -> torch.Tensor:
    """Compute ``(numerator + smooth) / (denominator + smooth)`` as float32.

    Explicitly handles the ``smooth=0`` and ``denominator=0`` edge case -
    this only happens when a class is absent from both prediction and
    target, which by convention counts as perfect agreement (returns
    ``1.0``) rather than the NaN a literal ``0/0`` division would produce.
    With the default ``smooth > 0``, the same case already resolves to
    ``1.0`` naturally through the formula - this branch only matters when
    the caller explicitly passes ``smooth=0``.
    """
    zero = torch.zeros((), dtype=torch.float32, device=denominator.device)
    if smooth == 0.0 and torch.isclose(denominator.to(torch.float32), zero):
        return torch.tensor(1.0, dtype=torch.float32, device=numerator.device)
    smooth_t = torch.as_tensor(smooth, dtype=torch.float32, device=numerator.device)
    return (numerator.to(torch.float32) + smooth_t) / (denominator.to(torch.float32) + smooth_t)


# ----------------------------------------------------------------------
# Public metrics
# ----------------------------------------------------------------------
def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: Optional[int] = None,
    smooth: float = DEFAULT_SMOOTH,
) -> torch.Tensor:
    """Dice similarity coefficient: ``2 * |A n B| / (|A| + |B|)``.

    Args:
        prediction: Predicted mask. Binary (``{0, 1}``) if ``class_id`` is
            ``None``, otherwise an integer label map.
        target: Ground-truth mask, same shape and calling convention as
            ``prediction``.
        class_id: If given, the class to extract from both tensors (via
            ``tensor == class_id``) before computing Dice. If ``None``,
            ``prediction``/``target`` are used directly as binary masks.
        smooth: Additive smoothing term (see module docstring /
            :data:`DEFAULT_SMOOTH`).

    Returns:
        A 0-dim ``float32`` tensor with the Dice score, in ``[0, 1]``.
        ``1.0`` both for a perfect match and for a class absent from both
        prediction and target (they agree there is nothing there).

    Raises:
        TypeError: If ``prediction`` or ``target`` is not a ``torch.Tensor``.
        ValueError: If shapes don't match, either tensor is empty,
            ``class_id`` is invalid, ``smooth`` is negative, or (in binary
            mode) either tensor contains values other than 0/1.
    """
    _validate_tensor_pair(prediction, target)
    _validate_class_id(class_id)
    _validate_smooth(smooth)

    pred_bin, target_bin = _binarize_pair(prediction, target, class_id)
    intersection = torch.sum(pred_bin * target_bin)
    denominator = torch.sum(pred_bin) + torch.sum(target_bin)

    if class_id is not None and denominator == 0:
        logger.debug(
            "Class %d is absent from both prediction and target; dice_score "
            "returns 1.0 by convention.",
            class_id,
        )

    return _safe_ratio(2.0 * intersection, denominator, smooth)


def iou_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: Optional[int] = None,
    smooth: float = DEFAULT_SMOOTH,
) -> torch.Tensor:
    """Intersection-over-Union (Jaccard index): ``|A n B| / |A u B|``.

    Args:
        prediction: Predicted mask. Binary (``{0, 1}``) if ``class_id`` is
            ``None``, otherwise an integer label map.
        target: Ground-truth mask, same shape and calling convention as
            ``prediction``.
        class_id: If given, the class to extract from both tensors (via
            ``tensor == class_id``) before computing IoU. If ``None``,
            ``prediction``/``target`` are used directly as binary masks.
        smooth: Additive smoothing term (see module docstring /
            :data:`DEFAULT_SMOOTH`).

    Returns:
        A 0-dim ``float32`` tensor with the IoU score, in ``[0, 1]``.
        ``1.0`` both for a perfect match and for a class absent from both
        prediction and target.

    Raises:
        TypeError: If ``prediction`` or ``target`` is not a ``torch.Tensor``.
        ValueError: If shapes don't match, either tensor is empty,
            ``class_id`` is invalid, ``smooth`` is negative, or (in binary
            mode) either tensor contains values other than 0/1.
    """
    _validate_tensor_pair(prediction, target)
    _validate_class_id(class_id)
    _validate_smooth(smooth)

    pred_bin, target_bin = _binarize_pair(prediction, target, class_id)
    intersection = torch.sum(pred_bin * target_bin)
    union = torch.sum(pred_bin) + torch.sum(target_bin) - intersection

    if class_id is not None and union == 0:
        logger.debug(
            "Class %d is absent from both prediction and target; iou_score "
            "returns 1.0 by convention.",
            class_id,
        )

    return _safe_ratio(intersection, union, smooth)


def dice_per_class(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = DEFAULT_SMOOTH,
) -> torch.Tensor:
    """Dice score computed independently for every class.

    Args:
        prediction: Integer label map (e.g. the argmax of a model's output).
        target: Ground-truth integer label map, same shape as ``prediction``.
        num_classes: Number of classes, evaluated as ``class_id in range(num_classes)``.
        smooth: Additive smoothing term, passed to each per-class :func:`dice_score` call.

    Returns:
        A 1D ``float32`` tensor of shape ``(num_classes,)``; index ``c``
        holds ``dice_score(prediction, target, class_id=c, smooth=smooth)``.

    Raises:
        TypeError: If ``prediction`` or ``target`` is not a ``torch.Tensor``.
        ValueError: If shapes don't match, either tensor is empty,
            ``num_classes`` is not a positive integer, or ``smooth`` is negative.
    """
    _validate_tensor_pair(prediction, target)
    _validate_num_classes(num_classes)
    _validate_smooth(smooth)

    scores = [
        dice_score(prediction, target, class_id=class_id, smooth=smooth)
        for class_id in range(num_classes)
    ]
    return torch.stack(scores)


def iou_per_class(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = DEFAULT_SMOOTH,
) -> torch.Tensor:
    """IoU score computed independently for every class.

    Args:
        prediction: Integer label map (e.g. the argmax of a model's output).
        target: Ground-truth integer label map, same shape as ``prediction``.
        num_classes: Number of classes, evaluated as ``class_id in range(num_classes)``.
        smooth: Additive smoothing term, passed to each per-class :func:`iou_score` call.

    Returns:
        A 1D ``float32`` tensor of shape ``(num_classes,)``; index ``c``
        holds ``iou_score(prediction, target, class_id=c, smooth=smooth)``.

    Raises:
        TypeError: If ``prediction`` or ``target`` is not a ``torch.Tensor``.
        ValueError: If shapes don't match, either tensor is empty,
            ``num_classes`` is not a positive integer, or ``smooth`` is negative.
    """
    _validate_tensor_pair(prediction, target)
    _validate_num_classes(num_classes)
    _validate_smooth(smooth)

    scores = [
        iou_score(prediction, target, class_id=class_id, smooth=smooth)
        for class_id in range(num_classes)
    ]
    return torch.stack(scores)


def _drop_background_if_needed(
    per_class_scores: torch.Tensor, include_background: bool, num_classes: int
) -> torch.Tensor:
    """Slice off index 0 (background) from a per-class score tensor, if requested."""
    if include_background:
        return per_class_scores
    if num_classes <= 1:
        raise ValueError(
            "Cannot exclude the background class (class_id=0) when "
            f"num_classes={num_classes} - there would be no classes left to average."
        )
    return per_class_scores[1:]


def mean_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = DEFAULT_SMOOTH,
    include_background: bool = True,
) -> torch.Tensor:
    """Mean Dice score across classes, optionally excluding background.

    Args:
        prediction: Integer label map (e.g. the argmax of a model's output).
        target: Ground-truth integer label map, same shape as ``prediction``.
        num_classes: Number of classes, evaluated as ``class_id in range(num_classes)``.
        smooth: Additive smoothing term, passed to each per-class :func:`dice_score` call.
        include_background: If ``False``, class 0 is excluded from the
            average (common in medical segmentation, where background
            dominates the volume and would otherwise inflate the score).

    Returns:
        A 0-dim ``float32`` tensor with the mean Dice score.

    Raises:
        TypeError: If ``prediction`` or ``target`` is not a ``torch.Tensor``.
        ValueError: If shapes don't match, ``num_classes`` is invalid,
            ``smooth`` is negative, or ``include_background=False`` with
            ``num_classes <= 1`` (nothing left to average).
    """
    per_class = dice_per_class(prediction, target, num_classes=num_classes, smooth=smooth)
    per_class = _drop_background_if_needed(per_class, include_background, num_classes)
    return torch.mean(per_class)


def mean_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = DEFAULT_SMOOTH,
    include_background: bool = True,
) -> torch.Tensor:
    """Mean IoU score across classes, optionally excluding background.

    Args:
        prediction: Integer label map (e.g. the argmax of a model's output).
        target: Ground-truth integer label map, same shape as ``prediction``.
        num_classes: Number of classes, evaluated as ``class_id in range(num_classes)``.
        smooth: Additive smoothing term, passed to each per-class :func:`iou_score` call.
        include_background: If ``False``, class 0 is excluded from the average.

    Returns:
        A 0-dim ``float32`` tensor with the mean IoU score.

    Raises:
        TypeError: If ``prediction`` or ``target`` is not a ``torch.Tensor``.
        ValueError: If shapes don't match, ``num_classes`` is invalid,
            ``smooth`` is negative, or ``include_background=False`` with
            ``num_classes <= 1`` (nothing left to average).
    """
    per_class = iou_per_class(prediction, target, num_classes=num_classes, smooth=smooth)
    per_class = _drop_background_if_needed(per_class, include_background, num_classes)
    return torch.mean(per_class)

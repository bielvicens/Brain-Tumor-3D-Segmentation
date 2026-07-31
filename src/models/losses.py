"""Loss functions for 3D multi-class medical image segmentation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss.

    Args:
        smooth: Numerical stabilizer added to numerator and denominator.
        ignore_index: Optional target class to exclude from the Dice
            computation.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError("smooth must be strictly positive.")

        if ignore_index is not None and not isinstance(ignore_index, int):
            raise TypeError("ignore_index must be an integer or None.")

        self.smooth = float(smooth)
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute soft multi-class Dice loss.

        Args:
            logits: Tensor with shape ``(N, C, D, H, W)``.
            target: Integer tensor with shape ``(N, D, H, W)``.

        Returns:
            Scalar Dice loss.
        """
        _validate_segmentation_inputs(logits, target)

        num_classes = logits.shape[1]

        if target.numel() > 0:
            min_label = int(target.min().item())
            max_label = int(target.max().item())

            if min_label < 0 or max_label >= num_classes:
                raise ValueError(
                    f"Target labels must be in [0, {num_classes - 1}], "
                    f"got [{min_label}, {max_label}]."
                )

        probabilities = F.softmax(logits, dim=1)

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=num_classes,
        )

        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).to(
            dtype=probabilities.dtype
        )

        if self.ignore_index is not None:
            if not 0 <= self.ignore_index < num_classes:
                raise ValueError(
                    f"ignore_index must be in [0, {num_classes - 1}], "
                    f"got {self.ignore_index}."
                )

            keep = [
                index
                for index in range(num_classes)
                if index != self.ignore_index
            ]

            probabilities = probabilities[:, keep]
            target_one_hot = target_one_hot[:, keep]

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            probabilities.shape[1],
            -1,
        )

        target_one_hot = target_one_hot.reshape(
            target_one_hot.shape[0],
            target_one_hot.shape[1],
            -1,
        )

        intersection = (probabilities * target_one_hot).sum(dim=2)
        denominator = (
            probabilities.sum(dim=2) + target_one_hot.sum(dim=2)
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class CrossEntropySegmentationLoss(nn.Module):
    """Cross-entropy loss for multi-class 3D segmentation."""

    def __init__(self, ignore_index: Optional[int] = None) -> None:
        super().__init__()

        self.ignore_index = (
            -100 if ignore_index is None else int(ignore_index)
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute cross-entropy segmentation loss."""
        _validate_segmentation_inputs(logits, target)

        num_classes = logits.shape[1]

        if target.numel() > 0:
            min_label = int(target.min().item())
            max_label = int(target.max().item())

            if min_label < 0 or max_label >= num_classes:
                raise ValueError(
                    f"Target labels must be in [0, {num_classes - 1}], "
                    f"got [{min_label}, {max_label}]."
                )

        return F.cross_entropy(
            logits,
            target.long(),
            ignore_index=self.ignore_index,
        )


class DiceCrossEntropyLoss(nn.Module):
    """Weighted combination of Dice and cross-entropy losses.

    ``loss = dice_weight * dice_loss + ce_weight * cross_entropy``

    Args:
        dice_weight: Weight applied to Dice loss.
        ce_weight: Weight applied to cross-entropy loss.
        smooth: Numerical stabilizer used by DiceLoss.
        ignore_index: Optional class excluded from both losses.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()

        if dice_weight < 0:
            raise ValueError("dice_weight must be non-negative.")

        if ce_weight < 0:
            raise ValueError("ce_weight must be non-negative.")

        if dice_weight == 0 and ce_weight == 0:
            raise ValueError(
                "At least one of dice_weight or ce_weight must be positive."
            )

        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)

        self.dice_loss = DiceLoss(
            smooth=smooth,
            ignore_index=ignore_index,
        )

        self.cross_entropy_loss = CrossEntropySegmentationLoss(
            ignore_index=ignore_index,
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the weighted Dice + cross-entropy loss."""
        dice = self.dice_loss(logits, target)
        cross_entropy = self.cross_entropy_loss(logits, target)

        return (
            self.dice_weight * dice
            + self.ce_weight * cross_entropy
        )


def _validate_segmentation_inputs(
    logits: Tensor,
    target: Tensor,
) -> None:
    """Validate the common input contract for segmentation losses."""
    if not isinstance(logits, Tensor):
        raise TypeError(
            f"logits must be a torch.Tensor, got {type(logits).__name__}."
        )

    if not isinstance(target, Tensor):
        raise TypeError(
            f"target must be a torch.Tensor, got {type(target).__name__}."
        )

    if logits.ndim != 5:
        raise ValueError(
            "logits must have shape (N, C, D, H, W), "
            f"got {tuple(logits.shape)}."
        )

    if target.ndim != 4:
        raise ValueError(
            "target must have shape (N, D, H, W), "
            f"got {tuple(target.shape)}."
        )

    expected_shape = (
        logits.shape[0],
        logits.shape[2],
        logits.shape[3],
        logits.shape[4],
    )

    if tuple(target.shape) != expected_shape:
        raise ValueError(
            "Target spatial shape must match logits: "
            f"expected {expected_shape}, got {tuple(target.shape)}."
        )

    if not torch.is_floating_point(logits):
        raise TypeError("logits must use a floating-point dtype.")

    if target.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("target must contain integer class labels.")
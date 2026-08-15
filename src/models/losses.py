"""Loss functions for 3D multi-class medical image segmentation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Weighted multi-class soft Dice loss focused on tumor classes.

    Background is excluded from the Dice loss by default because the main
    objective is accurate tumor segmentation.

    Args:
        smooth:
            Numerical stabilizer added to numerator and denominator.

        ignore_index:
            Optional target class to exclude from the Dice computation.

        class_weights:
            Optional weights for each class. The tensor must contain one
            weight per class.

            For BraTS:
                0 = Background
                1 = NCR
                2 = ED
                3 = ET

        include_background:
            Whether to include background in the Dice loss.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None,
        class_weights: Optional[Tensor] = None,
        include_background: bool = False,
    ) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError(
                "smooth must be strictly positive."
            )

        if (
            ignore_index is not None
            and not isinstance(ignore_index, int)
        ):
            raise TypeError(
                "ignore_index must be an integer or None."
            )

        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.include_background = include_background

        if class_weights is not None:

            if not isinstance(class_weights, Tensor):
                raise TypeError(
                    "class_weights must be a torch.Tensor or None."
                )

            if class_weights.ndim != 1:
                raise ValueError(
                    "class_weights must be a 1D tensor."
                )

            if torch.any(class_weights < 0):
                raise ValueError(
                    "class_weights must contain non-negative values."
                )

            self.register_buffer(
                "class_weights",
                class_weights.float(),
            )

        else:
            self.class_weights = None

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute weighted soft Dice loss."""

        _validate_segmentation_inputs(
            logits,
            target,
        )

        num_classes = logits.shape[1]

        # ---------------------------------------------------------
        # Validate target labels
        # ---------------------------------------------------------

        if target.numel() > 0:

            min_label = int(
                target.min().item()
            )

            max_label = int(
                target.max().item()
            )

            if min_label < 0 or max_label >= num_classes:
                raise ValueError(
                    f"Target labels must be in [0, {num_classes - 1}], "
                    f"got [{min_label}, {max_label}]."
                )

        # ---------------------------------------------------------
        # Validate class weights
        # ---------------------------------------------------------

        if (
            self.class_weights is not None
            and len(self.class_weights) != num_classes
        ):
            raise ValueError(
                "class_weights must contain exactly one weight "
                f"per class. Expected {num_classes}, "
                f"got {len(self.class_weights)}."
            )

        # ---------------------------------------------------------
        # Softmax probabilities
        # ---------------------------------------------------------

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        # ---------------------------------------------------------
        # One-hot target
        # ---------------------------------------------------------

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=num_classes,
        )

        target_one_hot = target_one_hot.permute(
            0,
            4,
            1,
            2,
            3,
        ).to(
            dtype=probabilities.dtype
        )

        # ---------------------------------------------------------
        # Select classes
        # ---------------------------------------------------------

        class_indices = list(
            range(num_classes)
        )

        # Exclude background by default.
        if not self.include_background:

            if 0 in class_indices:
                class_indices.remove(0)

        # Exclude ignore_index if requested.
        if self.ignore_index is not None:

            if not 0 <= self.ignore_index < num_classes:
                raise ValueError(
                    f"ignore_index must be in [0, {num_classes - 1}], "
                    f"got {self.ignore_index}."
                )

            if self.ignore_index in class_indices:
                class_indices.remove(
                    self.ignore_index
                )

        if not class_indices:
            raise ValueError(
                "No classes remain for Dice loss computation."
            )

        probabilities = probabilities[
            :,
            class_indices,
        ]

        target_one_hot = target_one_hot[
            :,
            class_indices,
        ]

        # ---------------------------------------------------------
        # Flatten spatial dimensions
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Dice per class
        # ---------------------------------------------------------

        intersection = (
            probabilities * target_one_hot
        ).sum(dim=2)

        denominator = (
            probabilities.sum(dim=2)
            + target_one_hot.sum(dim=2)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        # Average over batch first.
        dice = dice.mean(dim=0)

        # ---------------------------------------------------------
        # Class weighting
        # ---------------------------------------------------------

        if self.class_weights is not None:

            weights = self.class_weights[
                class_indices
            ].to(
                device=dice.device,
                dtype=dice.dtype,
            )

            weight_sum = weights.sum()

            if weight_sum <= 0:
                raise ValueError(
                    "Dice class weights must sum to a positive value."
                )

            loss = (
                (1.0 - dice) * weights
            ).sum() / weight_sum

        else:

            loss = (
                1.0 - dice
            ).mean()

        return loss


class CrossEntropySegmentationLoss(nn.Module):
    """Weighted cross-entropy loss for multi-class 3D segmentation."""

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        class_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.ignore_index = (
            -100 if ignore_index is None else int(ignore_index)
        )

        if class_weights is not None:
            if not isinstance(class_weights, Tensor):
                raise TypeError(
                    "class_weights must be a torch.Tensor or None."
                )

            if class_weights.ndim != 1:
                raise ValueError(
                    "class_weights must be a 1D tensor."
                )

            if torch.any(class_weights < 0):
                raise ValueError(
                    "class_weights must contain non-negative values."
                )

            self.register_buffer(
                "class_weights",
                class_weights.float(),
            )
        else:
            self.class_weights = None

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute weighted cross-entropy segmentation loss."""

        _validate_segmentation_inputs(
            logits,
            target,
        )

        num_classes = logits.shape[1]

        if target.numel() > 0:
            min_label = int(target.min().item())
            max_label = int(target.max().item())

            if min_label < 0 or max_label >= num_classes:
                raise ValueError(
                    f"Target labels must be in [0, {num_classes - 1}], "
                    f"got [{min_label}, {max_label}]."
                )

        if (
            self.class_weights is not None
            and len(self.class_weights) != num_classes
        ):
            raise ValueError(
                "class_weights must contain exactly one weight "
                f"per class. Expected {num_classes}, "
                f"got {len(self.class_weights)}."
            )

        return F.cross_entropy(
            logits,
            target.long(),
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )


class DiceCrossEntropyLoss(nn.Module):
    """Weighted combination of tumor-focused Dice and cross-entropy."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None,
        class_weights: Optional[Tensor] = None,
        include_background: bool = False,
    ) -> None:
        super().__init__()

        if dice_weight < 0:
            raise ValueError(
                "dice_weight must be non-negative."
            )

        if ce_weight < 0:
            raise ValueError(
                "ce_weight must be non-negative."
            )

        if dice_weight == 0 and ce_weight == 0:
            raise ValueError(
                "At least one of dice_weight or ce_weight "
                "must be positive."
            )

        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)

        self.dice_loss = DiceLoss(
            smooth=smooth,
            ignore_index=ignore_index,
            class_weights=class_weights,
            include_background=include_background,
        )

        self.cross_entropy_loss = CrossEntropySegmentationLoss(
            ignore_index=ignore_index,
            class_weights=class_weights,
        )

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute weighted Dice + cross-entropy loss."""

        dice = self.dice_loss(
            logits,
            target,
        )

        cross_entropy = self.cross_entropy_loss(
            logits,
            target,
        )

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
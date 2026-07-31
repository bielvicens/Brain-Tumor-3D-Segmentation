import pytest
import torch

from src.models.losses import (
    CrossEntropySegmentationLoss,
    DiceCrossEntropyLoss,
    DiceLoss,
)


def _make_logits(
    batch: int = 2,
    classes: int = 3,
    depth: int = 4,
    height: int = 4,
    width: int = 4,
) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(
        batch,
        classes,
        depth,
        height,
        width,
        requires_grad=True,
    )


def _make_target(
    batch: int = 2,
    classes: int = 3,
    depth: int = 4,
    height: int = 4,
    width: int = 4,
) -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randint(
        0,
        classes,
        (batch, depth, height, width),
    )


def test_dice_loss_returns_scalar() -> None:
    logits = _make_logits()
    target = _make_target()

    loss = DiceLoss()(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dice_loss_is_zero_for_perfect_prediction() -> None:
    target = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    logits = torch.full((1, 2, 2, 2, 2), -20.0)

    logits[:, 0] = 20.0

    loss = DiceLoss()(logits, target)

    assert loss.item() < 1e-5


def test_dice_loss_is_higher_for_bad_prediction() -> None:
    target = torch.zeros(1, 2, 2, 2, dtype=torch.long)

    good_logits = torch.full((1, 2, 2, 2, 2), -20.0)
    good_logits[:, 0] = 20.0

    bad_logits = torch.full((1, 2, 2, 2, 2), -20.0)
    bad_logits[:, 1] = 20.0

    good_loss = DiceLoss()(good_logits, target)
    bad_loss = DiceLoss()(bad_logits, target)

    assert bad_loss > good_loss


def test_dice_loss_supports_multiclass_segmentation() -> None:
    logits = _make_logits(classes=4)
    target = _make_target(classes=4)

    loss = DiceLoss()(logits, target)

    assert torch.isfinite(loss)


def test_dice_loss_supports_ignore_index() -> None:
    logits = _make_logits(classes=3)
    target = _make_target(classes=3)

    loss = DiceLoss(ignore_index=0)(logits, target)

    assert torch.isfinite(loss)


def test_dice_loss_rejects_invalid_logits_rank() -> None:
    logits = torch.randn(1, 3, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.long)

    with pytest.raises(ValueError, match="logits must have shape"):
        DiceLoss()(logits, target)


def test_dice_loss_rejects_invalid_target_rank() -> None:
    logits = _make_logits(batch=1)
    target = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)

    with pytest.raises(ValueError, match="target must have shape"):
        DiceLoss()(logits, target)


def test_dice_loss_rejects_shape_mismatch() -> None:
    logits = _make_logits(batch=1)
    target = torch.zeros(1, 3, 4, 4, dtype=torch.long)

    with pytest.raises(ValueError, match="spatial shape"):
        DiceLoss()(logits, target)


def test_dice_loss_rejects_invalid_target_labels() -> None:
    logits = _make_logits(batch=1, classes=3)
    target = torch.full((1, 4, 4, 4), 3, dtype=torch.long)

    with pytest.raises(ValueError, match="Target labels"):
        DiceLoss()(logits, target)


def test_dice_loss_backward_works() -> None:
    logits = _make_logits()
    target = _make_target()

    loss = DiceLoss()(logits, target)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_cross_entropy_returns_scalar() -> None:
    logits = _make_logits()
    target = _make_target()

    loss = CrossEntropySegmentationLoss()(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_cross_entropy_backward_works() -> None:
    logits = _make_logits()
    target = _make_target()

    loss = CrossEntropySegmentationLoss()(logits, target)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_cross_entropy_rejects_invalid_labels() -> None:
    logits = _make_logits(batch=1, classes=2)
    target = torch.full((1, 4, 4, 4), 2, dtype=torch.long)

    with pytest.raises(ValueError, match="Target labels"):
        CrossEntropySegmentationLoss()(logits, target)


def test_combined_loss_matches_weighted_components() -> None:
    logits = _make_logits()
    target = _make_target()

    dice = DiceLoss()(logits, target)
    ce = CrossEntropySegmentationLoss()(logits, target)

    combined = DiceCrossEntropyLoss(
        dice_weight=2.0,
        ce_weight=3.0,
    )(logits, target)

    expected = 2.0 * dice + 3.0 * ce

    assert torch.allclose(combined, expected)


def test_combined_loss_backward_works() -> None:
    logits = _make_logits()
    target = _make_target()

    loss = DiceCrossEntropyLoss()(logits, target)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dice_rejects_non_positive_smooth() -> None:
    with pytest.raises(ValueError, match="smooth"):
        DiceLoss(smooth=0)


def test_combined_loss_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="dice_weight"):
        DiceCrossEntropyLoss(dice_weight=-1)


def test_combined_loss_requires_positive_total_weight() -> None:
    with pytest.raises(ValueError, match="At least one"):
        DiceCrossEntropyLoss(
            dice_weight=0,
            ce_weight=0,
        )
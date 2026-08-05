"""Evaluate a trained BraTS 3D U-Net model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.builders import (
    build_datasets,
    build_model,
)
from src.utils import (
    ProjectConfig,
    load_checkpoint,
)
from src.utils.metrics import (
    mean_dice,
    mean_iou,
)
from src.inference import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained segmentation model.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate.",
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate(
    predictor: Predictor,
    dataloader: DataLoader,
    num_classes: int,
) -> tuple[float, float]:

    predictor.model.eval()

    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    for images, masks in dataloader:

        images = images.to(
            predictor.device,
            dtype=torch.float32,
        )

        masks = masks.to(
            predictor.device,
            dtype=torch.long,
        )

        predictions = predictor.predict_mask(images)

        dice = mean_dice(
            predictions,
            masks,
            num_classes=num_classes,
            include_background=False,
        )

        iou = mean_iou(
            predictions,
            masks,
            num_classes=num_classes,
            include_background=False,
        )

        total_dice += float(dice.item())
        total_iou += float(iou.item())
        num_batches += 1

    return (
        total_dice / num_batches,
        total_iou / num_batches,
    )


def main() -> None:

    args = parse_args()

    config = ProjectConfig()

    dataset = build_datasets(
        config=config,
        split="validation",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    model = build_model(config)

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else Path(config.checkpoint.directory)
        / config.experiment.name
        / config.checkpoint.best_model_name
    )

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        map_location=config.training.device,
    )

    predictor = Predictor(
        model=model,
        device=config.training.device,
    )

    dice, iou = evaluate(
        predictor=predictor,
        dataloader=dataloader,
        num_classes=4,
    )

    print("=" * 40)
    print(f"Mean Dice : {dice:.4f}")
    print(f"Mean IoU  : {iou:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()